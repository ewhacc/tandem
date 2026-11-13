# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2 helper GPU: the CPU engine's missed routes computed on a second GPU.

VLLM_MOE_W2_CPU_HELPER=<device index> (with VLLM_MOE_W2_CPU=1) keeps the
CPU engine's protocol (moe_w2_cpu: pinned request block, doorbell, ack) and
replaces its compute. Every base layer's 2-bit slot rows are copied to the
helper GPU at arm time (the same pinned rows the CPU engine reads; ~73 GiB
for DS4-Flash). Per (layer, token bucket) one CUDA graph on the helper
reads the pinned block, computes the masked routes (non-masked routes get
weight 0 and a dummy expert), and writes y back to the pinned block. The
engine's master thread launches the graph when the doorbell rings and
queues the ack as a stream write after it, so no host thread waits.

Warm-set mode (VLLM_MOE_W2_HELPER_GB=<GiB>): the helper holds a FIXED slice
of the package ranking (VLLM_MOE_W2_BASE_RAM_RANK, global greedy order) and
runs next to the CPU engine instead of replacing it. The slice skips the
first VLLM_MOE_W2_HELPER_SKIP experts (default: the main GPU pool's slot
count, the main cache's likely working set) and takes as many as fit in
HELPER_GB. Per decode layer the main GPU's misses split three ways: helper
experts to the helper (own pinned mask, y and doorbell), the rest to the
CPU engine (and its DIRECT link share). Vector-code planes (VLLM_MOE_W2_VQ)
decode on the helper with moe_w2_vq.
"""

import ctypes
import os

import torch
import triton

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEV = os.getenv("VLLM_MOE_W2_CPU_HELPER", "").strip()
BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


def enabled() -> bool:
    return _DEV != ""


def warm_gb() -> float:
    return float(os.getenv("VLLM_MOE_W2_HELPER_GB", "0") or 0)


def split_mode() -> bool:
    """VLLM_MOE_W2_HELPER_SPLIT=alt: the helper OWNS every other expert of
    the ranking's top 2n (ranks 2, 4, ...; n = what fits in HELPER_GB) and
    computes all decode routes to them, resident or not; the main GPU drops
    those routes (moe_w2_ep-style remap) and caches only the others, so the
    two GPUs split the hot set evenly and never hold the same expert.
    Default (unset): the helper takes only the main GPU's misses on a
    fixed slice after the main cache's share."""
    return os.getenv("VLLM_MOE_W2_HELPER_SPLIT", "") in ("alt", "dyn")


def dynamic() -> bool:
    """VLLM_MOE_W2_HELPER_SPLIT=dyn: alt split as the start layout, then
    the helper is a cache of the experts the main GPU does not serve,
    exclusive with it (moe_w2_cpu._DynHelper swaps by decayed counts of the
    routes the main GPU missed)."""
    return os.getenv("VLLM_MOE_W2_HELPER_SPLIT", "") == "dyn"


def remap(own: torch.Tensor, fb: torch.Tensor, topk_weights: torch.Tensor,
          topk_ids: torch.Tensor):
    """Split mode, main GPU side, from this call's ownership snapshot `own`
    ([E] i32, nonzero = helper's): -> (weights with the helper's routes
    zeroed, ids with them pointed at the layer's fallback main expert `fb`,
    expert_map with the helper's experts -1). Same contract as
    moe_w2_ep.remap."""
    E = own.shape[0]
    ar = torch.arange(E, dtype=torch.int32, device=own.device)
    emap = torch.where(own == 0, ar, torch.full_like(ar, -1))
    mine = own[topk_ids.long()] == 0
    w = torch.where(mine, topk_weights, torch.zeros_like(topk_weights))
    i = torch.where(mine, topk_ids, fb.to(topk_ids.dtype))
    return w, i, emap


def _rank_entries(keys, n_experts: int) -> list:
    path = os.getenv("VLLM_MOE_W2_BASE_RAM_RANK", "")
    if not path or not os.path.exists(path):
        raise RuntimeError("VLLM_MOE_W2_HELPER_GB needs the package ranking "
                           "(VLLM_MOE_W2_BASE_RAM_RANK)")
    ks = set(keys)
    ent = []
    with open(path) as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            li, _rk, ei, c = ln.split()[:4]
            li, ei = int(li), int(ei)
            if li in ks and 0 <= ei < n_experts:
                ent.append((-int(c), li, ei))
    ent.sort()
    return ent


def ranking(keys, n_experts: int) -> dict:
    """Layer key -> its experts, most-routed first (package ranking)."""
    out = {k: [] for k in keys}
    for _c, li, ei in _rank_entries(keys, n_experts):
        out[li].append(ei)
    return out


def warm_sets(keys, n_experts: int, slot_bytes: int, skip: int) -> list:
    """Per layer key (in `keys` order) the helper's expert ids: the global
    ranking (tsv `layer rank expert count`, count desc) minus its first
    `skip` entries, as many as fit in VLLM_MOE_W2_HELPER_GB."""
    ent = _rank_entries(keys, n_experts)
    n = int(warm_gb() * 2**30 // slot_bytes)
    take = ent[1:2 * n:2] if split_mode() else ent[skip:skip + n]
    sets = {k: [] for k in keys}
    for _c, li, ei in take:
        sets[li].append(ei)
    return [sorted(sets[k]) for k in keys]


class Helper:
    """GPU-resident experts + one captured graph per (layer, bucket)."""

    def __init__(self, S, host, slot_bytes: int, off_s13: int, off_c2: int,
                 off_s2: int, H: int, I: int, E: int, kcap: int, tcap: int,
                 limit, sets=None, fetch=None, msk=None, y=None, vq=None,
                 pool_rows=0):
        """sets (warm-set mode): per layer the expert ids held here, filled
        through fetch(i, ids) -> host row tensors; msk / y: the helper's own
        pinned mask and output blocks (default S.msk / S.y). vq: the raw
        e4m3 pattern table (numpy u8) for vector-code planes. pool_rows > 0
        (dynamic mode): one row pool shared by all layers, hmap[i] maps a
        route id to its pool row (pool_rows = not held); rows change at run
        time through load_row / publish / unpublish."""
        from vllm.model_executor.layers.quantization.utils import (
            moe_w2_cubit, moe_w2_triton)
        # vLLM tracks the current stream itself (patched set_stream); the
        # helper's capture context would leave it on the helper's stream
        from vllm.utils import torch_utils
        tls = torch_utils._current_stream_tls
        prev_stream = getattr(tls, "value", None)
        self.S, self.H, self.I, self.E = S, H, I, E
        self.msk_h = S.msk if msk is None else msk
        self.y_h = S.y if y is None else y
        self.sets = sets
        self.kcap, self.limit = kcap, limit
        self.slot_bytes = slot_bytes
        self.offs = (0, off_s13, off_c2, off_s2)
        self.dev = torch.device(f"cuda:{int(_DEV)}")
        self.buckets = [b for b in BUCKETS if b <= tcap]
        if self.buckets[-1] != tcap:
            self.buckets.append(tcap)
        with torch.cuda.device(self.dev):
            self.planes = []
            # per layer: route id -> row here (n_l = dummy), desc-build
            # residency (dummy reads as FP4-resident: its pairs get m=0)
            self.hmap, self.noslots, self.nrows = [], [], []
            self.pool_rows = pool_rows
            if pool_rows:
                N = pool_rows
                pl = torch.empty((N, slot_bytes), dtype=torch.uint8,
                                 device=self.dev)
                self.row_of = {}          # (layer idx, expert) -> pool row
                r = 0
                for i, ids in enumerate(sets):
                    for j in range(0, len(ids), 32):
                        rows = fetch(i, ids[j:j + 32])
                        for k, row in enumerate(rows):
                            pl[r].copy_(row)
                            self.row_of[(i, ids[j + k])] = r
                            r += 1
                self.free_rows = list(range(N - 1, r - 1, -1))
                self.pool = pl
                for i in range(len(sets)):
                    self.planes.append(pl.view(-1))
                    self.nrows.append(N)
            elif sets is None:
                for p in host:
                    buf = (ctypes.c_uint8 * (E * slot_bytes)).from_address(p)
                    self.planes.append(torch.frombuffer(buf, dtype=torch.uint8)
                                       .to(self.dev))
                    self.nrows.append(E)
            else:
                for i, ids in enumerate(sets):
                    n = len(ids)
                    pl = torch.empty((max(n, 1), slot_bytes), dtype=torch.uint8,
                                     device=self.dev)
                    for j in range(0, n, 32):
                        rows = fetch(i, ids[j:j + 32])
                        for r, row in enumerate(rows):
                            pl[j + r].copy_(row)
                    self.planes.append(pl.view(-1))
                    self.nrows.append(n)
            for n in self.nrows:
                hm = torch.full((E + 1,), n, dtype=torch.int32)
                if pool_rows:
                    li = len(self.hmap)
                    for e in sets[li]:
                        hm[e] = self.row_of[(li, e)]
                elif sets is None:
                    hm[:E] = torch.arange(E, dtype=torch.int32)
                else:
                    ids = sets[len(self.hmap)]
                    if ids:
                        hm[torch.tensor(ids)] = torch.arange(
                            n, dtype=torch.int32)
                self.hmap.append(hm.to(self.dev))
                if pool_rows and self.noslots:
                    self.noslots.append(self.noslots[0])
                    continue
                ns = torch.full((n + 1,), -1, dtype=torch.int32)
                ns[n] = 0
                self.noslots.append(ns.to(self.dev))
            self.fill_stream = torch.cuda.Stream(device=self.dev)
            self.native = False
            if vq is not None:
                from vllm.model_executor.layers.quantization.utils import (
                    moe_w2_vq)
                P = vq.size // 8
                self.vq_tab = torch.from_numpy(vq.copy()).view(P, 8).to(self.dev)
                cap = torch.cuda.get_device_capability(self.dev)
                dot = os.getenv("VLLM_MOE_W2_MMA_DOT", "auto").lower()
                if dot not in ("e4m3", "f16"):
                    dot = "e4m3" if cap == (8, 9) or cap[0] == 12 else "f16"
                vfns = moe_w2_vq.make_launchers((H, I), P, tiers=("vq",),
                                                dot=dot)
                worst = moe_w2_vq.self_test(vfns, P)
                self.fns = {}
                for (_t, K), fn in vfns.items():
                    self.fns[("w2", K)] = (
                        lambda d, n_rows, pairs, stream, f=fn:
                        f(d, self.vq_tab, n_rows, pairs, stream))
                self.native = True
                self.kind = f"VQ {dot}, parity {max(worst.values()):.1e}"
            else:
                self.fns = moe_w2_triton.make_launchers(
                    (H, I), fp8=(moe_w2_triton.dot_mode() == "fp8"))
                self.kind = "Triton"
            if vq is None and os.getenv("VLLM_MOE_W2_MMA", "1") == "1":
                from vllm.model_executor.layers.quantization.utils import (
                    moe_w2_mma)
                if moe_w2_mma.supported():
                    # NVRTC decode kernel loaded into this device's context
                    self.fns.update(moe_w2_mma.make_launchers(
                        (H, I), fallback=self.fns))
                    self.native = True
                    self.kind = "mma.sync"
            self.xd = torch.zeros((tcap, H), dtype=torch.bfloat16,
                                  device=self.dev)
            self.ids = torch.zeros(tcap * kcap, dtype=torch.int32,
                                   device=self.dev)
            self.w = torch.zeros(tcap * kcap, dtype=torch.float32,
                                 device=self.dev)
            self.msk = torch.zeros(tcap * kcap, dtype=torch.int32,
                                   device=self.dev)
            self.stream = torch.cuda.Stream(device=self.dev)
            self.cubit = moe_w2_cubit
            # eager warmup (Triton JIT) on every bucket, then capture
            for b in self.buckets:
                self._body(0, b)
            torch.cuda.synchronize(self.dev)
            pool = torch.cuda.graph_pool_handle()
            self.graphs = []
            for i in range(len(self.planes)):
                for b in self.buckets:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool, stream=self.stream):
                        self._body(i, b)
                    self.graphs.append(g)
            torch.cuda.synchronize(self.dev)
            cu = ctypes.CDLL("libcuda.so.1")
            ctx = ctypes.c_void_p()
            cu.cuCtxGetCurrent(ctypes.byref(ctx))
            self.ctx = ctx.value
        tls.value = prev_stream
        self.execs = [g.raw_cuda_graph_exec() for g in self.graphs]
        # token count -> bucket index
        self.bucket_of = [0] * (tcap + 1)
        j = 0
        for t in range(1, tcap + 1):
            while self.buckets[j] < t:
                j += 1
            self.bucket_of[t] = j
        logger.info("moe_w2 helper GPU %s armed (%s decode): %d layers, "
                    "%d experts (%.1f GiB of rows%s), %d graphs (buckets %s)",
                    self.dev, self.kind, len(self.planes), self._held(),
                    self._held() * slot_bytes / 2**30,
                    "" if sets is None else (
                        f", dynamic pool of {pool_rows}" if pool_rows
                        else ", fixed warm set"),
                    len(self.graphs), self.buckets)

    def _held(self) -> int:
        return len(self.row_of) if self.pool_rows else sum(self.nrows)

    def load_row(self, row: int, src: torch.Tensor) -> None:
        """Copy one host row into a pool row nothing maps (dynamic mode)."""
        with torch.cuda.device(self.dev), torch.cuda.stream(self.fill_stream):
            self.pool[row].copy_(src, non_blocking=True)

    def sync_loads(self) -> None:
        self.fill_stream.synchronize()

    def publish(self, i: int, e: int, row: int) -> None:
        """Map route (layer i, expert e) to its loaded pool row; call after
        sync_loads and before the main side marks e as the helper's."""
        self.row_of[(i, e)] = row
        with torch.cuda.device(self.dev), torch.cuda.stream(self.fill_stream):
            self.hmap[i][e:e + 1].fill_(row)
        self.fill_stream.synchronize()

    def unpublish(self, i: int, e: int) -> int:
        """Unmap (layer i, expert e) -> its row, now free. Only once no
        in-flight step can still route e here."""
        row = self.row_of.pop((i, e))
        with torch.cuda.device(self.dev), torch.cuda.stream(self.fill_stream):
            self.hmap[i][e:e + 1].fill_(self.pool_rows)
        self.fill_stream.synchronize()
        return row

    def _body(self, i: int, T: int) -> None:
        S, kc, E = self.S, self.kcap, self.E
        n = T * kc
        self.xd[:T].copy_(S.x[i, :T], non_blocking=True)
        self.ids[:n].copy_(S.ids[i, :n], non_blocking=True)
        self.w[:n].copy_(S.w[i, :n], non_blocking=True)
        self.msk[:n].copy_(self.msk_h[i, :n], non_blocking=True)
        on = self.msk[:n] != 0
        nl = self.nrows[i]
        w = torch.where(on, self.w[:n], 0.0).view(T, kc)
        if self.pool_rows:
            # align on expert ids (moe_align takes < 1024 experts), then
            # point each block at its pool row
            ids = torch.where(on, self.ids[:n].clamp(0, E), E).view(T, kc)
            y = self._forward(self.xd[:T], w, ids, self.planes[i], E,
                              self.noslots[i], rowmap=self.hmap[i], nrow=nl)
        else:
            loc = self.hmap[i][self.ids[:n].clamp(0, E).long()]
            ids = torch.where(on, loc, nl).view(T, kc)
            y = self._forward(self.xd[:T], w, ids, self.planes[i], nl,
                              self.noslots[i])
        self.y_h[i, :T].copy_(y, non_blocking=True)

    def _forward(self, x, w, ids, rows, E, noslot, rowmap=None, nrow=None):
        """Decode moe_w2 over slot-layout rows [E * slot_bytes]; routes with
        ids == E (not this engine's) are skipped. rowmap ([E + 1] i32, E ->
        nrow): rows are pool rows [nrow * slot_bytes], looked up per block."""
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size)
        cb, H, I, kc = self.cubit, self.H, self.I, self.kcap
        T = x.shape[0]
        sorted_ids, expert_blocks, num_post = moe_align_block_size(
            ids, 4, E + 1)
        slots = sorted_ids.numel()
        pairs = slots // 4
        dev = x.device
        xq = torch.zeros((T + 1, H), dtype=torch.float8_e4m3fn, device=dev)
        xs = torch.zeros((T + 1, H // 32), dtype=torch.float32, device=dev)
        cb._quant_a32(x, xq[:T], xs[:T], cb._G1)
        valid = sorted_ids < T * kc
        r = torch.where(valid, sorted_ids // kc,
                        torch.full_like(sorted_ids, T))
        a1 = xq.index_select(0, r)
        as1 = xs.index_select(0, r)
        c13 = torch.empty((slots, 2 * I), dtype=torch.bfloat16, device=dev)
        act = torch.empty((slots, I), dtype=torch.bfloat16, device=dev)
        a2 = torch.empty((slots, I), dtype=torch.float8_e4m3fn, device=dev)
        as2 = torch.empty((slots, I // 32), dtype=torch.float32, device=dev)
        c2 = torch.empty((slots, H), dtype=torch.bfloat16, device=dev)
        d = torch.empty((4, pairs, 6), dtype=torch.int64, device=dev)
        b = rows.data_ptr()
        sb = self.slot_bytes
        o = self.offs
        n_rows = E + 1
        if rowmap is not None:
            expert_blocks = rowmap[expert_blocks.clamp(0, E).long()]
            n_rows = nrow + 1
        cb._desc_build_kernel[(triton.cdiv(pairs, 256),)](
            expert_blocks, num_post, noslot, d,
            a1.data_ptr(), as1.data_ptr(), c13.data_ptr(),
            a2.data_ptr(), as2.data_ptr(), c2.data_ptr(),
            b + o[0], b + o[1], b + o[2], b + o[3], a1.data_ptr(),
            sb, sb, sb, sb, 0, 0,
            H, (H // 32) * 4, 4 * I, I, (I // 32) * 4, 2 * H,
            n_rows, pairs, pairs * 6, 4, BLOCK=256)
        self.fns[("w2", H)](d[0], 2 * I, pairs, None)
        if self.limit is None:
            torch.ops._C.silu_and_mul(act, c13)
        elif cb._SWIGLU_CLAMP_FP32:
            cb._silu_and_mul_clamp_fp32(act, c13, self.limit)
        else:
            torch.ops._C.silu_and_mul_with_clamp(act, c13, self.limit, 1.0,
                                                 0.0)
        cb._quant_a32(act, a2, as2, cb._G2)
        self.fns[("w2", I)](d[1], H, pairs, None)
        route = sorted_ids.clamp(max=T * kc - 1)
        ws = w.reshape(-1)[route]
        # dummy-expert rows were never computed (garbage): drop them by
        # destination, not by weight
        valid = valid & (ids.reshape(-1)[route] != E)
        ws = torch.where(valid, ws, torch.zeros_like(ws))
        dst = torch.where(valid, sorted_ids,
                          torch.full_like(sorted_ids, T * kc)).long()
        gath = torch.zeros((T * kc + 1, H), dtype=torch.float32, device=dev)
        gath.index_copy_(0, dst, c2.float() * ws.unsqueeze(1))
        return gath[:T * kc].view(T, kc, H).sum(dim=1)
