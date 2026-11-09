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


class Helper:
    """GPU-resident experts + one captured graph per (layer, bucket)."""

    def __init__(self, S, host, slot_bytes: int, off_s13: int, off_c2: int,
                 off_s2: int, H: int, I: int, E: int, kcap: int, tcap: int,
                 limit):
        from vllm.model_executor.layers.quantization.utils import (
            moe_w2_cubit, moe_w2_triton)
        self.S, self.H, self.I, self.E = S, H, I, E
        self.kcap, self.limit = kcap, limit
        self.slot_bytes = slot_bytes
        self.offs = (0, off_s13, off_c2, off_s2)
        self.dev = torch.device(f"cuda:{int(_DEV)}")
        self.buckets = [b for b in BUCKETS if b <= tcap]
        if self.buckets[-1] != tcap:
            self.buckets.append(tcap)
        with torch.cuda.device(self.dev):
            self.planes = []
            for p in host:
                buf = (ctypes.c_uint8 * (E * slot_bytes)).from_address(p)
                self.planes.append(torch.frombuffer(buf, dtype=torch.uint8)
                                   .to(self.dev))
            self.fns = moe_w2_triton.make_launchers(
                (H, I), fp8=(moe_w2_triton.dot_mode() == "fp8"))
            self.native = False
            if os.getenv("VLLM_MOE_W2_MMA", "1") == "1":
                from vllm.model_executor.layers.quantization.utils import (
                    moe_w2_mma)
                if moe_w2_mma.supported():
                    # NVRTC decode kernel loaded into this device's context
                    self.fns.update(moe_w2_mma.make_launchers(
                        (H, I), fallback=self.fns))
                    self.native = True
            # dummy expert E reads as FP4-resident: its pairs get m=0 in the
            # 2-bit tables and the (never launched) FP4 tier takes them
            self.noslot = torch.full((E + 1,), -1, dtype=torch.int32,
                                     device=self.dev)
            self.noslot[E] = 0
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
            for i in range(len(host)):
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
        self.execs = [g.raw_cuda_graph_exec() for g in self.graphs]
        # token count -> bucket index
        self.bucket_of = [0] * (tcap + 1)
        j = 0
        for t in range(1, tcap + 1):
            while self.buckets[j] < t:
                j += 1
            self.bucket_of[t] = j
        logger.info("moe_w2 helper GPU %s armed (%s decode): %d layers "
                    "(%.1f GiB of 2-bit rows), %d graphs (buckets %s)",
                    self.dev, "mma.sync" if self.native else "Triton",
                    len(host), len(host) * E * slot_bytes / 2**30,
                    len(self.graphs), self.buckets)

    def _body(self, i: int, T: int) -> None:
        S, kc, E = self.S, self.kcap, self.E
        n = T * kc
        self.xd[:T].copy_(S.x[i, :T], non_blocking=True)
        self.ids[:n].copy_(S.ids[i, :n], non_blocking=True)
        self.w[:n].copy_(S.w[i, :n], non_blocking=True)
        self.msk[:n].copy_(S.msk[i, :n], non_blocking=True)
        on = self.msk[:n] != 0
        ids = torch.where(on, self.ids[:n], E).view(T, kc)
        w = torch.where(on, self.w[:n], 0.0).view(T, kc)
        y = self._forward(self.xd[:T], w, ids, self.planes[i])
        S.y[i, :T].copy_(y, non_blocking=True)

    def _forward(self, x, w, ids, rows):
        """Decode moe_w2 over slot-layout rows [E * slot_bytes]; routes with
        ids == E (not missed) are skipped."""
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size)
        cb, H, I, E, kc = self.cubit, self.H, self.I, self.E, self.kcap
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
        cb._desc_build_kernel[(triton.cdiv(pairs, 256),)](
            expert_blocks, num_post, self.noslot, d,
            a1.data_ptr(), as1.data_ptr(), c13.data_ptr(),
            a2.data_ptr(), as2.data_ptr(), c2.data_ptr(),
            b + o[0], b + o[1], b + o[2], b + o[3], a1.data_ptr(),
            sb, sb, sb, sb, 0, 0,
            H, (H // 32) * 4, 4 * I, I, (I // 32) * 4, 2 * H,
            E + 1, pairs, pairs * 6, 4, BLOCK=256)
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
