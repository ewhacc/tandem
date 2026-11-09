# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2: missed experts computed on the CPU (DIRECT base cache, decode).

With VLLM_MOE_W2_CPU=1 a decode MoE layer no longer reads its non-resident
experts over PCIe. The descriptor build zeroes those pairs (the non-DIRECT
miss path), and the graph hands them to a CPU engine (moe_w2_cpu.cpp, built
with g++ at first use) through pinned memory:

  request  x (bf16), routing and a per-route CPU mask go to the layer's
           pinned block, then a doorbell word (seq * 128 + layer) is rung.
  CPU      a master thread + spinning workers compute the masked pairs from
           the same pinned 2-bit host rows the DIRECT path reads (AVX-512
           VNNI or AVX2, int8 activations per 32-group), write y (f32) and
           ack seq.
  join     after the layer's unpermute, one GPU thread polls the ack (with
           a timeout), then y is added to the layer output.

The GPU computes the resident experts while the CPU computes the misses.
Results are not bit-identical to the GPU path (CPU activation format and
summation order). Needs AVX-512 VNNI + VBMI, else AVX2 + FMA
(VLLM_MOE_W2_CPU_ISA=avx2 forces the AVX2 kernels). With VLLM_MOE_W2_VQ
the rows hold vector codes and the engine decodes them (AVX-512: one
8-weight table gather per group, ~0.55x the 2-bit kernel's rate at full
load on Genoa; AVX2: four 8-byte loads per row and k32, 0.87x on a 16c
Zen3, 0.48x on an 8c Zen2). Env:
VLLM_MOE_W2_CPU_THREADS (default: physical cores minus cores/16, at
least 1 and at most 4 held back: 8 cores -> 7, 64 -> 60).
"""

import ctypes
import hashlib
import os
import subprocess

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

_ON = (os.getenv("VLLM_MOE_W2_CPU", "0") == "1"
       and os.getenv("VLLM_MOE_W2_BASE_DIRECT", "0") == "1")
_TCAP = 32          # tokens per layer job; = TMAX in moe_w2_cpu.cpp (8 reqs x 4 rows)
_KCAP = 8           # routes per token in the pinned blocks
_HDR = 16
_HINT_MAX = 128     # predicted ids per hint (LOOKA T cap 8 x top-K <= 16)
_SPIN = 20_000_000  # ack polls before the GPU gives up (error counted)
# CPU/PCIe split: per layer, up to this many missed experts (those routed by
# the most tokens first) are read over PCIe by the DIRECT path; the rest go
# to the CPU. 0 = every miss on the CPU.
_PCIE = int(os.getenv("VLLM_MOE_W2_CPU_PCIE", "0"))
# load-adaptive split: round(FRAC x missed routes) experts per layer go over
# PCIe (capped by _PCIE when set). The CPU's cost grows with routes, PCIe's
# with distinct experts, so batched steps shift work to the link. 0.10 was
# best at concurrency 1..8 on Genoa (0 experts at c1, ~2 per layer at c8).
_FRAC = float(os.getenv("VLLM_MOE_W2_CPU_PCIE_FRAC", "0.10"))
_SPLIT = _PCIE > 0 or _FRAC > 0


@triton.jit
def _req_kernel(x_ptr, ids_ptr, w_ptr, slot_row_ptr, fslot_row_ptr, cpu_row_ptr,
                hx_addr, hids_addr, hw_addr, hmsk_addr, H, T, top_k, n_experts,
                n_pcie, fracq, rmap_ptr, HAS_F: tl.constexpr, SPLIT: tl.constexpr,
                ROWMAP: tl.constexpr,
                KCAP: tl.constexpr, R: tl.constexpr, EP: tl.constexpr,
                BLOCK: tl.constexpr):
    """grid (T, cdiv(H, BLOCK)): copy x row t into the layer's pinned
    block; program (0, 0) also writes every route and the CPU mask (routes
    resident in neither GPU pool). SPLIT: the missed experts routed by the
    most tokens stay on the DIRECT path, min(n_pcie, round(fracq / 65536 x
    missed routes)) of them; the CPU's experts are flagged in cpu_row for
    the descriptor build. ROWMAP (planes files): only pinned experts
    (rmap_ptr[e] >= 0) can go over the link; cold ones stay on the CPU."""
    t = tl.program_id(0)
    hb = tl.program_id(1)
    h = hb * BLOCK + tl.arange(0, BLOCK)
    m = h < H
    hx = hx_addr.to(tl.pointer_type(tl.int16))
    tl.store(hx + t * H + h, tl.load(x_ptr + t * H + h, mask=m), mask=m)
    if (t == 0) & (hb == 0):
        r = tl.arange(0, R)
        rt = r // KCAP
        k = r % KCAP
        v = (rt < T) & (k < top_k)
        e = tl.load(ids_ptr + rt * top_k + k, mask=v, other=0)
        # capture warmups route garbage: only real expert ids go anywhere
        v = v & (e >= 0) & (e < n_experts)
        e = tl.where(v, e, 0)
        res = tl.load(slot_row_ptr + e, mask=v, other=0) >= 0
        if HAS_F:
            res = res | (tl.load(fslot_row_ptr + e, mask=v, other=0) >= 0)
        miss = v & (res == 0)
        if SPLIT:
            # tokens per missed expert; rank distinct experts by
            # (count desc, id asc); the first n_pcie stay DIRECT
            same = (e[:, None] == e[None, :]) & miss[:, None] & miss[None, :]
            cnt = tl.sum(same.to(tl.int32), axis=1)
            if ROWMAP:
                link_ok = miss & (tl.load(rmap_ptr + e, mask=v, other=-1) >= 0)
            else:
                link_ok = miss
            first = link_ok & (tl.sum((same & (r[None, :] < r[:, None])).to(tl.int32),
                                      axis=1) == 0)
            ahead = (first[None, :] & ((cnt[None, :] > cnt[:, None])
                     | ((cnt[None, :] == cnt[:, None]) & (e[None, :] < e[:, None]))))
            rank = tl.sum(ahead.to(tl.int32), axis=1)
            n_miss = tl.sum(miss.to(tl.int32), axis=0)
            n_link = tl.minimum(n_pcie, (n_miss * fracq + 32768) >> 16)
            cpu = miss & ~(link_ok & (rank < n_link))
            ex = tl.arange(0, EP)
            tl.store(cpu_row_ptr + ex, tl.zeros((EP,), tl.int32), mask=ex < n_experts)
            tl.debug_barrier()
            tl.store(cpu_row_ptr + e, tl.full((R,), 1, tl.int32), mask=cpu)
        else:
            cpu = miss
        hi = hids_addr.to(tl.pointer_type(tl.int32))
        hw = hw_addr.to(tl.pointer_type(tl.float32))
        hm = hmsk_addr.to(tl.pointer_type(tl.int32))
        tl.store(hi + r, e.to(tl.int32))
        tl.store(hw + r, tl.load(w_ptr + rt * top_k + k, mask=v, other=0.0).to(tl.float32))
        tl.store(hm + r, cpu.to(tl.int32))


@triton.jit
def _ring_kernel(seq_ptr, bump, hdr_addr, glob_addr, key, T):
    """Single program: (first layer) advance seq, publish T, ring."""
    s = tl.load(seq_ptr) + bump
    tl.store(seq_ptr, s)
    hd = hdr_addr.to(tl.pointer_type(tl.int32))
    tl.store(hd + 2, T)
    s = tl.inline_asm_elementwise("fence.sc.sys;\n\tmov.b32 $0, $1;", "=r,r",
                                  [s], dtype=tl.int32, is_pure=False, pack=1)
    g = glob_addr.to(tl.pointer_type(tl.int64))
    tl.store(g, s.to(tl.int64) * 128 + key)


@triton.jit
def _hint_kernel(pred_ptr, n, hids_addr, hhdr_addr, hseq_ptr, BLOCK: tl.constexpr):
    """Single program: post a router-lookahead prediction (n expert ids) to
    the layer's hint block, then bump its seq. Hints only: a torn read on
    the host costs a wasted or missed readahead, never a wrong result."""
    i = tl.arange(0, BLOCK)
    hi = hids_addr.to(tl.pointer_type(tl.int32))
    tl.store(hi + i, tl.load(pred_ptr + i, mask=i < n, other=-1), mask=i < n)
    tl.debug_barrier()
    s = tl.load(hseq_ptr) + 1
    tl.store(hseq_ptr, s)
    hd = hhdr_addr.to(tl.pointer_type(tl.int32))
    tl.store(hd + 2, n)
    s = tl.inline_asm_elementwise("fence.sc.sys;\n\tmov.b32 $0, $1;", "=r,r",
                                  [s], dtype=tl.int32, is_pure=False, pack=1)
    tl.store(hd, s)


@triton.jit
def _gtime(dep):
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r",
                                     [dep], dtype=tl.int64, is_pure=False,
                                     pack=1)


@triton.jit
def _wait_kernel(seq_ptr, hdr_addr, err_ptr, wait_ptr, SPIN: tl.constexpr):
    """Poll the ack; wait_ptr[0] += ns spent here, wait_ptr[1] += 1."""
    s = tl.load(seq_ptr)
    t0 = _gtime(s)
    ack = hdr_addr.to(tl.pointer_type(tl.int32)) + 1
    a = tl.load(ack, volatile=True)
    i = 0
    while (a != s) & (i < SPIN):
        a = tl.load(ack, volatile=True)
        i += 1
    if a != s:
        tl.atomic_add(err_ptr, 1)
    tl.atomic_add(wait_ptr, _gtime(a) - t0)
    tl.atomic_add(wait_ptr + 1, 1)


@triton.jit
def _add_kernel(out_ptr, hy_addr, H, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = h < H
    hy = hy_addr.to(tl.pointer_type(tl.float32))
    o = tl.load(out_ptr + t * H + h, mask=m).to(tl.float32)
    o += tl.load(hy + t * H + h, mask=m, volatile=True)
    tl.store(out_ptr + t * H + h, o.to(out_ptr.dtype.element_ty), mask=m)


class _S:
    armed = False
    lib = None
    eng = None
    key_idx: dict = {}        # layer_key -> block index
    first_key = None
    H = 0
    hdr = glob = ids = w = msk = x = y = None
    seq = None                # [1] i32 device: forwards seen
    cpu_rows = None           # [n_keys, E] i32 device: CPU experts (split)
    rmap = None               # [n_keys, E] i32 device: pinned row per expert
                              # (-1 = cold, planes files) when RAM_GB is set
    cold = None               # {key: (rowmap, fds)}: planes-file cold rows
    hhdr = hids = hseq = None  # cold-read hints (moe_w2_looka COLD_HINT)
    E = 0
    err = None                # [1] i32 device: ack timeouts


def on() -> bool:
    return _ON and _S.armed


def _core_cpus() -> list[int]:
    """One logical CPU (the first SMT sibling) per physical core this
    process may run on, in core order."""
    allowed = sorted(os.sched_getaffinity(0))
    out, seen = [], set()
    for c in allowed:
        try:
            sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/"
                       "thread_siblings_list").read().strip()
        except OSError:
            sib = str(c)
        if sib not in seen:
            seen.add(sib)
            out.append(c)
    return out


def _physical_cores() -> int:
    return max(1, len(_core_cpus()))


def _build():
    flags = open("/proc/cpuinfo").read().split()
    isa = os.getenv("VLLM_MOE_W2_CPU_ISA", "")
    if not isa:
        isa = ("avx512" if all(f in flags for f in
                               ("avx512_vnni", "avx512vbmi", "avx512bw"))
               else "avx2")
    if isa == "avx2" and not all(f in flags for f in ("avx2", "fma")):
        raise RuntimeError("CPU lacks AVX2 + FMA")
    cflags = ["-march=native"] + (["-mno-avx512f"] if isa == "avx2" else [])
    src = os.path.join(os.path.dirname(__file__), "moe_w2_cpu.cpp")
    code = open(src, "rb").read() + " ".join(cflags).encode()
    tag = hashlib.sha256(code).hexdigest()[:12]
    d = os.path.join(os.path.expanduser("~/.cache/vllm"), "moe_w2_cpu")
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f"moe_w2_cpu-{tag}.so")
    if not os.path.exists(so):
        tmp = f"{so}.{os.getpid()}.tmp"
        subprocess.run(["g++", "-O3", *cflags, "-std=c++17", "-shared",
                        "-fPIC", "-o", tmp, src, "-lpthread", "-ldl"], check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.w2e_create.restype = ctypes.c_void_p
    lib.w2e_create.argtypes = (
        [ctypes.c_int, ctypes.c_void_p] + [ctypes.c_uint64] * 4
        + [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_int]
        + [ctypes.c_void_p] * 8)
    lib.w2e_create_gpu.restype = ctypes.c_void_p
    lib.w2e_create_gpu.argtypes = (
        [ctypes.c_int] + [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p])
    lib.w2e_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.w2e_destroy.argtypes = [ctypes.c_void_p]
    lib.w2e_isa.restype = ctypes.c_char_p
    lib.w2e_cold.restype = ctypes.c_int
    lib.w2e_cold.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.w2e_hint.restype = ctypes.c_int
    lib.w2e_hint.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
    lib.w2e_vq.restype = ctypes.c_int
    lib.w2e_vq.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                           ctypes.c_float]
    return lib


def _vq_table():
    """VLLM_MOE_W2_VQ table (raw e4m3 [P, 8]) -> (u8 [P, 8] int8(w / cs) ^
    0x80, cs): cs the power of two that puts max |w| in 64..127 (1/16 for
    the e2m1-range tables; 96% of the e4m3 entries stay exact)."""
    import math

    import numpy as np
    raw = np.fromfile(os.environ["VLLM_MOE_W2_VQ"], dtype=np.uint8)
    C = torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn).float().view(-1, 8)
    m = float(C.abs().max())
    cs = 2.0 ** -math.floor(math.log2(127 / m)) if m > 0 else 1.0
    q = (C / cs).round().clamp(-127, 127).to(torch.int8).view(torch.uint8) ^ 0x80
    return q.contiguous().numpy(), cs


def arm(btier) -> None:
    """After the layers are built, before graph capture."""
    if not _ON or _S.armed or btier is None:
        return
    try:
        from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
        keys = sorted(k for k, st in moe_w2_cubit._LAYERS.items()
                      if st.get("base"))
        if not keys:
            return
        st = moe_w2_cubit._LAYERS[keys[0]]
        H, I = st["K13"], st["K2"]
        assert st["N13"] == 2 * I and st["N2"] == H
        assert st["off_s13"] == st["N13"] * H // 4
        lim = st.get("swiglu_limit")
        host = []
        files = getattr(btier._store, "cpu_layout", None)
        # With planes files (RAM_GB) the DIRECT share reads only PINNED rows:
        # the split kernel offers the link pinned misses only (rmap >= 0);
        # cold experts always go to the CPU (it reads them from the files).
        cold = {}
        rmaps = []
        for k in keys:
            if files is not None:
                p, rmap, fds = files(k)
                cold[k] = (rmap, fds)
                rmaps.append(torch.from_numpy(rmap.astype("int32")))
            else:
                p = btier.direct_host_ptr(k)
            if not p:
                raise RuntimeError(f"layer {k}: no pinned slot-layout host rows")
            host.append(p)
        n = len(keys)
        pin = dict(dtype=torch.int32, pin_memory=True)
        _S.hdr = torch.zeros((n, _HDR), **pin)
        _S.glob = torch.zeros(8, dtype=torch.int64, pin_memory=True)
        _S.ids = torch.zeros((n, _TCAP * _KCAP), **pin)
        _S.w = torch.zeros((n, _TCAP * _KCAP), dtype=torch.float32,
                           pin_memory=True)
        _S.msk = torch.zeros((n, _TCAP * _KCAP), **pin)
        _S.x = torch.zeros((n, _TCAP, H), dtype=torch.bfloat16, pin_memory=True)
        _S.y = torch.zeros((n, _TCAP, H), dtype=torch.float32, pin_memory=True)
        _S.seq = torch.zeros(1, dtype=torch.int32, device=btier.dev)
        _S.err = torch.zeros(1, dtype=torch.int32, device=btier.dev)
        _S.wait = torch.zeros(2, dtype=torch.int64, device=btier.dev)
        _S.E = int(st["E"])
        _S.cpu_rows = torch.zeros((n, _S.E), dtype=torch.int32,
                                  device=btier.dev)
        _S.rmap = (torch.stack(rmaps).to(btier.dev)
                   if files is not None and _SPLIT else None)
        _S.key_idx = {k: i for i, k in enumerate(keys)}
        _S.first_key = keys[0]
        _S.H = H
        # Leave cores/16 (1..4) for vLLM's own threads. On an 8-core EPYC
        # 7F32 (rome, GSM8K c8) 7 threads beat 4 by 29%; on a 64-core
        # Genoa 60 beat 48
        pc = _physical_cores()
        thr = int(os.getenv("VLLM_MOE_W2_CPU_THREADS",
                            str(max(1, pc - max(1, min(4, pc // 16))))))
        _S.lib = _build()
        hp = (ctypes.c_uint64 * n)(*host)
        # VLLM_MOE_W2_CPU_PIN=1: one engine thread per physical core, from
        # the top. Off by default: on p32 it cost 13% at concurrency 1
        # (80 -> 70 tok/s) and gained 5% at 8 (vLLM's own threads compete)
        cores = _core_cpus()
        pin = (cores[-thr:] if os.getenv("VLLM_MOE_W2_CPU_PIN", "0") == "1"
               and thr <= len(cores) else [-1] * thr)
        cp = (ctypes.c_int * thr)(*pin)
        from vllm.model_executor.layers.quantization.utils import moe_w2_helper
        vq = moe_w2_cubit.vq_enabled()
        if vq and moe_w2_helper.enabled():
            raise RuntimeError("VLLM_MOE_W2_VQ: the helper GPU cannot decode "
                               "vector codes; use the CPU experts")
        if moe_w2_helper.enabled():
            hlp = moe_w2_helper.Helper(
                _S, host, int(btier.slot_bytes), st["off_s13"], st["off_c2"],
                st["off_s2"], H, I, int(st["E"]), _KCAP, _TCAP,
                float(lim) if lim is not None else None)
            ex = (ctypes.c_uint64 * len(hlp.execs))(*hlp.execs)
            bk = (ctypes.c_int * len(hlp.bucket_of))(*hlp.bucket_of)
            _S.helper = hlp
            _S.eng = _S.lib.w2e_create_gpu(
                n, _S.hdr.data_ptr(), _S.glob.data_ptr(), hlp.ctx,
                hlp.stream.cuda_stream, ex, len(hlp.buckets), bk)
            if not _S.eng:
                raise RuntimeError("helper GPU engine: libcuda symbols missing")
            _S.armed = True
            return
        _S.eng = _S.lib.w2e_create(
            n, hp, int(btier.slot_bytes), st["off_s13"], st["off_c2"],
            st["off_s2"], H, I, _KCAP, int(st["E"]),
            float(lim) if lim is not None else float("inf"), thr,
            _S.hdr.data_ptr(), _S.glob.data_ptr(), _S.ids.data_ptr(),
            _S.w.data_ptr(), _S.msk.data_ptr(), _S.x.data_ptr(),
            _S.y.data_ptr(), cp)
        if cold:
            drop = int(getattr(btier._store, "drop", False))
            # I/O threads overlap the cold reads with the hot experts' compute
            n_io = int(os.getenv("VLLM_MOE_W2_COLD_IO", "8"))
            _S.cold = cold      # rowmaps stay alive for the engine
            for i, k in enumerate(keys):
                rmap, fds = cold[k]
                rc = _S.lib.w2e_cold(_S.eng, i, rmap.ctypes.data,
                                     (ctypes.c_int * 4)(*fds), drop, n_io)
                if rc:
                    raise RuntimeError(f"w2e_cold layer {k}: rc {rc}")
            if os.getenv("VLLM_MOE_W2_COLD_HINT", "0") == "1":
                _S.hhdr = torch.zeros((n, _HDR), dtype=torch.int32,
                                      pin_memory=True)
                _S.hids = torch.zeros((n, _HINT_MAX), dtype=torch.int32,
                                      pin_memory=True)
                _S.hseq = torch.zeros(1, dtype=torch.int32, device=btier.dev)
                if _S.lib.w2e_hint(_S.eng, _S.hhdr.data_ptr(),
                                   _S.hids.data_ptr(), _HINT_MAX, n_io):
                    raise RuntimeError("w2e_hint failed")
        if vq:
            tab, cs = _vq_table()
            _S.vq_tab = tab
            rc = _S.lib.w2e_vq(_S.eng, tab.ctypes.data, tab.shape[0], cs)
            if rc:
                _S.lib.w2e_destroy(_S.eng)
                _S.eng = None
                raise RuntimeError(f"VLLM_MOE_W2_VQ: CPU engine refused the "
                                   f"table (rc {rc}, H={H} I={I})")
        _S.armed = True
        logger.info("moe_w2 CPU experts armed (%s%s): %d layers, %d threads, "
                    "H=%d I=%d, swiglu limit %s, PCIe share: up to %d "
                    "missed experts per layer (frac %.3f of missed routes), "
                    "pinned to CPUs %s", _S.lib.w2e_isa().decode(),
                    ", vector codes" if vq else "", n, thr,
                    H, I, lim, _PCIE, _FRAC,
                    f"{pin[0]}..{pin[-1]}" if pin[0] >= 0 else "none")
        from vllm.model_executor.layers.quantization.utils import tandem_stats
        tandem_stats.start(btier, {
            "isa": _S.lib.w2e_isa().decode(), "threads": thr,
            "gpu_cache": f"{btier.n_slots} experts",
            "model": os.path.basename(os.getenv("VLLM_TANDEM_MODEL", "").rstrip("/")),
            "shape": os.getenv("VLLM_TANDEM_SHAPE", "")})
    except Exception as e:  # noqa: BLE001 - optional feature, never kill boot
        if os.getenv("VLLM_MOE_W2_BASE_FILES", "0") == "1":
            # ... except over the planes-file tier: the GPU cannot read the
            # cold rows, so serving on would drop every cold expert
            raise RuntimeError(f"moe_w2 CPU experts (required by "
                               f"VLLM_MOE_W2_BASE_FILES) failed: {e}") from e
        logger.warning("moe_w2 CPU experts disabled: %s", e)


def request(layer_key: int, x: torch.Tensor, topk_ids: torch.Tensor,
            topk_weights: torch.Tensor, btier, ftier=None) -> bool:
    """Decode MoE input (before the descriptor build): hand this layer's
    non-resident routes to the CPU. False when the layer is not served."""
    T = x.shape[0]
    i = _S.key_idx.get(layer_key)
    if not on() or T > _TCAP or i is None:
        return False
    top_k = topk_ids.shape[1]
    H = x.shape[1]
    slot_row = btier.slot_table[layer_key]
    frow = ftier.slot_table[layer_key] if ftier is not None else slot_row
    xc = x.contiguous()
    _req_kernel[(T, triton.cdiv(H, 1024))](
        xc.view(torch.int16), topk_ids.contiguous(), topk_weights.contiguous(),
        slot_row, frow, _S.cpu_rows[i], _S.x[i].data_ptr(),
        _S.ids[i].data_ptr(), _S.w[i].data_ptr(), _S.msk[i].data_ptr(), H, T,
        top_k, _S.E, _PCIE if _PCIE > 0 else 1 << 20,
        int(_FRAC * 65536) if _FRAC > 0 else 1 << 20,
        _S.rmap[i] if _S.rmap is not None else slot_row,
        HAS_F=ftier is not None, SPLIT=_SPLIT, ROWMAP=_S.rmap is not None,
        # R sized to this call's routes: the SPLIT ranking is O(R^2) in one
        # program (R = cap x 8 = 256 cost ~9% at 1 user)
        KCAP=_KCAP, R=triton.next_power_of_2(T * _KCAP),
        EP=triton.next_power_of_2(_S.E),
        BLOCK=1024)
    _ring_kernel[(1,)](_S.seq, int(layer_key == _S.first_key),
                       _S.hdr[i].data_ptr(), _S.glob.data_ptr(), i, T)
    return True


def hint(layer_key: int, pred: torch.Tensor) -> None:
    """In-graph (decode): the router-lookahead prediction of layer_key's
    experts ([T, K] i32), posted while the previous layer runs; the engine
    reads the predicted cold experts ahead (posix_fadvise WILLNEED)."""
    i = _S.key_idx.get(layer_key)
    if not on() or _S.hhdr is None or i is None:
        return
    n = pred.numel()
    if n > _HINT_MAX:
        return
    _hint_kernel[(1,)](pred.contiguous(), n, _S.hids[i].data_ptr(),
                       _S.hhdr[i].data_ptr(), _S.hseq,
                       BLOCK=triton.next_power_of_2(_HINT_MAX), num_warps=1)


def rowmap(layer_key: int):
    """[E] i32 device pinned-row map of a planes-files layer (DIRECT reads
    host row rmap[e]; -1 = cold, always a CPU expert), else None."""
    return _S.rmap[_S.key_idx[layer_key]] if _S.rmap is not None else None


def cpu_row(layer_key: int):
    """[E] i32 device flags of the CPU's experts when the misses are split
    with the DIRECT path, else None (every miss on the CPU)."""
    return _S.cpu_rows[_S.key_idx[layer_key]] if _SPLIT else None


def join(layer_key: int, out: torch.Tensor) -> None:
    """After the layer's unpermute: wait for the CPU's ack, add its y."""
    i = _S.key_idx[layer_key]
    T, H = out.shape
    _wait_kernel[(1,)](_S.seq, _S.hdr[i].data_ptr(), _S.err, _S.wait,
                       SPIN=_SPIN)
    _add_kernel[(T, triton.cdiv(H, 1024))](out, _S.y[i].data_ptr(), H,
                                          BLOCK=1024)


def raw_stats() -> list | None:
    """The engine's cumulative counters (w2e_stats layout), or None."""
    if not on():
        return None
    s = (ctypes.c_uint64 * 33)()
    _S.lib.w2e_stats(_S.eng, s)
    return list(s)


def kpi_summary() -> str:
    s = raw_stats()
    if s is None:
        return ""
    jobs, pairs, ns, mx = list(s)[:4]
    ph = "/".join(f"{v / max(jobs, 1) / 1e3:.0f}" for v in list(s)[4:8])
    n_cold, cold_ns, io_err, wait_ns = list(s)[8:12]
    cold = (f", cold reads {n_cold} experts ({cold_ns / max(n_cold, 1) / 1e3:.0f}"
            f" us each, {cold_ns / max(jobs, 1) / 1e3:.0f} us/job, compute blocked "
            f"{wait_ns / max(jobs, 1) / 1e3:.0f} us/job, read-ahead hints {s[12]} (stale {s[13]} chunks"
            f", read {s[15]}; used {s[14]} = {100 * s[14] / max(s[12], 1):.0f}% of hints, "
            f"{100 * s[14] / max(n_cold, 1):.0f}% of cold reads; used/hinted by rank "
            + " ".join(f"{s[24 + r]}/{s[16 + r]}" for r in range(8)) + "), "
            f"io errors {io_err})"
            if n_cold or io_err else "")
    return (f"; CPU experts: {pairs} pairs in {jobs} layer jobs, "
            f"{ns / max(jobs, 1) / 1e3:.0f} us/job (max {mx / 1e3:.0f}; "
            f"prep/w13/act/w2 {ph}), GPU wait {_wait_us():.0f} us/layer, "
            f"ack timeouts {int(_S.err.item())}{cold}")


def _wait_us() -> float:
    ns, n = _S.wait.tolist()
    return ns / max(n, 1) / 1e3
