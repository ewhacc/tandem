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
                n_pcie, fracq, rmap_ptr, hown_ptr, hmsk_h_addr, hhdr_h_addr,
                hcnt_ptr,
                HAS_F: tl.constexpr, SPLIT: tl.constexpr,
                ROWMAP: tl.constexpr, HELP: tl.constexpr, HALL: tl.constexpr,
                CNT: tl.constexpr,
                KCAP: tl.constexpr, R: tl.constexpr, EP: tl.constexpr,
                BLOCK: tl.constexpr):
    """grid (T, cdiv(H, BLOCK)): copy x row t into the layer's pinned
    block; program (0, 0) also writes every route and the CPU mask (routes
    resident in neither GPU pool). SPLIT: the missed experts routed by the
    most tokens stay on the DIRECT path, min(n_pcie, round(fracq / 65536 x
    missed routes)) of them; the CPU's experts are flagged in cpu_row for
    the descriptor build. ROWMAP (planes files): only pinned experts
    (rmap_ptr[e] >= 0) can go over the link; cold ones stay on the CPU.
    HELP (helper warm set): missed routes to the helper's experts
    (hown_ptr[e] != 0) go to the helper's mask instead, before the split;
    HALL (alt split): every route to a helper expert, resident or not.
    CNT (dynamic helper): hcnt_ptr[e] += routes the main GPU does not
    serve (helper's and CPU's), the helper cache's hotness signal."""
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
        if HELP:
            hm = (v if HALL else miss) & (tl.load(hown_ptr + e, mask=v, other=0) != 0)
            miss = miss & ~hm
            hmh = hmsk_h_addr.to(tl.pointer_type(tl.int32))
            tl.store(hmh + r, hm.to(tl.int32))
            # helper route count -> its header word 3 (0: no job, no ring)
            hhd = hhdr_h_addr.to(tl.pointer_type(tl.int32))
            tl.store(hhd + 3, tl.sum(hm.to(tl.int32), axis=0))
            if CNT:
                tl.atomic_add(hcnt_ptr + e, tl.full((R,), 1, tl.int32),
                              mask=miss | hm)
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
            if HELP:
                # the main GPU skips the helper's routes like the CPU's
                tl.store(cpu_row_ptr + e, tl.full((R,), 1, tl.int32), mask=cpu | hm)
            else:
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
def _gring_kernel(seq_ptr, hdr_addr, glob_addr, key, T):
    """Helper doorbell (after _ring_kernel, same seq): with no helper
    routes (header word 3 == 0) ack directly instead of ringing."""
    s = tl.load(seq_ptr)
    hd = hdr_addr.to(tl.pointer_type(tl.int32))
    tl.store(hd + 2, T)
    n = tl.load(hd + 3, volatile=True)
    s = tl.inline_asm_elementwise("fence.sc.sys;\n\tmov.b32 $0, $1;", "=r,r",
                                  [s], dtype=tl.int32, is_pure=False, pack=1)
    if n == 0:
        tl.store(hd + 1, s)
    else:
        g = glob_addr.to(tl.pointer_type(tl.int64))
        tl.store(g, s.to(tl.int64) * 128 + key)


@triton.jit
def _add_if_kernel(out_ptr, hy_addr, hdr_addr, H, BLOCK: tl.constexpr):
    """_add_kernel when the helper had routes (header word 3 != 0)."""
    t = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(hdr_addr.to(tl.pointer_type(tl.int32)) + 3, volatile=True)
    m = (h < H) & (n != 0)
    hy = hy_addr.to(tl.pointer_type(tl.float32))
    o = tl.load(out_ptr + t * H + h, mask=m).to(tl.float32)
    o += tl.load(hy + t * H + h, mask=m, volatile=True)
    tl.store(out_ptr + t * H + h, o.to(out_ptr.dtype.element_ty), mask=m)


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
    # helper warm set next to the CPU engine (moe_w2_helper warm_sets)
    g_eng = None
    g_hdr = g_glob = g_msk = g_y = None
    g_own = None              # [n_keys, E] i32 pinned host: helper-held experts
    g_snap = None             # [n_keys, E] i32 device: per-call copy of g_own
    g_fb = None               # [n_keys] i32 device: fallback main expert
    g_cnt = None              # [n_keys, E] i32 device: non-main routes (dyn)
    g_dyn = None              # _DynHelper
    serving = False           # vLLM's graph capture is over
    g_wait = None
    g_split = False           # alt split: the helper owns half the hot set
    helper = None


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
        from vllm.model_executor.layers.quantization.utils import moe_w2_ep
        thr = int(os.getenv("VLLM_MOE_W2_CPU_THREADS",
                            str(max(1, (pc - max(1, min(4, pc // 16)))
                                    // moe_w2_ep.ids()[0]))))
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
        hybrid = moe_w2_helper.enabled() and moe_w2_helper.warm_gb() > 0
        if vq and moe_w2_helper.enabled() and not hybrid:
            raise RuntimeError("VLLM_MOE_W2_VQ: the helper GPU cannot decode "
                               "vector codes; use the CPU experts")
        if moe_w2_helper.enabled() and not hybrid:
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
        if hybrid:
            try:
                _arm_helper(btier, keys, st, H, I, lim, vq)
            except Exception as e:  # noqa: BLE001 - keep the CPU experts
                logger.warning("moe_w2 helper GPU disabled (CPU experts "
                               "stay on): %s", e)
                _S.g_eng = _S.g_dyn = _S.g_cnt = None
                _S.g_split = False
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


def _arm_helper(btier, keys, st, H, I, lim, vq) -> None:
    """Warm-set helper GPU next to the CPU engine: its own pinned mask, y
    and doorbell; x / ids / w are shared with the CPU's blocks."""
    import numpy as np

    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_ep, moe_w2_helper)
    n, E, sb = len(keys), int(st["E"]), int(btier.slot_bytes)
    skip = int(os.getenv("VLLM_MOE_W2_HELPER_SKIP", str(btier.n_slots)))
    # tensor parallel (moe_w2_ep): every rank runs its own helper on the
    # shared helper GPU over its own experts only; the rank's EP fallback
    # (the weight-0 target of other ranks' routes) stays on the main GPU
    ep_fb = None
    allow = None
    if moe_w2_ep.active():
        if not moe_w2_helper.split_mode():
            raise RuntimeError("helper GPU with tensor parallel needs "
                               "VLLM_MOE_W2_HELPER_SPLIT=alt or dyn")
        ep_fb = {k: int(moe_w2_ep.tables(k, E, "cpu")[1]) for k in keys}
        allow = (lambda li, ei: moe_w2_ep.local(ei) and ei != ep_fb[li])
    sets = moe_w2_helper.warm_sets(keys, E, sb, skip, allow)
    _S.g_split = moe_w2_helper.split_mode()
    store = btier._store

    def fetch(i, ids):
        return store.rows_for([(keys[i], e) for e in ids])

    pin = dict(dtype=torch.int32, pin_memory=True)
    _S.g_hdr = torch.zeros((n, _HDR), **pin)
    _S.g_glob = torch.zeros(8, dtype=torch.int64, pin_memory=True)
    _S.g_msk = torch.zeros((n, _TCAP * _KCAP), **pin)
    _S.g_y = torch.zeros((n, _TCAP, H), dtype=torch.float32, pin_memory=True)
    _S.g_wait = torch.zeros(2, dtype=torch.int64, device=btier.dev)
    own = torch.zeros((n, E), **pin)
    for i, ids in enumerate(sets):
        if ids:
            own[i, torch.tensor(ids)] = 1
    _S.g_own = own
    _S.g_snap = torch.zeros((n, E), dtype=torch.int32, device=btier.dev)
    # fallback for the helper's dropped routes: the layer's top-ranked
    # expert outside the start set; never handed to the helper
    rank = moe_w2_helper.ranking(keys, E)
    fb = []
    for i, k in enumerate(keys):
        held = set(sets[i])
        fb.append(ep_fb[k] if ep_fb is not None else
                  next((e for e in rank[k] if e not in held), 0))
    _S.g_fb = torch.tensor(fb, dtype=torch.int32, device=btier.dev)
    dyn = moe_w2_helper.dynamic()
    if dyn:
        _S.g_cnt = torch.zeros((n, E), dtype=torch.int32, device=btier.dev)
    raw = (np.fromfile(os.environ["VLLM_MOE_W2_VQ"], dtype=np.uint8)
           if vq else None)
    hlp = moe_w2_helper.Helper(
        _S, None, sb, st["off_s13"], st["off_c2"], st["off_s2"], H, I, E,
        _KCAP, _TCAP, float(lim) if lim is not None else None,
        sets=sets, fetch=fetch, msk=_S.g_msk, y=_S.g_y, vq=raw,
        pool_rows=int(moe_w2_helper.warm_gb() * 2**30 // sb) if dyn else 0)
    ex = (ctypes.c_uint64 * len(hlp.execs))(*hlp.execs)
    bk = (ctypes.c_int * len(hlp.bucket_of))(*hlp.bucket_of)
    _S.helper = hlp
    _S.g_eng = _S.lib.w2e_create_gpu(
        n, _S.g_hdr.data_ptr(), _S.g_glob.data_ptr(), hlp.ctx,
        hlp.stream.cuda_stream, ex, len(hlp.buckets), bk)
    if not _S.g_eng:
        raise RuntimeError("helper GPU engine: libcuda symbols missing")
    if dyn:
        _S.g_dyn = _DynHelper(btier, keys, hlp, fb)
    logger.info("moe_w2 helper warm set: %d experts %s (%.1f GiB), per "
                "layer %d..%d", sum(len(x) for x in sets),
                ("owned, every other of the top ranks (alt split)"
                 + (" as the start of a dynamic cache" if dyn else ""))
                if _S.g_split else f"after the top {skip}",
                sum(len(x) for x in sets) * sb / 2**30,
                min(len(x) for x in sets), max(len(x) for x in sets))


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
    if _S.g_eng and not _S.g_split:
        # split mode took this call's snapshot in helper_own (the remap and
        # this request must see the same ownership)
        _S.g_snap[i].copy_(_S.g_own[i], non_blocking=True)
    xc = x.contiguous()
    _req_kernel[(T, triton.cdiv(H, 1024))](
        xc.view(torch.int16), topk_ids.contiguous(), topk_weights.contiguous(),
        slot_row, frow, _S.cpu_rows[i], _S.x[i].data_ptr(),
        _S.ids[i].data_ptr(), _S.w[i].data_ptr(), _S.msk[i].data_ptr(), H, T,
        top_k, _S.E, _PCIE if _PCIE > 0 else 1 << 20,
        int(_FRAC * 65536) if _FRAC > 0 else 1 << 20,
        _S.rmap[i] if _S.rmap is not None else slot_row,
        _S.g_snap[i] if _S.g_eng else slot_row,
        _S.g_msk[i].data_ptr() if _S.g_eng else 0,
        _S.g_hdr[i].data_ptr() if _S.g_eng else 0,
        _S.g_cnt[i] if _S.g_cnt is not None else slot_row,
        HAS_F=ftier is not None, SPLIT=_SPLIT, ROWMAP=_S.rmap is not None,
        HELP=bool(_S.g_eng), HALL=bool(_S.g_eng) and _S.g_split,
        CNT=_S.g_cnt is not None,
        # R sized to this call's routes: the SPLIT ranking is O(R^2) in one
        # program (R = cap x 8 = 256 cost ~9% at 1 user)
        KCAP=_KCAP, R=triton.next_power_of_2(T * _KCAP),
        EP=triton.next_power_of_2(_S.E),
        BLOCK=1024)
    _ring_kernel[(1,)](_S.seq, int(layer_key == _S.first_key),
                       _S.hdr[i].data_ptr(), _S.glob.data_ptr(), i, T)
    if _S.g_eng:
        _gring_kernel[(1,)](_S.seq, _S.g_hdr[i].data_ptr(),
                            _S.g_glob.data_ptr(), i, T)
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


def helper_split(layer_key: int, T: int) -> bool:
    """Alt split on for this decode call: the main GPU drops the helper's
    routes (moe_w2_helper.remap) and request() gets the full routing."""
    return (_S.g_split and _S.g_eng is not None and on() and T <= _TCAP
            and layer_key in _S.key_idx)


def mark_serving() -> None:
    _S.serving = True


def helper_own(layer_key: int):
    """Split mode, once per decode call before the remap: snapshot the
    helper's ownership row (host-written by _DynHelper) to the device ->
    (own [E] i32, fallback expert [1] i32); request() reads the same copy."""
    i = _S.key_idx[layer_key]
    _S.g_snap[i].copy_(_S.g_own[i], non_blocking=True)
    return _S.g_snap[i], _S.g_fb[i:i + 1]


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
    if _S.g_eng:
        _wait_kernel[(1,)](_S.seq, _S.g_hdr[i].data_ptr(), _S.err, _S.g_wait,
                           SPIN=_SPIN)
        _add_if_kernel[(T, triton.cdiv(H, 1024))](
            out, _S.g_y[i].data_ptr(), _S.g_hdr[i].data_ptr(), H, BLOCK=1024)


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
            f"ack timeouts {int(_S.err.item())}{cold}{_helper_kpi()}")


def _helper_kpi() -> str:
    if not _S.g_eng:
        return ""
    s = (ctypes.c_uint64 * 33)()
    _S.lib.w2e_stats(_S.g_eng, s)
    ns, n = _S.g_wait.tolist()
    return (f"; helper GPU: {s[0]} layer jobs, wait {ns / max(n, 1) / 1e3:.0f}"
            f" us/layer" + (_S.g_dyn.kpi() if _S.g_dyn is not None else ""))


class _DynHelper:
    """Dynamic helper cache (VLLM_MOE_W2_HELPER_SPLIT=dyn), one rule: the
    helper holds the hottest experts the main GPU does not serve. Hotness =
    decayed count of routes the main GPU missed (_req_kernel CNT: helper's
    and CPU's), same half-life as the main pool. Each pass loads the
    hottest CPU-served experts into free rows and frees the coldest held
    ones that a candidate beats by MARGIN; an expert the main pool took
    (prefill fills) leaves the helper (YIELD, default on). Exclusive with
    the main GPU.

    Ownership change order (a route is never dropped or computed twice):
    admit = load row, sync, map it on the helper, THEN set g_own (host);
    evict = clear g_own, then free the row once the step counter shows
    every step that could have snapshotted the old bit has finished."""

    def __init__(self, btier, keys, hlp, fb):
        import threading

        import numpy as np
        self.np = np
        self.bt, self.keys, self.h = btier, list(keys), hlp
        n, E = _S.g_own.shape
        self.n, self.E = n, E
        self.own = _S.g_own.numpy()
        self.never = np.zeros((n, E), bool)
        self.never[np.arange(n), np.asarray(fb)] = True
        self.score = np.zeros((n, E))
        self.cnt_h = torch.zeros((n, E), dtype=torch.int32, pin_memory=True)
        self.last = np.zeros((n, E), np.int64)
        self.pending = []           # (seq to wait for, layer idx, expert)
        self.busy = np.zeros((n, E), bool)
        self.fill = int(os.getenv("VLLM_MOE_W2_HELPER_FILL", "16"))
        self.margin = float(os.getenv("VLLM_MOE_W2_HELPER_MARGIN", "1.25"))
        self.half = float(os.getenv("VLLM_MOE_W2_DELTA_DECAY_S", "30"))
        self.yield_ = os.getenv("VLLM_MOE_W2_HELPER_YIELD", "1") == "1"
        self.period = float(os.getenv("VLLM_MOE_W2_HELPER_PERIOD_MS", "10")) / 1e3
        self.loads = self.evicts = self.yields = self.cold_skips = 0
        self.side = torch.cuda.Stream(device=btier.dev)
        self.ev = torch.cuda.Event()
        rm = getattr(btier._store, "_rowmaps", None)
        # all experts pinned (no RAM_GB): no row maps; tensor parallel: -1 =
        # another rank's expert (never a candidate here)
        self.rowmaps = [rm.get(k) if rm else None for k in self.keys]
        self.th = threading.Thread(target=self._loop, name="moe_w2_dynhelper",
                                   daemon=True)
        self.th.start()

    def _seq(self) -> int:
        return int(_S.glob[0]) >> 7

    def _loop(self) -> None:
        import time
        # no CUDA calls from this thread until vLLM's graph capture is over
        # (global-mode capture fails on another thread's stream work): wait
        # until the worker finished capture (gpu_worker.compile_or_warm_up_model)
        while not _S.serving:
            time.sleep(0.5)
        t_last = time.monotonic()
        while True:
            time.sleep(self.period)
            try:
                now = time.monotonic()
                self._pass(0.5 ** ((now - t_last) / self.half))
                t_last = now
            except Exception as e:  # noqa: BLE001 - never kill serving
                logger.warning_once("moe_w2 dynamic helper pass failed: %s", e)

    def _pass(self, decay: float) -> None:
        np = self.np
        with torch.cuda.device(self.bt.dev), torch.cuda.stream(self.side):
            self.cnt_h.copy_(_S.g_cnt, non_blocking=True)
            self.ev.record(self.side)
        self.ev.synchronize()
        c = self.cnt_h.numpy().astype(np.int64)
        delta = c - self.last
        self.last = c
        self.score *= decay
        self.score += delta
        seq = self._seq()
        # rows of evicted experts: free once no running step can route there
        keep = []
        for need, i, e in self.pending:
            if seq >= need:
                self.h.free_rows.append(self.h.unpublish(i, e))
                self.busy[i, e] = False
            else:
                keep.append((need, i, e))
        self.pending = keep
        mirror = self.bt._mirror
        held = self.own != 0
        # the main pool took it (prefill fills): main wins, the helper lets
        # go. RH5: 25.8 tok/s against 25.3 with YIELD=0 (3.7x fewer loads,
        # but the helper then computes routes main holds: wait 16 -> 60 us)
        for i, k in (enumerate(self.keys) if self.yield_ else ()):
            row = held[i]
            if row.any():
                both = np.flatnonzero(row & (mirror[k].numpy() >= 0))
                for e in both:
                    self._evict(i, int(e))
                    self.yields += 1
        if not delta.any():
            return
        held = self.own != 0
        cand = (delta > 0) & ~held & ~self.never & ~self.busy
        for i, k in enumerate(self.keys):
            if cand[i].any():
                cand[i] &= mirror[k].numpy() < 0
        ci, ce = np.nonzero(cand)
        if ci.size == 0:
            return
        order = np.argsort(-self.score[ci, ce], kind="stable")[:self.fill]
        ci, ce = ci[order], ce[order]
        # evict the coldest held experts the candidates beat
        hs = np.where(held & ~self.busy, self.score, np.inf)
        # rows already on their way back count as free
        need = max(0, len(ci) - len(self.h.free_rows) - len(self.pending))
        if need:
            flat = np.argpartition(hs.ravel(), min(need, hs.size - 1))[:need]
            vic = sorted(flat, key=lambda f: hs.ravel()[f])
            best = sorted(self.score[ci, ce], reverse=True)
            for f, sc in zip(vic, best):
                vs = hs.ravel()[f]
                if not np.isfinite(vs) or sc <= vs * self.margin + 1.0:
                    break
                self._evict(*map(int, np.unravel_index(f, hs.shape)))
        # admit into free rows
        adm = []
        for i, e in zip(ci.tolist(), ce.tolist()):
            if not self.h.free_rows:
                break
            rmap = self.rowmaps[i]
            if rmap is not None and rmap[e] < 0:
                self.cold_skips += 1      # not pinned: stays a CPU expert
                continue
            adm.append((i, e, self.h.free_rows.pop()))
        if not adm:
            return
        with self.bt._lock:
            rows = self.bt._store.rows_for(
                [(self.keys[i], e) for i, e, _r in adm])
            for (i, e, r), src in zip(adm, rows):
                self.h.load_row(r, src)
            self.h.sync_loads()
        for i, e, r in adm:
            self.h.publish(i, e, r)
            self.own[i, e] = 1
            self.loads += 1

    def _evict(self, i: int, e: int) -> None:
        self.own[i, e] = 0
        self.busy[i, e] = True
        self.evicts += 1
        # steps up to the current one may hold the old bit; seq + 2 starts
        # only after seq + 1 started, i.e. after seq finished (its joins
        # waited for every helper job of that step)
        self.pending.append((self._seq() + 2, i, e))

    def kpi(self) -> str:
        return (f", dynamic: {len(self.h.row_of)} held, {self.loads} loads, "
                f"{self.evicts} evictions ({self.yields} to the main GPU)"
                + (f", {self.cold_skips} cold skips" if self.cold_skips else ""))


def _wait_us() -> float:
    ns, n = _S.wait.tolist()
    return ns / max(n, 1) / 1e3
