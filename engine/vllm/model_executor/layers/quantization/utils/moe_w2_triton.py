# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable Triton backend for the moe_w2 expert GEMMs (Phase 2).

The SM120 path runs hand-written SASS (moe_w2_cubit: QMMA.SF cubins, no
PTX). This module computes the SAME GEMMs from the SAME host-packed planes
and the SAME descriptor ABI, so every other part of the stack (planes,
caches, gate, replay, stores) is unchanged on other GPUs:

  w2  / w2mc4  2-bit codes {-4,-1,1,4} x UE8M0 block-32 scales; 6-field
               desc {a, as, b, bs, c, m_rows}; decode (m <= 4) and prefill
               (m <= 16) share one kernel through the m_rows mask
  w4           FP4 (e2m1 nibble plane, pack_fp4_fragment_major) x UE8M0
               scales; same 6-field desc (the FP4 recovery tier)
  w4q          split-FP4: radix-5 refinement planes over the 2-bit base;
               8-field desc {a, as, base, ref, bs, c, m_rows, pad}

Each launcher has the cubin call shape ``fn(desc, n_rows, pairs)`` with
grid (n_rows / BLOCK_N, pairs) and stores BF16 rows exactly where the
cubin does (a pair is computed by one tier only; m_rows == 0 exits).

Numerics: every operand entering tl.dot is exact (e4m3 activations, the
2-bit levels, e2m1 values), so the only rounding is the fp32 accumulation
order: identical inputs give identical outputs run to run (no atomics),
which keeps the replay contract (gate / base-miss replay) exact within one
backend. Results differ from the SASS kernels in summation order only.

Math per arch (``dot_mode``):
  "fp8"  native e4m3 tl.dot (SM89, SM90, SM100, SM120)
  "bf16" operands decoded to bf16 in registers, e4m3 activations decoded
         from their bit pattern by integer ops (SM80/SM86: no fp8 MMA and
         no fp8 type support in codegen)

Kernel structure and plane addressing follow the Ada port in
vllm-moet-sm89-w2 (overlay moe_w2_sm89.py, Apache-2.0): one 32-group per
K step, codes gathered through the byte-major k permutation so every code
byte is read once, and both scales folded per group as an [M, N] outer
product. Plane layouts: moe_w2_planes.pack_fragment_major /
pack_fp4_fragment_major / pack_quintal_fragment_major / pack_scales.
"""

import os

import torch
import triton
import triton.language as tl

_NUM_WARPS = int(os.getenv("VLLM_MOE_W2_TRITON_WARPS", "2"))
_BLOCK_N = int(os.getenv("VLLM_MOE_W2_TRITON_BLOCK_N", "16"))
if _BLOCK_N not in (16, 32, 64):
    raise ValueError("VLLM_MOE_W2_TRITON_BLOCK_N must be 16, 32 or 64")
_GATE = 2.5e-2          # op-level parity gate (worst |err| / max |ref|)


def dot_mode() -> str:
    m = os.getenv("VLLM_MOE_W2_TRITON_DOT", "auto").lower()
    if m in ("fp8", "bf16"):
        return m
    return "fp8" if torch.cuda.get_device_capability() >= (8, 9) else "bf16"


@triton.jit
def _load_a(a_ptr, offs_m, mask_m, k_glob, K: tl.constexpr,
            FP8: tl.constexpr):
    """[16, 32] e4m3 activations -> fp8 (FP8) or bf16 (exact decode)."""
    ptrs = a_ptr + offs_m[:, None] * K + k_glob[None, :]
    if FP8:
        out = tl.load(ptrs.to(tl.pointer_type(tl.float8e4nv)),
                      mask=mask_m[:, None], other=0.0)
    else:
        bits = tl.load(ptrs.to(tl.pointer_type(tl.uint8)),
                       mask=mask_m[:, None], other=0).to(tl.int32)
        s = bits >> 7
        e = (bits >> 3) & 15
        m = bits & 7
        # normal: (1 + m/8) 2^(e-7) -> f32 bits; subnormal: m * 2^-9
        nrm = ((s << 31) | ((e + 120) << 23) | (m << 20)).to(
            tl.float32, bitcast=True)
        sub = m.to(tl.float32) * 0.001953125
        sub = tl.where(s > 0, -sub, sub)
        out = tl.where(e > 0, nrm, sub).to(tl.bfloat16)
    return out


@triton.jit
def _scale_b(bs_ptr, n_block, r16, g, K: tl.constexpr):
    sb = tl.load(bs_ptr + (n_block * (K // 32) + g) * 16 + r16).to(tl.int32)
    w_s = (sb << 23).to(tl.float32, bitcast=True)        # 2^(sb-127)
    return tl.where(sb == 0, 5.877471754111438e-39, w_s)


@triton.jit
def moe_w2_mm_tl(desc_ptr, n_rows, K: tl.constexpr, BLOCK_N: tl.constexpr,
                 FP8: tl.constexpr):
    tl.static_assert(K % 64 == 0)
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)
    d = desc_ptr + pid_p * 6
    m_rows = tl.load(d + 5)
    if m_rows == 0:
        return
    a_ptr = tl.load(d + 0).to(tl.pointer_type(tl.uint8))
    as_ptr = tl.load(d + 1).to(tl.pointer_type(tl.float32))
    b_ptr = tl.load(d + 2).to(tl.pointer_type(tl.uint8))
    bs_ptr = tl.load(d + 3).to(tl.pointer_type(tl.uint8))
    c_ptr = tl.load(d + 4).to(tl.pointer_type(tl.bfloat16))

    offs_m = tl.arange(0, 16)
    mask_m = offs_m < m_rows
    n_abs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_block = n_abs // 16
    r16 = n_abs % 16
    row_off = (r16 % 8) * 32 + (r16 // 8) * 4
    b_idx = tl.arange(0, 8)
    byte_col = (b_idx // 2) * 8 + (b_idx % 2)
    f = tl.arange(0, 32)
    k_perm = ((f // 4) % 2) * 16 + (f // 8) * 4 + (f % 4)
    shift = (f % 4) * 2

    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for g in range(0, K // 32):
        kb = g // 2
        k32 = g % 2
        a = _load_a(a_ptr, offs_m, mask_m, kb * 64 + k32 * 32 + k_perm, K,
                    FP8)
        a_s = tl.load(as_ptr + offs_m * (K // 32) + g, mask=mask_m,
                      other=0.0)
        blk = (n_block * (K // 64) + kb) * 256
        cb = tl.load(b_ptr + blk[:, None] + row_off[:, None]
                     + (byte_col + k32 * 2)[None, :]).to(tl.int32)
        ex = tl.reshape(tl.broadcast_to(cb[:, :, None], (BLOCK_N, 8, 4)),
                        (BLOCK_N, 32))
        codes = (ex >> shift[None, :]) & 3
        mag = tl.where((codes == 0) | (codes == 3), 4.0, 1.0)
        wv = tl.where(codes < 2, -mag, mag)
        if FP8:
            w = wv.to(tl.float8e4nv)
        else:
            w = wv.to(tl.bfloat16)
        w_s = _scale_b(bs_ptr, n_block, r16, g, K)
        acc += tl.dot(a, tl.trans(w)) * (a_s[:, None] * w_s[None, :])
    tl.store(c_ptr + offs_m[:, None] * n_rows + n_abs[None, :],
             acc.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def moe_w4_mm_tl(desc_ptr, n_rows, K: tl.constexpr, BLOCK_N: tl.constexpr,
                 FP8: tl.constexpr):
    """FP4 tier: e2m1 nibble plane (pack_fp4_fragment_major: 512-byte
    (nb, kb) blocks, row offset g*64 + tile*8, even k4 in the low nibble)."""
    tl.static_assert(K % 64 == 0)
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)
    d = desc_ptr + pid_p * 6
    m_rows = tl.load(d + 5)
    if m_rows == 0:
        return
    a_ptr = tl.load(d + 0).to(tl.pointer_type(tl.uint8))
    as_ptr = tl.load(d + 1).to(tl.pointer_type(tl.float32))
    b_ptr = tl.load(d + 2).to(tl.pointer_type(tl.uint8))
    bs_ptr = tl.load(d + 3).to(tl.pointer_type(tl.uint8))
    c_ptr = tl.load(d + 4).to(tl.pointer_type(tl.bfloat16))

    offs_m = tl.arange(0, 16)
    mask_m = offs_m < m_rows
    n_abs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_block = n_abs // 16
    r16 = n_abs % 16
    row_off = (r16 % 8) * 64 + (r16 // 8) * 8
    f = tl.arange(0, 32)
    byte_col = (f // 8) * 16 + ((f // 4) % 2) * 2 + (f % 4) // 2
    nib_shift = (f % 2) * 4
    k_perm = ((f // 4) % 2) * 16 + (f // 8) * 4 + (f % 4)

    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for g in range(0, K // 32):
        kb = g // 2
        k32 = g % 2
        a = _load_a(a_ptr, offs_m, mask_m, kb * 64 + k32 * 32 + k_perm, K,
                    FP8)
        a_s = tl.load(as_ptr + offs_m * (K // 32) + g, mask=mask_m,
                      other=0.0)
        blk = (n_block * (K // 64) + kb) * 512
        wb = tl.load(b_ptr + blk[:, None] + row_off[:, None]
                     + (byte_col + k32 * 4)[None, :]).to(tl.int32)
        nib = (wb >> nib_shift[None, :]) & 0xF
        s = nib >> 3
        e2 = (nib >> 1) & 3
        m = nib & 1
        if FP8:
            # e2m1 -> its exact e4m3 bit pattern
            pat = tl.where(e2 > 0, ((e2 + 6) << 3) | (m << 2),
                           tl.where(m > 0, 48, 0)) | (s << 7)
            w = pat.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        else:
            # value = (e2 > 0 ? (1 + m/2) 2^(e2-1) : m/2), exact in bf16
            v = tl.where(e2 > 0, (2 + m).to(tl.float32)
                         * (1 << e2).to(tl.float32) * 0.25,
                         m.to(tl.float32) * 0.5)
            w = tl.where(s > 0, -v, v).to(tl.bfloat16)
        w_s = _scale_b(bs_ptr, n_block, r16, g, K)
        acc += tl.dot(a, tl.trans(w)) * (a_s[:, None] * w_s[None, :])
    tl.store(c_ptr + offs_m[:, None] * n_rows + n_abs[None, :],
             acc.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def moe_w4q_mm_tl(desc_ptr, n_rows, K: tl.constexpr, FP8: tl.constexpr):
    """Split-FP4: 2-bit base codes (class + sign) plus radix-5 refinement
    words (10 bits per 4 elements; 320-byte (nb, kb) blocks: 8 bytes per
    lane, then 2 bytes per lane at +256). Decoded values are exact e2m1."""
    tl.static_assert(K % 64 == 0)
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)
    d = desc_ptr + pid_p * 8
    m_rows = tl.load(d + 6)
    if m_rows == 0:
        return
    a_ptr = tl.load(d + 0).to(tl.pointer_type(tl.uint8))
    as_ptr = tl.load(d + 1).to(tl.pointer_type(tl.float32))
    b_ptr = tl.load(d + 2).to(tl.pointer_type(tl.uint8))
    q_ptr = tl.load(d + 3).to(tl.pointer_type(tl.uint8))
    bs_ptr = tl.load(d + 4).to(tl.pointer_type(tl.uint8))
    c_ptr = tl.load(d + 5).to(tl.pointer_type(tl.bfloat16))

    offs_m = tl.arange(0, 16)
    mask_m = offs_m < m_rows
    r = tl.arange(0, 16)
    g_row = r % 8
    tile = r // 8
    row_off = g_row * 32 + tile * 4
    b_idx = tl.arange(0, 8)
    t = b_idx // 2
    half = b_idx % 2
    lane = g_row[:, None] * 4 + t[None, :]
    byte_col = t * 8 + half
    f = tl.arange(0, 32)
    k_perm = ((f // 4) % 2) * 16 + (f // 8) * 4 + (f % 4)
    shift = (f % 4) * 2
    n_block = pid_n + 0 * r

    acc = tl.zeros((16, 16), dtype=tl.float32)
    for kg in range(0, K // 32):
        kb = kg // 2
        k32 = kg % 2
        a = _load_a(a_ptr, offs_m, mask_m, kb * 64 + k32 * 32 + k_perm, K,
                    FP8)
        a_s = tl.load(as_ptr + offs_m * (K // 32) + kg, mask=mask_m,
                      other=0.0)
        bblk = (pid_n * (K // 64) + kb) * 256
        cb = tl.load(b_ptr + bblk + row_off[:, None]
                     + (byte_col + k32 * 2)[None, :]).to(tl.int32)
        ex = tl.reshape(tl.broadcast_to(cb[:, :, None], (16, 8, 4)),
                        (16, 32))
        codes = (ex >> shift[None, :]) & 3

        word_idx = tile[:, None] * 4 + k32 * 2 + half[None, :]
        bit = word_idx * 10
        byte_pos = bit // 8
        bit_off = bit - byte_pos * 8
        qbase = q_ptr + (pid_n * (K // 64) + kb) * 320
        b0 = tl.load(tl.where(byte_pos < 8, qbase + lane * 8 + byte_pos,
                              qbase + 256 + lane * 2 + (byte_pos - 8))
                     ).to(tl.int32)
        bp1 = byte_pos + 1
        b1 = tl.load(tl.where(bp1 < 8, qbase + lane * 8 + bp1,
                              qbase + 256 + lane * 2 + (bp1 - 8))
                     ).to(tl.int32)
        bp2 = byte_pos + 2
        b2 = tl.load(tl.where(bp2 < 8, qbase + lane * 8 + bp2,
                              qbase + 256 + lane * 2 + (bp2 - 8)),
                     mask=bp2 < 10, other=0).to(tl.int32)
        words8 = ((b0 | (b1 << 8) | (b2 << 16)) >> bit_off) & 1023
        words = tl.reshape(tl.broadcast_to(words8[:, :, None], (16, 8, 4)),
                           (16, 32))
        q0 = words // 5
        d0 = words - q0 * 5
        q1 = q0 // 5
        d1 = q0 - q1 * 5
        q2 = q1 // 5
        d2 = q1 - q2 * 5
        q3 = q2 // 5
        d3 = q2 - q3 * 5
        sel = (f % 4)[None, :]
        digit = tl.where(sel == 0, d0, tl.where(sel == 1, d1,
                                                tl.where(sel == 2, d2, d3)))
        big = (codes == 0) | (codes == 3)
        mi = digit + tl.where(big, 5, 0)
        val = tl.where(mi == 0, 0.0, tl.where(mi == 1, 0.5,
              tl.where(mi == 2, 1.0, tl.where(mi == 3, 1.5,
              tl.where(mi == 4, 2.0, tl.where(mi == 5, 3.0,
              tl.where(mi == 6, 4.0, 6.0)))))))
        wv = tl.where(codes < 2, -val, val)
        if FP8:
            w = wv.to(tl.float8e4nv)
        else:
            w = wv.to(tl.bfloat16)
        w_s = _scale_b(bs_ptr, n_block, r, kg, K)
        acc += tl.dot(a, tl.trans(w)) * (a_s[:, None] * w_s[None, :])
    tl.store(c_ptr + offs_m[:, None] * n_rows + (pid_n * 16 + r)[None, :],
             acc.to(tl.bfloat16), mask=mask_m[:, None])


def _run(kern, grid, desc, n_rows, K, stream, **kw):
    cur = torch.cuda.current_stream()
    if stream is not None and not isinstance(stream, int):
        stream = stream.value or 0      # ctypes.c_void_p from moe_w2_cubit
    if stream is None or stream == cur.cuda_stream:
        kern[grid](desc, n_rows, K=K, num_warps=_NUM_WARPS, **kw)
        return
    with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
        kern[grid](desc, n_rows, K=K, num_warps=_NUM_WARPS, **kw)


def make_launcher(tier: str, K: int, fp8: bool):
    """fn(desc, n_rows, pairs, stream=None) for one (tier, K)."""
    if tier in ("w2", "w2mc4"):
        kern, bn, extra = moe_w2_mm_tl, _BLOCK_N, {"BLOCK_N": _BLOCK_N}
    elif tier == "w4":
        kern, bn, extra = moe_w4_mm_tl, _BLOCK_N, {"BLOCK_N": _BLOCK_N}
    elif tier == "w4q":
        kern, bn, extra = moe_w4q_mm_tl, 16, {}
    else:
        raise ValueError(tier)

    def launch(desc, n_rows, pairs, stream=None):
        if n_rows % bn:
            raise ValueError(f"moe_w2 triton {tier}: n_rows={n_rows} not a "
                             f"multiple of {bn}")
        _run(kern, (n_rows // bn, pairs), desc, n_rows, K, stream,
             FP8=fp8, **extra)

    launch.tier, launch.K, launch.fp8 = tier, K, fp8
    return launch


def make_launchers(ks, fp8: bool) -> dict:
    fns = {}
    for k in ks:
        for tier in ("w2", "w2mc4", "w4", "w4q"):
            fns[(tier, k)] = make_launcher(tier, k, fp8)
    return fns


# --------------------------------------------------------------------------
# op-level parity cases (random planes through the real packers)
# --------------------------------------------------------------------------

def _act(g, m_rows, k, ue8m0=False):
    a = torch.randn(m_rows, k, generator=g) * 0.5
    ab = a.view(m_rows, k // 32, 32)
    a_s = ab.abs().amax(-1).clamp_min(1e-10) / 448.0
    if ue8m0:
        # per_token_group_quant_fp8(use_ue8m0=True): scale rounded up to 2^e
        a_s = torch.exp2(torch.ceil(torch.log2(a_s)))
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(
        torch.float8_e4m3fn).view(m_rows, k)
    return a8, a_s.float().contiguous()


def op_case(tier: str, k: int, n_rows: int, m_rows: int, seed: int = 7,
            ue8m0: bool = False):
    # vLLM sets a CUDA default device during model load: build the case and
    # its reference on the CPU explicitly
    with torch.device("cpu"):
        return _op_case(tier, k, n_rows, m_rows, seed, ue8m0)


def _op_case(tier: str, k: int, n_rows: int, m_rows: int, seed: int,
             ue8m0: bool = False):
    """(desc, bufs, ref): one pair, random weights packed by the serving
    packers; ref = fp32 dequant matmul on the CPU. Keep `bufs` alive while
    the kernel runs (desc holds raw pointers)."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_planes as P
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(seed)
    sexp = torch.randint(120, 132, (n_rows, k // 32), dtype=torch.uint8,
                         generator=g)
    a8, a_s = _act(g, m_rows, k, ue8m0)
    scale = torch.exp2(sexp.float() - 127.0).repeat_interleave(32, 1)
    if tier in ("w2", "w2mc4"):
        codes = torch.randint(0, 4, (n_rows, k), dtype=torch.uint8,
                              generator=g)
        w = torch.tensor([-4.0, -1.0, 1.0, 4.0])[codes.long()] * scale
        plane = P.pack_fragment_major(codes)
    elif tier == "w4":
        nib = torch.randint(0, 16, (n_rows, k), dtype=torch.uint8,
                            generator=g)
        w = P._E2M1_VALS.float().cpu()[nib.long()] * scale
        plane = P.pack_fp4_fragment_major(nib)
    elif tier == "w4q":
        # base codes + quintal refinement of the same nibbles: exact e2m1
        nib = torch.randint(0, 16, (n_rows, k), dtype=torch.uint8,
                            generator=g)
        w = P.quintal_dequant(nib) * scale
        plane = P.pack_fragment_major(P._NIBBLE_TO_CODE[nib.long()])
        qplane = P.pack_quintal_fragment_major(nib)
    else:
        raise ValueError(tier)
    ref = (a8.float() * a_s.repeat_interleave(32, 1)) @ w.T
    bufs = (a8.view(torch.uint8).to(dev), a_s.to(dev), plane.to(dev),
            P.pack_scales(sexp).to(dev),
            torch.zeros(m_rows, n_rows, dtype=torch.bfloat16, device=dev))
    ptrs = [bufs[0].data_ptr(), bufs[1].data_ptr(), bufs[2].data_ptr(),
            bufs[3].data_ptr(), bufs[4].data_ptr(), m_rows]
    if tier == "w4q":
        # 8-field ABI {a, as, base, ref, bs, c, m_rows, pad}
        bufs = bufs + (qplane.to(dev),)
        ptrs = ptrs[:3] + [bufs[5].data_ptr()] + ptrs[3:] + [0]
    desc = torch.tensor([ptrs], dtype=torch.int64, device=dev)
    return desc, bufs, ref


def worst_rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    got = got.float().cpu()
    if not bool(torch.isfinite(got).all()):
        return float("inf")
    return (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)


def self_test(fns: dict, ks, n_rows: int = 64) -> dict:
    """Boot parity gate: every registered w2/w4/w4q launcher at every K, decode
    (m=4) and prefill (m=16) rows, against the fp32 reference. Raises on a
    miss: a Triton codegen or layout bug fails here, attributed, instead of
    as silent output corruption."""
    out = {}
    for k in ks:
        for tier in ("w2", "w4", "w4q"):
            if (tier, k) not in fns:
                continue
            for m in (4, 16):
                desc, bufs, ref = op_case(tier, k, n_rows, m)
                fns[(tier, k)](desc, n_rows, 1)
                torch.cuda.synchronize()
                w = worst_rel(bufs[4], ref)
                out[(tier, k, m)] = w
                if not w < _GATE:
                    raise RuntimeError(
                        f"moe_w2 triton backend parity FAILED: {tier} K={k} "
                        f"m={m} worst_rel={w:.3e} (gate {_GATE}); "
                        f"dot={'fp8' if fns[(tier, k)].fp8 else 'bf16'}, "
                        f"triton {triton.__version__}, torch "
                        f"{torch.__version__}")
    return out
