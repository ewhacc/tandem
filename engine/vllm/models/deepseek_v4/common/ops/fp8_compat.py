# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Software OCP e4m3 for Triton kernels on GPUs without fp8e4nv (SM80/86).

Triton rejects ``tl.float8e4nv`` below SM89 (only fp8e4b15/fp8e5 exist
there). The DeepSeek-V4 cache / indexer / o_proj quantizers only need the
e4m3 BYTES, so below SM89 they encode them with integer math instead and
store through uint8 pointers. ``SW_FP8`` is a Triton constexpr global: the
hardware branch is not compiled where it is unsupported, and SM89+ keeps
the hardware conversion unchanged. VLLM_TRITON_SW_FP8=1 forces the
software path anywhere (for testing it against the hardware one).
"""

import os

import torch

from vllm.triton_utils import tl, triton


def _sw_fp8() -> bool:
    if os.getenv("VLLM_TRITON_SW_FP8", "0") == "1":
        return True
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() < (8, 9)


SW_FP8 = tl.constexpr(_sw_fp8())


@triton.jit
def f32_to_e4m3_u8(x):
    """fp32 (|x| <= 448, finite) -> OCP e4m3 byte, round to nearest even,
    saturating; bit-identical to the hardware cvt.rn.satfinite."""
    xb = x.to(tl.uint32, bitcast=True)
    sign = (xb >> 24) & 0x80
    a = tl.abs(x)
    ab = a.to(tl.uint32, bitcast=True)
    e = (ab >> 23).to(tl.int32) - 127
    mant = ab & 0x7FFFFF
    keep = mant >> 20
    rem = mant & 0xFFFFF
    up = (rem > 0x80000) | ((rem == 0x80000) & ((keep & 1) == 1))
    # a mantissa carry (keep = 7 -> 8) rolls into the exponent field
    normal = (((e + 7).to(tl.uint32)) << 3) + keep + up.to(tl.uint32)
    normal = tl.minimum(normal, 0x7E)                     # 448, satfinite
    # subnormals (a < 2^-6): k * 2^-9, k = rint(a * 512) (k = 8 is 2^-6)
    k = ((a * 512.0 + 12582912.0) - 12582912.0).to(tl.uint32)
    mag = tl.where(e < -6, k, normal)
    return (mag | sign).to(tl.uint8)


@triton.jit
def e4m3_u8_to_f32(b):
    """OCP e4m3 byte -> fp32 (exact)."""
    b = b.to(tl.uint32)
    s = b >> 7
    e = (b >> 3) & 15
    m = b & 7
    nrm = ((s << 31) | ((e + 120) << 23) | (m << 20)).to(tl.float32,
                                                          bitcast=True)
    sub = m.to(tl.float32) * 0.001953125
    sub = tl.where(s > 0, -sub, sub)
    return tl.where(e > 0, nrm, sub)


def dsv4_has_cutedsl() -> bool:
    """CuTe-DSL DSv4 kernels emit hardware e4m3 conversions that ptxas
    rejects below SM89; the Triton versions (software e4m3) run there."""
    from vllm.utils.import_utils import has_cutedsl

    return has_cutedsl() and not SW_FP8
