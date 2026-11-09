# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native block-scaled FP8 grouped matmul for Ada (SM89).

Ported from vllm-sm89-ds4 (Apache-2.0): the DeepSeek-V4 o_proj wo_a einsum
where DeepGEMM has no pre-SM90 kernel."""

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op


_BLOCK_K = 128
_BLOCK_N_SCALE = 128


@triton.jit
def _ada_fp8_grouped_mm_kernel(
    A,
    A_SCALE,
    B,
    B_SCALE,
    C,
    M,
    H,
    N,
    K,
    stride_am,
    stride_ah,
    stride_ak,
    stride_asm,
    stride_ash,
    stride_ask,
    stride_bn,
    stride_bk,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_ch,
    stride_cn,
    B_SCALE_E8M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N_SCALE: tl.constexpr,
):
    """Compute ``bhr,hdr->bhd`` without materializing dequantized inputs."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_n = pid % num_pid_n
    pid_h = (pid // num_pid_n) % H
    pid_m = pid // (num_pid_n * H)

    # Repeating the final partial M tile avoids a masked tensor-core operand.
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        A
        + offs_m[:, None] * stride_am
        + pid_h * stride_ah
        + offs_k[None, :] * stride_ak
    )
    b_rows = pid_h * N + offs_n
    b_ptrs = B + offs_k[:, None] * stride_bk + b_rows[None, :] * stride_bn

    as_ptrs = A_SCALE + offs_m * stride_asm + pid_h * stride_ash
    bs_rows = pid_h * (N // BLOCK_N_SCALE) + offs_n // BLOCK_N_SCALE
    bs_ptrs = B_SCALE + bs_rows * stride_bsn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_scale = tl.load(as_ptrs + k_block * stride_ask)
        b_scale = tl.load(bs_ptrs + k_block * stride_bsk)
        if B_SCALE_E8M0:
            scale_bits = b_scale.to(tl.int32) << 23
            b_scale = scale_bits.to(tl.float32, bitcast=True)

        block_acc = tl.dot(a, b, out_dtype=tl.float32)
        accumulator += block_acc * a_scale[:, None] * b_scale[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = (
        C
        + offs_cm[:, None] * stride_cm
        + pid_h * stride_ch
        + offs_n[None, :] * stride_cn
    )
    c_mask = (offs_cm[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)


def _kernel_config(num_tokens: int) -> dict[str, int]:
    # Decode needs enough N tiles to occupy the Ada GPU. Larger prefills
    # benefit more from wider tiles and weight reuse across token rows.
    if num_tokens <= 16:
        return {
            "BLOCK_M": 16,
            "BLOCK_N": 64,
            "num_warps": 4,
            "num_stages": 3,
        }
    if num_tokens <= 64:
        return {
            "BLOCK_M": 32,
            "BLOCK_N": 64,
            "num_warps": 4,
            "num_stages": 3,
        }
    return {
        "BLOCK_M": 64,
        "BLOCK_N": 128,
        "num_warps": 8,
        "num_stages": 3,
    }


def _ada_fp8_grouped_mm_impl(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    assert a.ndim == 3 and a.dtype == torch.float8_e4m3fn
    assert a_scale.ndim == 3 and a_scale.dtype == torch.float32
    assert b.ndim == 2 and b.dtype == torch.float8_e4m3fn
    assert b_scale.ndim == 2
    assert b_scale.dtype in (torch.float32, torch.float8_e8m0fnu)

    m, h, k = a.shape
    assert m > 0 and h > 0
    assert a_scale.shape == (m, h, k // _BLOCK_K)
    assert k % _BLOCK_K == 0
    assert b.shape[0] % h == 0 and b.shape[1] == k
    n = b.shape[0] // h
    assert n % _BLOCK_N_SCALE == 0
    assert b_scale.shape == (h * n // _BLOCK_N_SCALE, k // _BLOCK_K)

    out = torch.empty((m, h, n), dtype=torch.bfloat16, device=a.device)
    b_scale_e8m0 = b_scale.dtype == torch.float8_e8m0fnu
    kernel_b_scale = b_scale.view(torch.uint8) if b_scale_e8m0 else b_scale
    config = _kernel_config(m)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]) * h * triton.cdiv(n, meta["BLOCK_N"]),)

    _ada_fp8_grouped_mm_kernel[grid](
        a,
        a_scale,
        b,
        kernel_b_scale,
        out,
        m,
        h,
        n,
        k,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a_scale.stride(0),
        a_scale.stride(1),
        a_scale.stride(2),
        b.stride(0),
        b.stride(1),
        kernel_b_scale.stride(0),
        kernel_b_scale.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B_SCALE_E8M0=b_scale_e8m0,
        BLOCK_K=_BLOCK_K,
        BLOCK_N_SCALE=_BLOCK_N_SCALE,
        **config,
    )
    return out


def _ada_fp8_grouped_mm_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    del a_scale, b_scale
    m, h, _ = a.shape
    return torch.empty((m, h, b.shape[0] // h), dtype=torch.bfloat16, device=a.device)


direct_register_custom_op(
    op_name="ada_fp8_grouped_mm",
    op_func=_ada_fp8_grouped_mm_impl,
    fake_impl=_ada_fp8_grouped_mm_fake,
)


def ada_fp8_grouped_mm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """Run the native SM89 grouped 128x128 block-scaled FP8 matmul."""
    return torch.ops.vllm.ada_fp8_grouped_mm(a, a_scale, b, b_scale)
