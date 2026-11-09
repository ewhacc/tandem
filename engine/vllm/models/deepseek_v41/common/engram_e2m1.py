# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_ENGRAM_E2M1=1: requantize the Engram tables to 4-bit (MXFP4-style)
in place after loading, to measure the quality of a 4-bit Engram store.

The checkpoint rows are e4m3 with one UE8M0 scale per 32 values (8.25
bits/value). Per 32-block we pick a new UE8M0 exponent t and round each
value to e2m1 ({0, .5, 1, 1.5, 2, 3, 4, 6}); e2m1 values are exact e4m3
bytes, so the result is written back into the same FP8 storage with the
new scale byte and the lookup kernel serves it unchanged. t is the better
(block squared error) of the no-clip exponent ceil(log2(amax / 6)) and one
below it (saturating the largest values at 6).

pack_nibbles turns such rows into the packed store (config.json
"engram_embed_format": "e2m1", common/engram.py PACKED lookup): one byte per
two values, low nibble = even column, nibble = sign << 3 | grid index.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_ROWS = 1 << 18


def enabled() -> bool:
    return os.getenv("VLLM_ENGRAM_E2M1", "0") == "1"


def _round_e2m1(y: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(_GRID, device=y.device)
    mids = (grid[1:] + grid[:-1]) / 2
    a = y.abs().clamp(max=6.0)
    return grid[torch.bucketize(a, mids)] * torch.sign(y)


@torch.no_grad()
def _requant(w: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """w [R, D] e4m3, s [R, D/32] uint8 (UE8M0) on GPU -> new (w, s), err2, ref2."""
    R, D = w.shape
    x = w.float().view(R, D // 32, 32) * torch.exp2(s.float() - 127)[..., None]
    amax = x.abs().amax(-1)
    t0 = torch.ceil(torch.log2(amax.clamp_min(1e-30) / 6.0))
    best_q = best_t = best_e = None
    for t in (t0, t0 - 1):
        t = t.clamp(-127, 127)
        q = _round_e2m1(x / torch.exp2(t)[..., None])
        e = (q * torch.exp2(t)[..., None] - x).square().sum(-1)
        if best_e is None:
            best_q, best_t, best_e = q, t, e
        else:
            m = e < best_e
            best_q = torch.where(m[..., None], q, best_q)
            best_t = torch.where(m, t, best_t)
            best_e = torch.where(m, e, best_e)
    zero = amax == 0
    best_t = torch.where(zero, s.float() - 127, best_t)
    w2 = best_q.view(R, D).to(torch.float8_e4m3fn)
    s2 = (best_t + 127).to(torch.uint8)
    return w2, s2, float(best_e.sum()), float(x.square().sum())


def pack_nibbles(w: torch.Tensor) -> torch.Tensor:
    """[R, D] e4m3 rows holding e2m1 grid values -> [R, D/2] uint8."""
    v = w.float()
    grid = torch.tensor(_GRID, device=v.device)
    idx = torch.bucketize(v.abs(), (grid[1:] + grid[:-1]) / 2)
    assert bool((grid[idx] == v.abs()).all()), "rows are not on the e2m1 grid"
    code = (idx | ((v < 0).to(idx.dtype) << 3)).to(torch.uint8)
    return code[:, 0::2] | (code[:, 1::2] << 4)


def requant_packed(w: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """fp8 rows + scales (GPU) -> packed nibble rows, new scales, err2, ref2."""
    w2, s2, e, r = _requant(w, s)
    return pack_nibbles(w2), s2, e, r


@torch.no_grad()
def requant_table(weight: torch.Tensor, scales: torch.Tensor, name: str) -> None:
    """In place over a (possibly pinned host) table, chunked through the GPU."""
    dev = torch.device("cuda")
    err = ref = 0.0
    for r0 in range(0, weight.shape[0], _ROWS):
        r1 = min(r0 + _ROWS, weight.shape[0])
        w, s = weight[r0:r1].to(dev), scales[r0:r1].to(dev)
        w2, s2, e, r = _requant(w, s)
        weight[r0:r1].copy_(w2)
        scales[r0:r1].copy_(s2)
        err += e
        ref += r
    logger.info("Engram E2M1: %s %d x %d requantized to 4 bit, rel RMS error %.4f",
                name, weight.shape[0], weight.shape[1], (err / max(ref, 1e-30)) ** 0.5)
