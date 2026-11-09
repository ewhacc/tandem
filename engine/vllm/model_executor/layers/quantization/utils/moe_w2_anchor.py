# SPDX-License-Identifier: Apache-2.0
"""Mode 1b: semantic anchor caching for the base expert cache.

Experts routed by the request's anchor span (the leading system + tool
prompt, absolute positions < VLLM_MOE_ANCHOR_TOKENS) earn an anchor score.
The LRU eviction key of an owner becomes

    owner_tick + BETA * min(score, CAP) / CAP

so anchor experts outlive long decode stretches without being pinned. At
most MAX_FRAC of the pool may carry the boost (highest scores win), so the
anchor set can never starve the decode working set. Scores decay with a
wall-clock half-life. Eviction order only: outputs stay exact.

    VLLM_MOE_ANCHOR=1              enable (Mode 1b)
    VLLM_MOE_ANCHOR_TOKENS=2048    anchor span (absolute prompt positions)
    VLLM_MOE_ANCHOR_BETA=2000      max boost, in ticks (~decode steps)
    VLLM_MOE_ANCHOR_CAP=8          score that earns the full boost
    VLLM_MOE_ANCHOR_HALFLIFE_S=600
    VLLM_MOE_ANCHOR_MAX_FRAC=0.25
"""
import os
import time

import torch

_ON = os.getenv("VLLM_MOE_ANCHOR", "0") == "1"
_TOKENS = int(os.getenv("VLLM_MOE_ANCHOR_TOKENS", "2048"))
_BETA = float(os.getenv("VLLM_MOE_ANCHOR_BETA", "2000"))
_CAP = float(os.getenv("VLLM_MOE_ANCHOR_CAP", "8"))
_HALF = float(os.getenv("VLLM_MOE_ANCHOR_HALFLIFE_S", "600"))
_FRAC = float(os.getenv("VLLM_MOE_ANCHOR_MAX_FRAC", "0.25"))


class _State:
    mask = None          # [T] bool (GPU) for the current eager forward
    score = None         # [L, E] float32 (host)
    t_decay = 0.0
    n_marked = 0         # anchor routings recorded
    n_boosted = 0        # owners carrying a boost at the last eviction
    evict_anchor = 0     # evictions that took an anchor-scored owner
    evict_total = 0


_S = _State()


def enabled() -> bool:
    return _ON


def set_positions(positions: torch.Tensor | None) -> None:
    """Runner, before an eager (prefill) forward: positions of the
    scheduled tokens. None for decode-sized steps (no marking)."""
    if not _ON:
        return
    _S.mask = None if positions is None else positions < _TOKENS


def _decay(now: float) -> None:
    if _S.score is None:
        return
    dt = now - _S.t_decay
    if dt > 1.0:
        _S.score.mul_(0.5 ** (dt / _HALF))
        _S.t_decay = now


def note(tier, layer_key: int, topk_ids: torch.Tensor) -> None:
    """Prefill path (eager): credit the anchor tokens' routed experts."""
    m = _S.mask
    if not _ON or m is None:
        return
    n = min(m.shape[0], topk_ids.shape[0])     # rows past n are padding
    E = tier._mirror.shape[1]
    ids = topk_ids[:n][m[:n]].reshape(-1).long()
    cnt = torch.bincount(ids, minlength=E).float().cpu()
    if _S.score is None:
        _S.score = torch.zeros(tier._mirror.shape, dtype=torch.float32)
        _S.t_decay = time.monotonic()
    _decay(time.monotonic())
    _S.score[layer_key] += cnt
    _S.n_marked += int(ids.numel())


def boost(li: torch.Tensor, ei: torch.Tensor, n_slots: int) -> torch.Tensor | None:
    """Per-slot additive LRU key boost (ticks), or None. li/ei: owner
    (layer, expert) per slot, clamped >= 0; caller masks free slots."""
    if not _ON or _S.score is None:
        return None
    raw = _S.score[li, ei]
    nz = raw > 0.5
    lim = int(_FRAC * n_slots)
    if int(nz.sum()) > lim:
        # rank by the raw score (the clamped one ties at CAP)
        keep = torch.zeros_like(nz)
        if lim > 0:
            keep[torch.topk(torch.where(nz, raw, torch.zeros_like(raw)),
                            lim).indices] = True
        nz &= keep
    s = raw.clamp(max=_CAP)
    _S.n_boosted = int(nz.sum())
    return torch.where(nz, s * (_BETA / _CAP), torch.zeros_like(s)).double()


def note_evictions(li: torch.Tensor, ei: torch.Tensor) -> None:
    if not _ON or _S.score is None or li.numel() == 0:
        return
    _S.evict_total += int(li.numel())
    _S.evict_anchor += int((_S.score[li, ei] > 0.5).sum())


def kpi_summary() -> str:
    if not _ON:
        return ""
    n_set = 0 if _S.score is None else int((_S.score > 0.5).sum())
    return (f" | anchor set={n_set} boosted={_S.n_boosted} "
            f"marked={_S.n_marked} evict_anchor={_S.evict_anchor}/"
            f"{_S.evict_total}")


def metrics() -> dict:
    if not _ON:
        return {}
    return {"anchor_set": 0 if _S.score is None
            else int((_S.score > 0.5).sum()),
            "anchor_boosted": _S.n_boosted,
            "anchor_marked": _S.n_marked,
            "anchor_evict": _S.evict_anchor,
            "evict_total": _S.evict_total}
