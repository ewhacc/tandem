# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert ownership across tensor-parallel ranks (VLLM_MOE_W2_EP=1).

vLLM runs the model at TP=W as usual (attention, shared experts and the
MoE output all-reduce are unchanged), but the moe_w2 routed experts are
not split along the intermediate dimension: each rank owns whole experts,
e % W == rank, served from the TP=1 planes (the Tandem package as
shipped; a VQ super-block of 256 k cannot be cut at V4.1's I/2 = 1152).
A rank computes only its own experts' pairs; routes to other ranks' experts
get weight 0 and are dropped by moe_align_block_size (expert_map with
ignore_invalid_experts). Each rank's output is the partial sum over its
experts, which the TP all-reduce adds up exactly as it adds the K-split
partial sums of plain TP.

Expert caches, the pinned RAM tier and the CPU experts are per rank and
see only the rank's own experts.
"""

import functools
import os

import torch


def enabled() -> bool:
    return os.getenv("VLLM_MOE_W2_EP", "0") == "1"


@functools.cache
def ids() -> tuple[int, int]:
    """(world, rank) of the expert split; (1, 0) when off or outside TP."""
    if not enabled():
        return 1, 0
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        return (get_tensor_model_parallel_world_size(),
                get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001 - no TP group (offline tools)
        return 1, 0


def active() -> bool:
    return ids()[0] > 1


def local(e: int) -> bool:
    w, r = ids()
    return e % w == r


_RANK_COUNTS: dict[int, dict[int, int]] | None = None


def _counts() -> dict[int, dict[int, int]]:
    """Calibration routing counts per layer (the package ranking tsv)."""
    global _RANK_COUNTS
    if _RANK_COUNTS is None:
        _RANK_COUNTS = {}
        path = os.getenv("VLLM_MOE_W2_BASE_RAM_RANK", "")
        if path and os.path.exists(path):
            with open(path) as f:
                for ln in f:
                    if ln.startswith("#") or not ln.strip():
                        continue
                    li, _rk, ei, c = ln.split()[:4]
                    _RANK_COUNTS.setdefault(int(li), {})[int(ei)] = int(c)
    return _RANK_COUNTS


_TABLES: dict = {}


def tables(layer_key: int, n_experts: int, dev):
    """(expert_map int32 [E]: e for own experts, -1 else; fallback id, a
    0-d int tensor: the layer's most-routed own expert) for one layer. The
    fallback replaces dropped routes in the ids that feed seen marking,
    prefill fetches and the CPU experts (weight 0, already hot), so the
    other ranks' experts are never fetched or counted here."""
    key = (layer_key, n_experts, str(dev))
    t = _TABLES.get(key)
    if t is None:
        w, r = ids()
        own = [e for e in range(n_experts) if e % w == r]
        emap = torch.full((n_experts,), -1, dtype=torch.int32)
        emap[own] = torch.tensor(own, dtype=torch.int32)
        cnt = _counts().get(layer_key, {})
        fb = max(own, key=lambda e: (cnt.get(e, 0), -e))
        t = (emap.to(dev), torch.tensor(fb, dtype=torch.int32, device=dev))
        _TABLES[key] = t
    return t


def remap(layer_key: int, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
          n_experts: int):
    """-> (weights with other ranks' routes zeroed, ids with those routes
    set to the fallback, expert_map). Capture-safe (fixed shapes)."""
    emap, fb = tables(layer_key, n_experts, topk_ids.device)
    own = emap[topk_ids.long()] >= 0
    w = torch.where(own, topk_weights, torch.zeros_like(topk_weights))
    i = torch.where(own, topk_ids, fb.to(topk_ids.dtype))
    return w, i, emap
