# SPDX-License-Identifier: Apache-2.0
"""E3 "Slipstream" shadow rows (Model Runner V2, base cache).

VLLM_MOE_W2_SHADOW_FROM=S (0 = off): in a speculative verify step, the rows
of a request with in-request index >= S are SHADOW rows. A shadow row's
routed experts that are not resident in the base pool are DROPPED - weight 0,
id redirected to a resident expert of the layer (the _apply_topp trick: the
pair is never fetched, read over PCIe or sent to the CPU, and contributes an
exact 0) - instead of being served on the miss path. Rows < S are exact.

A shadow row that dropped anything in any layer is DIRTY. Its output, and
every later row's (their attention reads its KV), is lossy, so acceptance is
truncated at each request's first dirty row: num_sampled <= index of that
row. Up to there the output is bit-identical to plain speculative decoding.
A shadow row that dropped nothing is exact and is accepted as usual
(accept-if-clean): "shadow or real" is decided after the forward.

The router still sees the shadow row's true (drifted) input, so the route
trace (VLLM_MOE_W2_ROUTE_TRACE, recorded before the drop) holds the shadow
rows' own routing: the drift measurement and the prefetch signal.

Graph-safe: persistent buffers sized at load, rewritten in place each step.
"""

import os

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_FROM = int(os.getenv("VLLM_MOE_W2_SHADOW_FROM", "0") or 0)
# mask/dirty/local/req: per token row; stats (GPU int64): shadow rows,
# dirty rows, drops, steps truncated, tokens cut
_B: dict = {}
_LOG_EVERY = 500
_steps = [0]


def enabled() -> bool:
    return _FROM > 0


def alloc(max_num_tokens: int, max_num_reqs: int, device) -> None:
    if not enabled() or _B:
        return
    n = max_num_tokens
    _B["mask"] = torch.zeros(n, dtype=torch.bool, device=device)
    _B["dirty"] = torch.zeros(n, dtype=torch.bool, device=device)
    _B["local"] = torch.zeros(n, dtype=torch.int32, device=device)
    _B["req"] = torch.zeros(n, dtype=torch.int64, device=device)
    _B["host"] = torch.zeros((3, n), dtype=torch.int64, pin_memory=True)
    _B["stats"] = torch.zeros(5, dtype=torch.int64, device=device)
    _B["max_reqs"] = max_num_reqs
    _B["n"] = 0
    logger.info("moe_w2 shadow rows: verify rows >= %d drop their "
                "non-resident experts; acceptance truncated at the first "
                "dirty row", _FROM)


def set_rows(input_batch) -> None:
    """Host, before the forward: mark this step's shadow rows."""
    if not _B:
        return
    qsl = input_batch.query_start_loc_np
    nreq = input_batch.num_reqs
    ndraft = input_batch.num_draft_tokens_per_req
    N = int(qsl[nreq])
    h = _B["host"]
    if ndraft is None or N == 0:
        h[0, :N] = 0
    else:
        counts = np.diff(qsl[:nreq + 1]).astype(np.int64)
        req = np.repeat(np.arange(nreq, dtype=np.int64), counts)
        local = np.arange(N, dtype=np.int64) - np.repeat(
            qsl[:nreq].astype(np.int64), counts)
        spec = np.repeat(np.asarray(ndraft[:nreq]) > 0, counts)
        h[0, :N] = torch.from_numpy((spec & (local >= _FROM)).astype(np.int64))
        h[1, :N] = torch.from_numpy(local)
        h[2, :N] = torch.from_numpy(req)
    _B["n"] = N
    # padded rows (cudagraph size > N) keep mask 0: never shadow
    _B["mask"].zero_()
    _B["mask"][:N].copy_(h[0, :N].bool(), non_blocking=True)
    _B["local"][:N].copy_(h[1, :N], non_blocking=True)
    _B["req"][:N].copy_(h[2, :N], non_blocking=True)
    _B["dirty"].zero_()


def apply(topk_weights: torch.Tensor, topk_ids: torch.Tensor,
          slot_row: torch.Tensor):
    """In the MoE forward, before moe_align: drop shadow rows' misses."""
    T = topk_ids.shape[0]
    m = _B["mask"][:T]
    drop = (slot_row[topk_ids.long()] < 0) & m.unsqueeze(1)
    _B["dirty"][:T].logical_or_(drop.any(dim=1))
    _B["stats"][2:3].add_(drop.sum())
    # first resident expert of the layer (0 if none: then one host read)
    r = torch.argmax((slot_row >= 0).to(torch.int8)).to(topk_ids.dtype)
    ids = torch.where(drop, r, topk_ids)
    w = torch.where(drop, torch.zeros_like(topk_weights), topk_weights)
    return w, ids


def truncate(num_sampled: torch.Tensor, num_rejected: torch.Tensor,
             num_reqs: int) -> None:
    """After rejection sampling: cap each request's accepted count at the
    in-request index of its first dirty row (in place)."""
    N = _B["n"]
    if N == 0 or num_reqs == 0:
        return
    dirty = _B["dirty"][:N]
    if torch.distributed.is_initialized():
        from vllm.distributed.parallel_state import get_tp_group
        tp = get_tp_group()
        if tp.world_size > 1:
            # ranks may differ in residency: a row is dirty if any rank
            # dropped one of its experts
            d = dirty.to(torch.int32)
            torch.distributed.all_reduce(
                d, op=torch.distributed.ReduceOp.MAX,
                group=tp.device_group)
            dirty = d.bool()
    local = _B["local"][:N].to(torch.int64)
    big = torch.iinfo(torch.int64).max
    val = torch.where(dirty, local, torch.full_like(local, big))
    first = torch.full((num_reqs,), big, dtype=torch.int64,
                       device=dirty.device)
    first.scatter_reduce_(0, _B["req"][:N], val, reduce="amin")
    old = num_sampled[:num_reqs]
    new = torch.minimum(old.to(torch.int64), first).to(old.dtype)
    cut = old - new
    num_rejected[:num_reqs] += cut
    num_sampled[:num_reqs].copy_(new)
    s = _B["stats"]
    s[0].add_(_B["mask"][:N].sum())
    s[1].add_(dirty.sum())
    s[3].add_((cut > 0).sum())
    s[4].add_(cut.sum())
    _steps[0] += 1
    if _steps[0] % _LOG_EVERY == 0:
        v = s.tolist()
        logger.info(
            "KPI shadow: %d shadow rows, %.1f%% dirty, %.1f drops/shadow "
            "row, %d steps truncated, %d tokens cut (cumulative, %d steps)",
            v[0], 100.0 * v[1] / max(v[0], 1), v[2] / max(v[0], 1),
            v[3], v[4], _steps[0])
