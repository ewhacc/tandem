# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1 benchmark metrics for the moe_w2 expert tiers (one JSONL stream).

Every executed step reports into a window; every VLLM_MOE_METRICS_EVERY steps
one JSON line is appended to VLLM_MOE_METRICS_PATH. Off (zero work beyond a
falsy check per hook) unless the path is set.

Per window:
  steps / decode_steps / tokens        what was executed
  base_miss_pairs, base_zero_miss      first-pass misses of the BASE cache;
                                       zero_miss = steps with no miss (the
                                       throughput KPI: replays are per step)
  base_replays                         mandatory miss->fetch->replay passes
  gate_decisions / gate_fires          confidence-gate firing
  gate_promotes / gate_replays         FP4 force-promotes and their replays
  h2d_bytes_{base,fp4}                 bytes moved into each GPU pool
  prefetched_{base,fp4}                of which speculative (lookahead)
  demand_filled_base                   of which in-graph demand fill
  base_routed_pairs, base_hit_rate     decode routings and the fraction
                                       found resident (no miss, no demand fill)
  step_ms_mean                         host wall time between step ends
"""

import json
import os
import time

_PATH = os.getenv("VLLM_MOE_METRICS_PATH", "")
_EVERY = max(int(os.getenv("VLLM_MOE_METRICS_EVERY", "200")), 1)
_MODE = os.getenv("VLLM_MOE_CACHE_MODE", "")


def enabled() -> bool:
    return bool(_PATH)


class _Window:
    def __init__(self):
        self.reset()
        self.last_t = None
        self.prev = {}          # tier tag -> (n_promoted, n_prefetched)

    def reset(self):
        self.steps = 0
        self.decode_steps = 0
        self.tokens = 0
        self.base_miss_pairs = 0
        self.base_zero_miss = 0
        self.base_replays = 0
        self.gate_decisions = 0
        self.gate_fires = 0
        self.gate_promotes = 0
        self.gate_replays = 0
        self.h2d = {"base": 0, "fp4": 0}
        self.prefetched = {"base": 0, "fp4": 0}
        self.demand = 0
        self.routed = 0
        self.served = {"base": [0.0, 0.0], "fp4": [0.0, 0.0]}
        self.phase_ms = {}      # runner host phase -> summed wall ms
        self.step_ms = 0.0
        self.timed = 0


TOPK = 0    # routed experts per token, set by moe_w2_cubit's forward
_mark_t = None


def mark(phase: str | None = None) -> None:
    """Runner phase timer: mark() opens a step, mark(name) charges the wall
    time since the previous mark to `name` (host view; a phase that syncs
    on the GPU also absorbs the GPU time it waits for)."""
    global _mark_t
    if not _PATH:
        return
    now = time.perf_counter()
    if phase is None and _entry_t is not None:
        # execute_model entry -> first moe_w2 hook: vLLM input prep
        _W.phase_ms["prep"] = _W.phase_ms.get("prep", 0.0) + (now - _entry_t) * 1e3
    if phase is not None and _mark_t is not None:
        _W.phase_ms[phase] = _W.phase_ms.get(phase, 0.0) + (now - _mark_t) * 1e3
    _mark_t = now
_entry_t = None


def mark_entry() -> None:
    """execute_model entry: charges the time since the previous step's last
    mark (sampling, RPC, scheduler, idle) to "gap" and opens "prep"."""
    global _entry_t
    if not _PATH:
        return
    now = time.perf_counter()
    if _mark_t is not None:
        _W.phase_ms["gap"] = _W.phase_ms.get("gap", 0.0) + (now - _mark_t) * 1e3
    _entry_t = now


_W = _Window()
_seq = 0


def on_base_step(miss_pairs: int, replayed: bool) -> None:
    """Base-cache first-pass outcome of one step (from DeltaTier.kpi_step)."""
    if not _PATH:
        return
    _W.base_miss_pairs += int(miss_pairs)
    _W.base_zero_miss += int(miss_pairs == 0)


def on_base_replays(n: int) -> None:
    if _PATH:
        _W.base_replays += int(n)


def on_gate_decision(fired: bool) -> None:
    if _PATH:
        _W.gate_decisions += 1
        _W.gate_fires += int(fired)


def on_gate_promote(n_promoted: int) -> None:
    """One promote->replay iteration of a fired step."""
    if _PATH:
        _W.gate_promotes += int(n_promoted)
        _W.gate_replays += int(n_promoted > 0)


def _tier_delta(tier, tag: str) -> None:
    if tier is None:
        return
    cur = (int(tier._n_promoted), int(getattr(tier, "_kpi_prefetched", 0)),
           int(getattr(tier, "_kpi_demand", 0)),
           float(getattr(tier, "_m_hits", 0.0)),
           float(getattr(tier, "_m_active", 0.0)))
    p = _W.prev.get(tag)
    if p is not None:
        # bytes crossing PCIe per fetch: the FP4 host row under R2'
        row = tier._cvt.src_row if getattr(tier, "_cvt", None) else tier.slot_bytes
        _W.h2d[tag] += max(cur[0] - p[0], 0) * int(row)
        # _kpi_prefetched is reset by the tier's own KPI window; a decrease
        # means it rolled over, so count the new value from zero.
        d = cur[1] - p[1]
        _W.prefetched[tag] += d if d >= 0 else cur[1]
        _W.demand += max(cur[2] - p[2], 0)
        # routings (manager-consumed seen windows) served from this pool:
        # base = not read from host by DIRECT, fp4 = upgraded to FP4
        _W.served[tag][0] += max(cur[3] - p[3], 0.0)
        _W.served[tag][1] += max(cur[4] - p[4], 0.0)
    _W.prev[tag] = cur


def end_step(base_tier, fp4_tier, num_tokens: int, num_reqs: int) -> None:
    """Runner hook after a step is fully executed (replays included)."""
    global _seq
    if not _PATH:
        return
    now = time.perf_counter()
    if _W.last_t is not None:
        _W.step_ms += (now - _W.last_t) * 1e3
        _W.timed += 1
    _W.last_t = now
    _W.steps += 1
    _W.tokens += int(num_tokens)
    _W.decode_steps += int(num_tokens <= num_reqs)
    if base_tier is None:
        # fully resident base (Mode 0): no step can miss
        _W.base_zero_miss += 1
    elif num_tokens <= num_reqs:
        _W.routed += int(num_tokens) * base_tier.n_layers * TOPK
    _tier_delta(base_tier, "base")
    _tier_delta(fp4_tier, "fp4")
    if _W.steps < _EVERY:
        return
    _seq += 1
    rec = dict(
        t=round(time.time(), 3), seq=_seq, mode=_MODE, steps=_W.steps,
        decode_steps=_W.decode_steps, tokens=_W.tokens,
        base_miss_pairs=_W.base_miss_pairs,
        base_zero_miss_steps=_W.base_zero_miss,
        base_replays=_W.base_replays,
        gate_decisions=_W.gate_decisions, gate_fires=_W.gate_fires,
        gate_promotes=_W.gate_promotes, gate_replays=_W.gate_replays,
        h2d_bytes_base=_W.h2d["base"], h2d_bytes_fp4=_W.h2d["fp4"],
        prefetched_base=_W.prefetched["base"],
        prefetched_fp4=_W.prefetched["fp4"],
        demand_filled_base=_W.demand,
        phase_ms={k: round(v / _W.steps, 3) for k, v in _W.phase_ms.items()},
        base_routed_pairs=_W.routed,
        # a demand fill is a miss served in-graph; a lookahead prefetch that
        # landed in time is a hit
        base_hit_rate=(round(1.0 - (_W.base_miss_pairs + _W.demand)
                             / _W.routed, 4) if _W.routed else None),
        step_ms_mean=round(_W.step_ms / max(_W.timed, 1), 3),
        base_slots=(base_tier.n_slots if base_tier is not None else 0),
        fp4_slots=(fp4_tier.n_slots if fp4_tier is not None else 0),
        base_pool_served=(round(_W.served["base"][0] / _W.served["base"][1], 4)
                          if _W.served["base"][1] else None),
        fp4_served=(round(_W.served["fp4"][0] / _W.served["fp4"][1], 4)
                    if _W.served["fp4"][1] else None),
    )
    try:
        with open(_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass  # observability must never break serving
    _W.reset()
