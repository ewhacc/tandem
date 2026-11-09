# SPDX-License-Identifier: Apache-2.0
"""moe_w2 hooks of the GPU model runner (V1), in one place.

The runner calls four entry points around its model forward; everything
moe_w2 does per step (manager drain, prefetch ring, anchors, base-cache
miss -> fetch -> replay fixed point, confidence-gate FP4 re-forward, step
close + metrics) lives here. `forward()` is a runner closure that re-runs
the step's forward under the same forward context (one CUDA-graph replay).
"""
import os
import time
import types

import torch

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import moe_w2_delta
from vllm.model_executor.layers.quantization.utils import (
    moe_w2_metrics as _w2m)

try:
    from vllm.model_executor.layers.quantization.utils import moe_w2_gate
except Exception:  # noqa: BLE001
    import os
    if os.getenv("VLLM_MOE_W2_GATE", "0") == "1":
        raise
    moe_w2_gate = None

logger = init_logger(__name__)


def promote_consensus(
    tier,
    tp_group,
    local_miss: int,
    *,
    pin: bool,
    where: str,
) -> None:
    """Complete local base-cache I/O on every TP rank before replay."""
    error = None
    if local_miss > 0:
        try:
            tier.force_promote(max_promote=None, pin=pin)
        except BaseException as e:  # preserve consensus after local I/O fault
            error = e
    if tp_group.world_size > 1:
        ok = torch.tensor(
            [0 if error is not None else 1],
            dtype=torch.int32,
            device=tier.dev,
        )
        torch.distributed.all_reduce(
            ok,
            op=torch.distributed.ReduceOp.MIN,
            group=tp_group.device_group,
        )
        all_ok = bool(ok.item())
    else:
        all_ok = error is None
    if not all_ok:
        message = (
            f"moe_w2 mandatory base-cache promotion failed before {where}; "
            "refusing to replay or return zero-contribution logits")
        if error is not None:
            raise RuntimeError(message) from error
        raise RuntimeError(message + " (failure occurred on a peer TP rank)")


_moe_w2_promote_consensus = promote_consensus


def mark_entry() -> None:
    """execute_model entry: metrics ("gap" / "prep" host phases), then the
    deferred gate's hand-off - hold the stream behind the doorbell so this
    step's input prep overlaps the pending step's GPU work, or (no doorbell)
    resolve the pending verdict before any persistent buffer is touched."""
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_metrics as _w2m)
    _w2m.mark_entry()
    if _PENDING is None or _PENDING.get("out") is None:
        return
    if _DOORBELL_ON and _doorbell_arm(_PENDING):
        return
    resolve_pending()


def pre_forward(runner, scheduler_output, input_ids, num_scheduled_tokens,
                max_num_scheduled_tokens) -> None:
    """Before the step's forward: drain the managers, reserve the prefetch
    ring, set the anchor span, run the draft-affinity prefetch."""
    # Manager residency commits are scheduled after the previous step.
    # Drain them before this forward so slot-table/pool mutation cannot
    # overlap graph readers.
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_metrics as _w2m)
    _w2m.mark()
    resolve_pending()   # deferred gate: doorbell-held step's verdict
    _w2m.mark("resolve")
    for _w2_tier in (moe_w2_delta._BASE_TIER, moe_w2_delta._TIER):
        if _w2_tier is None:
            continue
        if _w2_tier._async_fill():
            # DIRECT_ASYNC: the manager only plans (host state, under the
            # tier lock); its GPU work is issued here, stream-ordered.
            _w2_tier.flush_fills()
        else:
            _w2_tier.wait_manager_idle()
    _w2m.mark("mgr_wait")
    _zero_null_blocks(runner)
    from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
    if moe_w2_cubit._ROUTE_TRACE:
        n = int(scheduler_output.total_num_scheduled_tokens)
        moe_w2_cubit.route_trace_step(input_ids[:n], runner.positions[:n])

    # moe_w2 Mode 2b: reserve this step's prefetch ring (decode-sized
    # steps only; prefill fetches per layer via ensure_resident).
    if (moe_w2_delta._BASE_TIER is not None
            and not runner.is_pooling_model
            and max_num_scheduled_tokens <= 4):
        from vllm.model_executor.layers.quantization.utils import (
            moe_w2_vprefetch)
        if moe_w2_vprefetch.enabled():
            moe_w2_vprefetch.begin_step(moe_w2_delta._BASE_TIER)
    _w2m.mark("ring_begin")
    # moe_w2 Mode 1b: anchor span of eager (prefill) forwards
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_anchor)
    if moe_w2_anchor.enabled() and moe_w2_delta._BASE_TIER is not None:
        moe_w2_anchor.set_positions(
            runner.positions[:num_scheduled_tokens]
            if max_num_scheduled_tokens > 4 else None)

    # moe_w2 draft-affinity PREFETCH (VLLM_MOE_W2_PREFETCH=1): before the
    # forward, fold the PREVIOUS step's in-graph routing log into the
    # token->experts table and prefetch this step's predicted experts on
    # the tier's side stream. Under MTP this step's input ids ARE last
    # step's sampled+draft tokens — the draft signal. Decode-shaped
    # steps only (a prefill chunk would poison the table; prefill
    # prefetches via ensure_resident anyway).
    if moe_w2_delta._BASE_TIER is not None and not runner.is_pooling_model:
        try:
            _w2d = moe_w2_delta
            _btier = _w2d._BASE_TIER
            if (
                _btier is not None
                and _btier.route_log is not None
                and max_num_scheduled_tokens <= 4
            ):
                _n_real = min(
                    scheduler_output.total_num_scheduled_tokens,
                    _btier.route_log.shape[1])
                _btier.draft_prefetch(input_ids[:_n_real])
        except Exception as e:  # noqa: BLE001 - never crash serving
            logger.warning_once("moe_w2 draft prefetch skipped: %s", e)



_DBG_VERIFY = int(os.getenv("VLLM_MOE_W2_DEBUG_VERIFY", "0") or 0)
_dbg_steps = 0


def _debug_verify(btier, where: str, nonconv: bool = False) -> None:
    """Debug (VLLM_MOE_W2_DEBUG_VERIFY=N): every N steps compare up to 64
    mapped pool slots with their host-store rows, and on a non-converged
    replay log the routed pairs that are still unmapped."""
    global _dbg_steps
    _dbg_steps += 1
    if _dbg_steps % _DBG_VERIFY and not nonconv:
        return
    torch.cuda.synchronize()
    if nonconv:
        st = btier.slot_table.cpu()
        um = getattr(btier, "_dbg_unmapped", [])
        logger.warning(
            "moe_w2 DEBUG %s: last snapshot unmapped %s (fetch candidates "
            "%s, store layers %d/%d, slot_table now %s)", where, um[:32],
            getattr(btier, "_dbg_cand", None), len(btier._store),
            btier.n_layers, [int(st[l, e]) for l, e in um[:8]])
    with btier._lock:
        mirror = btier._mirror.clone()
        st = btier.slot_table.cpu()
    nz = (mirror >= 0).nonzero()
    if nz.shape[0] == 0:
        return
    pick = nz[torch.randperm(nz.shape[0])[:64]]
    bad, tbl = [], 0
    for li, ei in pick.tolist():
        slot = int(mirror[li, ei])
        if int(st[li, ei]) != slot:
            tbl += 1
        dev_row = btier.pool[slot].cpu()
        host_row = btier._store.rows_for([(li, ei)])[0].cpu()
        if not torch.equal(dev_row, host_row[:dev_row.numel()]):
            bad.append((li, ei, slot))
    logger.warning("moe_w2 DEBUG verify %s step %d: %d/%d slots differ from "
                   "store %s; slot_table!=mirror %d", where, _dbg_steps,
                   len(bad), pick.shape[0], bad[:8], tbl)


def pp_base_replay(runner, forward, model_output):
    """Pipeline parallel: repair this stage's base-cache misses (segment
    re-run to a fixed point) before its activations go downstream."""
    # moe_w2 BASE cache under PIPELINE parallelism: a miss is LOCAL to the
    # stage — its desc kernels zeroed the missing pairs' contributions and
    # bumped ITS miss counter, and the stage still holds this step's
    # inputs (static graph buffers / intermediate_tensors). Correctness
    # therefore needs only a SEGMENT re-run before the activations are
    # consumed downstream — ~1/pp_size of a forward, unlike the gate's
    # full-pipeline replay (the gate's signal, confidence, exists only at
    # the logits; the base's signal is per-stage). No PP collective: each
    # stage fixes its own segment; TP ranks WITHIN a stage decide via
    # MAX-reduce and replay together (the segment contains TP
    # collectives). Downstream stages never see the zeroed activations,
    # so no cross-stage coordination is needed. KV rewrites in the replay
    # are idempotent (same slot mapping, corrected values) — this covers
    # MTP verify steps too, same as the inline TP path. PP==1 keeps the
    # post-logits path below (unchanged).
    if (
        moe_w2_delta._BASE_TIER is not None
        and get_pp_group().world_size > 1
        and not runner.is_pooling_model
    ):
        try:
            _w2d = moe_w2_delta
            _btier = _w2d._BASE_TIER
            if _btier is not None:
                # open this step's pin scope: slots touched by any pass
                # below are ineligible for eviction until the next step
                _btier.step_begin()
                _miss = int(_btier.miss_count.item())
                _tp = get_tp_group()
                if _tp.world_size > 1:
                    _t = torch.tensor([_miss], device=_btier.dev)
                    torch.distributed.all_reduce(
                        _t, op=torch.distributed.ReduceOp.MAX,
                        group=_tp.device_group)
                    _max_miss = int(_t.item())
                else:
                    _max_miss = _miss
                _moe_w2_promote_consensus(
                    _btier,
                    _tp,
                    _miss,
                    pin=_w2d.fp_continue(0, _max_miss),
                    where="PP base segment replay",
                )
                _btier.kpi_step(
                    _max_miss, _max_miss > _w2d.base_miss_tol())
                # Replay to a FIXED POINT (bounded): the corrected early
                # layers can re-route later layers to experts the first
                # pass never fetched — those SECOND-ORDER misses zero
                # contributions inside the replay itself, making logits
                # depend on pool content (measured as cross-request
                # greedy nondeterminism). Re-read the counter after each
                # replay; fetch + replay again until miss-free (typically
                # converges in one extra pass).
                _replays = 0
                while _w2d.fp_continue(_replays, _max_miss):
                    _replays += 1
                    model_output = forward()
                    _miss = int(_btier.miss_count.item())
                    if _tp.world_size > 1:
                        _t = torch.tensor([_miss], device=_btier.dev)
                        torch.distributed.all_reduce(
                            _t, op=torch.distributed.ReduceOp.MAX,
                            group=_tp.device_group)
                        _max_miss = int(_t.item())
                    else:
                        _max_miss = _miss
                    _moe_w2_promote_consensus(
                        _btier,
                        _tp,
                        _miss,
                        pin=_w2d.fp_continue(_replays, _max_miss),
                        where="PP base fixed-point replay",
                    )
                _w2d.fp_validate_complete(_max_miss)
                if _replays:
                    _btier.kpi_fp(_replays, _max_miss)
                # PP lifecycle closes both tiers once in the worker
                # barrier after first-pass send and optional gate replay.
        except Exception as e:
            logger.exception(
                "moe_w2 mandatory PP base-cache replay failed")
            raise RuntimeError(
                "mandatory moe_w2 PP base-cache replay failed; "
                "refusing zero-contribution activations") from e

    return model_output


_DIRECT_LOG_EVERY = 500


def _direct_step_close(btier):
    """DIRECT step close, no host sync: fold this step's miss count into a
    device total (the counter is zeroed in-graph by the next forward), and
    record the step's end event — host-side pool mutations wait on it
    (DeltaTier._direct_fence) so a slot is never rewritten while a graph
    that may still read it is in flight. One .item() per 500 steps for the
    KPI line."""
    st = getattr(btier, "_direct_kpi", None)
    if st is None:
        st = btier._direct_kpi = {
            "tot": torch.zeros(1, dtype=torch.int64,
                               device=btier.miss_count.device),
            "steps": 0, "last": 0}
    st["tot"] += btier.miss_count
    ev = torch.cuda.Event()
    ev.record()
    btier._direct_ev = ev
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_vprefetch)
    moe_w2_vprefetch.snapshot()
    st["steps"] += 1
    if st["steps"] % _DIRECT_LOG_EVERY == 0:
        tot = int(st["tot"].item())
        logger.info(
            "[base] KPI DIRECT: %.2f host-served pairs/step over the last %d "
            "steps (no replays; cumulative %d pairs in %d steps) - pool "
            "%d slots, %d async fills%s", (tot - st["last"]) / _DIRECT_LOG_EVERY,
            _DIRECT_LOG_EVERY, tot, st["steps"], btier.n_slots,
            getattr(btier, "_kpi_async_fills", 0),
            moe_w2_vprefetch.kpi_summary() + _cpu_kpi())
        st["last"] = tot


def _cpu_kpi() -> str:
    from vllm.model_executor.layers.quantization.utils import moe_w2_cpu
    from vllm.models.deepseek_v41.common import engram_files
    e = engram_files.kpi()
    return moe_w2_cpu.kpi_summary() + (f"; {e}" if e else "")


def _gate_fire(runner, forward, logits, hidden_states, aux_hidden_states,
               logits_indices, _tp, _or_tp, _and_tp, _dump_pre, active):
    """One fired step: promote -> replay to a fixed point (GATE_FP_MAX),
    shared by the synchronous gate and the deferred one (GATE_DEFER).
    `active[0]` is True while a TP collective replay is in flight."""
    sample_hidden_states = hidden_states[logits_indices]
    _gate_promoted_total = 0
    # FIXED-POINT fire (2026-07-13): a single replay is
    # only FIRST-order — upgraded early layers re-route
    # later layers onto still-cold experts INSIDE the
    # replay, and those decide the token at 2-bit
    # (measured on GPQA: tau=1.00 scored 66.2% vs 72.2%
    # for a pool big enough to hide the residue behind
    # standing coverage; GSM8K, with stable routing,
    # showed no gap). Iterate promote->replay until no
    # rank promotes (routed set fully upgraded) or the
    # bound trips — restoring tau->1 convergence to the
    # mxfp4 ceiling regardless of pool size (above the
    # fire floor).
    _gate_iters = 0
    _step_budget = moe_w2_gate.step_promote_budget()
    # DIRECT: a replay's new base misses are read from
    # host in-graph (exact) - no base repair passes
    _gate_btier = (None if moe_w2_delta._BASE_DIRECT
                   else moe_w2_delta._BASE_TIER)

    def _gate_base_miss():
        """(this rank's, worst rank's) base misses."""
        local = int(_gate_btier.miss_count.item())
        if _tp.world_size <= 1:
            return local, local
        t = torch.tensor([local], device=_gate_btier.dev)
        torch.distributed.all_reduce(
            t, op=torch.distributed.ReduceOp.MAX,
            group=_tp.device_group)
        return local, int(t.item())

    # P11 fire-stall decomposition (measurement mode,
    # VLLM_MOE_W2_GATE_FIRE_TIMING=1 + DUMP): host wall
    # clock per promote->replay iteration, explicit syncs
    # at the phase boundaries. TP/single-GPU inline path
    # only (the PP worker-driven replay is out of scope).
    _fire_timing = moe_w2_gate.fire_timing_enabled()
    _fire_t = [] if _fire_timing else None
    while _gate_iters < moe_w2_gate.fire_fp_max():
        # Promote this rank's COLD routed experts
        # (per-rank shard side effect, never a per-rank
        # replay gate). GATE_MAX_PROMOTE is a per-STEP
        # budget shared across the FP passes — the
        # per-CALL reading multiplied the cap by the
        # loop count (measured on a 204-slot pool:
        # 64+64+48 promotions in ONE step = the whole
        # pool cycled per fire, [pinned 204, free 0]
        # refusals, 12% pool hit-rate = pure churn).
        if _fire_timing:
            # Drain queued work (the decision sync already
            # emptied the stream; this catches the dump
            # argmax) so promote/replay attribute cleanly.
            moe_w2_delta.LAST_FORCE_TIMING = None
            torch.cuda.synchronize()
            _ft_t0 = time.perf_counter()
        promote_error = None
        try:
            n_promoted = moe_w2_gate.force_promote_step(
                max_promote=_step_budget)
        except BaseException as e:
            promote_error = e
            n_promoted = 0
        if _fire_timing:
            _ft_prom_ms = (time.perf_counter()
                           - _ft_t0) * 1e3
        if not _and_tp(promote_error is None):
            logger.error(
                "moe_w2 TP gate promotion failed on at "
                "least one rank; skipping optional "
                "re-forward%s",
                f": {promote_error}"
                if promote_error else "")
            break
        if _step_budget is not None:
            _step_budget = max(
                _step_budget - n_promoted, 0)
        _gate_promoted_total += n_promoted
        # Replay only if SOME rank upgraded a cold
        # expert -> the result can actually change; if
        # everything routed was already FP4 the replay
        # is wasted HBM bandwidth. A refused/exhausted
        # promotion pass (n=0 with budget left, or
        # budget 0) ends the loop the same way — extra
        # replays cannot change the result either.
        if not (_or_tp(n_promoted > 0)
                and moe_w2_gate.reforward_enabled()):
            if _fire_timing:
                # terminal pass: promote cost paid, no
                # replay follows (n==0 or budget spent)
                _fire_t.append(dict(
                    promote_ms=round(_ft_prom_ms, 3),
                    n=int(n_promoted), replay_ms=0.0,
                    delta=moe_w2_delta.LAST_FORCE_TIMING))
            break
        _gate_iters += 1
        active[0] = True
        regated_output = forward()
        if runner.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = regated_output
        else:
            hidden_states = regated_output
        sample_hidden_states = hidden_states[logits_indices]
        logits = runner.model.compute_logits(sample_hidden_states)
        # The gate loop runs AFTER mandatory base replay.
        # Its changed early-layer numerics can route a later
        # layer onto a new, non-resident base expert. The
        # descriptor then zeroes that contribution, so the
        # step is repaired the same way the base loop
        # repairs its own second-order misses: fetch the
        # pairs, replay, re-read — under ONE shared policy
        # (moe_w2_delta.gate_repair_continue). Only the
        # residue left when that policy stops is fail-open,
        # and strict refuses it below.
        if _gate_btier is not None:
            _gate_miss_local, _gate_miss = _gate_base_miss()
            _gate_repairs = 0
            while moe_w2_delta.gate_repair_continue(
                    _gate_repairs, _gate_miss):
                # Same consensus as the base loop: every
                # rank completes its own pack I/O (pinned
                # for the replay that follows) before any
                # rank re-enters the collective.
                _moe_w2_promote_consensus(
                    _gate_btier,
                    _tp,
                    _gate_miss_local,
                    pin=True,
                    where="TP gate base repair",
                )
                _gate_repairs += 1
                _repair_out = forward()
                if runner.use_aux_hidden_state_outputs:
                    hidden_states, aux_hidden_states = (
                        _repair_out)
                else:
                    hidden_states = _repair_out
                sample_hidden_states = hidden_states[
                    logits_indices]
                logits = runner.model.compute_logits(
                    sample_hidden_states)
                _gate_miss_local, _gate_miss = (
                    _gate_base_miss())
            if _gate_repairs:
                _gate_btier.kpi_gate_repair(_gate_repairs)
                # First repair of the process is INFO: it
                # says the gate is re-routing off a
                # partial base pool, which is a pool-
                # sizing signal and used to be a crash.
                if not getattr(runner, "_moe_w2_gate_repair_seen", False):
                    runner._moe_w2_gate_repair_seen = True
                    logger.info(
                        "moe_w2 gate base repair engaged: "
                        "%d pass(es), %d missing pairs "
                        "left (later repairs at DEBUG)",
                        _gate_repairs, _gate_miss)
                else:
                    logger.debug(
                        "moe_w2 gate base repair: %d "
                        "pass(es), %d missing pairs left",
                        _gate_repairs, _gate_miss)
            moe_w2_delta.gate_validate_base_clean(
                _gate_miss)
        active[0] = False
        if _fire_timing:
            torch.cuda.synchronize()
            _fire_t.append(dict(
                promote_ms=round(_ft_prom_ms, 3),
                n=int(n_promoted),
                replay_ms=round(
                    (time.perf_counter() - _ft_t0) * 1e3
                    - _ft_prom_ms, 3),
                delta=moe_w2_delta.LAST_FORCE_TIMING))
    if _dump_pre is not None:
        # post-loop logits are the final (possibly
        # re-decided) step logits; equal pre/post is a
        # no-flip label, not skipped.
        moe_w2_gate.dump_step(
            _dump_pre, logits.argmax(dim=-1),
            _gate_promoted_total, _gate_iters,
            timing=_fire_t)
    return logits, hidden_states, aux_hidden_states, sample_hidden_states


# P1.7 deferred gate (VLLM_MOE_W2_GATE_DEFER=1, async scheduling, TP/1 GPU).
# The synchronous gate reads each decode step's verdict with .item() inside
# execute_model, so the worker cannot sample N or prepare N+1 while the GPU
# runs N (measured on V4.1 TP2: ~8 ms of a ~19 ms step). Deferred: step N
# only records the device-side signal and samples as usual; the verdict is
# RESOLVED on the worker's main thread before its next RPC (execute_model of
# N+1, before any of N's persistent input buffers are overwritten) or when
# it goes idle. A fire then promotes, replays N in place (same graph, same
# buffers), re-samples into the SAME device tensor that N+1's input prep
# reads, and redoes N's host copy. Semantics equal the synchronous gate's:
# N+1 sees the replayed KV and the re-decided token. N's output is held on
# the output thread until resolution (emission lags < 1 step).
_GATE_DEFER = os.getenv("VLLM_MOE_W2_GATE_DEFER", "0") == "1"
_DEFER_IDLE_S = float(os.getenv("VLLM_MOE_W2_GATE_DEFER_IDLE_MS", "2")) / 1e3


class _GateHold:
    """N's verdict hand-off between the worker main thread (resolve) and the
    async output thread (AsyncGPUModelRunnerOutput.get_output)."""

    def __init__(self):
        import threading
        self.resolved = threading.Event()
        self.recopy_ev = None
        # the worker thread that resolves; with the uniproc executor (TP1)
        # the engine core waits for N's output on that same thread, before
        # it ever calls execute_model(N+1): resolve inline there
        self.owner = threading.get_ident()

    def wait(self) -> None:
        import threading
        if (not self.resolved.is_set()
                and threading.get_ident() == self.owner):
            resolve_pending()
        self.resolved.wait()
        if self.recopy_ev is not None:
            self.recopy_ev.synchronize()


_PENDING = None   # dict for the step whose verdict is not resolved yet


_DEFER_LOGGED = [False]


def defer_eligible(runner, spec_decode_metadata) -> bool:
    if not _GATE_DEFER:
        return False
    if not _DEFER_LOGGED[0]:
        _DEFER_LOGGED[0] = True
        logger.info("moe_w2 deferred gate (P1.7): %s",
                    "armed" if runner.use_async_scheduling
                    and get_pp_group().world_size == 1 else
                    "INACTIVE (needs async scheduling, no PP) - sync gate")
    if not (runner.use_async_scheduling
            and spec_decode_metadata is None
            and get_pp_group().world_size == 1):
        return False
    ib = runner.input_batch
    if ib.max_num_logprobs or not ib.sampling_metadata.no_penalties:
        return False   # stale logprobs / CPU output-id readers after a re-decision
    return True


def defer_armed() -> bool:
    return _PENDING is not None and _PENDING.get("out") is not None


def attach_output(runner, async_output, sampled_token_ids) -> None:
    """sample_tokens(N) built N's async output: hold it until N resolves."""
    if _PENDING is None or _PENDING.get("out") is not None:
        return
    hold = _GateHold()
    async_output._moe_w2_hold = hold
    _PENDING.update(out=async_output, hold=hold, sampled=sampled_token_ids,
                    copy_stream=runner._get_or_create_async_output_copy_stream())


@torch.inference_mode()
def resolve_pending() -> None:
    """Resolve the armed step's verdict (worker main thread). Idempotent.
    Inference mode: the worker busy loop also resolves outside
    execute_model (idle timeout), and the re-decision updates inference
    tensors in place."""
    global _PENDING
    st = _PENDING
    if st is None or st.get("out") is None:
        return
    _PENDING = None
    hold = st["hold"]
    db = st.get("doorbell")
    try:
        if db is None:
            _resolve(st)
        else:
            _doorbell_resolve(st, db)
    finally:
        hold.resolved.set()


# Doorbell (P1.7b, VLLM_MOE_W2_GATE_DOORBELL=1 with GATE_DEFER): resolving
# at the next RPC's entry still serializes N+1's host input prep (~5 ms on
# V4.1) behind N's GPU work - measured no better than the synchronous gate.
# Instead N+1's entry enqueues a stream wait on a device flag
# (cuStreamWaitValue32) at the head of the compute stream: N+1's input prep
# runs on the host while the GPU finishes N, and every GPU op it issues
# (H2D input copies, prev-token gathers) queues behind the wait, so N's
# persistent input buffers stay intact. At N+1's pre_forward the verdict is
# resolved on a FIX stream (ordered after N's sampling by an event recorded
# before the wait); a fire promotes, replays N, re-samples and recopies
# there, and once that work is done the host releases the flag.
# The flag lives in pinned HOST memory and is released by a CPU store, never
# by a stream write: the first launch of a not-yet-used kernel while a
# stream is held stalls the host behind a device-wide barrier that a stream
# write would queue behind too (a permanent deadlock, V41N/V41O/V41R); a CPU
# store always gets through. Such stalls (and hidden host syncs inside prep)
# are bounded by a watchdog that releases the flag after
# GATE_DOORBELL_TIMEOUT_MS (prep) / _RESOLVE_TIMEOUT_MS (resolve); that
# step's verdict is dropped, and the doorbell turns off after
# _DOORBELL_MAX_STALLS of them. To keep first launches out of held windows,
# the doorbell holds only after one fire (and one verify replay, when on)
# has resolved unheld.
_DOORBELL_ON = (_GATE_DEFER
                and os.getenv("VLLM_MOE_W2_GATE_DOORBELL", "1") == "1")
_DOORBELL_TIMEOUT_S = float(
    os.getenv("VLLM_MOE_W2_GATE_DOORBELL_TIMEOUT_MS", "1000")) / 1e3
# resolve itself (a fire: NVMe promotion + up to 3 replays) may take a few
# hundred ms; a resolve that blocks on work queued behind the held stream
# never ends - stacks are dumped and the flag released after this long
_DOORBELL_RESOLVE_TIMEOUT_S = float(
    os.getenv("VLLM_MOE_W2_GATE_DOORBELL_RESOLVE_TIMEOUT_MS", "3000")) / 1e3
_DOORBELL_MAX_STALLS = int(
    os.getenv("VLLM_MOE_W2_GATE_DOORBELL_MAX_STALLS", "3"))
# resolve paths that have run at least once without the doorbell
_WARM = {"fire": False, "verify": False}


class _Doorbell:

    def __init__(self, dev):
        import threading
        from cuda.bindings import driver as cu
        self.cu = cu
        self.dev = dev
        # pinned host word (UVA-addressable): released by CPU stores
        self.flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.ptr = cu.CUdeviceptr(self.flag.data_ptr())
        # raw view of the word: the watchdog thread runs outside inference
        # mode, where a torch in-place write to this (inference) tensor raises
        import ctypes
        self.word = ctypes.c_int32.from_address(self.flag.data_ptr())
        self.fix = torch.cuda.Stream(device=dev)
        self.stalls = 0
        self.k = 0               # last armed value
        self.lock = threading.Lock()
        self.armed = False       # waiting for resolve to start
        self.broken = False      # too many stalls: doorbell off for good
        self.phase = None
        self.deadline = None     # watchdog deadline (perf_counter) or None
        self.wd_cv = threading.Condition(self.lock)
        self.wd = threading.Thread(target=self._watch, daemon=True,
                                   name="moe_w2_doorbell_wd")
        self.wd.start()

    def _check(self, res, what):
        err = res[0] if isinstance(res, tuple) else res
        if err != self.cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"moe_w2 doorbell {what}: {err}")

    def wait_on(self, stream, k):
        self._check(self.cu.cuStreamWaitValue32(
            self.cu.CUstream(stream.cuda_stream), self.ptr, k,
            self.cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
            "wait")

    def ring(self, k):
        self.word.value = k      # CPU store; the GPU polls the host word

    def released(self, k) -> bool:
        return self.word.value >= k

    def _watch(self):
        while True:
            try:
                self._watch_loop()
            except Exception:  # noqa: BLE001 - the watchdog must not die
                logger.exception("moe_w2 doorbell watchdog error")

    def _watch_loop(self):
        import sys
        import time as _t
        import traceback
        while True:
            with self.lock:
                while self.deadline is None:
                    self.wd_cv.wait()
                k, phase, dl = self.k, self.phase, self.deadline
                left = dl - _t.perf_counter()
                if left > 0:
                    self.wd_cv.wait(timeout=left)
                    continue
                if self.deadline != dl:
                    continue
                # held stream not released in time: prep (or resolve) is
                # blocked behind it (host sync, first kernel launch)
                self.deadline = None
                self.armed = False
                self.stalls += 1
                self.broken = self.stalls >= _DOORBELL_MAX_STALLS
                me = __import__("threading").get_ident()
                frames = "\n".join(
                    f"--- thread {tid}\n" + "".join(traceback.format_stack(f))
                    for tid, f in sys._current_frames().items() if tid != me)
                self.ring(k)
            logger.error(
                "moe_w2 doorbell: %s did not finish within its deadline "
                "(stall %d/%d%s) - released; that step's gate verdict is "
                "dropped. Thread stacks:\n%s", phase, self.stalls,
                _DOORBELL_MAX_STALLS,
                ", doorbell DISABLED" if self.broken else "",
                frames[-20000:])

    def set_deadline(self, phase, timeout_s):
        import time as _t
        self.phase = phase
        self.deadline = (None if timeout_s is None
                         else _t.perf_counter() + timeout_s)
        self.wd_cv.notify()


_DB = [None]


def _doorbell_arm(st) -> bool:
    """execute_model(N+1) entry with N pending: hold the compute stream."""
    try:
        db = _DB[0]
        if db is None:
            db = _DB[0] = _Doorbell(st["logits"].device)
            logger.info("moe_w2 deferred gate: doorbell armed (prep of N+1 "
                        "overlaps N; timeout %.0f ms)",
                        _DOORBELL_TIMEOUT_S * 1e3)
        # hold only over resolve paths that already ran unheld: a fire is
        # possible only with tau > 0, a verify replay only when it is on
        if db.broken or (st.get("tau", 0) > 0 and not _WARM["fire"]) or (
                _DEFER_VERIFY and not _WARM["verify"]):
            return False          # resolve at entry, unheld (warm-up)
        main = torch.cuda.current_stream()
        pre = torch.cuda.Event()
        pre.record(main)          # N's forward + sampling, before the wait
        with db.lock:
            db.k += 1
            db.wait_on(main, db.k)
            st["doorbell"] = (db, db.k, pre)
            db.armed = True
            db.set_deadline("step input prep", _DOORBELL_TIMEOUT_S)
        return True
    except Exception as e:  # noqa: BLE001 - fall back to entry resolve
        logger.warning("moe_w2 doorbell unavailable (%s) - resolving at "
                       "entry", e)
        global _DOORBELL_ON
        _DOORBELL_ON = False
        return False


def _doorbell_resolve(st, dbk) -> None:
    db, k, pre = dbk
    with db.lock:
        stale = not (db.armed and db.k == k)   # watchdog already released
        db.armed = False
        db.set_deadline("gate resolve", _DOORBELL_RESOLVE_TIMEOUT_S)
    # a stale step still resolves (its TP collectives must match the other
    # ranks'), but with its verdict dropped on every rank
    st["db_stale"] = stale

    def release():
        # once the fix stream's work (replay / re-sample / recopy) is done;
        # _resolve calls this as soon as the verdict is settled, so the
        # step's remaining host bookkeeping overlaps N+1 on the GPU
        if st.get("db_released"):
            return
        st["db_released"] = True
        done = torch.cuda.Event()
        done.record(db.fix)
        done.synchronize()
        with db.lock:
            db.set_deadline(None, None)
        db.ring(k)               # idempotent: the flag only grows

    st["release"] = release
    try:
        with torch.cuda.stream(db.fix):
            db.fix.wait_event(pre)
            _resolve(st)
    finally:
        if not st.get("db_released"):
            with db.lock:
                db.set_deadline(None, None)
            db.ring(k)


# Debug (VLLM_MOE_W2_GATE_DEFER_VERIFY=1): at every resolve, replay N with
# nothing to fix and require bit-identical logits - proves N's persistent
# input buffers were intact when the verdict was settled (the doorbell's
# whole point: N+1's prep ran in between). Costs one forward per step.
# (_FILE: runtime toggle, "1"/"0", re-read every 64 resolves)
_DEFER_VERIFY = os.getenv("VLLM_MOE_W2_GATE_DEFER_VERIFY", "0") == "1"
_DEFER_VERIFY_FILE = os.getenv("VLLM_MOE_W2_GATE_DEFER_VERIFY_FILE")
_VERIFY_N = [0, 0]   # checked, mismatched
_VERIFY_POLL = [0]


def _verify_on() -> bool:
    global _DEFER_VERIFY
    if _DEFER_VERIFY_FILE:
        _VERIFY_POLL[0] += 1
        if _VERIFY_POLL[0] % 64 == 1:
            try:
                with open(_DEFER_VERIFY_FILE) as f:
                    _DEFER_VERIFY = f.read().strip() == "1"
            except OSError:
                pass
    return _DEFER_VERIFY


def _verify_stage(name: str) -> None:
    # Diag (V41O hang): poll the current stream's progress past `name`
    # instead of blocking, and say which stage never completes.
    ev = torch.cuda.Event()
    ev.record()
    t0 = time.perf_counter()
    while not ev.query():
        if time.perf_counter() - t0 > 3.0:
            from vllm.utils.torch_utils import current_stream as _vcs
            logger.error(
                "moe_w2 VERIFY diag: stage '%s' not done after 3 s "
                "(torch stream %#x, vllm stream %#x)", name,
                torch.cuda.current_stream().cuda_stream,
                _vcs().cuda_stream)
            ev.synchronize()
            break
        time.sleep(0.001)
    else:
        if _VERIFY_N[0] < 3:
            logger.info("moe_w2 VERIFY diag: stage '%s' done in %.1f ms",
                        name, (time.perf_counter() - t0) * 1e3)


def _defer_verify(st) -> None:
    runner = st["runner"]
    _verify_stage("entry")
    out = st["forward"]()
    _verify_stage("forward")
    hs = out[0] if runner.use_aux_hidden_state_outputs else out
    lg = runner.model.compute_logits(hs[st["logits_indices"]])
    _verify_stage("compute_logits")
    if _VERIFY_N[0] < 3:
        ms = torch.cuda.memory_stats()
        logger.info("moe_w2 VERIFY diag: alloc retries %d, device allocs %d",
                    ms.get("num_alloc_retries", -1),
                    ms.get("num_device_alloc", -1))
    neq = (lg != st["logits"]).any()
    _verify_stage("compare")
    _VERIFY_N[0] += 1
    if bool(neq.item()):
        _VERIFY_N[1] += 1
        logger.error("moe_w2 deferred gate VERIFY: replay of the pending "
                     "step differs (max |d| %.4g; rows %s, max scheduled "
                     "tokens %s, verify #%d, batch %s -> %s, stale %s, "
                     "computed %s)",
                     float((lg.float() - st["logits"].float()).abs().max()),
                     st.get("n_tokens"), st.get("max_sched"), _VERIFY_N[0],
                     st.get("req_ids"), tuple(runner.input_batch.req_ids),
                     st.get("db_stale"), st.get("ncomp"))
    if _VERIFY_N[0] % 500 == 0:
        logger.info("moe_w2 deferred gate VERIFY: %d/%d replays identical",
                    _VERIFY_N[0] - _VERIFY_N[1], _VERIFY_N[0])


def resolve_held() -> None:
    """Settle a pending step whose doorbell still holds the stream (an
    execute_model that returned before pre_forward)."""
    if _PENDING is not None and _PENDING.get("doorbell") is not None:
        resolve_pending()


def _resolve(st) -> None:
    runner = st["runner"]
    logits = st["logits"]
    _tp = get_tp_group()

    def _or_tp(flag: bool) -> bool:
        if _tp.world_size <= 1:
            return flag
        t = torch.tensor([1 if flag else 0], device=logits.device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX,
                                     group=_tp.device_group)
        return bool(t.item())

    def _and_tp(flag: bool) -> bool:
        if _tp.world_size <= 1:
            return flag
        t = torch.tensor([1 if flag else 0], device=logits.device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN,
                                     group=_tp.device_group)
        return bool(t.item())

    st["worst_ev"].synchronize()
    fire = moe_w2_gate.record_decision(
        float(st["worst_cpu"].item()), st["tau"], st["rows"], False,
        audit_coin=False)
    # one TP reduction settles both the fire and the (debug) verify replay:
    # each rank reads the verify toggle file on its own clock, and a replay
    # on one rank only would hang in the forward's collectives
    verify = _verify_on()
    stale = bool(st.get("db_stale"))
    if _tp.world_size > 1:
        t = torch.tensor([int(fire), int(verify), int(stale)],
                         device=logits.device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX,
                                     group=_tp.device_group)
        fire, verify, stale = (bool(x) for x in t.tolist())
    if verify and st.get("doorbell") is not None and not _WARM["verify"]:
        # verify was switched on after this step armed the doorbell: its
        # first replay would launch new kernels under the held stream (a
        # watchdog stall, V41S/V41U) - skip it; the next step arms unheld
        verify = False
    if stale:
        # the watchdog released the held stream during N+1's prep on some
        # rank: N's buffers may be overwritten - no replay, no re-decision
        fire = verify = False
    if not (fire or verify) and st.get("release") is not None:
        st["release"]()          # nothing to redo: let N+1 run now
    if verify:
        _defer_verify(st)
        if st.get("doorbell") is None:
            _WARM["verify"] = True
    if fire:
        _dump_pre = (logits.argmax(dim=-1)
                     if moe_w2_gate.dump_enabled() else None)
        _active = [False]
        try:
            logits = _gate_fire(
                runner, st["forward"], logits, st["hidden_states"],
                st["aux_hidden_states"], st["logits_indices"], _tp,
                _or_tp, _and_tp, _dump_pre, _active)[0]
        except Exception as e:
            if _active[0]:
                logger.exception("moe_w2 deferred gate failed during TP replay")
                raise RuntimeError(
                    "moe_w2 TP gate replay failed after collective entry; "
                    "worker must terminate") from e
            logger.warning("moe_w2 deferred gate skipped: %s", e)
            fire = False
    dbk = st.get("doorbell")
    if fire and dbk is not None:
        # the watchdog may have released the held stream mid-resolve on some
        # rank: N+1's prep has then consumed N's token - drop the
        # re-decision on every rank (tokens must stay identical across TP)
        late = dbk[0].released(dbk[1])
        if _tp.world_size > 1:
            t = torch.tensor([int(late)], device=logits.device)
            torch.distributed.all_reduce(
                t, op=torch.distributed.ReduceOp.MAX, group=_tp.device_group)
            late = bool(t.item())
        if late:
            logger.warning("moe_w2 doorbell: fire verdict dropped (released "
                           "by the watchdog during resolve)")
            fire = False
    if fire:
        # re-decide N into the tensor N+1's input prep reads, then redo
        # N's host copy (the output thread waits for recopy_ev)
        new = runner.sampler(logits=logits,
                             sampling_metadata=st["sampling_metadata"])
        sampled = st["sampled"]
        sampled.copy_(new.sampled_token_ids.view_as(sampled))
        out = st["out"]
        ev = torch.cuda.Event(blocking=True)
        cs = st["copy_stream"]
        with torch.cuda.stream(cs):
            cs.wait_stream(torch.cuda.current_stream())
            out.sampled_token_ids_cpu.copy_(sampled, non_blocking=True)
            ev.record()
        st["hold"].recopy_ev = ev
        if st.get("doorbell") is None:
            _WARM["fire"] = True
        if (moe_w2_delta._BASE_DIRECT and moe_w2_delta._BASE_TIER is not None
                and getattr(moe_w2_delta._BASE_TIER, "_direct_ev", None)
                is not None):
            moe_w2_delta._BASE_TIER._direct_ev.record()
    if st.get("release") is not None:
        st["release"]()
    if moe_w2_delta._TIER is not None:
        try:
            moe_w2_delta._TIER.step_end()
        except Exception:  # noqa: BLE001 - never crash serving
            pass


def post_forward(runner, forward, logits, hidden_states, aux_hidden_states,
                 sample_hidden_states, logits_indices, spec_decode_metadata,
                 max_num_scheduled_tokens, scheduler_output):
    """After the first forward (PP==1 base fixed point, confidence gate,
    step close). Returns the final (logits, hidden_states,
    aux_hidden_states, sample_hidden_states)."""
    _w2m.mark("launch")
    # Debug: re-run the step's forward N extra times with nothing to fix
    # (any config). Outputs must not change; a change means the model's
    # forward is not idempotent and miss->replay cannot be exact on it.
    _force = int(os.getenv("VLLM_MOE_W2_DEBUG_FORCE_REPLAY", "0") or 0)
    if _force and logits is not None and not runner.is_pooling_model:
        for _ in range(_force):
            _re_out = forward()
            if runner.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = _re_out
            else:
                hidden_states = _re_out
            sample_hidden_states = hidden_states[logits_indices]
            logits = runner.model.compute_logits(sample_hidden_states)
    # moe_w2 BASE cache (VLLM_MOE_W2_BASE_CACHE_GB>0): the 2-bit base is
    # host-resident and the GPU pool may MISS routed experts — the desc
    # kernel zeroed their contributions and bumped the tier's miss
    # counter. Fetch the missing experts synchronously and replay the
    # step's graph ONCE (mandatory for correctness, unlike the optional
    # gate re-forward below). TP-only: all ranks decide via OR-reduce and
    # replay together (the re-forward is a collective). Any restoration
    # failure is fatal: returning zero-contribution logits is silent model
    # corruption.
    if (
        moe_w2_delta._BASE_DIRECT
        and moe_w2_delta._BASE_TIER is not None
        and logits is not None
        and not runner.is_pooling_model
    ):
        # DIRECT miss path: the graph already served every miss from the
        # pinned host rows (bit-exact), so nothing here waits on the GPU.
        _direct_step_close(moe_w2_delta._BASE_TIER)
    elif (
        moe_w2_delta._BASE_TIER is not None
        and logits is not None
        and not runner.is_pooling_model
        and get_pp_group().world_size == 1
    ):
        try:
            _w2d = moe_w2_delta
            _btier = _w2d._BASE_TIER
            if _btier is not None:
                # open this step's pin scope: slots touched by any pass
                # below are ineligible for eviction until the next step
                _btier.step_begin()
                _miss = int(_btier.miss_count.item())
                _tp = get_tp_group()
                if _tp.world_size > 1:
                    _t = torch.tensor([_miss], device=logits.device)
                    torch.distributed.all_reduce(
                        _t, op=torch.distributed.ReduceOp.MAX,
                        group=_tp.device_group)
                    _max_miss = int(_t.item())
                else:
                    _max_miss = _miss
                # Fetch this rank's missing experts whenever it has any
                # (they join the pool for future steps); replay only when
                # the worst rank exceeds the tolerance — the same reduced
                # max on every rank, so the collective replay stays in
                # lockstep. Tolerated steps keep their logits: <= TOL of
                # the step's ~top_k*n_layers weighted contributions were
                # zeroed (bounded approximation, delta/gate class).
                # Pin only when a replay will actually re-read the step
                # (otherwise the fetch serves FUTURE steps).  Every TP
                # rank participates in the status consensus even when its
                # local miss count is zero.
                _moe_w2_promote_consensus(
                    _btier,
                    _tp,
                    _miss,
                    pin=_w2d.fp_continue(0, _max_miss),
                    where="TP base replay",
                )
                # KPI: per-step replay rate + missing pairs (windowed
                # INFO line) — the pool-sizing signal.
                _btier.kpi_step(
                    _max_miss, _max_miss > _w2d.base_miss_tol())
                # Replay to a FIXED POINT (bounded): corrected early
                # layers can re-route later layers onto experts the
                # first pass never fetched; those second-order misses
                # zero contributions inside the replay itself and made
                # greedy output depend on pool content (cross-request
                # nondeterminism). Re-read the counter after each
                # replay; fetch + replay until miss-free (typically one
                # extra pass).
                _replays = 0
                while _w2d.fp_continue(_replays, _max_miss):
                    _replays += 1
                    _re_out = forward()
                    if runner.use_aux_hidden_state_outputs:
                        hidden_states, aux_hidden_states = _re_out
                    else:
                        hidden_states = _re_out
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = runner.model.compute_logits(sample_hidden_states)
                    _miss = int(_btier.miss_count.item())
                    if _tp.world_size > 1:
                        _t = torch.tensor([_miss], device=logits.device)
                        torch.distributed.all_reduce(
                            _t, op=torch.distributed.ReduceOp.MAX,
                            group=_tp.device_group)
                        _max_miss = int(_t.item())
                    else:
                        _max_miss = _miss
                    _moe_w2_promote_consensus(
                        _btier,
                        _tp,
                        _miss,
                        pin=_w2d.fp_continue(_replays, _max_miss),
                        where="TP base fixed-point replay",
                    )
                if _DBG_VERIFY:
                    _debug_verify(_btier, "post-replay",
                                  nonconv=_max_miss > _w2d.base_miss_tol())
                _w2d.fp_validate_complete(_max_miss)
                if _replays:
                    _btier.kpi_fp(_replays, _max_miss)
        except moe_w2_delta.StrictReplayNonConvergence as e:
            # Fail CLOSED per step, not per engine: the requests in this
            # step finish with an error (scheduler, via the flag on this
            # step's ModelRunnerOutput) and none of these logits becomes
            # a served token; every other request keeps being served.
            # Raising here used to take the whole engine down (measured:
            # GLM TP4 pager, 30 GiB pool, first GSM8K prefill).
            runner._moe_w2_step_error = str(e)
            logger.error(
                "moe_w2 strict base-cache replay did not converge - "
                "failing this step's requests closed (engine stays up): "
                "%s", e)
        except Exception as e:
            logger.exception("moe_w2 mandatory base-cache replay failed")
            raise RuntimeError(
                "mandatory moe_w2 base-cache replay failed; refusing "
                "zero-contribution logits") from e

    _w2m.mark("miss_check")
    # Confidence-gated FP4 re-forward (VLLM_MOE_W2_GATE=1; default OFF, so
    # the serving path is byte-for-byte unchanged). When this step's 2-bit
    # top-1 is low-confidence, pull its routed COLD experts up to FP4
    # (force_promote) and replay the graph ONCE so the step is re-decided
    # at FP4. Fires on real DECODE: a SPEC-DECODE verify step (MTP on) OR a
    # pure single-token decode (MTP off); prefill is excluded (it has
    # spec_decode_metadata None AND many scheduled tokens). Under MTP
    # `logits` covers the base+draft positions; should_reforward fires if
    # ANY is low-conf, the replay recomputes ALL of them at FP4, and the
    # downstream draft verification then accepts against the FP4 target.
    # Last PP rank only. Fail-safe: any error keeps the 2-bit logits.
    if (
        moe_w2_gate is not None
        and moe_w2_gate.enabled()
        and logits is not None
        and not runner.is_pooling_model
        and get_pp_group().is_last_rank
        and (spec_decode_metadata is not None or max_num_scheduled_tokens <= 1)
    ):
        gate_collective_active = False
        try:
            # The re-forward is a COLLECTIVE (TP: all-reduce per layer; PP:
            # a full pipeline pass), so all participating ranks must replay
            # together or it desyncs/hangs, and a PARTIAL (last-stage-only)
            # replay corrupts output.
            #   - PP (world_size>1): only RECORD the decision here; the
            #     worker broadcasts it and drives gate_reforward() on EVERY
            #     rank. MTP+PP stays fail-safed to the 2-bit logits (the
            #     sparse-MLA verify re-run is not idempotent).
            #   - TP / single GPU: re-forward INLINE. OR-reduce `fire`
            #     across TP so every rank agrees (logits are replicated
            #     post lm_head all-gather).
            if get_pp_group().world_size > 1:
                decision_error = None
                try:
                    local_fire = (
                        spec_decode_metadata is None
                        and moe_w2_gate.should_reforward(logits)
                        and moe_w2_gate.reforward_enabled()
                    )
                except BaseException as e:
                    decision_error = e
                    local_fire = False
                _tp = get_tp_group()
                ok = torch.tensor(
                    [0 if decision_error is not None else 1],
                    dtype=torch.int32, device=logits.device)
                fire_t = torch.tensor(
                    [1 if local_fire else 0],
                    dtype=torch.int32, device=logits.device)
                if _tp.world_size > 1:
                    torch.distributed.all_reduce(
                        ok, op=torch.distributed.ReduceOp.MIN,
                        group=_tp.device_group)
                    torch.distributed.all_reduce(
                        fire_t, op=torch.distributed.ReduceOp.MAX,
                        group=_tp.device_group)
                if not bool(ok.item()):
                    logger.error(
                        "moe_w2 PP gate decision failed on at least one "
                        "TP rank; skipping optional re-forward%s",
                        f": {decision_error}" if decision_error else "")
                    runner._gate_fire = False
                else:
                    runner._gate_fire = bool(fire_t.item())
            elif defer_eligible(runner, spec_decode_metadata):
                global _PENDING
                resolve_pending()   # never two in flight
                worst, rows, tau = moe_w2_gate.signal_worst(logits)
                if worst is not None:
                    worst_cpu = torch.empty((), dtype=worst.dtype,
                                            pin_memory=True)
                    worst_cpu.copy_(worst, non_blocking=True)
                    ev = torch.cuda.Event(blocking=True)
                    ev.record()
                    _PENDING = dict(
                        runner=runner, forward=forward, logits=logits,
                        hidden_states=hidden_states,
                        aux_hidden_states=aux_hidden_states,
                        logits_indices=logits_indices, worst_cpu=worst_cpu,
                        worst_ev=ev, rows=rows, tau=tau, out=None,
                        sampling_metadata=runner.input_batch.sampling_metadata,
                        n_tokens=int(logits_indices.shape[0]),
                        max_sched=int(max_num_scheduled_tokens),
                        req_ids=tuple(runner.input_batch.req_ids),
                        ncomp=runner.input_batch.num_computed_tokens_cpu[
                            :runner.input_batch.num_reqs].tolist())
            else:
                _tp = get_tp_group()

                def _or_tp(flag: bool) -> bool:
                    if _tp.world_size <= 1:
                        return flag
                    t = torch.tensor([1 if flag else 0], device=logits.device)
                    torch.distributed.all_reduce(
                        t, op=torch.distributed.ReduceOp.MAX,
                        group=_tp.device_group)
                    return bool(t.item())

                def _and_tp(flag: bool) -> bool:
                    if _tp.world_size <= 1:
                        return flag
                    t = torch.tensor(
                        [1 if flag else 0], device=logits.device)
                    torch.distributed.all_reduce(
                        t, op=torch.distributed.ReduceOp.MIN,
                        group=_tp.device_group)
                    return bool(t.item())

                decision_error = None
                try:
                    fire = moe_w2_gate.should_reforward(
                        logits, spec=spec_decode_metadata)
                except BaseException as e:
                    decision_error = e
                    fire = False
                if not _and_tp(decision_error is None):
                    logger.error(
                        "moe_w2 TP gate decision failed on at least one "
                        "rank; skipping optional re-forward%s",
                        f": {decision_error}" if decision_error else "")
                    fire = False
                fire = _or_tp(fire)
                # P9 dump (VLLM_MOE_W2_GATE_DUMP): snapshot the pre-replay
                # per-row argmax so the post-loop comparison yields flip
                # labels. GPU-side argmax here; the sync happens once in
                # dump_step. Disabled => zero extra work.
                _dump_pre = None
                if fire and moe_w2_gate.dump_enabled():
                    _dump_pre = logits.argmax(dim=-1)
                    if moe_w2_gate.hcap_enabled():
                        # 9b: decision-time h_l snapshot (pre-FP-loop —
                        # replays must not overwrite the signal the
                        # gate saw). Keyed by the gate step counter,
                        # matching the dump row written below.
                        from vllm.model_executor.layers.quantization \
                            .utils import moe_w2_cubit
                        _h = moe_w2_cubit.hcap_snapshot()
                        if _h is not None:
                            moe_w2_gate.hcap_store(
                                moe_w2_gate.stats()["steps"], _h)
                if fire:
                    _active = [False]
                    try:
                        (logits, hidden_states, aux_hidden_states,
                         sample_hidden_states) = _gate_fire(
                            runner, forward, logits, hidden_states,
                            aux_hidden_states, logits_indices, _tp,
                            _or_tp, _and_tp, _dump_pre, _active)
                    finally:
                        gate_collective_active = _active[0]
        except Exception as e:
            if gate_collective_active:
                logger.exception(
                    "moe_w2 confidence gate failed during TP replay")
                raise RuntimeError(
                    "moe_w2 TP gate replay failed after collective "
                    "entry; worker must terminate") from e
            logger.warning(
                "moe_w2 confidence gate preflight skipped: %s", e)

    _w2m.mark("gate")
    if (moe_w2_delta._BASE_DIRECT and moe_w2_delta._BASE_TIER is not None
            and getattr(moe_w2_delta._BASE_TIER, "_direct_ev", None)
            is not None and logits is not None):
        # gate replays ran after the step-close event: move the fence
        # (DeltaTier._direct_fence) past them
        moe_w2_delta._BASE_TIER._direct_ev.record()
    # moe_w2: the step is fully executed (base replays + gate replay
    # done) — close both tiers' SEEN windows so the eviction exclusion
    # and force_promote's hit-pinning stay scoped to ONE step. Without
    # this a prefill's accumulated marks (61 ensure_resident layers)
    # pinned the whole pool at the next decode step (measured: [pool
    # 2123, pinned 2123], hundreds of unpromotable experts, corrupted
    # first decode tokens).
    if ((moe_w2_delta._BASE_TIER is not None
         or moe_w2_delta._TIER is not None)
            and not runner.is_pooling_model
            and get_pp_group().world_size == 1):
        try:
            _w2d_end = moe_w2_delta
            if _w2d_end._BASE_TIER is not None:
                _w2d_end._BASE_TIER.step_end()
            # under PP a pending gate fire consumes the FP4 window in
            # gate_reforward() (driven by the worker AFTER this call);
            # force_promote zeroes it there.
            if _w2d_end._TIER is not None and not getattr(
                    runner, "_gate_fire", False) and _PENDING is None:
                _w2d_end._TIER.step_end()
            _w2m.mark("step_end")
            from vllm.model_executor.layers.quantization.utils import (
                moe_w2_metrics)
            if moe_w2_metrics.enabled():
                moe_w2_metrics.end_step(
                    _w2d_end._BASE_TIER, _w2d_end._TIER,
                    scheduler_output.total_num_scheduled_tokens,
                    len(scheduler_output.num_scheduled_tokens))
        except Exception:  # noqa: BLE001 - never crash serving
            pass

    return logits, hidden_states, aux_hidden_states, sample_hidden_states


def on_model_loaded(runner, load_dummy_weights: bool) -> None:
    """End of load_model: arm the pool-floor guard (real weights only) and
    the LOOKA lookahead counters (router weights now exist)."""
    runner._moe_w2_floor_check_armed = not load_dummy_weights
    # One INFO line the first time a gate re-forward re-routes onto a base
    # pair outside the pool (0.24 set this in GPUModelRunner.__init__).
    runner._moe_w2_gate_repair_seen = False
    if int(os.environ.get("VLLM_MOE_W2_DEBUG_NANTRACE", "0") or 0) > 0:
        _arm_nan_trace(runner.model)
    if moe_w2_delta._BASE_TIER is None:
        return
    try:
        from vllm.model_executor.layers.quantization.utils import (
            moe_w2_looka)
        moe_w2_looka.arm(runner.model, moe_w2_delta._BASE_TIER.n_layers,
                         moe_w2_delta._BASE_TIER.dev)
    except Exception as e:  # noqa: BLE001 - never fatal
        logger.warning("moe_w2 LOOKA arm failed: %s", e)


_NT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


_ZERO_NULL = int(os.environ.get("VLLM_MOE_W2_ZERO_NULL_BLOCK", "0") or 0)
_ZERO_NULL_STEPS = [0]


def _zero_null_blocks(runner) -> None:
    """VLLM_MOE_W2_ZERO_NULL_BLOCK=1|2: zero block 0 (the block pool's null
    block) of every KV cache tensor before the first (1) or every (2) step.

    Padded sparse-attention index entries (-1) make the DeepSeek-V4.1 SM120
    sparse MLA kernel load the null block's record and scale it by zero; any
    NaN bit pattern left there (dummy/warmup writes, other cache views sharing
    the buffer) turns into NaN outputs for every token and head."""
    if not _ZERO_NULL or (_ZERO_NULL == 1 and _ZERO_NULL_STEPS[0]):
        return
    caches = [t for t in getattr(runner, "kv_caches", []) or []
              if isinstance(t, torch.Tensor) and t.dim() >= 2]
    if not caches:
        # V2 runner: the profiling run steps before the KV caches exist;
        # do not spend the one-shot zeroing on it
        return
    _ZERO_NULL_STEPS[0] += 1
    log = _ZERO_NULL_STEPS[0] <= 3
    dirty = 0
    for t in caches:
        if log and t.dtype in _NT_DTYPES:
            dirty += int((~torch.isfinite(t[0])).sum().item())
        elif log and t.dtype in (torch.uint8, torch.float8_e4m3fn):
            dirty += int(((t[0].view(torch.uint8) & 0x7F) == 0x7F).sum().item())
        t[0].zero_()
    if log:
        logger.warning("moe_w2: zeroed the null block of %d KV caches "
                       "(step %d, nan-pattern elements before: %d)",
                       len(caches), _ZERO_NULL_STEPS[0], dirty)


def _nonfinite(obj) -> tuple[int, int]:
    """(non-finite count, element count) over the float tensors in obj."""
    bad = tot = 0
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, torch.Tensor):
            if o.dtype in _NT_DTYPES and o.numel():
                bad += int((~torch.isfinite(o)).sum().item())
                tot += o.numel()
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
        elif isinstance(o, dict):
            stack.extend(o.values())
        elif hasattr(o, "__dataclass_fields__"):
            stack.extend(getattr(o, f) for f in o.__dataclass_fields__)
    return bad, tot


def _arm_nan_trace(model) -> None:
    """VLLM_MOE_W2_DEBUG_NANTRACE=N: forward hooks on every module log the
    first N modules whose output turns non-finite while all their inputs are
    finite (eager only: hooks do not run inside captured CUDA graphs)."""
    budget = [int(os.environ["VLLM_MOE_W2_DEBUG_NANTRACE"])]
    calls = [0]

    def hook(mod, args, kwargs, out):
        if budget[0] <= 0 or torch.cuda.is_current_stream_capturing():
            return
        calls[0] += 1
        ob, ot = _nonfinite(out)
        if not ob:
            return
        ib, it = _nonfinite((args, kwargs))
        if ib:
            return
        budget[0] -= 1
        shapes = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
        amax = [float(a.float().abs().max()) for a in args
                if isinstance(a, torch.Tensor) and a.dtype in _NT_DTYPES
                and a.numel()]
        logger.warning(
            "moe_w2 NANTRACE call %d %s (%s): out nonfinite %d/%d, "
            "inputs finite shapes=%s amax=%s", calls[0],
            getattr(mod, "_nantrace_name", "?"), type(mod).__name__, ob, ot,
            shapes, amax)

    n = 0
    for name, mod in model.named_modules():
        mod._nantrace_name = name
        mod.register_forward_hook(hook, with_kwargs=True)
        n += 1
    logger.warning("moe_w2 NANTRACE armed on %d modules", n)


def check_pool_floor(runner) -> None:
    """Refuse a base-cache pool or FP4 need-pool sized below its per-step
    working set (VLLM_MOE_W2_FORCE_POOL=1 downgrades to a warning). Runs
    after the auto pool is sized and before cudagraph capture."""
    if not getattr(runner, "_moe_w2_floor_check_armed", False):
        return
    if moe_w2_delta._BASE_TIER is None and moe_w2_delta._TIER is None:
        return
    try:
        n_spec = 0
        if runner.speculative_config is not None:
            n_spec = int(getattr(runner.speculative_config,
                                 "num_speculative_tokens", 0) or 0)
        topk = int(getattr(runner.model_config.hf_text_config,
                           "num_experts_per_tok", 0) or 0)
        if topk > 0:
            moe_w2_delta.check_pool_floor(
                topk, n_spec, int(runner.scheduler_config.max_num_seqs))
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 - guard must not break boot
        logger.warning("moe_w2 pool-floor check skipped: %s", e)


def post_kv_init(runner) -> None:
    """Worker, after the KV cache exists and BEFORE cudagraph capture (the
    graphs bake pool pointers and slot-table contents): size the FP4 auto
    pool, check the pool floors, warm-start the base pool, arm the in-graph
    prefetcher and load a static FP4 membership."""
    try:
        if moe_w2_delta._TIER is not None:
            moe_w2_delta._TIER.finalize_auto()
    except Exception as e:  # noqa: BLE001 - opt-in path, never fatal
        logger.warning("moe_w2 delta auto-sizing failed: %s", e)
    check_pool_floor(runner)
    try:
        if moe_w2_delta._BASE_TIER is not None:
            moe_w2_delta._BASE_TIER.preload_pool()
    except Exception as e:  # noqa: BLE001 - warm-start is best-effort
        logger.warning("moe_w2 pool warm-start failed: %s", e)
    try:
        from vllm.model_executor.layers.quantization.utils import (
            moe_w2_vprefetch)
        if moe_w2_delta._BASE_TIER is not None:
            moe_w2_vprefetch.arm(runner.get_model(), moe_w2_delta._BASE_TIER,
                                 input_ids=(runner.input_ids.gpu
                                            if hasattr(runner, "input_ids")
                                            else runner.input_buffers.input_ids))
    except Exception as e:  # noqa: BLE001 - optional, never fatal
        logger.warning("moe_w2 VPREFETCH arm failed: %s", e)
    from vllm.model_executor.layers.quantization.utils import moe_w2_cpu
    moe_w2_cpu.arm(moe_w2_delta._BASE_TIER)
    # A bad static list must fail the boot (fail-fast contract).
    if moe_w2_delta._TIER is not None:
        moe_w2_delta._TIER.load_static_membership()


# Model Runner V2 (needed by DSpark, E3). Only the host-sync-free subset
# is wired: the DIRECT base cache (misses read over PCIe or computed on the
# CPU in-graph) with its async fills, gate off. The replay base cache, the
# confidence gate and the prefetch lanes need the V1 runner.
def v2_on_model_loaded(runner, load_dummy_weights: bool) -> None:
    """End of the V2 load_model: refuse what V2 cannot run, then the V1
    load hook (runner attributes it reads exist on both runners)."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_gate
    if moe_w2_delta._BASE_TIER is not None and not moe_w2_delta._BASE_DIRECT:
        raise RuntimeError(
            "moe_w2 on Model Runner V2 supports the DIRECT base cache only "
            "(VLLM_MOE_W2_BASE_DIRECT=1); the miss->replay path needs V1")
    from vllm.model_executor.layers.quantization.utils import moe_w2_shadow
    if moe_w2_gate.enabled():
        # v2_gate: synchronous TP gate only (no PP worker-driven replay, no
        # deferred doorbell gate, no sharded sampler)
        why = ("pipeline parallel" if get_pp_group().world_size > 1 else
               "VLLM_MOE_W2_GATE_DEFER" if _GATE_DEFER else
               "sharded sampling" if getattr(runner, "batch_sharder", None)
               is not None else
               "shadow rows" if moe_w2_shadow.enabled() else None)
        if why:
            raise RuntimeError(
                f"moe_w2 confidence gate on Model Runner V2 does not support "
                f"{why}; run with VLLM_MOE_W2_GATE=0 or the V1 runner")
    on_model_loaded(runner, load_dummy_weights)
    if moe_w2_shadow.enabled():
        if moe_w2_delta._BASE_TIER is None:
            raise RuntimeError("VLLM_MOE_W2_SHADOW_FROM needs the base cache")
        moe_w2_shadow.alloc(runner.max_num_tokens, runner.max_num_reqs,
                            runner.device)


def v2_pre_forward(runner, input_batch) -> None:
    """V2: before the forward - issue the planned fills (stream-ordered) or
    drain the manager, zero the null blocks, route-trace step marker, mark
    the shadow rows."""
    # a gated step whose sample_tokens never ran (failed step): close it
    v2_post_sample(runner)
    num_tokens = input_batch.num_tokens
    t0 = time.perf_counter() if _GPROF else 0.0
    if _GPROF and _GP["t_exit"] is not None:
        _gprof_add("gap", t0 - _GP["t_exit"])
        _GP["t_exit"] = None
        _gprof_step()
    for tier in (moe_w2_delta._BASE_TIER, moe_w2_delta._TIER):
        if tier is None:
            continue
        if tier._async_fill():
            tier.flush_fills()
        else:
            tier.wait_manager_idle()
    if _GPROF:
        _gprof_add("mgr", time.perf_counter() - t0)
    _zero_null_blocks(runner)
    from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
    if moe_w2_cubit._ROUTE_TRACE:
        moe_w2_cubit.route_trace_step(
            runner.input_buffers.input_ids[:num_tokens],
            runner.input_buffers.positions[:num_tokens])
    from vllm.model_executor.layers.quantization.utils import moe_w2_shadow
    if moe_w2_shadow.enabled():
        moe_w2_shadow.set_rows(input_batch)


def v2_post_forward(runner, num_tokens: int, num_reqs: int) -> None:
    """V2: after the forward - DIRECT step close (fence event, KPI) and
    close both tiers' seen windows (wakes the managers). With the gate on,
    the windows stay open until v2_post_sample: force_promote reads this
    step's routed experts from them."""
    if moe_w2_delta._BASE_TIER is None and moe_w2_delta._TIER is None:
        return
    if moe_w2_delta._BASE_TIER is not None:
        _direct_step_close(moe_w2_delta._BASE_TIER)
    runner._moe_w2_open = (num_tokens, num_reqs)
    if moe_w2_gate is None or not moe_w2_gate.enabled():
        v2_post_sample(runner)


# VLLM_MOE_W2_GATE_PROF=1: host-time split of gated V2 steps, logged every
# 500 steps: wait (forward still on the GPU), decide, close (v2_post_sample),
# gap (gate exit -> next forward's pre-hook), mgr (pre-hook manager wait).
# VLLM_MOE_W2_GATE_NOREAD=1: compute the gate signal but never read it (no
# sync, never fires) - the gate's cost without the wait.
_GPROF = os.getenv("VLLM_MOE_W2_GATE_PROF", "0") == "1"
_NOREAD = os.getenv("VLLM_MOE_W2_GATE_NOREAD", "0") == "1"
_GP = {"n": 0, "wait": 0.0, "decide": 0.0, "close": 0.0, "gap": 0.0,
       "mgr": 0.0, "t_exit": None}


def _gprof_add(key: str, dt: float) -> None:
    _GP[key] += dt


def _gprof_step() -> None:
    _GP["n"] += 1
    if _GP["n"] % 500 == 0:
        n = 500
        logger.info(
            "moe_w2 GATE-PROF over %d steps (ms/step): wait %.2f decide %.2f "
            "close %.2f gap %.2f (mgr %.2f)", n,
            *(1e3 * _GP[k] / n for k in ("wait", "decide", "close", "gap",
                                        "mgr")))
        for k in ("wait", "decide", "close", "gap", "mgr"):
            _GP[k] = 0.0


def v2_post_sample(runner) -> None:
    """V2: the step is fully executed (gate replays included): move the
    DIRECT fence past any replay, close both tiers' seen windows, metrics."""
    st = getattr(runner, "_moe_w2_open", None)
    if st is None:
        return
    runner._moe_w2_open = None
    t0 = time.perf_counter() if _GPROF else 0.0
    _v2_post_sample(st)
    if _GPROF and _GP["t_exit"] is not None:
        _gprof_add("close", time.perf_counter() - t0)


def _v2_post_sample(st) -> None:
    btier = moe_w2_delta._BASE_TIER
    if btier is not None:
        if getattr(btier, "_direct_ev", None) is not None:
            btier._direct_ev.record()
        btier.step_end()
    if moe_w2_delta._TIER is not None:
        moe_w2_delta._TIER.step_end()
    from vllm.model_executor.layers.quantization.utils import moe_w2_metrics
    if moe_w2_metrics.enabled():
        moe_w2_metrics.end_step(btier, moe_w2_delta._TIER, *st)


def _v2_spec(input_batch, input_ids):
    """The fields of V1's SpecDecodeMetadata the gate's spec mask reads:
    drafts per request and the draft ids (a verify request's rows are
    [last token, d1..dk], so its drafts are input ids 1..k)."""
    nd = input_batch.num_draft_tokens_per_req
    if not input_batch.num_draft_tokens or nd is None:
        return None
    nd = [int(n) for n in nd[:input_batch.num_reqs]]
    qs = input_batch.query_start_loc_np
    idx = [i for r, n in enumerate(nd) for i in range(qs[r] + 1, qs[r] + 1 + n)]
    # pinned + non_blocking: a pageable H2D copy would wait for the forward
    idx = torch.tensor(idx, dtype=torch.long).pin_memory().to(
        input_ids.device, non_blocking=True)
    ids = input_ids[idx]
    return types.SimpleNamespace(num_draft_tokens=nd, draft_token_ids=ids)


def v2_gate(runner, forward, hidden_states, aux_hidden_states, input_batch):
    """V2 synchronous confidence gate (decode / verify steps): decide on the
    step's logits, and on a fire promote the cold routed experts to FP4 and
    replay to a fixed point (_gate_fire). Returns (logits or None,
    hidden_states, aux_hidden_states); None = gate off for this step, the
    sampler computes the logits as usual."""
    if (moe_w2_gate is None or not moe_w2_gate.enabled() or forward is None
            or input_batch.num_reqs == 0
            or not (input_batch.num_draft_tokens > 0
                    or input_batch.num_tokens == input_batch.num_reqs)):
        return None, hidden_states, aux_hidden_states
    logits_indices = input_batch.logits_indices
    if _GPROF:
        t0 = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter()
        _gprof_add("wait", t1 - t0)
    logits = runner.model.compute_logits(hidden_states[logits_indices])
    if _NOREAD:
        moe_w2_gate.signal_worst(
            logits, _v2_spec(input_batch, runner.input_buffers.input_ids))
        if _GPROF:
            _gprof_add("decide", time.perf_counter() - t1)
            _GP["t_exit"] = time.perf_counter()
        return logits, hidden_states, aux_hidden_states
    _tp = get_tp_group()

    def _or_tp(flag: bool) -> bool:
        if _tp.world_size <= 1:
            return flag
        t = torch.tensor([1 if flag else 0], device=logits.device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX,
                                     group=_tp.device_group)
        return bool(t.item())

    def _and_tp(flag: bool) -> bool:
        if _tp.world_size <= 1:
            return flag
        t = torch.tensor([1 if flag else 0], device=logits.device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN,
                                     group=_tp.device_group)
        return bool(t.item())

    decision_error = None
    try:
        fire = moe_w2_gate.should_reforward(
            logits, spec=_v2_spec(input_batch,
                                  runner.input_buffers.input_ids))
    except BaseException as e:
        decision_error = e
        fire = False
    if not _and_tp(decision_error is None):
        logger.error("moe_w2 TP gate decision failed on at least one rank; "
                     "skipping optional re-forward%s",
                     f": {decision_error}" if decision_error else "")
        fire = False
    if _GPROF:
        _gprof_add("decide", time.perf_counter() - t1)
        _GP["t_exit"] = time.perf_counter()
    if not _or_tp(fire):
        return logits, hidden_states, aux_hidden_states
    dump_pre = (logits.argmax(dim=-1) if moe_w2_gate.dump_enabled()
                else None)
    active = [False]
    try:
        logits, hidden_states, aux_hidden_states, _ = _gate_fire(
            runner, forward, logits, hidden_states, aux_hidden_states,
            logits_indices, _tp, _or_tp, _and_tp, dump_pre, active)
    except Exception as e:
        if active[0]:
            raise RuntimeError("moe_w2 TP gate replay failed after "
                               "collective entry; worker must terminate") from e
        logger.warning("moe_w2 confidence gate fire skipped: %s", e)
    return logits, hidden_states, aux_hidden_states


# VLLM_MOE_W2_GATE_LAZY=1 (V2, TP1, uniproc executor): the gate's verdict
# is copied to the host without a wait and read at the next step's entry
# (or when this step's output is read, whichever comes first). The engine
# reads a step's output only after it has launched the next step, so the
# host keeps running ahead of the GPU. A fire restores the request state
# the step advanced, rebuilds the attention tables the drafter shares,
# replays at FP4 and redoes sample + draft; the step's pending output is
# swapped for the redone one before anyone reads it.
_LAZY = os.getenv("VLLM_MOE_W2_GATE_LAZY", "0") == "1"


def v2_lazy_ok(runner, input_batch, grammar_output) -> bool:
    if not _LAZY or moe_w2_gate is None or not moe_w2_gate.enabled():
        return False
    if (get_tp_group().world_size > 1 or grammar_output is not None
            or input_batch.has_structured_output_reqs
            or input_batch.num_reqs == 0
            or not (input_batch.num_draft_tokens > 0
                    or input_batch.num_tokens == input_batch.num_reqs)):
        return False
    return bool((input_batch.idx_mapping_np[:input_batch.num_reqs] >= 0).all())


def v2_gate_begin(runner, forward, hidden_states, aux_hidden_states,
                  input_batch):
    """Logits + gate signal on the GPU, the decision copied to pinned host
    memory without a wait. Returns (logits, pending)."""
    logits = runner.model.compute_logits(
        hidden_states[input_batch.logits_indices])
    is_spec = input_batch.num_draft_tokens > 0
    worst, rows, tau = moe_w2_gate.signal_worst(
        logits, _v2_spec(input_batch, runner.input_buffers.input_ids))
    p = types.SimpleNamespace(
        worst=None, ev=None, tau=tau, rows=rows, is_spec=is_spec,
        logits=logits, hidden=hidden_states, aux=aux_hidden_states,
        input_batch=input_batch, forward=forward, snap=None, finish=None,
        async_output=None)
    if worst is not None:
        p.worst = torch.empty((), dtype=worst.dtype, pin_memory=True)
        p.worst.copy_(worst, non_blocking=True)
        p.ev = torch.cuda.Event()
        p.ev.record()
    return logits, p


def v2_snapshot(runner, p, input_batch) -> None:
    """Per-request state postprocess_sampled advances (restored on a fire)."""
    idx = input_batch.idx_mapping[:input_batch.num_reqs]
    rs = runner.req_states
    bins = runner.sampler.penalties_state.output_bin_counts
    p.snap = (idx, rs.num_computed_tokens.gpu[idx].clone(),
              rs.last_sampled_tokens[idx].clone(),
              rs.total_len.gpu[idx].clone(),
              bins[idx].clone() if bins is not None else None)


def _v2_restore(runner, snap) -> None:
    idx, nc, ls, tl, bins = snap
    rs = runner.req_states
    rs.num_computed_tokens.gpu[idx] = nc
    rs.last_sampled_tokens[idx] = ls
    rs.total_len.gpu[idx] = tl
    if bins is not None:
        runner.sampler.penalties_state.output_bin_counts[idx] = bins


def v2_gate_arm(runner, p, finish, async_output) -> None:
    p.finish = finish
    p.async_output = async_output
    runner._moe_w2_lazy = p
    get_output = async_output.get_output

    def _get_output():
        if getattr(runner, "_moe_w2_lazy", None) is p:
            v2_gate_resolve(runner)
        return get_output()

    async_output.get_output = _get_output


def v2_gate_resolve(runner) -> None:
    """Read the pending verdict (the forward is done by now on most steps);
    on a fire redo the step; then close it (v2_post_sample)."""
    p = getattr(runner, "_moe_w2_lazy", None)
    if p is None:
        return
    runner._moe_w2_lazy = None
    fire = False
    if p.worst is not None:
        p.ev.synchronize()
        fire = moe_w2_gate.record_decision(float(p.worst), p.tau, p.rows,
                                           p.is_spec)
    if fire and p.forward is not None:
        _v2_lazy_redo(runner, p)
    v2_post_sample(runner)


def _v2_lazy_redo(runner, p) -> None:
    ib = p.input_batch
    _v2_restore(runner, p.snap)
    runner.prepare_attn(ib)
    logits, hidden, aux = p.logits, p.hidden, p.aux
    try:
        logits, hidden, aux, _ = _gate_fire(
            runner, p.forward, logits, hidden, aux, ib.logits_indices,
            get_tp_group(), lambda f: f, lambda f: f, None, [False])
    except Exception as e:
        logger.warning("moe_w2 lazy gate fire skipped: %s", e)
    new = p.finish(hidden, aux, logits, ib)
    p.async_output.__dict__.update(new.__dict__)


def post_warmup(runner) -> None:
    """End of compile_or_warm_up_model (all dummy runs done): the warmup
    forwards marked `seen` but never closed a step, so discard that window
    and its pins before the first served step (DeltaTier.discard_window)."""
    for tier in (moe_w2_delta._BASE_TIER, moe_w2_delta._TIER):
        if tier is not None:
            tier.discard_window()
