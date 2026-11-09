# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-graph vertical prefetch for the moe_w2 BASE cache (Phase 1 Mode 2b).

At decode layer L the router of layer L+n is applied to layer L's MoE input.
The predicted experts of L+n that are not resident are fetched INSIDE the
same captured step, on a side stream, so layer L+n finds them resident in
the first pass instead of missing and forcing a replay. Side-stream kernels:

  claim   for each predicted (L+n, e): atomic CAS slot_table -1 -> CLAIMED
          (-2) so duplicate predictions fetch once, then take a slot from the
          per-step reserve ring (bounded by the per-window budget). The take
          is logged as (layer, expert) per ring position for the host.
  gather  copy each claimed expert's pinned host row into its pool slot
          (zero-copy UVA reads, 16-byte words).
  publish write the slot into slot_table (a claim without a slot rolls back
          to -1).

The main stream waits on the side stream's event right before layer L+n
builds its descriptors.

Host side (outside the graph, under the tier lock):
  begin_step  reserve R ring slots: free slots first, then the coldest
              evictable owners. Victims are UNMAPPED for the step (the graph
              may overwrite them) but remembered.
  reconcile   adopt every ring take into the tier's host bookkeeping. Called
              before any host fetch (force_promote) so the miss path never
              re-fetches a pair the graph already mapped, and at step end.
  end_step    unused ring slots are MAPPED BACK to their previous owner (the
              bytes were never touched), so an unused reservation costs
              nothing; slots that were free go back to the free list.

Correctness: a late or wrong prediction is an ordinary miss, restored by the
strict miss -> fetch -> replay path, so greedy outputs are unchanged.

In-graph demand fill (VLLM_MOE_DEMAND=1, usable without the lookahead):
right before a decode layer builds its descriptors, the same claim -> gather
-> publish runs on the MAIN stream for the layer's actual routed experts. The
layer stalls on the copy instead of missing, so later layers route on
correct activations and the step needs no replay unless the ring runs out.

STAGE (VLLM_MOE_VPREFETCH_STAGE=1, DIRECT base cache; E1 disposable
prefetch): predicted experts resident in neither pool are copied into a
separate staging buffer of _STAGE_SLOTS slots instead of base-pool slots.
A per-step stage table [layers, E] (reset at the first MoE layer) maps
(layer, expert) -> staging slot once the copy has landed; the descriptor
build serves base slot > staged slot > pinned host row (DIRECT). Nothing
is adopted, evicted or synchronised on the host: the staging buffer never
displaces pool residents, a late copy just leaves the DIRECT read, and the
forward never waits for the side stream until its last MoE layer.

NEXT (VLLM_MOE_VPREFETCH_NEXT=1, with STAGE): a one-token-ahead lane. After
decode MoE layer L of step t, the experts layer L will route at step t+1
are predicted from x[t, L] and copied into staging slots owned by layer L
(_NEXT_BUDGET per layer), on a second side stream with a whole step of
lead time and no join. The predictor is the router of L, or, with
_NEXT_LR > 0, a per-layer linear model learned online in the graph:
z = W_L x/rms(x) + b_L (init from the router), one SGD step per decode step
on BCE against the routing that step t+1 actually took (labels are free one
step later). Stage-table rows are cleared on the forward stream right after
their layer's MoE, so entries never outlive their slot contents.

Env: VLLM_MOE_VPREFETCH=1, _N (lookahead layers, default 2), _K (predicted
experts per token, default 8), _BUDGET (fetches per target layer, default 2),
_RING (reserved slots per step, default 64; 160 with demand fill), _CHUNKS
(CTAs per row, 8). VLLM_MOE_DEMAND=1, VLLM_MOE_DEMAND_CHUNKS (CTAs per row, 32).
"""

import os
import re
import time

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

CLAIMED = tl.constexpr(-2)

_LOOKA = os.getenv("VLLM_MOE_VPREFETCH", "0") == "1"
_DEMAND = os.getenv("VLLM_MOE_DEMAND", "0") == "1"
_ENABLED = _LOOKA or _DEMAND
_N = max(1, int(os.getenv("VLLM_MOE_VPREFETCH_N", "2")))
_K = max(1, int(os.getenv("VLLM_MOE_VPREFETCH_K", "8")))
_BUDGET = max(0, int(os.getenv("VLLM_MOE_VPREFETCH_BUDGET", "2")))
_RING = max(1, int(os.getenv("VLLM_MOE_VPREFETCH_RING",
                             "160" if _DEMAND else "64")))
_DCHUNKS = max(1, int(os.getenv("VLLM_MOE_DEMAND_CHUNKS", "32")))
_DEMAND_TAG = 1 << 16    # added to the logged layer of demand-fill takes
# Reserve per step adapts to use: 2 x EMA(takes) + 16, within [_RING_MIN,
# _RING]. Reserving and restoring unused slots is host time on every step.
_RING_MIN = max(1, min(_RING, int(os.getenv("VLLM_MOE_VPREFETCH_RING_MIN",
                                            "32"))))
# Opt-in: close the step's ring and reserve the next one on the tier's
# manager thread instead of inline. Measured slower (1a 32 GiB + demand:
# 24.3 -> 28.8 ms/step): the runner then waits ~4 ms in wait_manager_idle
# and its own host work slows under GIL contention.
_DEFER = os.getenv("VLLM_MOE_VPREFETCH_DEFER", "0") == "1"
_CHUNKS = max(1, int(os.getenv("VLLM_MOE_VPREFETCH_CHUNKS", "8")))
_TCAP = 4      # decode-sized steps only (the launcher captures 1, 2, 4)
# Hash-routed layers (DS4 layers 0-2: experts = tid2eid[token]) are fetched
# EXACTLY from the step's token ids at the first MoE layer.
_HASH = os.getenv("VLLM_MOE_VPREFETCH_HASH", "0") == "1"
# ASYNC ring (with the DIRECT_ASYNC base cache, lookahead only): a
# persistent ring of free slots topped up by the manager and written at the
# runner's step-start flush; the graph claims from it and never waits for a
# fetch (an unpublished expert is read from host by the DIRECT path); the
# claim log is adopted from a non-blocking snapshot. No per-step host sync.
_STAGE = (_LOOKA and not _DEMAND
          and os.getenv("VLLM_MOE_W2_BASE_DIRECT", "0") == "1"
          and os.getenv("VLLM_MOE_VPREFETCH_STAGE", "0") == "1")
_STAGE_SLOTS = max(1, int(os.getenv("VLLM_MOE_VPREFETCH_STAGE_SLOTS", "160")))
# side-stream priority (CUDA: lower = higher priority, -1 high on most
# GPUs): lets prefetch CTAs be scheduled ahead of the forward's queued work
_PRIO = int(os.getenv("VLLM_MOE_VPREFETCH_PRIO", "0"))
# STAGE copy timing: 0 = copy right after the claim (issued at the MoE
# input of the source layer, overlapping its DIRECT reads); 1 = hold the
# copy until the source layer's MoE has finished (the PCIe is then idle
# through the next attention block)
_AFTER = os.getenv("VLLM_MOE_VPREFETCH_AFTER", "0") == "1"
# STAGE: the target layer waits for its own staging copies before its
# descriptor build (a late correct copy otherwise moves the same bytes twice:
# the copy, and the DIRECT read that did not wait for it)
_SJOIN = os.getenv("VLLM_MOE_VPREFETCH_STAGE_JOIN", "0") == "1"
_NEXT = _STAGE and os.getenv("VLLM_MOE_VPREFETCH_NEXT", "0") == "1"
_NEXT_BUDGET = max(1, int(os.getenv("VLLM_MOE_VPREFETCH_NEXT_BUDGET", "1")))
_NEXT_CHUNKS = max(1, int(os.getenv("VLLM_MOE_VPREFETCH_NEXT_CHUNKS", "4")))
_NEXT_LR = float(os.getenv("VLLM_MOE_VPREFETCH_NEXT_LR", "0"))
# learned predictor only: copy a prediction only if its probability is at
# least this (wrong copies cost full PCIe time; > 1 disables the copies)
_NEXT_MIN_P = float(os.getenv("VLLM_MOE_VPREFETCH_NEXT_MIN_P", "0"))
if _NEXT:
    # a late vertical publish could land after its row was cleared and leave
    # an entry to a slot the next step rewrites: the target must join
    _SJOIN = True
_ASYNC = (_LOOKA and not _DEMAND and not _STAGE
          and os.getenv("VLLM_MOE_W2_BASE_DIRECT", "0") == "1"
          and os.getenv("VLLM_MOE_W2_DIRECT_ASYNC", "1") == "1")


# ------------------------------------------------------------------ kernels

@triton.jit
def _claim_kernel(pred_ptr, n_pred, slot_row_ptr, ring_ptr, ring_len,
                  ring_head_ptr, budget, out_ptr, log_l_ptr, log_e_ptr,
                  layer, tag):
    """Single program. pred: [n_pred] expert ids (best first, may repeat).
    out[i] = slot taken for pred[i], or -1 (resident / duplicate / no slot /
    over budget). ring_head counts takes this step across all layers; the
    take at ring position h is logged as (layer + tag, e)."""
    taken = 0
    for i in range(0, n_pred):
        e = tl.load(pred_ptr + i)
        s = -1
        if taken < budget:
            old = tl.atomic_cas(slot_row_ptr + e, -1, CLAIMED)
            if old == -1:
                h = tl.atomic_add(ring_head_ptr, 1)
                if h < ring_len:
                    s = tl.load(ring_ptr + h)
                    tl.store(log_l_ptr + h, layer + tag)
                    tl.store(log_e_ptr + h, e)
                    taken += 1
                else:
                    tl.atomic_xchg(slot_row_ptr + e, -1)   # ring exhausted
        tl.store(out_ptr + i, s)


@triton.jit
def _claim_ring_kernel(pred_ptr, n_pred, slot_row_ptr, ring_ptr, ring_len,
                       head_ptr, tail_ptr, budget, out_ptr, log_l_ptr,
                       log_e_ptr, layer):
    """ASYNC ring claim (single program). head/tail are monotonic counts:
    ring position h % ring_len holds a usable slot while h < tail. Takes are
    logged at the same position for the host's adoption."""
    taken = 0
    h = tl.load(head_ptr)
    t = tl.load(tail_ptr)
    for i in range(0, n_pred):
        e = tl.load(pred_ptr + i)
        s = -1
        if taken < budget:
            if h < t:
                old = tl.atomic_cas(slot_row_ptr + e, -1, CLAIMED)
                if old == -1:
                    pos = h % ring_len
                    s = tl.load(ring_ptr + pos)
                    tl.store(log_l_ptr + pos, layer)
                    tl.store(log_e_ptr + pos, e)
                    h += 1
                    taken += 1
        tl.store(out_ptr + i, s)
    tl.store(head_ptr, h)


@triton.jit
def _stage_claim_kernel(pred_ptr, n_pred, slot_row_ptr, fslot_row_ptr,
                        stage_row_ptr, head_ptr, n_slots, budget, out_ptr,
                        tot_ptr, HAS_F: tl.constexpr):
    """STAGE claim (single program). Skips experts resident in the base
    pool (or the FP4 pool), dedups through a CAS on the stage table, and
    takes staging slots in prediction order up to `budget`."""
    taken = 0
    for i in range(0, n_pred):
        e = tl.load(pred_ptr + i)
        s = -1
        res = tl.load(slot_row_ptr + e) >= 0
        if HAS_F:
            res = res | (tl.load(fslot_row_ptr + e) >= 0)
        if (taken < budget) & (res == 0):
            old = tl.atomic_cas(stage_row_ptr + e, -1, CLAIMED)
            if old == -1:
                h = tl.atomic_add(head_ptr, 1)
                if h < n_slots:
                    s = h
                    taken += 1
                    tl.atomic_add(tot_ptr, 1)
                else:
                    tl.atomic_xchg(stage_row_ptr + e, -1)
        tl.store(out_ptr + i, s)


@triton.jit
def _next_claim_kernel(pred_ptr, n_pred, slot_row_ptr, fslot_row_ptr,
                       stage_row_ptr, base, budget, out_ptr, tot_ptr,
                       step_ptr, pstep_ptr, HAS_F: tl.constexpr):
    """NEXT claim (single program): like _stage_claim_kernel, but the slots
    are the layer's own [base, base + budget). pred < 0: skip. A replay of
    the step (the layer already predicted at this step) claims nothing: its
    first pass's copies stay in place."""
    lim = tl.where(tl.load(pstep_ptr) == tl.load(step_ptr), 0, budget)
    taken = 0
    for i in range(0, n_pred):
        e = tl.load(pred_ptr + i)
        s = -1
        res = (e < 0) | (tl.load(slot_row_ptr + tl.maximum(e, 0)) >= 0)
        if HAS_F:
            res = res | (tl.load(fslot_row_ptr + tl.maximum(e, 0)) >= 0)
        if (taken < lim) & (res == 0):
            if tl.atomic_cas(stage_row_ptr + e, -1, CLAIMED) == -1:
                s = base + taken
                taken += 1
                tl.atomic_add(tot_ptr, 1)
        tl.store(out_ptr + i, s)


@triton.jit
def _row_clear_kernel(row_ptr, n, nv, step_ptr, pstep_ptr,
                      BLOCK: tl.constexpr):
    """NEXT: clear a stage row after its layer's MoE. In a replay of the
    step only the vertical lane's entries (slots < nv, or claimed) go; the
    next-token entries of the first pass stay (no claim replaces them)."""
    i = tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(row_ptr + i, mask=m, other=-1)
    replay = tl.load(pstep_ptr) == tl.load(step_ptr)
    keep = replay & (v >= nv)
    tl.store(row_ptr + i, tl.where(keep, v, -1), mask=m)


@triton.jit
def _gather_kernel(pred_ptr, slots_ptr, host_base, row_bytes, pool_ptr,
                   slot_bytes, BLOCK: tl.constexpr):
    """grid (n_pred, chunks). Copies host row pred[i] (pinned, UVA) into
    pool slot slots[i] when slots[i] >= 0, as int32 words."""
    i = tl.program_id(0)
    c = tl.program_id(1)
    s = tl.load(slots_ptr + i)
    if s < 0:
        return
    e = tl.load(pred_ptr + i).to(tl.int64)
    src = (host_base + e * row_bytes).to(tl.pointer_type(tl.int32))
    dst = (pool_ptr + s.to(tl.int64) * slot_bytes).to(tl.pointer_type(tl.int32))
    n_words = row_bytes // 4
    per_chunk = tl.cdiv(n_words, tl.num_programs(1))
    lo = c * per_chunk
    hi = tl.minimum(lo + per_chunk, n_words)
    for off in range(lo, hi, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < hi
        tl.store(dst + idx, tl.load(src + idx, mask=m), mask=m)


@triton.jit
def _publish_kernel(pred_ptr, slots_ptr, n_pred, slot_row_ptr):
    i = tl.program_id(0)
    if i < n_pred:
        s = tl.load(slots_ptr + i)
        if s >= 0:
            e = tl.load(pred_ptr + i)
            tl.store(slot_row_ptr + e, s)


@triton.jit
def _gather_cvt_kernel(pred_ptr, slots_ptr, host_base, src_row_bytes, pool_ptr,
                       slot_bytes, c13, s13, c2, lut_ptr, BLOCK: tl.constexpr):
    """grid (n_pred, chunks). Like _gather_kernel, but the host row is an
    FP4 slot row [fp4_13 | sc13 | fp4_2 | sc2] and the pool slot receives
    the 2-bit row [codes13 | sc13 | codes2 | sc2] (R2': 2-bit derived on the
    fly). Both code planes share one element order (moe_w2_planes
    pack_fragment_major / pack_fp4_fragment_major), so each output byte is
    lut[in byte 2i] | lut[in byte 2i+1] << 4, lut = 2 nibbles -> 2 codes;
    scale sections are copied verbatim (MXFP4 scales are kept as-is)."""
    i = tl.program_id(0)
    c = tl.program_id(1)
    s = tl.load(slots_ptr + i)
    if s < 0:
        return
    e = tl.load(pred_ptr + i).to(tl.int64)
    src = (host_base + e * src_row_bytes).to(tl.pointer_type(tl.int32))
    dst = (pool_ptr + s.to(tl.int64) * slot_bytes).to(tl.pointer_type(tl.int32))
    n_words = slot_bytes // 4
    per_chunk = tl.cdiv(n_words, tl.num_programs(1))
    lo = c * per_chunk
    hi = tl.minimum(lo + per_chunk, n_words)
    w13 = c13 // 4
    w_s13 = (c13 + s13) // 4
    w_c2 = (c13 + s13 + c2) // 4
    for off in range(lo, hi, BLOCK):
        w = off + tl.arange(0, BLOCK)
        m = w < hi
        in_c13 = w < w13
        in_c2 = (w >= w_s13) & (w < w_c2)
        is_code = in_c13 | in_c2
        # input word of this output word (codes: 2 input words per output)
        a = tl.where(in_c13, 2 * w,
            tl.where(w < w_s13, w + w13,
            tl.where(in_c2, w + w13 + (w - w_s13),
                     w + w13 + (c2 // 4))))
        x0 = tl.load(src + a, mask=m, other=0)
        x1 = tl.load(src + a + 1, mask=m & is_code, other=0)
        o = tl.zeros_like(x0)
        for k in tl.static_range(4):
            xw = x0 if k < 2 else x1
            sh = (k % 2) * 16
            b_lo = (xw >> sh) & 0xFF
            b_hi = (xw >> (sh + 8)) & 0xFF
            v = tl.load(lut_ptr + b_lo) | (tl.load(lut_ptr + b_hi) << 4)
            o = o | (v << (8 * k))
        tl.store(dst + w, tl.where(is_code, o, x0), mask=m)


class Cvt:
    """R2' geometry: 2-bit slot sections derived from FP4 host rows."""

    def __init__(self, c13: int, s13: int, c2: int, s2: int, src_row: int,
                 dev):
        assert src_row == 2 * c13 + s13 + 2 * c2 + s2, "FP4 row layout"
        assert c13 % 4 == 0 and s13 % 4 == 0 and c2 % 4 == 0 and s2 % 4 == 0
        self.c13, self.s13, self.c2, self.s2 = c13, s13, c2, s2
        self.src_row = src_row
        self.dst_row = c13 + s13 + c2 + s2
        code = torch.tensor([2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 1, 0, 0, 0],
                            dtype=torch.int32)
        b = torch.arange(256)
        # one FP4 byte (2 nibbles, low first) -> 4 bits of 2-bit codes
        self.lut = (code[b & 15] | (code[b >> 4] << 2)).to(torch.int32).to(dev)


def gather(pred, out_slots, host_row_base, row_bytes, pool, chunks, cvt=None):
    """Copy (or, with cvt, convert) each claimed host row into its slot."""
    n = pred.numel()
    if cvt is None:
        _gather_kernel[(n, chunks)](pred, out_slots, host_row_base, row_bytes,
                                    pool, pool.shape[1], BLOCK=1024)
    else:
        _gather_cvt_kernel[(n, chunks)](
            pred, out_slots, host_row_base, cvt.src_row, pool, pool.shape[1],
            cvt.c13, cvt.s13, cvt.c2, cvt.lut, BLOCK=512)


def write_slot_cvt(pool, slot: int, row: torch.Tensor, cvt: Cvt,
                   chunks: int = 32) -> None:
    """Host fetch paths (R2'): convert one pinned FP4 host row into pool
    slot `slot` on the current stream. The row must stay alive until the
    stream passes this point (same contract as copy_(non_blocking=True))."""
    dev = pool.device
    pred = torch.zeros(1, dtype=torch.int32, device=dev)
    slots = torch.full((1,), slot, dtype=torch.int32, device=dev)
    _gather_cvt_kernel[(1, chunks)](
        pred, slots, row.data_ptr(), cvt.src_row, pool, pool.shape[1],
        cvt.c13, cvt.s13, cvt.c2, cvt.lut, BLOCK=512)


def issue(pred, slot_row, ring, ring_head, budget, host_row_base, row_bytes,
          pool, out_slots, log_l, log_e, layer, chunks=_CHUNKS, tag=0,
          cvt=None):
    """Enqueue claim -> gather -> publish on the CURRENT stream. Static
    shapes; capture-safe."""
    n = pred.numel()
    if _ASYNC:
        _claim_ring_kernel[(1,)](pred, n, slot_row, ring, ring.numel(),
                                 ring_head, _S.tail, budget, out_slots,
                                 log_l, log_e, layer)
    else:
        _claim_kernel[(1,)](pred, n, slot_row, ring, ring.numel(),
                            ring_head, budget, out_slots, log_l, log_e,
                            layer, tag)
    gather(pred, out_slots, host_row_base, row_bytes, pool, chunks, cvt)
    _publish_kernel[(n,)](pred, out_slots, n, slot_row)


# ------------------------------------------------------------------ state

class _S:
    armed = False
    gate_w: dict = {}          # target layer_key -> [E, H] router weight
    gate_b: dict = {}          # target layer_key -> [E] f32 bias (or absent)
    hash_tab: dict = {}        # hash layer_key -> tid2eid [vocab, top_k]
    input_ids = None           # runner's persistent input_ids buffer (GPU)
    first_key = None           # first moe_w2 layer_key of the forward
    scoring = "sqrtsoftplus"
    host_base: dict = {}       # layer_key -> UVA address of pinned [E, row]
    row_bytes = 0
    cvt = None                 # Cvt when the host rows are FP4 (R2')
    pred: dict = {}            # target key -> [TCAP*K] i32
    out: dict = {}             # target key -> [TCAP*K] i32
    dpred: dict = {}           # layer key -> [TCAP*top_k] i32 (demand fill)
    dout: dict = {}            # layer key -> [TCAP*top_k] i32
    ev: dict = {}              # target key -> Event recorded on the side stream
    xbuf: dict = {}            # target key -> [TCAP, H] x snapshot (ASYNC)
    side = None
    ring = None                # [R] i32 slots reserved this step
    head = None                # [1] i32 takes this step
    log_l = None               # [R] i32
    log_e = None               # [R] i32
    ring_host = None           # [R] i64 CPU: reserved slot per ring position
    prev_li = None             # [R] i64 CPU: previous owner layer (-1: free)
    prev_ei = None             # [R] i64 CPU: previous owner expert
    consumed = 0               # ring positions already reconciled
    active = False             # a ring is reserved for the current step
    n_issued = 0
    n_adopted = 0
    n_restored = 0
    n_demand = 0
    t_ms = {"begin": 0.0, "reconcile": 0.0, "end": 0.0}   # host time
    n_steps = 0
    start = 0                  # first usable ring position this step
    ev_step = None             # end of the step's GPU work (deferred roll)
    roll = False               # manager owes end_step + begin_step
    prepared = False           # ring reserved ahead by the manager
    ema_takes = float(_RING)   # starts at the full ring, adapts down
    # ASYNC ring (host side; manager + runner flush under btier._lock)
    tail = None                # [1] i32 device: slots provided (monotonic)
    tail_host = 0              # host count of slots provided
    adopted = 0                # ring takes adopted into host bookkeeping
    ring_shadow = None         # [R] i64 CPU: slot at each ring position
    ring_queue: list = []      # [(pos, slot)] to write at the next flush
    snap = None                # pinned [1 + 2R] i32: head | log_l | log_e
    snap_ev = None
    snap_pending = False
    last_key = None            # last moe_w2 layer key (final join)
    n_conflict = 0
    # STAGE (disposable staging buffer)
    stage_pool = None          # [_STAGE_SLOTS, slot_bytes] u8
    stage_tab = None           # [n_layer_keys, E] i32: staging slot | -1
    stage_head = None          # [1] i32 takes this step
    stage_tot = None           # [1] i32 takes, cumulative
    stage_hits = None          # [2] i32 staged pairs served, late (cumul.)
    pending: dict = {}         # source layer key -> [(pred, out, tgt)] (AFTER)
    ev_after = None            # Event: source layer's MoE done (AFTER)
    joins: set = set()         # target keys with a staging copy to wait for
    # NEXT (one-token-ahead lane)
    side2 = None
    nbase: dict = {}           # layer key -> first staging slot of the layer
    nx: dict = {}              # layer key -> [TCAP, H] x of this step
    nT: dict = {}              # layer key -> T, set at the MoE input (decode)
    ev_next = None
    nw: dict = {}              # layer key -> [E, H] f32 learned weight (LR > 0)
    nb: dict = {}              # layer key -> [E] f32 bias
    nc: dict = {}              # layer key -> [1] f32 input scale (0: unset)
    nxp: dict = {}             # layer key -> [TCAP, H] f32 scaled x predicted from
    npp: dict = {}             # layer key -> [TCAP, E] f32 its probabilities
    ny: dict = {}              # layer key -> [TCAP, E] f32 multi-hot labels
    nargs = None               # (btier, ftier) of the decode forward
    ev_owed = None             # side2: the last layer's prediction has read x
    owed = False               # ev_owed recorded in this forward
    next_tot = None            # [1] i32 NEXT takes, cumulative
    nstep = None               # [1] i32 decode steps (runner, not replays)
    npstep = None              # [n_keys] i32 step of each layer's prediction


def enabled() -> bool:
    return _ENABLED and _S.armed


def shutdown() -> None:
    _S.armed = False
    _S.gate_w = {}
    _S.gate_b = {}
    _S.hash_tab = {}
    _S.input_ids = None
    _S.host_base = {}
    _S.pred = {}
    _S.out = {}
    _S.dpred = {}
    _S.dout = {}
    _S.ev = {}
    _S.side = None
    _S.active = False


def arm(model, btier, input_ids=None) -> None:
    """Collect the routers per moe_w2 layer_key and allocate the persistent
    in-graph buffers. Called once after weight load, before graph capture.
    Hash-routed layers (gate.tid2eid) are never targets: their routing does
    not come from the router logits."""
    if not _ENABLED or _S.armed or btier is None:
        return
    try:
        from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
        store = btier._store
        if not hasattr(store, "_layers") or getattr(store, "_rowmaps", None):
            logger.warning("moe_w2 VPREFETCH: needs the full-layout pinned "
                           "host store (no pack/arena, no EP) - disabled")
            return
        tl_to_key = {st.get("tl_idx"): key
                     for key, st in moe_w2_cubit._LAYERS.items()
                     if st.get("tl_idx") is not None}
        pat = re.compile(r"\.layers\.(\d+)\.(?:mlp|ffn)\.gate\."
                         r"(weight|e_score_correction_bias|tid2eid)$")
        hashed = set()
        for name, p in list(model.named_parameters()) + list(
                model.named_buffers()):
            m = pat.search(name)
            if m is None:
                continue
            key = tl_to_key.get(int(m.group(1)))
            if key is None:
                continue
            kind = m.group(2)
            if kind == "weight":
                _S.gate_w[key] = p.data
            elif kind == "tid2eid":
                hashed.add(key)
                _S.hash_tab[key] = p.data
            else:
                _S.gate_b[key] = p.data.float()
        for key in hashed:
            _S.gate_w.pop(key, None)
            _S.gate_b.pop(key, None)
        for mod in model.modules():
            sf = getattr(mod, "scoring_func", None)
            if isinstance(sf, str):
                _S.scoring = sf
                break
        if not _LOOKA:
            _S.gate_w = {}
            _S.gate_b = {}
        elif not _S.gate_w:
            logger.warning("moe_w2 VPREFETCH: no router weights found - "
                           "disabled")
            return
        dev = btier.dev
        _S.first_key = min(store._layers) if store._layers else None
        if not (_HASH and input_ids is not None):
            _S.hash_tab = {}
        # the first MoE layer is already running when the forward reaches it
        _S.hash_tab.pop(_S.first_key, None)
        _S.input_ids = input_ids
        for key in list(_S.gate_w) + list(_S.hash_tab):
            if key not in store._layers:
                _S.gate_w.pop(key, None)
                _S.hash_tab.pop(key, None)
                continue
            _S.pred[key] = torch.zeros(_TCAP * max(_K, 16), dtype=torch.int32,
                                       device=dev)
            _S.out[key] = torch.full((_TCAP * max(_K, 16),), -1, dtype=torch.int32,
                                     device=dev)
            _S.ev[key] = torch.cuda.Event()
        if _DEMAND:
            for key in store._layers:
                _S.dpred[key] = torch.zeros(_TCAP * 16, dtype=torch.int32,
                                            device=dev)
                _S.dout[key] = torch.full((_TCAP * 16,), -1,
                                          dtype=torch.int32, device=dev)
        _S.host_base = {k: t.data_ptr() for k, t in store._layers.items()}
        _S.cvt = getattr(btier, "_cvt", None)
        _S.row_bytes = (_S.cvt.src_row if _S.cvt is not None
                        else int(btier.slot_bytes))
        _S.side = torch.cuda.Stream(dev, priority=_PRIO)
        _S.ring = torch.zeros(_RING, dtype=torch.int32, device=dev)
        _S.head = torch.full((1,), _RING, dtype=torch.int32, device=dev)
        _S.log_l = torch.full((_RING,), -1, dtype=torch.int32, device=dev)
        _S.log_e = torch.full((_RING,), -1, dtype=torch.int32, device=dev)
        _S.last_key = max(store._layers) if store._layers else None
        if _STAGE:
            n_keys = int(btier.slot_table.shape[0])
            nkeys = sorted(_S.gate_w) if _NEXT else []
            _S.nbase = {k: _STAGE_SLOTS + i * _NEXT_BUDGET
                        for i, k in enumerate(nkeys)}
            _S.stage_pool = torch.empty(
                (_STAGE_SLOTS + len(nkeys) * _NEXT_BUDGET,
                 int(btier.slot_bytes)), dtype=torch.uint8, device=dev)
            _S.stage_tab = torch.full((n_keys, int(btier.slot_table.shape[1])),
                                      -1, dtype=torch.int32, device=dev)
            if _NEXT:
                _arm_next(nkeys, dev)
            _S.stage_head = torch.zeros(1, dtype=torch.int32, device=dev)
            _S.stage_tot = torch.zeros(1, dtype=torch.int32, device=dev)
            _S.stage_hits = torch.zeros(3, dtype=torch.int32, device=dev)
        if _ASYNC:
            _S.head.zero_()
            _S.tail = torch.zeros(1, dtype=torch.int32, device=dev)
            _S.ring_shadow = torch.full((_RING,), -1, dtype=torch.int64)
            _S.snap = torch.zeros(1 + 2 * _RING, dtype=torch.int32
                                  ).pin_memory()
            _S.snap_ev = torch.cuda.Event()
        _S.armed = True
        logger.info(
            "moe_w2 VPREFETCH armed: %d router-lookahead targets, hash "
            "layers %s (exact token-id prefetch: %s), lookahead n=%d, "
            "top-%d/token, budget %d/layer, ring %d slots, scoring %s; "
            "in-graph demand fill: %s; async ring (DIRECT, no stalls, no "
            "host sync): %s; disposable staging: %s",
            len(_S.gate_w), sorted(hashed), sorted(_S.hash_tab), _N, _K,
            _BUDGET, _RING, _S.scoring, _DEMAND, _ASYNC,
            f"{_STAGE_SLOTS} slots ({_STAGE_SLOTS * int(btier.slot_bytes) / 2**30:.2f} GiB)"
            if _STAGE else False)
        if _NEXT:
            logger.info(
                "moe_w2 VPREFETCH next-token lane: %d layers x %d slots "
                "(%.2f GiB), %s", len(_S.nbase), _NEXT_BUDGET,
                len(_S.nbase) * _NEXT_BUDGET * int(btier.slot_bytes) / 2**30,
                f"online SGD lr {_NEXT_LR} ({len(_S.nw)} x [E, H] f32)"
                if _NEXT_LR > 0 else "router predictor")
    except Exception as e:  # noqa: BLE001 - optional feature, never kill boot
        logger.warning("moe_w2 VPREFETCH arm failed: %s", e)


def _arm_next(keys, dev) -> None:
    _S.side2 = torch.cuda.Stream(dev, priority=0)
    _S.ev_next = torch.cuda.Event()
    _S.next_tot = torch.zeros(1, dtype=torch.int32, device=dev)
    _S.nstep = torch.ones(1, dtype=torch.int32, device=dev)
    _S.npstep = torch.zeros(int(_S.stage_tab.shape[0]), dtype=torch.int32,
                            device=dev)
    for k in keys:
        W = _S.gate_w[k]
        _S.pred[("n", k)] = torch.zeros(_TCAP * _K, dtype=torch.int32,
                                        device=dev)
        _S.out[("n", k)] = torch.full((_TCAP * _K,), -1, dtype=torch.int32,
                                      device=dev)
        _S.nx[k] = torch.zeros((_TCAP, W.shape[1]), dtype=W.dtype,
                               device=dev)
        if _NEXT_LR > 0:
            # z = W xs + b, xs = x * c / rms(x) with c the rms of the layer's
            # first decode input (set once, in-graph): W starts at the router
            _S.nw[k] = W.float().clone()
            _S.nc[k] = torch.zeros(1, dtype=torch.float32, device=dev)
            _S.ny[k] = torch.zeros((_TCAP, W.shape[0]), dtype=torch.float32,
                                   device=dev)
            b = _S.gate_b.get(k)
            _S.nb[k] = (b.clone() if b is not None
                        else torch.zeros(W.shape[0], device=dev))
            _S.nxp[k] = torch.zeros((_TCAP, W.shape[1]), dtype=torch.float32,
                                    device=dev)
            _S.npp[k] = torch.zeros((_TCAP, W.shape[0]), dtype=torch.float32,
                                    device=dev)
    _S.ev_owed = torch.cuda.Event()


# ------------------------------------------------------------- in-graph

def _scores(x, key):
    logits = (x.to(_S.gate_w[key].dtype) @ _S.gate_w[key].t()).float()
    if _S.scoring == "sqrtsoftplus":
        s = torch.sqrt(torch.nn.functional.softplus(logits))
    elif _S.scoring == "softmax":
        s = torch.softmax(logits, dim=-1)
    else:
        s = torch.sigmoid(logits)
    b = _S.gate_b.get(key)
    return s + b if b is not None else s


def stage_on() -> bool:
    return _STAGE and _S.armed


def stage_reset(layer_key: int, T: int) -> None:
    """First MoE layer of a decode forward (main stream; the previous
    forward joined the side stream at its last layer): empty the staging
    buffer. With NEXT the rows were cleared layer by layer (the last one is
    cleared here) and the next-token entries stay; the last layer's
    next-token fetch, owed by the previous forward, is issued here."""
    if not (stage_on() and layer_key == _S.first_key and T <= _TCAP):
        return
    _S.stage_head.zero_()
    if not _NEXT:
        _S.stage_tab.fill_(-1)
        return
    _clear_row(_S.last_key)
    if _S.last_key in _S.nbase:
        _next_issue(_S.last_key, T)


def stage_args(layer_key: int, slot_row: torch.Tensor):
    """(stage row, staging pool, hit counter, first next-token slot, STAGE
    flag) for a decode descriptor build; placeholders when staging is off."""
    if not stage_on():
        return slot_row, slot_row, slot_row, 0, False
    return (_S.stage_tab[layer_key], _S.stage_pool, _S.stage_hits,
            _STAGE_SLOTS, True)


def _issue_stage(pred, tgt, budget, btier, ftier, src=None) -> None:
    """Side stream: claim staging slots for `pred` (layer tgt), copy the
    host rows, then publish them in the stage table. With AFTER the copy
    and publish are deferred to after_moe(src)."""
    n = pred.numel()
    out = _S.out[tgt][:n]
    frow = (ftier.slot_table[tgt] if ftier is not None
            else btier.slot_table[tgt])
    _stage_claim_kernel[(1,)](pred, n, btier.slot_table[tgt], frow,
                              _S.stage_tab[tgt], _S.stage_head,
                              _STAGE_SLOTS, budget, out, _S.stage_tot,
                              HAS_F=ftier is not None)
    if _AFTER and src is not None:
        _S.pending.setdefault(src, []).append((pred, out, tgt))
        return
    _copy_publish(pred, out, tgt)


def _copy_publish(pred, out, tgt) -> None:
    n = pred.numel()
    gather(pred, out, _S.host_base[tgt], _S.row_bytes, _S.stage_pool,
           _CHUNKS, _S.cvt)
    _publish_kernel[(n,)](pred, out, n, _S.stage_tab[tgt])
    if _SJOIN:
        _S.ev[tgt].record(_S.side)
        _S.joins.add(tgt)


def next_at(layer_key: int, x: torch.Tensor, topk_ids: torch.Tensor,
            btier, ftier=None) -> None:
    """Decode MoE input of layer_key (main stream): snapshot x for the
    next-token prediction made after this layer's MoE, and, with the online
    learner, take one SGD step on the prediction made a step ago for this
    step's routing (topk_ids)."""
    T = x.shape[0]
    if not _NEXT or T > _TCAP:
        return
    _S.nT[layer_key] = T
    _S.nargs = (btier, ftier)
    if layer_key not in _S.nbase:
        return
    main = torch.cuda.current_stream()
    if layer_key == _S.last_key and _S.owed:
        # the prediction issued at this forward's first layer reads nx[last]
        _S.owed = False
        main.wait_event(_S.ev_owed)
    _S.nx[layer_key][:T].copy_(x)
    if _NEXT_LR > 0:
        y = _S.ny[layer_key][:T]
        y.zero_()
        y.scatter_(1, topk_ids.long(), 1.0)
        _S.side2.wait_stream(main)
        with torch.cuda.stream(_S.side2):
            # BCE gradient in the xs parameterization (dz = -lr g |xs|^2/c^2
            # matches SGD on W' = W c over x/rms)
            # only on the prediction made one step ago: not in a replay of
            # this step (which predicted already), nor across a gap. The
            # last layer's is made at this step's first layer (from the
            # previous step's x): same step (replays repeat its update).
            lag = 0 if layer_key == _S.last_key else 1
            ok = (_S.nstep == _S.npstep[layer_key:layer_key + 1] + lag)
            g = (_S.npp[layer_key][:T] - y) * ok.float()
            c = _S.nc[layer_key]
            _S.nw[layer_key].addmm_((g / (c * c).clamp_min(1e-12)).t(),
                                    _S.nxp[layer_key][:T], alpha=-_NEXT_LR)
            _S.nb[layer_key].add_(g.sum(0), alpha=-_NEXT_LR)


def _next_issue(key: int, T: int) -> None:
    """Fork the side2 stream here (after `key`'s MoE): predict `key`'s
    experts for the next step, claim its next-token slots, copy, publish."""
    if _S.nargs is None:
        return
    btier, ftier = _S.nargs
    _S.ev_next.record()
    _S.side2.wait_event(_S.ev_next)
    with torch.cuda.stream(_S.side2):
        x = _S.nx[key][:T]
        if _NEXT_LR > 0:
            xf = x.float()
            r = xf.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            c = _S.nc[key]
            c.copy_(torch.where(c > 0, c, r.mean()))
            xs = xf * (c / r)
            z = torch.addmm(_S.nb[key], xs, _S.nw[key].t())
            _S.nxp[key][:T].copy_(xs)
            torch.sigmoid(z, out=_S.npp[key][:T])
        else:
            z = _scores(x, key)
        if key == _S.last_key:
            _S.ev_owed.record(_S.side2)
            _S.owed = True
        n = T * _K
        pred = _S.pred[("n", key)][:n]
        v, idx = torch.topk(z, _K, dim=-1)
        if _NEXT_LR > 0 and _NEXT_MIN_P > 0:
            idx = torch.where(torch.sigmoid(v) >= _NEXT_MIN_P, idx, -1)
        pred.copy_(idx.reshape(-1).to(torch.int32))
        out = _S.out[("n", key)][:n]
        frow = (ftier.slot_table[key] if ftier is not None
                else btier.slot_table[key])
        pstep = _S.npstep[key:key + 1]
        _next_claim_kernel[(1,)](pred, n, btier.slot_table[key], frow,
                                 _S.stage_tab[key], _S.nbase[key],
                                 _NEXT_BUDGET, out, _S.next_tot, _S.nstep,
                                 pstep, HAS_F=ftier is not None)
        gather(pred, out, _S.host_base[key], _S.row_bytes, _S.stage_pool,
               _NEXT_CHUNKS, _S.cvt)
        _publish_kernel[(n,)](pred, out, n, _S.stage_tab[key])
        pstep.copy_(_S.nstep)


def _clear_row(key: int) -> None:
    row = _S.stage_tab[key]
    _row_clear_kernel[(1,)](row, row.numel(), _STAGE_SLOTS, _S.nstep,
                            _S.npstep[key:key + 1],
                            BLOCK=triton.next_power_of_2(row.numel()))


def after_moe(layer_key: int) -> None:
    """End of decode MoE layer `layer_key` (main stream): release the
    copies held for it (AFTER); NEXT: clear the layer's stage row and fork
    its next-token fetch (the last layer's is issued at the next forward's
    first layer; the side2 fork rejoins here so the forward ends joined)."""
    T = _S.nT.pop(layer_key, None) if _NEXT else None
    if T is not None and layer_key != _S.last_key:
        _clear_row(layer_key)
        if layer_key in _S.nbase:
            _next_issue(layer_key, T)
    elif T is not None:
        torch.cuda.current_stream().wait_stream(_S.side2)
    jobs = _S.pending.pop(layer_key, None) if _S.pending else None
    if not jobs:
        return
    if _S.ev_after is None:
        _S.ev_after = torch.cuda.Event()
    _S.ev_after.record()
    _S.side.wait_event(_S.ev_after)
    with torch.cuda.stream(_S.side):
        for pred, out, tgt in jobs:
            _copy_publish(pred, out, tgt)


def issue_at(layer_key: int, x: torch.Tensor, btier, ftier=None) -> None:
    """Decode MoE layer `layer_key`: predict layer_key+N and enqueue its
    fetch on the side stream. Python branching specializes at capture."""
    tgt = layer_key + _N
    if tgt not in _S.gate_w or x.shape[0] > _TCAP:
        return
    T = x.shape[0]
    n = T * _K
    main = torch.cuda.current_stream()
    if _STAGE:
        xb = _S.xbuf.get(tgt)
        if xb is None or xb.shape[1] != x.shape[1] or xb.dtype != x.dtype:
            xb = _S.xbuf[tgt] = torch.empty((_TCAP, x.shape[1]),
                                            dtype=x.dtype, device=x.device)
        xb[:T].copy_(x)
        _S.side.wait_stream(main)
        with torch.cuda.stream(_S.side):
            ids = torch.topk(_scores(xb[:T], tgt), _K, dim=-1).indices
            pred = _S.pred[tgt][:n]
            # token-major, best first: with T > 1 the budget favours token 0
            pred.copy_(ids.reshape(-1).to(torch.int32))
            _issue_stage(pred, tgt, _BUDGET, btier, ftier, layer_key)
        return
    if _ASYNC:
        # the forward stream only snapshots x and forks; prediction, claim
        # and gather all run on the side stream (nothing waits for them)
        xb = _S.xbuf.get(tgt)
        if xb is None or xb.shape[1] != x.shape[1] or xb.dtype != x.dtype:
            xb = _S.xbuf[tgt] = torch.empty((_TCAP, x.shape[1]),
                                            dtype=x.dtype, device=x.device)
        xb[:T].copy_(x)
        _S.side.wait_stream(main)
        with torch.cuda.stream(_S.side):
            ids = torch.topk(_scores(xb[:T], tgt), _K, dim=-1).indices
            pred = _S.pred[tgt][:n]
            pred.copy_(ids.reshape(-1).to(torch.int32))
            issue(pred, btier.slot_table[tgt], _S.ring, _S.head, _BUDGET,
                  _S.host_base[tgt], _S.row_bytes, btier.pool,
                  _S.out[tgt][:n], _S.log_l, _S.log_e, tgt, cvt=_S.cvt)
        return
    ids = torch.topk(_scores(x, tgt), _K, dim=-1).indices
    pred = _S.pred[tgt][:n]
    pred.copy_(ids.reshape(-1).to(torch.int32))
    _S.side.wait_stream(main)
    with torch.cuda.stream(_S.side):
        issue(pred, btier.slot_table[tgt], _S.ring, _S.head, _BUDGET,
              _S.host_base[tgt], _S.row_bytes, btier.pool, _S.out[tgt][:n],
              _S.log_l, _S.log_e, tgt, cvt=_S.cvt)
        _S.ev[tgt].record(_S.side)


def issue_hash(layer_key: int, T: int, btier, ftier=None) -> None:
    """At the first MoE layer: fetch every later hash layer's experts,
    which are tid2eid[token] - known exactly from the step's input ids."""
    if layer_key != _S.first_key or not _S.hash_tab or T > _TCAP:
        return
    ids = _S.input_ids[:T].long()
    main = torch.cuda.current_stream()
    _S.side.wait_stream(main)
    for key, tab in _S.hash_tab.items():
        exp = tab[ids].reshape(-1).to(torch.int32)          # [T * top_k]
        n = exp.numel()
        pred = _S.pred[key][:n]
        pred.copy_(exp)
        with torch.cuda.stream(_S.side):
            if _STAGE:
                _issue_stage(pred, key, n, btier, ftier, layer_key)
                continue
            issue(pred, btier.slot_table[key], _S.ring, _S.head, n,
                  _S.host_base[key], _S.row_bytes, btier.pool,
                  _S.out[key][:n], _S.log_l, _S.log_e, key, cvt=_S.cvt)
            _S.ev[key].record(_S.side)


def demand_at(layer_key: int, topk_ids: torch.Tensor, btier) -> None:
    """Before layer_key's descriptor build, on the main stream: fetch every
    routed expert that is still not resident (in-graph demand fill)."""
    if not _DEMAND or topk_ids.shape[0] > _TCAP or layer_key not in _S.dpred:
        return
    n = topk_ids.numel()
    pred = _S.dpred[layer_key][:n]
    pred.copy_(topk_ids.reshape(-1).to(torch.int32))
    issue(pred, btier.slot_table[layer_key], _S.ring, _S.head, n,
          _S.host_base[layer_key], _S.row_bytes, btier.pool,
          _S.dout[layer_key][:n], _S.log_l, _S.log_e, layer_key,
          chunks=_DCHUNKS, tag=_DEMAND_TAG, cvt=_S.cvt)


def join_at(layer_key: int, T: int) -> None:
    """Before layer_key's descriptor build: wait for its prefetch, if one
    was issued in this forward (static at capture)."""
    if T > _TCAP:
        return
    if _STAGE and _SJOIN and layer_key in _S.joins:
        _S.joins.discard(layer_key)
        torch.cuda.current_stream().wait_event(_S.ev[layer_key])
    if _ASYNC or _STAGE:
        # never wait for a prefetch (DIRECT serves unpublished experts from
        # host); join the side stream once, at the last layer, so the
        # captured fork rejoins before the forward ends
        if layer_key == _S.last_key:
            torch.cuda.current_stream().wait_stream(_S.side)
        return
    if layer_key in _S.hash_tab or (layer_key in _S.gate_w
                                    and layer_key - _N >= 0):
        torch.cuda.current_stream().wait_event(_S.ev[layer_key])


# ----------------------------------------------------------------- host

def begin_step(btier) -> None:
    """Reserve the ring for the next decode forward (runner, before launch;
    no forward in flight). The previous step's ring must be closed."""
    if _NEXT and enabled():
        _S.nstep.add_(1)
    if not enabled() or _ASYNC or _STAGE:
        return
    if _S.prepared and _S.active:
        _S.prepared = False             # reserved by the manager already
        return
    if _S.active:
        end_step(btier)
    t0 = time.perf_counter()
    with btier._lock:
        want = min(_RING, max(_RING_MIN, int(2 * _S.ema_takes) + 16))
        slots = btier._take_slots_batch(want, min_cold=2)
        n = len(slots)
        # usable slots sit at the END of the ring: positions R-n..R-1, and
        # the head starts at R-n (earlier positions are unreachable)
        ring = torch.full((_RING,), -1, dtype=torch.int64)
        pli = torch.full((_RING,), -1, dtype=torch.int64)
        pei = torch.full((_RING,), -1, dtype=torch.int64)
        if n:
            idx = torch.tensor(slots, dtype=torch.int64)
            ring[_RING - n:] = idx
            # unmapped by _take_slots_batch; hold the slots as hard-excluded
            # free markers (not in the free list) for the step
            pli[_RING - n:] = btier._owner_li[idx].long()
            pei[_RING - n:] = btier._owner_ei[idx].long()
            btier._owner_li[idx] = -1
            btier._owner_ei[idx] = -1
    _S.ring_host, _S.prev_li, _S.prev_ei = ring, pli, pei
    if n:
        _S.ring[_RING - n:].copy_(ring[_RING - n:].to(torch.int32),
                                  non_blocking=False)
    _S.head.fill_(_RING - n)
    _S.consumed = _RING - n
    _S.start = _RING - n
    _S.active = True
    _S.n_steps += 1
    _S.t_ms["begin"] += (time.perf_counter() - t0) * 1e3


def reconcile(btier, pin: bool = True) -> int:
    """Adopt ring takes made by the graph so far into host bookkeeping.
    Caller must not hold btier._lock. pin=False after the step's pin scope
    closed (deferred roll on the manager thread)."""
    if not (enabled() and _S.active) or _ASYNC:
        return 0
    t0 = time.perf_counter()
    try:
        return _reconcile(btier, pin)
    finally:
        _S.t_ms["reconcile"] += (time.perf_counter() - t0) * 1e3


def _reconcile(btier, pin: bool) -> int:
    torch.cuda.current_stream().synchronize()
    head = min(int(_S.head.item()), _RING)
    if head <= _S.consumed:
        return 0
    lo = _S.consumed
    li = _S.log_l[lo:head].long().cpu()
    ei = _S.log_e[lo:head].long().cpu()
    dem = li >= _DEMAND_TAG
    n_dem = int(dem.sum())
    li = torch.where(dem, li - _DEMAND_TAG, li)
    sl = _S.ring_host[lo:head]
    with btier._lock:
        # (layer, expert) pairs are unique within a step (CAS claim), and
        # so are ring slots: plain index writes
        btier._mirror[li, ei] = sl.to(btier._mirror.dtype)
        btier._owner_li[sl] = li.to(btier._owner_li.dtype)
        btier._owner_ei[sl] = ei.to(btier._owner_ei.dtype)
        btier._owner_tick[sl] = btier._tick
        if pin:
            btier._step_pin_mask[sl] = True
        btier._freq[li, ei] += 1.0
        btier._n_promoted += head - lo
        btier._win_promoted += head - lo
        btier._kpi_prefetched += head - lo - n_dem
        btier._kpi_demand = getattr(btier, "_kpi_demand", 0) + n_dem
    btier._fire_mutate_hooks()
    _S.n_adopted += head - lo - n_dem
    _S.n_demand += n_dem
    _S.consumed = head
    return head - lo


def step_end_hook(btier) -> None:
    """DeltaTier.step_end (runner thread, step fully executed). Deferred:
    mark the step's GPU end and let the manager pass (woken by step_end)
    close the ring and reserve the next one."""
    if not (enabled() and _S.active):
        return
    if _DEFER and btier._thread is not None:
        if _S.ev_step is None:
            _S.ev_step = torch.cuda.Event()
        _S.ev_step.record()
        _S.roll = True
        _S.prepared = False
        return
    end_step(btier)


def manager_roll(btier) -> None:
    """Manager thread, at the start of a pass (the runner is between
    steps): close the finished step's ring, reserve the next step's. Its GPU
    writes complete before the pass returns, i.e. before the runner's
    wait_manager_idle lets the next forward launch."""
    if not _S.roll:
        return
    _S.roll = False
    _S.ev_step.synchronize()
    end_step(btier, pin=False)
    begin_step(btier)
    _S.prepared = True
    torch.cuda.current_stream().synchronize()


def end_step(btier, pin: bool = True) -> None:
    """Close the step's ring: adopt remaining takes, map unused reserved
    slots back to their previous owner (bytes untouched) or free them."""
    if not (enabled() and _S.active):
        return
    reconcile(btier, pin)
    t0 = time.perf_counter()
    takes = _S.consumed - _S.start
    _S.ema_takes = 0.9 * _S.ema_takes + 0.1 * takes
    n_restore = 0
    with btier._lock:
        sl = _S.ring_host[_S.consumed:]
        pli = _S.prev_li[_S.consumed:]
        pei = _S.prev_ei[_S.consumed:]
        used = sl >= 0
        sl, pli, pei = sl[used], pli[used], pei[used]
        if sl.numel():
            had = pli >= 0
            back = had.clone()
            back[had] = btier._mirror[pli[had], pei[had]] < 0
            rs, rl, re = sl[back], pli[back], pei[back]
            n_restore = int(rs.numel())
            if n_restore:
                btier._mirror[rl, re] = rs.to(btier._mirror.dtype)
                btier._owner_li[rs] = rl.to(btier._owner_li.dtype)
                btier._owner_ei[rs] = re.to(btier._owner_ei.dtype)
                btier._owner_tick[rs] = btier._tick
                dev = btier.dev
                btier.slot_table[rl.to(dev), re.to(dev)] = rs.to(
                    device=dev, dtype=torch.int32)
            btier._free.extend(sl[~back].tolist())
    if n_restore:
        btier._fire_mutate_hooks()
    _S.n_restored += n_restore
    _S.head.fill_(_RING)            # no ring until the next begin_step
    _S.active = False
    _S.t_ms["end"] += (time.perf_counter() - t0) * 1e3


# ------------------------------------------------------- async ring (host)

def async_on() -> bool:
    return _ASYNC and _S.armed


def ring_topup(btier) -> None:
    """Manager pass, btier._lock held: provide up to _RING free slots in
    the ring. Capacity: positions not yet adopted are never rewritten.
    Victims are unmapped (deferred) at the next flush, before the ring
    entries become visible; the slots are held ownerless meanwhile (hard-
    excluded from eviction, not in the free list)."""
    if not async_on():
        return
    k = _RING - (_S.tail_host - _S.adopted)
    if k <= 0:
        return
    slots = btier._take_slots_batch(k, defer_unmap=True)
    if not slots:
        return
    idx = torch.tensor(slots, dtype=torch.int64)
    btier._owner_li[idx] = -1
    btier._owner_ei[idx] = -1
    for sl in slots:
        pos = _S.tail_host % _RING
        _S.ring_shadow[pos] = sl
        _S.ring_queue.append((pos, sl))
        _S.tail_host += 1


def flush_ring(btier) -> None:
    """Runner, step start, btier._lock held, after the victim unmaps on the
    forward stream: write the queued ring entries, then publish the tail."""
    if not (async_on() and _S.ring_queue):
        return
    q, _S.ring_queue = _S.ring_queue, []
    t = torch.tensor(q, dtype=torch.int64)
    dev = btier.dev
    _S.ring.index_copy_(0, t[:, 0].pin_memory().to(dev, non_blocking=True),
                        t[:, 1].to(torch.int32).pin_memory().to(
                            dev, non_blocking=True))
    _S.tail.copy_(torch.tensor([_S.tail_host], dtype=torch.int32
                               ).pin_memory(), non_blocking=True)


def snapshot() -> None:
    """Runner, after a decode step's launch: copy head and the claim log to
    pinned memory (forward stream, non-blocking). Skipped while the last
    snapshot is unconsumed; head is monotonic, nothing is lost."""
    if not async_on() or _S.snap_pending:
        return
    R = _RING
    _S.snap[0:1].copy_(_S.head, non_blocking=True)
    _S.snap[1:1 + R].copy_(_S.log_l, non_blocking=True)
    _S.snap[1 + R:].copy_(_S.log_e, non_blocking=True)
    _S.snap_ev.record()
    _S.snap_pending = True


def adopt(btier) -> int:
    """Manager pass, btier._lock held, BEFORE the seen window is consumed
    (so prefetched experts count as hits, not promotion candidates).
    Adopts the graph's ring takes from a landed snapshot. A pair the host
    mapped elsewhere meanwhile (an async fill) keeps the host's slot: its
    device entry is reset to host truth at the next flush and the ring
    slot is freed (reuse goes through a later flush's stream order)."""
    if not (async_on() and _S.snap_pending) or not _S.snap_ev.query():
        return 0
    R = _RING
    head = int(_S.snap[0])
    _S.snap_pending = False
    n = head - _S.adopted
    if n <= 0:
        return 0
    pos = torch.arange(_S.adopted, head, dtype=torch.int64) % R
    li = _S.snap[1:1 + R][pos].long()
    ei = _S.snap[1 + R:][pos].long()
    sl = _S.ring_shadow[pos]
    _S.adopted = head
    free = btier._mirror[li, ei] < 0
    a_li, a_ei, a_sl = li[free], ei[free], sl[free]
    btier._mirror[a_li, a_ei] = a_sl.to(btier._mirror.dtype)
    btier._owner_li[a_sl] = a_li.to(btier._owner_li.dtype)
    btier._owner_ei[a_sl] = a_ei.to(btier._owner_ei.dtype)
    btier._owner_tick[a_sl] = btier._tick
    btier._freq[a_li, a_ei] += 1.0
    btier._kpi_prefetched += int(a_sl.numel())
    if not bool(free.all()):
        c = ~free
        btier._unmap_queue.append(torch.stack([li[c], ei[c]], 1))
        btier._free.extend(sl[c].tolist())
        _S.n_conflict += int(c.sum())
    _S.n_adopted += n
    return n


def kpi_summary() -> str:
    if not enabled():
        return ""
    if _STAGE:
        nxt = (f"; next-token lane staged {int(_S.next_tot.item())}, served "
               f"{int(_S.stage_hits[2])} (budget {_NEXT_BUDGET}/layer, lr "
               f"{_NEXT_LR}, min p {_NEXT_MIN_P})" if _NEXT else "")
        return (f"; VPREFETCH stage: staged {int(_S.stage_tot.item())}, "
                f"served {int(_S.stage_hits[0]) - int(_S.stage_hits[2])}, "
                f"late {int(_S.stage_hits[1])}{nxt} "
                f"(n={_N}, top-{_K}, "
                f"budget {_BUDGET}/layer, {_STAGE_SLOTS} slots, hash "
                f"{sorted(_S.hash_tab)})")
    if _ASYNC:
        return (f"; VPREFETCH async: adopted {_S.n_adopted}, conflicts "
                f"{_S.n_conflict}, ring provided {_S.tail_host} (n={_N}, "
                f"top-{_K}, budget {_BUDGET}/layer, ring {_RING})")
    return (f"; VPREFETCH adopted {_S.n_adopted}, demand-filled "
            f"{_S.n_demand}, ring restored "
            f"{_S.n_restored} (lookahead {_LOOKA}, n={_N}, budget "
            f"{_BUDGET}/layer; demand fill {_DEMAND}; reserve EMA "
            f"{_S.ema_takes:.1f} takes/step); host ms/step "
            + "/".join(f"{k} {v / max(_S.n_steps, 1):.2f}"
                       for k, v in _S.t_ms.items()))
