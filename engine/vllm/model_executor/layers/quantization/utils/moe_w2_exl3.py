# SPDX-License-Identifier: Apache-2.0
"""EXL3 trellis base tier for the moe_w2 stack (vLLM-Moet).

Serves routed-expert GEMMs from EXL3 packs (built by exl3-ab/
build_exl3_pack.py) via exllamav3's grouped multi-matrix kernel
`exl3_mgemm` (pointer-table expert indirection, in-kernel weighted
reduction). Designed to REPLACE the 2-bit scalar base tier while leaving
the FP4 need-pool, gate and miss machinery untouched: experts currently
served by the FP4 pool are masked out of the EXL3 call (negative indices
or compaction) and their contribution is added by the existing moe_w4
path — the composition is additive per token.

This module is deliberately vLLM-import-free so it can be parity-tested
standalone (tools/test_moe_w2_exl3.py) and imported by moe_w2_cubit.

Env:
  VLLM_MOE_W2_EXL3_PACK   pack directory (exl3-l{li:02d}.safetensors)
  VLLM_MOE_W2_EXL3_REPO   exllamav3 checkout providing the CUDA ext
                          (default /root/workspace/exllamav3)
  VLLM_MOE_W2_EXL3_W2K    down-proj K variant: 1|2|3 (default 2)
"""
from __future__ import annotations

import importlib
import os
import sys
import types

import torch
import torch.nn.functional as F

_EXT = None
# m-bucketing of the batched (prefill/fallback) mgemm shapes: pad m > 8 to
# the next multiple, collapsing the coop-autotune key space (see _batched
# docstring). 0/1 disables.
_M_BUCKET = int(os.environ.get("VLLM_MOE_W2_EXL3_M_BUCKET", "16"))
# Prefill route over exl3_moe_dual (mission PREFILL-TRELLIS-GLM, [M]
# 2026-08-24): union-decode fused MoE-FFN — decodes each routed expert's
# trellis ONCE and runs MMA over its token rows, vs _batched's m*k slot
# expansion that re-decodes the trellis per (token, expert) assignment.
# Measured 8.08x vs the mgemm expansion on the DS4 prefill anatomy ([T]
# 8253865); geometry (K/N) is runtime, so GLM's H=6144 serves from the
# same compiled instance pair. Opt-in per cell.
_MOE_PREFILL = os.environ.get("VLLM_MOE_W2_EXL3_MOE_PREFILL", "0") == "1"
# Per-call fused cap: the kernel stages at most this many token rows per
# expert (temp buffers are [concurrency, 128, H|I]); batches whose densest
# expert exceeds it split on the token axis and recurse.
_MOE_PREFILL_ROWS = 128
# Prefill route over the m16x4 SASS canon pair (tor D, mission
# PREFILL-TRELLIS-GLM F-D2): expert-major driver — trellis decoded once
# per (expert, 64-row slot), tensor-core HMMA over 4 slabs of 16 rows.
# Lab window vs the exl3_moe_dual equivalent on the same planes/shape:
# 2.09x full-route (28.9 -> 13.8 ms/layer @chunk 2048 top-8, DESIGN
# 5w). GLM base-only (H=6144, plan (2,2,1)); needs the x4 cubin set in
# VLLM_MOE_W2_EXL3_WAVE_M8_CUBIN_DIR. Takes precedence over
# _MOE_PREFILL for eligible chunks; falls through when unavailable.
_WAVE_PREFILL = os.environ.get("VLLM_MOE_W2_EXL3_WAVE_PREFILL", "0") == "1"


def build_x4_slots(counts: torch.Tensor, tok_sorted: torch.Tensor,
                   wts_sorted: torch.Tensor):
    """Expert-major slot builder for the x4 prefill route: every expert
    is cut into ceil(count/64) slots of <=64 sorted assignment rows;
    slots pack into groups of 6 (pad slots ride expert 0 with all rows
    empty — staged as zeros by the ext, skipped by the scatter). Empty
    rows carry token -1 / weight 0. Pure tensor function
    (device-agnostic), unit-tested in tools/test_wave_x4_builder.py.
    Returns (row_map (6G,64) int32, row_wts (6G,64) half,
    slot_expert (6G,) long, G) or None when nothing routed."""
    E = counts.shape[0]
    dev = counts.device
    seg = (counts + 63) // 64
    n_slots = int(seg.sum().item())
    if n_slots == 0:
        return None
    G = (n_slots + 5) // 6
    S = 6 * G
    slot_e = torch.repeat_interleave(torch.arange(E, device=dev), seg)
    cum_seg = torch.cumsum(seg, 0) - seg
    segix = (torch.arange(n_slots, device=dev)
             - torch.repeat_interleave(cum_seg, seg))
    starts = torch.cumsum(counts, 0) - counts
    base = starts[slot_e] + segix * 64
    idx = base.unsqueeze(1) + torch.arange(64, device=dev)
    limit = (starts[slot_e] + counts[slot_e]).unsqueeze(1)
    valid = idx < limit
    idxc = idx.clamp(max=max(int(tok_sorted.numel()) - 1, 0))
    neg1 = torch.tensor(-1, dtype=torch.int32, device=dev)
    row_map = torch.full((S, 64), -1, dtype=torch.int32, device=dev)
    row_map[:n_slots] = torch.where(valid, tok_sorted[idxc].to(torch.int32),
                                    neg1)
    zero = torch.zeros((), dtype=wts_sorted.dtype, device=dev)
    row_wts = torch.zeros(S, 64, dtype=torch.half, device=dev)
    row_wts[:n_slots] = torch.where(valid, wts_sorted[idxc], zero).half()
    slot_e_pad = torch.zeros(S, dtype=torch.long, device=dev)
    slot_e_pad[:n_slots] = slot_e
    return row_map, row_wts, slot_e_pad, G


def delta_slot_geom(dk13: int = 2, dk2: int = 2, hidden: int = 4096,
                    inter: int = 2048) -> dict:
    """Byte geometry of one Δ-pool slot (pack v3). Layout (u8, contiguous):
    w13 section [tr1|suh1|svh1|tr3|suh3|svh3] then w2 section
    [tr2|suh2|svh2]. Trellis bytes = (k/16)*(n/16)*(16*dk words)*2; sidecars
    are fp16 per-channel vectors. All offsets are 16-byte aligned for DS4
    dims. Returns dict(dk13, dk2, sec13, sec2, slot_bytes, offs={proj:
    (tr, suh, svh)}) with offsets relative to the SLOT start.

    Per-projection delta K ([P] 2026-08-02, the 1.67 track's winning
    allocation is base (2,2,1) + Δ (2,2,3)): the w13 streams and the w2
    stream may carry different trellis bitrates; dk13 == dk2 reproduces
    the legacy uniform layout byte-for-byte."""
    tr13 = (hidden // 16) * (inter // 16) * dk13 * 32
    tr2 = (inter // 16) * (hidden // 16) * dk2 * 32
    suh13, svh13 = hidden * 2, inter * 2
    suh2, svh2 = inter * 2, hidden * 2
    one13 = tr13 + suh13 + svh13
    sec13 = 2 * one13
    sec2 = tr2 + suh2 + svh2
    offs = {
        "g": (0, tr13, tr13 + suh13),
        "u": (one13, one13 + tr13, one13 + tr13 + suh13),
        "d": (sec13, sec13 + tr2, sec13 + tr2 + suh2),
    }
    return dict(dk13=dk13, dk2=dk2, sec13=sec13, sec2=sec2,
                slot_bytes=sec13 + sec2, offs=offs)


def delta_ptr_tables(pool_base: int, slot_bytes: int,
                     slot_row: torch.Tensor, geom: dict) -> dict:
    """Per-step Δ pointer tables into the pool: for each projection,
    int64 [E] tensors (trellis, suh, svh) = pool_base + slot*slot_bytes +
    section offset. slot -1 (not pooled) is clamped to slot 0 — those
    entries are never selected (negative-index skip + zero weights in
    forward_topk_dual) but must still point at mapped memory. Pure
    elementwise int64 arithmetic: capture-safe."""
    sp = slot_row.clamp_min(0).to(torch.int64) * slot_bytes + pool_base
    return {proj: (sp + otr, sp + osuh, sp + osvh)
            for proj, (otr, osuh, osvh) in geom["offs"].items()}


def build_m8_groups(topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                    cap: int = 6, rows: int = 8) -> list:
    """Partition one decode step's routed experts into the fixed-size
    groups the M=8-native decode wave serves (charter M8 §2bis: the m8
    kernels are FIXED-SHAPE na=6, so the UNION of the step's experts —
    |U| ∈ [k, T*k] — is split into ceil(|U|/cap) groups and the launcher
    runs once per group; every union expert is decoded exactly ONCE per
    step regardless of the partition, which is the whole M8 value).

    topk_ids [T, k] int, topk_weights [T, k] float — any device; the
    grouping is data-dependent HOST logic (this is an eager-only path),
    so both are copied to CPU here and the returned tensors live on CPU.

    Returns a list of per-group triples:
      experts     (cap,)      int64 — pack expert ids feeding the group's
                  slot pointer columns. Deterministic: the union is
                  ascending, chunked in order. The last group is padded
                  to `cap` with REPEATS of the group's first expert
                  (§2bis: real experts with all rows -1) — a repeated
                  in-group expert re-reads bytes that are already being
                  streamed for this group (L2-warm) instead of pulling a
                  foreign expert's 12.6 MB, and its rows are all -1 so
                  stage_in zeroes them and out skips them: exactly zero
                  contribution.
      row_map     (cap, rows) int32 — [expert-in-group][row] = token
                  index into x, or -1 (empty row). Rows of an expert are
                  its routed tokens in ascending token order, packed
                  from row 0 (T <= rows always holds: a token yields at
                  most ONE row per expert).
      row_weights (cap, rows) float16 — the routing weight of (token,
                  expert); exactly 0.0 where row_map == -1. A duplicate
                  expert inside one token's topk row (not produced by
                  real routing, but legal input) accumulates its weights
                  into the single (expert, token) row in fp32 before the
                  half cast.
    """
    T, k = topk_ids.shape
    assert T <= rows, f"M8 grouping serves T<={rows} tokens, got {T}"
    ids = topk_ids.detach().to("cpu", torch.int64)
    wts = topk_weights.detach().to("cpu", torch.float32)
    per: dict[int, dict[int, float]] = {}
    for t in range(T):
        for j in range(k):
            e = int(ids[t, j])
            tw = per.setdefault(e, {})
            tw[t] = tw.get(t, 0.0) + float(wts[t, j])
    union = sorted(per)
    groups = []
    for g0 in range(0, len(union), cap):
        chunk = union[g0:g0 + cap]
        experts = torch.tensor(chunk + [chunk[0]] * (cap - len(chunk)),
                               dtype=torch.int64)
        row_map = torch.full((cap, rows), -1, dtype=torch.int32)
        row_weights = torch.zeros(cap, rows, dtype=torch.float32)
        for s, e in enumerate(chunk):
            for r, (t, w) in enumerate(sorted(per[e].items())):
                row_map[s, r] = t
                row_weights[s, r] = w
        groups.append((experts, row_map, row_weights.to(torch.half)))
    return groups


def build_m8_groups_graph(topk_ids: torch.Tensor,
                          topk_weights: torch.Tensor,
                          cap: int = 6, rows: int = 8,
                          groups: "int | None" = None):
    """Capture-safe (fixed-shape, device-only) builder of the M8/M16
    groups — the in-graph counterpart of build_m8_groups (charter CS [E]
    2026-08-10, F0c: parity 20000/20000 bit-identical with the host
    builder, including (t,e) duplicates, token overlaps, partial and
    fully-empty groups).

    G (`groups`) MUST cover the worst-case union: ceil(min(T*k, E_reach)
    / cap), where E_reach is the number of experts this rank can see
    after the ownership transform (owned set; E when unsharded). The
    legacy default G=T is a TOP-6 invariant (union <= 6T) — [O]
    2026-08-25 finding: top-8 models (GLM) can exceed 6T slots and the
    scatters below then write OUT OF BOUNDS (silent). Callers pass
    `groups` from Exl3BaseTier._wave_groups_for; the default stays only
    for backward compatibility of top-6 call sites. Real groups come
    first, surplus groups are EMPTY (all row_map -1, experts = repeat of
    union[0] so slot pointers stay dereferenceable; the m8g/m16g canon
    guard exits those CTAs in the prologue). Rows are PREFIX-PACKED from
    row 0 — the guard's emptiness test reads row 0 only, so this packing
    is a hard contract, not a convention.

    Everything is sort/cumsum/scatter/where on the input device with
    static shapes — no host syncs, no data-dependent shapes: legal
    inside cudagraph capture. Duplicate (t,e) weights accumulate in
    fp64 -> fp32 -> half, matching the host builder's chain bit-exactly.

    Returns (experts [G, cap] i64, row_map [G, cap, rows] i32,
    row_weights [G, cap, rows] f16), all on topk_ids.device."""
    T, k = topk_ids.shape
    G = groups if groups is not None else T
    S = G * cap
    N = T * k
    dev = topk_ids.device

    flat = topk_ids.reshape(N).to(torch.int64)
    wts = topk_weights.reshape(N).to(torch.float64)

    svals, sidx = torch.sort(flat, stable=True)
    first = torch.ones(N, dtype=torch.bool, device=dev)
    first[1:] = svals[1:] != svals[:-1]
    rank_sorted = torch.cumsum(first.to(torch.int64), 0) - 1
    U = rank_sorted[-1] + 1                              # device scalar
    rank = torch.empty(N, dtype=torch.int64, device=dev)
    rank[sidx] = rank_sorted

    exp_of_rank = torch.zeros(S, dtype=torch.int64, device=dev)
    exp_of_rank.scatter_(0, rank_sorted, svals)

    u_ix = torch.arange(S, device=dev)
    g_first = (u_ix // cap) * cap
    pad_started = exp_of_rank[torch.clamp(g_first, max=S - 1)]
    experts = torch.where(u_ix < U, exp_of_rank,
                          torch.where(g_first < U, pad_started,
                                      exp_of_rank[0]))

    tok = torch.arange(T, device=dev, dtype=torch.int64)
    tok_of_el = tok.repeat_interleave(k)
    cell = rank * rows + tok_of_el
    w_dense = torch.zeros(S * rows, dtype=torch.float64, device=dev)
    w_dense.scatter_add_(0, cell, wts)
    w_dense = w_dense.to(torch.float32).reshape(S, rows)
    p_dense = torch.zeros(S * rows, dtype=torch.int64, device=dev)
    p_dense.scatter_(0, cell, torch.ones_like(cell))
    present = p_dense.reshape(S, rows).to(torch.bool)

    row_idx = torch.cumsum(present.to(torch.int64), 1) - 1
    row_map = torch.full((S, rows), -1, dtype=torch.int64, device=dev)
    row_w = torch.zeros(S, rows, dtype=torch.float32, device=dev)
    tgt = torch.where(present, row_idx, torch.zeros_like(row_idx))
    src_t = torch.arange(rows, device=dev,
                         dtype=torch.int64).unsqueeze(0).expand(S, rows)
    row_map.scatter_(1, tgt, torch.where(present, src_t,
                                         torch.full_like(src_t, -1)))
    row_w.scatter_(1, tgt, torch.where(present, w_dense,
                                       torch.zeros_like(w_dense)))
    # scatter order among duplicate target 0 is undefined -> repair col 0
    any_present = present.any(1)
    first_tok = torch.argmax(present.to(torch.int8), 1)
    fw = w_dense.gather(1, first_tok.unsqueeze(1)).squeeze(1)
    row_map[:, 0] = torch.where(any_present, first_tok,
                                torch.full_like(first_tok, -1))
    row_w[:, 0] = torch.where(any_present, fw, torch.zeros_like(fw))

    return (experts.reshape(G, cap),
            row_map.reshape(G, cap, rows).to(torch.int32),
            row_w.reshape(G, cap, rows).to(torch.half))


def _load_ext():
    """Load exllamav3_ext without executing the heavy exllamav3 package
    __init__ (tokenizers/generators are irrelevant here)."""
    global _EXT
    if _EXT is not None:
        return _EXT
    repo = os.environ.get("VLLM_MOE_W2_EXL3_REPO",
                          "/root/workspace/exllamav3")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    if repo not in sys.path:
        sys.path.insert(0, repo)
    for name, path in [("exllamav3", f"{repo}/exllamav3")]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [path]
            m.__package__ = name
            sys.modules[name] = m
    ext_mod = importlib.import_module("exllamav3.ext")
    _EXT = ext_mod.exllamav3_ext
    return _EXT


class Exl3BaseTier:
    """Resident EXL3 base tier: per-layer pointer tables over pack tensors.

    forward_topk() computes the full routed-expert block for a batch of
    tokens: y[t] = sum_j w[t,j] * expert_{ids[t,j]}(x[t]), skipping any
    (t, j) marked in fp4_mask (those are served by the FP4 tier outside).
    """

    def __init__(self, pack_dir: str, layers, device: str = "cuda:0",
                 w13_k: int = 2, w2_k: int | None = None,
                 num_experts: int = 256, hidden: int = 4096,
                 inter: int = 2048, act_limit: float = 10.0,
                 experts=None, paged: bool = False):
        """experts: optional expert-ownership subset (EP sharding / partial
        residency). None = all E resident (TP1 full, the historical
        behavior). A sequence of GLOBAL expert ids = uniform subset for
        every layer; a dict {layer_idx: sequence} = per-layer subsets
        (heat-ordered partial residency). Non-owned experts stay addressable
        in every pointer table — their pointers ALIAS the anchor (first
        owned) expert's tensors, the established wrong-but-finite clamp
        pattern — but the caller contract is that non-owned (t, j) pairs
        always carry weight 0 (weight-side zeroing, same as fp4_mask), so
        they contribute exactly nothing while never dereferencing unmapped
        memory. Only owned experts' bytes are loaded (VRAM scales with
        len(experts)).

        paged (BASE cache over the EXL3 tier, 2026-08-20 [F]): NO pack
        tensors are loaded to the GPU — base bytes live in the base
        DeltaTier's pool slots and every base pointer (both the _tables
        g/u/d tuples and the ptab base rows) derives from that tier's
        slot_table, recomputed at pool-MUTATION time by refresh_base_ptrs
        (the refresh_ptabs idiom applied to the base stream). `experts`
        keeps its ownership meaning (the rank's PAGEABLE set); residency
        within it is dynamic. Missing experts point at pool slot 0
        (wrong-but-finite) and the forward glue zeroes their weights from
        the same slot_table in-graph, so a fetch+replay serves them
        exactly."""
        from safetensors import safe_open
        self.dev = device
        self.E = num_experts
        self.H = hidden
        self.I = inter
        self.act_limit = act_limit
        self.w13_k = w13_k
        self.w2_k = w2_k or int(os.environ.get("VLLM_MOE_W2_EXL3_W2K", "2"))
        self.ext = _load_ext()
        self._store: dict[str, torch.Tensor] = {}
        self._tables: dict[int, dict] = {}
        # Resident zero output-hadamard (svh) vectors. Pointing a non-pooled
        # expert's Δ svh at these zeros its Δ contribution exactly (svh is a
        # post-decode multiply, hadamard_inner) — base-only fallback for the
        # decode wave without a kernel-side delta-enable. svh length = the
        # projection's OUTPUT width: up (g/u) -> I, down (d) -> H.
        self._zero_svh_up = torch.zeros(self.I, dtype=torch.half, device=device)
        self._zero_svh_dn = torch.zeros(self.H, dtype=torch.half, device=device)
        # Persistent decode-wave pointer tables (init_ptabs); None until the
        # serving fast path is wired (first eager forward, before capture).
        self._ptab: dict[int, torch.Tensor] | None = None
        # M8 decode-wave state (charter M8 §2bis): cached ext-capability
        # probe and the lazily allocated per-tier work buffers, reused
        # across groups AND layers — stage_in_m8 zeroes the -1 rows, so
        # they are never cleared between calls.
        self._wave_m8_avail: bool | None = None
        self._m8_bufs: tuple | None = None
        # capture-safe M8 (charter CS [E] 2026-08-10): guarded-cubin probe
        # + persistent fixed-shape buffers of the in-graph route (see
        # forward_topk_wave_m8_graph).
        self._wave_m8_guarded: bool | None = None
        self._m8_gbufs: dict | None = None
        # M16 decode-wave state (tor C [O] 2026-08-25, F-C2): rodzina m16
        # (GLM base-only, okno T w [2,16], DUAL-SET z m8 — patrz dispatch)
        # + jej bufory (A-side w layoucie PRZEPLATANYM — ABI kanonow m16).
        self._wave_m16_avail: bool | None = None
        self._wave_m16_guarded: bool | None = None
        self._m16_bufs: tuple | None = None
        self._m16_gbufs: dict | None = None
        # paged-base state (init_base_paged); None = resident base.
        self.paged = paged
        self._base_pairs: list | None = None
        self._base_keys: torch.Tensor | None = None

        # per-layer ownership (EP sharding / heat residency); values are
        # sorted LongTensors of global expert ids, E-long when full.
        self.owned: dict[int, torch.Tensor] = {}
        for li in layers:
            if experts is None:
                own = list(range(self.E))
            elif isinstance(experts, dict):
                own = experts.get(li)
                own = (list(range(self.E)) if own is None
                       else sorted(int(e) for e in own))
            else:
                own = sorted(int(e) for e in experts)
            if not own:
                raise ValueError(
                    f"moe_w2 EXL3: empty expert subset for layer {li}")
            own_set = set(own)
            anchor = own[0]
            self.owned[li] = torch.tensor(own, dtype=torch.long,
                                          device=device)
            if paged:
                # No resident bytes: placeholder [E] pointer tables, filled
                # by refresh_base_ptrs from the base tier's slot_table
                # before the first forward (init_base_paged runs at build).
                self._tables[li] = {
                    proj: tuple(
                        torch.zeros(self.E, dtype=torch.int64, device=device)
                        for _ in range(3))
                    for proj in ("g", "u", "d")
                }
                continue
            path = os.path.join(pack_dir, f"exl3-l{li:02d}.safetensors")
            keep = {}
            with safe_open(path, "pt") as f:
                for ei in own:
                    for proj, kb in (("w1", self.w13_k), ("w3", self.w13_k),
                                     ("w2", self.w2_k)):
                        for part in ("trellis", "suh", "svh"):
                            k = f"e{ei}.{proj}.k{kb}.{part}"
                            keep[k] = f.get_tensor(k).to(device)
            # non-owned experts: alias the anchor's tensors so every [E]
            # pointer table stays fully dereferenceable (wrong-but-finite);
            # the weight-side zeroing contract keeps their contribution 0.
            if len(own) < self.E:
                for ei in range(self.E):
                    if ei in own_set:
                        continue
                    for proj, kb in (("w1", self.w13_k), ("w3", self.w13_k),
                                     ("w2", self.w2_k)):
                        for part in ("trellis", "suh", "svh"):
                            keep[f"e{ei}.{proj}.k{kb}.{part}"] = \
                                keep[f"e{anchor}.{proj}.k{kb}.{part}"]
            # layer-prefixed keys: multi-layer tiers must not overwrite the
            # previous layer's tensors (the pointer tables would dangle)
            self._store.update({f"l{li}.{k}": v for k, v in keep.items()})

            def tbl(proj, kb, part):
                return torch.tensor(
                    [keep[f"e{e}.{proj}.k{kb}.{part}"].data_ptr()
                     for e in range(self.E)],
                    dtype=torch.int64, device=device)

            self._tables[li] = {
                "g": tuple(tbl("w1", self.w13_k, p)
                           for p in ("trellis", "suh", "svh")),
                "u": tuple(tbl("w3", self.w13_k, p)
                           for p in ("trellis", "suh", "svh")),
                "d": tuple(tbl("w2", self.w2_k, p)
                           for p in ("trellis", "suh", "svh")),
            }

    def total_bytes(self) -> int:
        # dedup by storage: non-owned aliases must not double-count
        seen: set[int] = set()
        tot = 0
        for t in self._store.values():
            p = t.data_ptr()
            if p in seen:
                continue
            seen.add(p)
            tot += t.numel() * t.element_size()
        return tot

    def _mgemm(self, A, tabs, C, A_had, idx, wts, kb, mults=(0, 0),
               num_tokens=1):
        # mults = (mcg_mult, mul1_mult) uint32 codebook selectors: (0, 0) =
        # 3INST (the base pack); (0, 0x83DCD12D) = MUL1 (the delta pack).
        self.ext.exl3_mgemm(A, tabs[0], C, tabs[1], A_had, tabs[2], idx, wts,
                            kb, -1, mults[0], mults[1], -1, -1, 0, num_tokens)

    def _batched(self, li: int, x: torch.Tensor, topk_ids: torch.Tensor,
                 base_w2_wts: torch.Tensor,
                 dtabs: dict | None = None, dk13: int = 2, dk2: int = 2,
                 dmults: tuple = (0, 0x83DCD12D),
                 delta_w2_wts: torch.Tensor | None = None,
                 delta_gu_sel: torch.Tensor | None = None) -> torch.Tensor:
        """Shared batched core: ALL m tokens x k pairs in ONE mgemm per
        (projection x stream) — 6 calls per LAYER instead of 6 per token
        (the measured per-call cost is a fixed ~8-16 us barrier train, so
        the per-token loop was the dominant decode/prefill dispatch cost;
        see internal/R_EXL3_KERNEL_ANATOMY_2026-08-03.md).

        Slot layout is token-major ([m*k] = m groups of k); the kernel's
        num_tokens grouped reduction (v1.3.0) sums each group into row t.
        Pair selection is WEIGHT-side only (zeroed w2 reduction weights),
        never index-side -1 skips — same reliability rule the dual path
        established. Capture-safe: fixed shapes, no host syncs.

        base_w2_wts / delta_w2_wts: [m, k] float weights for the base /
        delta down-proj reduction (already masked by the caller).
        delta_gu_sel: [m, k] 0/1 gate for the Δ g/u contribution pre-
        activation (unified semantics); None with dtabs = ungated add
        (dual semantics).

        m-BUCKETING (2026-08-14 [B]): chunked-prefill compositions produce
        an effectively unbounded stream of novel m values, and the coop
        autotuner pays a measured 30-120 s FIRST CALL per (size_m, ...) key
        ([L] 2026-08-11 root-cause; re-measured as ~100 s engine-wide
        stalls, twice per C4 cell, TP2 EP boots 2026-08-14). Rows above the
        decode window (m > 8) are padded to the next multiple of
        VLLM_MOE_W2_EXL3_M_BUCKET (default 16, 0/1 disables): pad rows
        carry row-0 ids at weight 0 — exactly zero contribution, the
        weight-side idiom — and the result is sliced back to the caller's
        m. The shape space collapses to <= BATCHED/bucket keys, so every
        shape tunes once per cache lifetime instead of once per prompt
        composition."""
        t = self._tables[li]
        m0, k = topk_ids.shape
        if _M_BUCKET > 1 and m0 > 8:
            m_pad = -(-m0 // _M_BUCKET) * _M_BUCKET
            if m_pad != m0:
                pad = m_pad - m0
                x = torch.cat([x, torch.zeros(
                    pad, x.shape[1], dtype=x.dtype, device=x.device)])
                topk_ids = torch.cat(
                    [topk_ids, topk_ids[:1].expand(pad, k)])
                base_w2_wts = torch.cat([base_w2_wts, torch.zeros(
                    pad, k, dtype=base_w2_wts.dtype,
                    device=base_w2_wts.device)])
                if delta_w2_wts is not None:
                    delta_w2_wts = torch.cat(
                        [delta_w2_wts, torch.zeros(
                            pad, k, dtype=delta_w2_wts.dtype,
                            device=delta_w2_wts.device)])
                if delta_gu_sel is not None:
                    delta_gu_sel = torch.cat(
                        [delta_gu_sel, torch.zeros(
                            pad, k, dtype=delta_gu_sel.dtype,
                            device=delta_gu_sel.device)])
        m, k = topk_ids.shape
        bszm = m * k
        xh = x.half()
        ids = topk_ids.reshape(1, bszm).contiguous()
        # .contiguous() is load-bearing: at m == 1 the expand+reshape can
        # legally return a stride-0 VIEW (one real row aliased k times) and
        # the kernel indexes A + j*H assuming a dense [bszm, 1, H] layout —
        # slots j >= 1 would read out of bounds (NaN cascade; measured).
        A = xh.unsqueeze(1).expand(m, k, self.H) \
            .reshape(bszm, 1, self.H).contiguous()
        w_base = base_w2_wts.half().reshape(1, bszm).contiguous()

        Cg = torch.zeros(bszm, 1, self.I, dtype=torch.half, device=self.dev)
        Cu = torch.zeros(bszm, 1, self.I, dtype=torch.half, device=self.dev)
        Ah = torch.empty(bszm, 1, self.H, dtype=torch.half, device=self.dev)
        Ah2 = torch.empty(bszm, 1, self.I, dtype=torch.half, device=self.dev)
        self._mgemm(A, t["g"], Cg, Ah, ids, None, self.w13_k)
        self._mgemm(A, t["u"], Cu, Ah, ids, None, self.w13_k)

        if dtabs is not None:
            Cg2 = torch.zeros_like(Cg)
            Cu2 = torch.zeros_like(Cu)
            self._mgemm(A, dtabs["g"], Cg2, Ah, ids, None, dk13, dmults)
            self._mgemm(A, dtabs["u"], Cu2, Ah, ids, None, dk13, dmults)
            if delta_gu_sel is not None:
                sel = delta_gu_sel.half().reshape(bszm, 1, 1)
                g = Cg.float() + sel * Cg2.float()
                u = Cu.float() + sel * Cu2.float()
            else:
                g = Cg.float() + Cg2.float()
                u = Cu.float() + Cu2.float()
        else:
            g = Cg.float()
            u = Cu.float()
        h = (F.silu(g.clamp(max=self.act_limit))
             * u.clamp(min=-self.act_limit, max=self.act_limit)).half()

        Cd = torch.zeros(bszm, 1, self.H, dtype=torch.float, device=self.dev)
        self._mgemm(h, t["d"], Cd, Ah2, ids, w_base, self.w2_k,
                    num_tokens=m)
        y = Cd[:m0, 0]
        if dtabs is not None:
            w_d = delta_w2_wts.half().reshape(1, bszm).contiguous()
            Cd2 = torch.zeros_like(Cd)
            self._mgemm(h, dtabs["d"], Cd2, Ah2, ids, w_d, dk2, dmults,
                        num_tokens=m)
            y = y + Cd2[:m0, 0]
        return y

    def _graph_cap_tokens(self) -> int:
        """Largest CUDA-graph-captured token count for this engine (0 when
        graphs are off / offline). The fused prefill route does host syncs
        (argsort/bincount/.item()) and dynamic shapes, so it must engage
        strictly ABOVE every capturable size: a warmup taking the fused
        route for a size that later captures via _batched would leave the
        mgemm autotune cold and fire it inside capture (the 4b8618e26
        crash class), and capturing the fused route itself is illegal."""
        cap = getattr(self, "_graph_cap", None)
        if cap is None:
            cap = 0
            try:
                from vllm.config import get_current_vllm_config
                cfg = get_current_vllm_config()
                if (cfg is not None and cfg.model_config is not None
                        and not cfg.model_config.enforce_eager):
                    comp = cfg.compilation_config
                    mode = getattr(comp, "cudagraph_mode", None)
                    if (mode is not None
                            and getattr(mode, "name", "NONE") != "NONE"):
                        cap = int(comp.max_cudagraph_capture_size or 0)
            except Exception:  # noqa: BLE001 - offline tools: no config
                cap = 0
            self._graph_cap = cap
        return cap

    def _moe_dual_bufs(self, E: int):
        """Lazy per-tier scratch for exl3_moe_dual: 4x state [conc, 128, H]
        + 4x intermediate [conc, 128, I] halves and the 9 null int64
        pointer tables that ride the (unused) delta stream on base-only
        cells. Allocated on first fused prefill — the engine's profiling
        run takes this route (T = max_num_batched_tokens > graph cap), so
        the bytes are accounted before the KV pool is sized."""
        bufs = getattr(self, "_dual_bufs", None)
        if bufs is None:
            conc = int(self.ext.exl3_moe_max_concurrency(
                torch.cuda.current_device()))
            st = tuple(torch.empty(conc, _MOE_PREFILL_ROWS, self.H,
                                   dtype=torch.half, device=self.dev)
                       for _ in range(4))
            it = tuple(torch.empty(conc, _MOE_PREFILL_ROWS, self.I,
                                   dtype=torch.half, device=self.dev)
                       for _ in range(4))
            null = torch.zeros(E, dtype=torch.int64, device=self.dev)
            bufs = (st, it, tuple(null for _ in range(9)))
            self._dual_bufs = bufs
        return bufs

    def _fused_prefill(self, li: int, x: torch.Tensor,
                       topk_ids: torch.Tensor,
                       wts: torch.Tensor) -> torch.Tensor:
        """T-above-graph-cap route: one exl3_moe_dual call for the whole
        chunk — expert-major union decode (trellis decoded once per ACTIVE
        expert, MMA over its token rows) instead of _batched's m*k slot
        expansion (trellis re-decoded per assignment). Base-only: the
        delta stream rides null pointer tables (the compiled single-plan
        instance pair). EP compaction falls out naturally: only pairs with
        a nonzero weight enter the sorted assignment lists, so non-owned
        sentinel pairs (ownership transform, weight 0), base-miss-zeroed
        pairs (pager bres) and fp4-masked pairs vanish instead of
        computing dead slots — the algebra is identical (they contributed
        exactly 0 in _batched). Eager-only by construction (host syncs);
        the caller gates on the graph capture envelope."""
        if (self.w13_k, self.w2_k) != (2, 1):
            raise ValueError(
                "VLLM_MOE_W2_EXL3_MOE_PREFILL=1: exl3_moe_dual ships a "
                "single compiled base-plan instance (2,2,1) 3INST; this "
                f"tier serves w13_k={self.w13_k}, w2_k={self.w2_k} — "
                "unset the flag or serve the (2,2,1) base pack")
        t = self._tables[li]
        m0, k = topk_ids.shape
        dev = self.dev
        y = torch.zeros(m0, self.H, dtype=torch.float, device=dev)
        flat_wts = wts.reshape(-1)
        live = flat_wts != 0
        flat_ids = topk_ids.reshape(-1)[live]
        if flat_ids.numel() == 0:
            return y
        flat_wts = flat_wts[live]
        flat_tok = torch.arange(
            m0, device=dev).repeat_interleave(k)[live]
        order = flat_ids.argsort()
        E = t["g"][0].shape[0]
        counts = torch.bincount(flat_ids[order], minlength=E + 1)
        # Host syncs are legal here (route engages only above the graph
        # capture envelope, i.e. eager prefill shapes).
        if int(counts.max().item()) > _MOE_PREFILL_ROWS:
            # Densest expert exceeds the per-call staging rows: split the
            # token range and recurse — per-expert load only shrinks.
            # Rare on flat routing (needs >128 of the chunk's tokens on
            # one expert); worst case degenerates toward per-token cost.
            mid = m0 // 2
            y[:mid] = self._fused_prefill(
                li, x[:mid], topk_ids[:mid], wts[:mid])
            y[mid:] = self._fused_prefill(
                li, x[mid:], topk_ids[mid:], wts[mid:])
            return y
        num_active = int((counts[:E] > 0).sum().item())
        st, it, null9 = self._moe_dual_bufs(E)
        self.ext.exl3_moe_dual(
            x.half().contiguous(), y, counts,
            flat_tok[order].contiguous(),
            flat_wts[order].half().contiguous(),
            *st, *it,
            0,                                   # MOE_ACT_SILU (gated)
            self.w13_k, self.w13_k, self.w2_k,   # base plan (2,2,1) cb0
            2, 2, 3,                             # delta selector (null)
            *t["g"], *t["u"], *t["d"],
            *null9,
            False, False,                        # base: 3INST
            False, True,                         # delta selector: MUL1
            float(self.act_limit), num_active)
        return y

    def _x4_bufs(self):
        """Lazy scratch of the x4 route (one group of 6 experts in
        flight): a16x4/di16x4 = rotated A-frag ABIs of the canons
        (ABI-ROT / ABI-ROT-k1, DESIGN 5v''/5v'''), c_up/c_down = flat
        canon outputs. ~20 MB total, shared across layers. (A
        two-stream group-interleave variant was probed and measured a
        wash — 13.61 vs 13.82 ms/layer, the [K]-class co-residency
        penalty eats the gap-fill gains — so the launcher stays
        single-stream; DESIGN 5w''')."""
        bufs = getattr(self, "_x4_scratch", None)
        if bufs is None:
            hv = torch.half
            bufs = (torch.empty(12, 4, 384, 256, dtype=hv, device=self.dev),
                    torch.empty(12, 64, self.I, dtype=hv, device=self.dev),
                    torch.empty(6, 4, 128, 256, dtype=hv, device=self.dev),
                    torch.empty(6, 64, self.H, dtype=hv, device=self.dev))
            self._x4_scratch = bufs
        return bufs

    def _x4_tabs(self, li: int):
        """Pointer tables in x4 slot order: up = interleaved (w1, w3)
        per expert (2E,), down = (E,). Rebuilt FRESH on every call —
        the paged-base pool mutates trellis addresses on fetch/evict,
        so any cached derived table goes stale after the first pager
        mutation (found on silicon: boot-B needle 0/4 vs baseline 4/4;
        the moe_dual route reads the tier tables per call for the same
        reason). Cost: 3x stack+reshape of (E,) int64 — noise next to
        the layer itself."""
        t = self._tables[li]
        up = tuple(torch.stack((g, u), dim=1).reshape(-1)
                   for g, u in zip(t["g"], t["u"]))
        dn = tuple(d.contiguous() for d in t["d"])
        return up + dn

    def _wave_prefill(self, li: int, x: torch.Tensor,
                      topk_ids: torch.Tensor,
                      wts: torch.Tensor) -> torch.Tensor:
        """Prefill-native x4 route (tor D F-D2): expert-major driver over
        the m16x4 canon pair — trellis decoded once per (expert, 64-row
        slot), HMMA over 4 slabs of 16 rows; 5 phases per group of 6
        experts inside ext.exl3_decode_wave_x4_layer (one call per
        layer). Eager-only (host syncs), engaged above the graph
        capture envelope. Seam parity vs the moe_dual route 3/3
        (rel 6.5e-4, repeat-band 1e-9 class — DESIGN 5w)."""
        t = self._tables[li]                     # ensure/refresh first
        m0, k = topk_ids.shape
        dev = self.dev
        y = torch.zeros(m0, self.H, dtype=torch.float, device=dev)
        flat_wts = wts.reshape(-1)
        live = flat_wts != 0
        flat_ids = topk_ids.reshape(-1)[live]
        if flat_ids.numel() == 0:
            return y
        flat_wts = flat_wts[live]
        flat_tok = torch.arange(
            m0, device=dev).repeat_interleave(k)[live]
        order = flat_ids.argsort()
        E = t["g"][0].shape[0]
        counts = torch.bincount(flat_ids[order], minlength=E)[:E]
        built = build_x4_slots(counts, flat_tok[order],
                               flat_wts[order])
        if built is None:
            return y
        row_map, row_wts, slot_e, G = built
        up_tr, up_suh, up_svh, dn_tr, dn_suh, dn_svh = self._x4_tabs(li)
        se2 = (slot_e.unsqueeze(1) * 2
               + torch.arange(2, device=dev)).reshape(-1)
        self.ext.exl3_decode_wave_x4_layer(
            x.half().contiguous(), y, row_map, row_wts,
            up_tr[se2].view(G, 12), up_suh[se2].view(G, 12),
            up_svh[se2].view(G, 12),
            dn_tr[slot_e].view(G, 6), dn_suh[slot_e].view(G, 6),
            dn_svh[slot_e].view(G, 6),
            *self._x4_bufs(), float(self.act_limit))
        return y

    def _wave_prefill_eligible(self, topk_ids: torch.Tensor) -> bool:
        if not _WAVE_PREFILL:
            return False
        # getattr: cells mounting an FD build without the x4 launcher
        # must fall through to the moe_dual route, not crash.
        avail = getattr(self.ext, "exl3_wave_x4_available", None)
        return (avail is not None
                and (self.w13_k, self.w2_k) == (2, 1)
                and topk_ids.shape[0] > max(8, self._graph_cap_tokens())
                and not torch.cuda.is_current_stream_capturing()
                and bool(avail(self.H)))

    @torch.inference_mode()
    def forward_topk(self, li: int, x: torch.Tensor, topk_ids: torch.Tensor,
                     topk_weights: torch.Tensor,
                     fp4_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x [m, H] (any float dtype); topk_ids [m, k] long; topk_weights
        [m, k] float; fp4_mask [m, k] bool, True = served by FP4 tier.
        Returns y [m, H] float32 with masked contributions EXCLUDED
        (weight-side zeroing at fixed shapes — no compaction, no host
        syncs, batch-of-tokens in one mgemm per projection). With
        VLLM_MOE_W2_EXL3_MOE_PREFILL=1, chunks strictly above the CUDA
        graph capture envelope (and never during capture) take the fused
        expert-major union-decode route instead (_fused_prefill)."""
        wts = topk_weights
        if fp4_mask is not None:
            wts = wts.masked_fill(fp4_mask, 0.0)
        if self._wave_prefill_eligible(topk_ids):
            return self._wave_prefill(li, x, topk_ids, wts)
        if (_MOE_PREFILL
                and topk_ids.shape[0] > max(8, self._graph_cap_tokens())
                and not torch.cuda.is_current_stream_capturing()):
            return self._fused_prefill(li, x, topk_ids, wts)
        return self._batched(li, x, topk_ids, wts)

    @torch.inference_mode()
    def forward_topk_unified(self, li: int, x: torch.Tensor,
                             topk_ids: torch.Tensor,
                             topk_weights: torch.Tensor,
                             dmask: torch.Tensor, dtabs: dict,
                             dk13: int = 2, dk2: int = 2,
                             dmults: tuple = (0, 0x83DCD12D)) -> torch.Tensor:
        """P6-perf: ONE pass serving base AND Δ-pool together — 6 mgemm
        calls per token instead of forward_topk(masked) + forward_topk_dual
        = 9 (which also computed the base g/u twice for pooled pairs).

        Per token, all k pairs at fixed shapes (capture-safe):
          g = g_base + dmask*g_delta;  u likewise  (Δ zeroed pre-activation
          for non-pooled pairs — their slot-0 reads are finite garbage)
          h = silu(clamp(g)) * clamp(u)
          y = w2_base(h, weights=w_full) + w2_delta(h, weights=w_masked)
        Algebraically identical to the split path: pooled pairs get the
        full (Wb+Δ) block on the upgraded h; non-pooled pairs get the pure
        base block (delta w2 weight = 0). Batched: 6 mgemm per LAYER."""
        wts_d = topk_weights.masked_fill(~dmask, 0.0)
        return self._batched(li, x, topk_ids, topk_weights,
                             dtabs=dtabs, dk13=dk13, dk2=dk2, dmults=dmults,
                             delta_w2_wts=wts_d, delta_gu_sel=dmask)

    @torch.inference_mode()
    def forward_topk_dual(self, li: int, x: torch.Tensor,
                          topk_ids: torch.Tensor,
                          topk_weights: torch.Tensor,
                          dmask: torch.Tensor, dtabs: dict,
                          dk13: int = 2, dk2: int = 2,
                          dmults: tuple = (0, 0x83DCD12D)) -> torch.Tensor:
        """Δ-pool contribution: full expert block y = sum_j w[t,j] *
        expert_{base+Δ}(x[t]) for the (t, j) pairs where dmask is True
        (Δ-resident in the pool). Per projection the two trellis streams
        are summed PRE-activation (weight-level additivity: (Wb+Δ)x =
        Wb x + Δ x), then the SwiGLU clamp applies to the sum — matching
        the delta pack's block validation semantics exactly.

        CAPTURE-SAFE: fixed shapes, no compaction, no host syncs. Pair
        selection is weight-side only: EVERY pair computes base+Δ g/u/h
        (non-pooled experts read slot-0 Δ bytes via the clamped pointer
        tables — wrong-but-finite, never unmapped/NaN), and the down-proj
        reduction weights are zeroed for non-pooled pairs, so their
        contribution is exactly 0. (Index-side -1 skip is NOT used: the
        mgemm negative-skip is not reliable mid-array in this fan-out
        mode — measured, parity test stage 1.)

        dtabs: {"g"|"u"|"d": (trellis_ptrs, suh_ptrs, svh_ptrs)} int64 [E]
        tables pointing into the pool slots (built by the caller from
        slot_table + the Δ slot geometry). dmults = (mcg_mult, mul1_mult)
        codebook selectors for the delta stream (pack v3 quantizes with
        MUL1; the base pack is 3INST). Batched: 6 mgemm per LAYER.
        """
        wts_d = topk_weights.masked_fill(~dmask, 0.0)
        return self._batched(li, x, topk_ids, wts_d,
                             dtabs=dtabs, dk13=dk13, dk2=dk2, dmults=dmults,
                             delta_w2_wts=wts_d, delta_gu_sel=None)

    @torch.inference_mode()
    def forward_topk_wave(self, li: int, x: torch.Tensor,
                          topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                          dtabs: dict | None = None,
                          dmask: torch.Tensor | None = None) -> torch.Tensor:
        """Decode path (M=1) via exl3_decode_wave_dual: the whole MoE layer
        in five flat launches instead of six mgemm barrier-trains (track 3.0
        wave; sonda verdict 2026-08-05).

        x [1, H]; topk_ids [1, k] long; topk_weights [1, k] float.
        dmask [1, k] bool selects the Δ-pooled pairs (True = base+Δ). Non-
        pooled pairs fall back to base-only by pointing their Δ output-
        hadamard (svh) at a resident zero vector, which zeros the Δ g/u
        (pre-SwiGLU) and the Δ down contribution exactly — no kernel-side
        delta-enable needed. dmask=None serves every active expert base+Δ
        (== forward_topk_dual all-True). Returns y [1, H] float32 =
        sum_j w[0,j] * expert_{ids[0,j]}(x[0]) with per-pair base/base+Δ.

        The kernel hardcodes the pack-v3 serving pair: base (2,2,1) on cb0
        (3INST) with a K1 down, Δ (2,2,3) on cb2 (MUL1) with a K3 down — so
        the tier must be built w13_k=2, w2_k=1 and dtabs must carry dk13=2,
        dk2=3.

        dtabs=None = the SERVING fast path over the persistent ptab stack
        (init_ptabs): the six slot-major pointer arrays the kernel expects
        (ups [gb,ub,gd,ud] -> 4*na; downs [db,dd] -> 2*na) come out of ONE
        int64 index_select over the per-layer [18, E] table — the Δ rows
        (incl. the zero-svh mask for non-pooled experts) are maintained at
        pool-MUTATION time (refresh_ptabs via the delta tier's mutate hook),
        not per step. The glue anatomy [V] 2026-08-05 measured the per-step
        compose (13 int64 adds + 15 gathers + 6 cats + wheres per LAYER) at
        ~2.0 ms/step of ~1 µs launches; the fast path replaces it with one
        kernel per layer. Values are bit-identical: same pointer arithmetic,
        moved from per-step to per-mutation.

        dtabs given = legacy per-step compose from the Δ pool tables (dmask
        semantics as above) — kept for parity tests and non-serving callers.
        Both paths: capture-safe, no host syncs."""
        assert x.shape[0] == 1, "decode wave is M=1 only"
        assert self.w13_k == 2 and self.w2_k == 1, \
            "wave serves the base (2,2,1) pair (w13_k=2, w2_k=1)"
        ids = topk_ids[0].to(torch.long)
        na = int(ids.numel())
        if dtabs is None:
            # fast path: one gather over the persistent per-layer ptab.
            # Row-major [rows, na] flattens to exactly the slot-major
            # concatenations the kernel wants (see _PTAB_ROWS in init_ptabs).
            assert self._ptab is not None, \
                "forward_topk_wave fast path before init_ptabs()"
            cols = self._ptab[li].index_select(1, ids)   # [18, na]
            up_tr = cols[0:4].reshape(-1)
            up_suh = cols[4:8].reshape(-1)
            up_svh = cols[8:12].reshape(-1)
            dn_tr = cols[12:14].reshape(-1)
            dn_suh = cols[14:16].reshape(-1)
            dn_svh = cols[16:18].reshape(-1)
        else:
            t = self._tables[li]
            g, u, d = t["g"], t["u"], t["d"]
            dg, du, dd = dtabs["g"], dtabs["u"], dtabs["d"]

            # Δ output-hadamard tables, masked to base-only where not pooled
            dg_svh, du_svh, dd_svh = dg[2][ids], du[2][ids], dd[2][ids]
            if dmask is not None:
                m = dmask[0].to(torch.bool)
                zu = dg_svh.new_full((), self._zero_svh_up.data_ptr())
                zd = dd_svh.new_full((), self._zero_svh_dn.data_ptr())
                dg_svh = torch.where(m, dg_svh, zu)
                du_svh = torch.where(m, du_svh, zu)
                dd_svh = torch.where(m, dd_svh, zd)

            # ups slot-major: [base_gate, base_up, delta_gate, delta_up] x na
            up_tr = torch.cat([g[0][ids], u[0][ids], dg[0][ids], du[0][ids]])
            up_suh = torch.cat([g[1][ids], u[1][ids], dg[1][ids], du[1][ids]])
            up_svh = torch.cat([g[2][ids], u[2][ids], dg_svh, du_svh])
            # downs slot-major: [base_down, delta_down] x na
            dn_tr = torch.cat([d[0][ids], dd[0][ids]])
            dn_suh = torch.cat([d[1][ids], dd[1][ids]])
            dn_svh = torch.cat([d[2][ids], dd_svh])

        xh = x.half().reshape(1, self.H).contiguous()
        out = torch.zeros(1, self.H, dtype=torch.float, device=self.dev)
        w = topk_weights[0].to(torch.half).contiguous()
        a_had = torch.empty(4 * na, self.H, dtype=torch.half, device=self.dev)
        c_up = torch.empty(4 * na, self.I, dtype=torch.half, device=self.dev)
        d_in = torch.empty(2 * na, self.I, dtype=torch.half, device=self.dev)
        c_down = torch.empty(2 * na, self.H, dtype=torch.half, device=self.dev)
        self.ext.exl3_decode_wave_dual(
            xh, out, w,
            up_tr.contiguous(), up_suh.contiguous(), up_svh.contiguous(),
            dn_tr.contiguous(), dn_suh.contiguous(), dn_svh.contiguous(),
            a_had, c_up, d_in, c_down,
            self.act_limit, na)
        return out

    # ---- M=8-native decode wave (charter M8 F2 tor B, [K] 2026-08-09) ----

    def ext_wave_m8_available(self) -> bool:
        """True when the loaded ext exposes the M=8-native decode-wave
        launcher AND its loader resolved the m8 cubins (§2bis: lazy
        cuModuleLoad on first launch; exl3_wave_m8_available() -> bool).
        An older ext build without the symbol (AttributeError) or a probe
        that raises degrades to False — the dispatch then keeps the
        existing fallbacks (the AFRAG env-flag-with-degradation pattern).
        Cached: env + ext build cannot change within a process."""
        if self._wave_m8_avail is None:
            try:
                # [N] F-B2: sonda per-geometria (H=6144 -> zestaw GLM
                # base-only uni_k6144+k1_n6144; H=4096 -> pelny dualny).
                self._wave_m8_avail = bool(
                    self.ext.exl3_wave_m8_available(self.H))
            except Exception:
                self._wave_m8_avail = False
        return self._wave_m8_avail

    def ext_wave_m8_guarded(self) -> bool:
        """True when the loader resolved the GUARDED m8 canon set
        (*_m8g.cubin, empty-group prologue guard, 4-ptr ABI) — the hard
        prerequisite of the capture-safe route: with fixed G=T groups
        per captured size, empty groups must cost ~4.5 us (guard exit),
        not ~100 us (full decode). Same degradation contract as
        ext_wave_m8_available (old ext / plain cubins -> False)."""
        if self._wave_m8_guarded is None:
            try:
                self._wave_m8_guarded = bool(
                    self.ext.exl3_wave_m8_guarded(self.H))
            except Exception:
                self._wave_m8_guarded = False
        return self._wave_m8_guarded

    def _m8_workbufs(self) -> tuple:
        """Lazy per-tier m8 work buffers (a_had, c_up, d_in, c_down), the
        §2bis shapes: 24 up / 12 dn slots x 8 rows. Allocated once and
        reused across groups AND layers WITHOUT clearing — stage_in_m8
        zeroes the -1 rows itself (§2bis), so stale rows are never read."""
        if self._m8_bufs is None:
            self._m8_bufs = (
                torch.empty(24, 8, self.H, dtype=torch.half, device=self.dev),
                torch.empty(24, 8, self.I, dtype=torch.half, device=self.dev),
                torch.empty(12, 8, self.I, dtype=torch.half, device=self.dev),
                torch.empty(12, 8, self.H, dtype=torch.half, device=self.dev),
            )
            if self.H == 6144:
                # [N] F-B2 base-only: ucb2/dn3 wylaczone => kwadranty
                # delta c_up[12:24) i c_down[6:12) NIGDY nie sa pisane,
                # a act/out czytaja je addytywnie — zero RAZ przy
                # alokacji chroni przed 0*NaN ze smieci torch.empty
                # (swieze wiersze -1 zeruje stage_in tylko w bazie).
                self._m8_bufs[1][12:].zero_()
                self._m8_bufs[3][6:].zero_()
        return self._m8_bufs

    @torch.inference_mode()
    def forward_topk_wave_m8(self, li: int, x: torch.Tensor,
                             topk_ids: torch.Tensor,
                             topk_weights: torch.Tensor):
        """Decode step for M ∈ [1, 8] tokens via exl3_decode_wave_dual_m8:
        the union of the step's routed experts is partitioned into fixed
        groups of 6 (build_m8_groups) and the m8 launcher runs once per
        group — each union expert's trellis is decoded ONCE (HMMA over 8
        token rows), vs the M·k expansion's per-(token, expert) decode.

        x [T, H]; topk_ids [T, k] int; topk_weights [T, k] float. Returns
        (out [T, H] float32, n_groups). out is allocated AND zeroed here,
        once per call (= once per layer per step): the ext contract
        (§2bis) is that the kernel only ever atomic-accumulates into out
        — accumulation ACROSS groups is how a token collects the sum
        over all its experts — so callers must not pre-zero or reuse it.

        Serving pair and pointer plumbing are exactly the M=1 wave's:
        pack-v3 base (2,2,1) cb0 + Δ (2,2,3) cb2 over the persistent
        ptab stack (init_ptabs must have run) — one index_select over
        the group's 6 expert columns yields the slot-major 24-up/12-dn
        pointer arrays (ups [gb,ub,gd,ud] x 6, downs [db,dd] x 6). The
        Δ zero-svh bake for non-pooled experts rides along unchanged.

        EAGER-ONLY: the grouping is data-dependent host logic (a D2H
        sync on topk); the dispatch never takes this route during
        cudagraph capture. Work buffers are per-tier and dirty by design
        (see _m8_workbufs)."""
        T = x.shape[0]
        assert 1 <= T <= 8, "decode wave m8 serves T in [1, 8]"
        assert self.w13_k == 2 and self.w2_k == 1, \
            "wave m8 serves the base (2,2,1) pair (w13_k=2, w2_k=1)"
        assert self._ptab is not None, \
            "forward_topk_wave_m8 before init_ptabs()"
        groups = build_m8_groups(topk_ids, topk_weights)
        xh = x.half().reshape(T, self.H).contiguous()
        out = torch.zeros(T, self.H, dtype=torch.float, device=self.dev)
        a_had, c_up, d_in, c_down = self._m8_workbufs()
        ptab = self._ptab[li]
        for experts, row_map, row_weights in groups:
            cols = ptab.index_select(1, experts.to(self.dev))   # [18, 6]
            self.ext.exl3_decode_wave_dual_m8(
                xh, out,
                row_map.to(self.dev).contiguous(),
                row_weights.to(self.dev).contiguous(),
                cols[0:4].reshape(-1).contiguous(),
                cols[4:8].reshape(-1).contiguous(),
                cols[8:12].reshape(-1).contiguous(),
                cols[12:14].reshape(-1).contiguous(),
                cols[14:16].reshape(-1).contiguous(),
                cols[16:18].reshape(-1).contiguous(),
                a_had, c_up, d_in, c_down,
                self.act_limit, 6)
        return out, len(groups)

    def _wave_groups_for(self, li: int, T: int, k: int,
                         cap: int = 6) -> int:
        """Liczba grup trasy wave-M8/M16 dla kroku T tokenow top-k —
        POKRYWA najgorszy przypadek unii: ceil(min(T*k, E_reach)/cap),
        gdzie E_reach = |owned[li]| (po transformie ownership routed idy
        to podzbior owned + sentinel z owned; E przy pelnym wladaniu).
        Fix znaleziska [O] 2026-08-25: stare G=T bylo inwariantem top-6
        (union <= 6T) — top-8 (GLM) moze go zlamac => silent OOB w
        builderze. Dla top-6 zwraca dokladnie T (bez zmian dla DS4).
        Statyczne per (li, T, k) — capture-safe (numel = metadana hosta,
        zero syncow)."""
        reach = self.owned[li].numel() if li in self.owned else self.E
        u_max = min(T * k, reach)
        return -(-u_max // cap)

    def _m8_graph_bufs(self, g_max: int = 8) -> dict:
        """Persistent buffers of the capture-safe M8 route: the ext call
        args must live at STABLE addresses (kernel launch params are
        frozen into the captured graph), so the in-graph builder/gather
        WRITE into these each step and the ext reads them by address —
        the init_ptabs idiom extended to the per-step M8 state.
        cols layout (G, 18, 6) row-major: for group g the six slot-major
        pointer arrays the ext takes are CONTIGUOUS row-slices
        (cols[g,0:4] flat = 24 up_tr, [4:8] suh, [8:12] svh, [12:14]
        dn_tr, [14:16] dn_suh, [16:18] dn_svh) — zero per-call copies.
        g_max: maksymalne G tej trasy (fix [O]: dla top-8 > 8; sizing
        raz, przy pierwszym forwardzie — profile-run poprzedza capture)."""
        if self._m8_gbufs is None:
            self._m8_gbufs = {
                "rmap": torch.full((g_max, 6, 8), -1, dtype=torch.int32,
                                   device=self.dev),
                "rw": torch.zeros(g_max, 6, 8, dtype=torch.half,
                                  device=self.dev),
                "cols": torch.zeros(g_max, 18, 6, dtype=torch.int64,
                                    device=self.dev),
                "out": torch.zeros(8, self.H, dtype=torch.float,
                                   device=self.dev),
                "xh": torch.zeros(8, self.H, dtype=torch.half,
                                  device=self.dev),
            }
        assert self._m8_gbufs["rmap"].shape[0] >= g_max, \
            "m8 graph bufs undersized (alloc before capture at max T)"
        return self._m8_gbufs

    @torch.inference_mode()
    def forward_topk_wave_m8_graph(self, li: int, x: torch.Tensor,
                                   topk_ids: torch.Tensor,
                                   topk_weights: torch.Tensor):
        """Capture-safe decode step for T in [2, 8] tokens on the
        M=8-native wave (charter CS [E] 2026-08-10): fixed G = T ext
        calls per step, groups built ON DEVICE (build_m8_groups_graph —
        no host syncs, static shapes), pointers gathered from the
        persistent ptab into persistent per-group buffers. Surplus
        groups are empty (all rows -1): the m8g canon guard exits them
        at ~4.5 us/canon, stage kernels write/skip zeros. Requires
        ext_wave_m8_guarded().

        Legal both inside cudagraph capture and eager (the eager route
        forward_topk_wave_m8 stays preferred outside graphs: its host
        builder emits only the real groups)."""
        T = x.shape[0]
        assert 2 <= T <= 8, "graph wave m8 serves T in [2, 8]"
        assert self.w13_k == 2 and self.w2_k == 1, \
            "wave m8 serves the base (2,2,1) pair (w13_k=2, w2_k=1)"
        assert self._ptab is not None, \
            "forward_topk_wave_m8_graph before init_ptabs()"
        k = topk_ids.shape[1]
        # fix [O] 2026-08-25: G pokrywa najgorsza unie (top-8 > 6T slots;
        # top-6: G == T bez zmian). Sizing bufow po maksimum okna trasy.
        G = self._wave_groups_for(li, T, k)
        bufs = self._m8_graph_bufs(self._wave_groups_for(li, 8, k))
        a_had, c_up, d_in, c_down = self._m8_workbufs()
        ptab = self._ptab[li]

        experts, rmap, rw = build_m8_groups_graph(topk_ids, topk_weights,
                                                  groups=G)
        bufs["rmap"][:G].copy_(rmap)
        bufs["rw"][:G].copy_(rw)
        # (18, G*6) -> (G, 18, 6) into the persistent slot-major buffer
        cols_all = ptab.index_select(1, experts.reshape(-1))
        bufs["cols"][:G].copy_(cols_all.view(18, G, 6).permute(1, 0, 2))
        xh = bufs["xh"][:T]
        xh.copy_(x.reshape(T, self.H))
        out = bufs["out"][:T]
        out.zero_()

        for g in range(G):
            cg = bufs["cols"][g]
            self.ext.exl3_decode_wave_dual_m8(
                xh, out,
                bufs["rmap"][g],
                bufs["rw"][g],
                cg[0:4].reshape(-1),
                cg[4:8].reshape(-1),
                cg[8:12].reshape(-1),
                cg[12:14].reshape(-1),
                cg[14:16].reshape(-1),
                cg[16:18].reshape(-1),
                a_had, c_up, d_in, c_down,
                self.act_limit, 6)
        return out, G

    # ---- M=16-native decode wave (tor C [O] 2026-08-25, F-C2) ----

    def ext_wave_m16_available(self) -> bool:
        """Sonda zestawu m16 (GLM base-only H=6144): loader per-geometria
        w ext (uni_m16[g]_k6144 + k1_m16[g]_n6144 z katalogu
        VLLM_MOE_W2_EXL3_WAVE_M8_CUBIN_DIR). Degradacja jak m8
        (stary ext / brak plikow -> False)."""
        if self._wave_m16_avail is None:
            try:
                self._wave_m16_avail = bool(
                    self.ext.exl3_wave_m16_available(self.H))
            except Exception:
                self._wave_m16_avail = False
        return self._wave_m16_avail

    def ext_wave_m16_guarded(self) -> bool:
        """True gdy loader ma GUARDED zestaw m16 (*_m16g*, guard pustej
        grupy rows-16) — warunek trasy w grafach (stale G grup)."""
        if self._wave_m16_guarded is None:
            try:
                self._wave_m16_guarded = bool(
                    self.ext.exl3_wave_m16_guarded(self.H))
            except Exception:
                self._wave_m16_guarded = False
        return self._wave_m16_guarded

    def _m16_workbufs(self) -> tuple:
        """Lazy per-tier m16 work buffers (a_had16, c_up, d_in16, c_down)
        — 12/24/12/12 slotow x 16 wierszy; a_had16/d_in16 w LAYOUCIE
        PRZEPLATANYM (ABI kanonow m16, DESIGN 5u — stage-kernele ext
        pisza go same). CALOSC zerowana RAZ przy alokacji: kwadranty
        delta c_up[12:24) i gorna polowa c_down[6:12) NIGDY nie sa
        pisane w base-only (act/out czytaja je addytywnie — 0*0=0);
        wiersze/pary -1 nadpisuje stage_in_m16 zerami przy kazdym kroku
        (bufor brudny miedzy grupami by design, jak m8)."""
        if self._m16_bufs is None:
            self._m16_bufs = (
                torch.zeros(12, 16, self.H, dtype=torch.half,
                            device=self.dev),
                torch.zeros(24, 16, self.I, dtype=torch.half,
                            device=self.dev),
                torch.zeros(12, 16, self.I, dtype=torch.half,
                            device=self.dev),
                torch.zeros(12, 16, self.H, dtype=torch.half,
                            device=self.dev),
            )
        return self._m16_bufs

    def _m16_graph_bufs(self, g_max: int) -> dict:
        """Persistent buffery trasy m16 w grafach (adresy stabilne —
        wzorzec _m8_graph_bufs); rmap/rw rows=16, out/xh do T=16."""
        if self._m16_gbufs is None:
            self._m16_gbufs = {
                "rmap": torch.full((g_max, 6, 16), -1, dtype=torch.int32,
                                   device=self.dev),
                "rw": torch.zeros(g_max, 6, 16, dtype=torch.half,
                                  device=self.dev),
                "cols": torch.zeros(g_max, 18, 6, dtype=torch.int64,
                                    device=self.dev),
                "out": torch.zeros(16, self.H, dtype=torch.float,
                                   device=self.dev),
                "xh": torch.zeros(16, self.H, dtype=torch.half,
                                  device=self.dev),
            }
        assert self._m16_gbufs["rmap"].shape[0] >= g_max, \
            "m16 graph bufs undersized (alloc before capture at max T)"
        return self._m16_gbufs

    @torch.inference_mode()
    def forward_topk_wave_m16_graph(self, li: int, x: torch.Tensor,
                                    topk_ids: torch.Tensor,
                                    topk_weights: torch.Tensor):
        """Capture-safe decode step for T in [2, 16] tokens on the
        M=16-native wave (tor C F-C2): G = _wave_groups_for(li, T, k)
        ext calls per step (fix [O]: pokrywa najgorsza unie top-k;
        puste grupy wychodza guardem m16g), groups built ON DEVICE
        (build_m8_groups_graph rows=16), pointers gathered from the
        persistent ptab. DUAL-SET: dispatch kieruje tu T w [9,16]
        (T<=8 zostaje na trasie m8 — kara m16 @T<=8 ~15%, DESIGN 5u');
        okno [2,8] wspierane dla parity/A-B. Requires
        ext_wave_m16_guarded(). Legal in-graph and eager."""
        T = x.shape[0]
        assert 2 <= T <= 16, "graph wave m16 serves T in [2, 16]"
        assert self.w13_k == 2 and self.w2_k == 1, \
            "wave m16 serves the base (2,2,1) pair (w13_k=2, w2_k=1)"
        assert self._ptab is not None, \
            "forward_topk_wave_m16_graph before init_ptabs()"
        k = topk_ids.shape[1]
        G = self._wave_groups_for(li, T, k)
        bufs = self._m16_graph_bufs(self._wave_groups_for(li, 16, k))
        a16, c_up, di16, c_dn = self._m16_workbufs()
        ptab = self._ptab[li]

        experts, rmap, rw = build_m8_groups_graph(topk_ids, topk_weights,
                                                  rows=16, groups=G)
        bufs["rmap"][:G].copy_(rmap)
        bufs["rw"][:G].copy_(rw)
        cols_all = ptab.index_select(1, experts.reshape(-1))
        bufs["cols"][:G].copy_(cols_all.view(18, G, 6).permute(1, 0, 2))
        xh = bufs["xh"][:T]
        xh.copy_(x.reshape(T, self.H))
        out = bufs["out"][:T]
        out.zero_()

        for g in range(G):
            cg = bufs["cols"][g]
            self.ext.exl3_decode_wave_dual_m16(
                xh, out,
                bufs["rmap"][g],
                bufs["rw"][g],
                cg[0:4].reshape(-1),
                cg[4:8].reshape(-1),
                cg[8:12].reshape(-1),
                cg[12:14].reshape(-1),
                cg[14:16].reshape(-1),
                cg[16:18].reshape(-1),
                a16, c_up, di16, c_dn,
                self.act_limit, 6)
        return out, G

    # ---- persistent decode-wave pointer tables (glue fix [V] 2026-08-05) --
    # Row layout of the per-layer [18, E] int64 table. Rows are grouped so a
    # single index_select(1, ids) -> [18, na] yields, via contiguous row-major
    # slices, exactly the six slot-major arrays exl3_decode_wave_dual takes:
    #   0-3   up_tr  = [g.tr,  u.tr,  dg.tr,  du.tr ]
    #   4-7   up_suh = [g.suh, u.suh, dg.suh, du.suh]
    #   8-11  up_svh = [g.svh, u.svh, dg.svh*, du.svh*]
    #   12-13 dn_tr  = [d.tr,  dd.tr ]
    #   14-15 dn_suh = [d.suh, dd.suh]
    #   16-17 dn_svh = [d.svh, dd.svh*]
    # (*) Δ svh rows carry the baked base-only mask: a non-pooled expert's
    # entry points at the resident zero vector instead of the pool — the same
    # semantics the per-step dmask/torch.where used to apply, moved to
    # pool-mutation time.

    def init_ptabs_base_only(self, pairs) -> None:
        """[N] F-B2 (tor B K=6144): ptab stack bez puli Δ — wiersze bazowe
        jak w init_ptabs (statyczne tensory packa; na komorce paged
        odswieza je hub bazy), wiersze Δ STATYCZNE: tr/suh ALIASOWANE na
        odpowiedniki bazowe (waznosc wskaznikow: stage_in w trybie
        base-only czyta tylko sloty bazowe, ale gather wysyla pelne
        tablice — alias trzyma je legalnymi), svh Δ = rezydentne wektory
        ZEROWE (wiersze 10/11/17 — act/out mnoza zerowe kwadranty c_up/
        c_down przez te wektory: 0*0=0, zadnego NaN). Zero hookow — stack
        w pelni statyczny; pool mutacje nie istnieja bez ftier."""
        assert not torch.cuda.is_current_stream_capturing(), \
            "init_ptabs_base_only during capture"
        self._ptab_pairs = sorted(pairs)
        L = len(self._ptab_pairs)
        stack = torch.zeros(L, 18, self.E, dtype=torch.int64, device=self.dev)
        zu = self._zero_svh_up.data_ptr()
        zd = self._zero_svh_dn.data_ptr()
        for ix, (_lk, li) in enumerate(self._ptab_pairs):
            t = self._tables[li]
            for row, src in ((0, t["g"][0]), (1, t["u"][0]),
                             (4, t["g"][1]), (5, t["u"][1]),
                             (8, t["g"][2]), (9, t["u"][2]),
                             (12, t["d"][0]), (14, t["d"][1]),
                             (16, t["d"][2])):
                stack[ix, row] = src
            stack[ix, 2] = stack[ix, 0]      # dg.tr  <- alias g.tr
            stack[ix, 3] = stack[ix, 1]      # du.tr  <- alias u.tr
            stack[ix, 6] = stack[ix, 4]      # dg.suh <- alias g.suh
            stack[ix, 7] = stack[ix, 5]      # du.suh <- alias u.suh
            stack[ix, 10] = zu               # dg.svh <- ZERO
            stack[ix, 11] = zu               # du.svh <- ZERO
            stack[ix, 13] = stack[ix, 12]    # dd.tr  <- alias d.tr
            stack[ix, 15] = stack[ix, 14]    # dd.suh <- alias d.suh
            stack[ix, 17] = zd               # dd.svh <- ZERO
        self._ptab_stack = stack
        self._ptab = {li: stack[ix]
                      for ix, (_lk, li) in enumerate(self._ptab_pairs)}

    def init_ptabs(self, ftier, geom: dict, pairs) -> None:
        """Build the persistent ptab stack and hook refreshes into the delta
        tier. `pairs` = [(layer_key, li), ...] mapping the delta tier's
        slot_table rows (layer_key) to this tier's pack layers (li). Base
        rows are static (resident pack tensors); Δ rows are (re)computed by
        refresh_ptabs, called once now and then from ftier's mutate hook —
        pool mutations only ever happen in eager host code (gate promote,
        prefill ensure, prefetch), never inside a capture/replay, so the
        captured gather always reads current pointers by address."""
        assert not torch.cuda.is_current_stream_capturing(), \
            "init_ptabs during capture (dummy-run should have initialized)"
        self._ptab_pairs = sorted(pairs)
        self._ptab_keys = torch.tensor([lk for lk, _ in self._ptab_pairs],
                                       dtype=torch.long, device=self.dev)
        L = len(self._ptab_pairs)
        stack = torch.zeros(L, 18, self.E, dtype=torch.int64, device=self.dev)
        for ix, (_lk, li) in enumerate(self._ptab_pairs):
            t = self._tables[li]
            for row, src in ((0, t["g"][0]), (1, t["u"][0]),
                             (4, t["g"][1]), (5, t["u"][1]),
                             (8, t["g"][2]), (9, t["u"][2]),
                             (12, t["d"][0]), (14, t["d"][1]),
                             (16, t["d"][2])):
                stack[ix, row] = src
        self._ptab_stack = stack
        self._ptab = {li: stack[ix]
                      for ix, (_lk, li) in enumerate(self._ptab_pairs)}
        self._ptab_pool_base = ftier.pool.data_ptr()
        self._ptab_slot_bytes = int(ftier.slot_bytes)
        self._ptab_offs = geom["offs"]
        self._ptab_zu = torch.tensor(self._zero_svh_up.data_ptr(),
                                     dtype=torch.int64, device=self.dev)
        self._ptab_zd = torch.tensor(self._zero_svh_dn.data_ptr(),
                                     dtype=torch.int64, device=self.dev)
        self.refresh_ptabs(ftier)
        ftier.mutate_hooks.append(lambda: self.refresh_ptabs(ftier))

    @torch.inference_mode()
    def refresh_ptabs(self, ftier) -> None:
        """Recompute the Δ rows of the ptab stack from the CURRENT slot_table
        — vectorized over all layers (~14 small kernels total per mutation
        event vs 13 int64 kernels x 43 layers x every step before). Same
        pointer arithmetic as delta_ptr_tables + the dmask zero-svh bake.
        inference_mode: the stack is an inference tensor (created inside the
        runner's inference context) and mutation hooks may fire from host
        paths outside it (manager tick) — the decorator keeps the in-place
        row writes legal from any caller."""
        st = ftier.slot_table.index_select(0, self._ptab_keys)   # [L, E] i32
        pooled = st >= 0
        sp = (st.clamp_min(0).to(torch.int64) * self._ptab_slot_bytes
              + self._ptab_pool_base)
        offs = self._ptab_offs
        P = self._ptab_stack
        (g_tr, g_suh, g_svh) = offs["g"]
        (u_tr, u_suh, u_svh) = offs["u"]
        (d_tr, d_suh, d_svh) = offs["d"]
        P[:, 2] = sp + g_tr
        P[:, 3] = sp + u_tr
        P[:, 6] = sp + g_suh
        P[:, 7] = sp + u_suh
        P[:, 10] = torch.where(pooled, sp + g_svh, self._ptab_zu)
        P[:, 11] = torch.where(pooled, sp + u_svh, self._ptab_zu)
        P[:, 13] = sp + d_tr
        P[:, 15] = sp + d_suh
        P[:, 17] = torch.where(pooled, sp + d_svh, self._ptab_zd)

    # ---- paged base (BASE cache over EXL3, [F] 2026-08-20) ---------------
    # The base stream's pointers become slot-table-derived exactly like the
    # Δ rows above: base bytes live in the base DeltaTier's pool ([n_slots,
    # base_slot_bytes], sections [w13: tr1|suh1|svh1|tr3|suh3|svh3 | w2:
    # tr|suh|svh] = delta_slot_geom(w13_k, w2_k) offsets), and every
    # mutation of that pool (fetch, eviction, prefill ensure) refreshes
    # BOTH pointer surfaces: the per-layer _tables g/u/d tuples (mgemm
    # batched/prefill paths — device tensors the kernel dereferences at
    # run time) and the ptab stack base rows (decode-wave paths). Missing
    # experts clamp to slot 0 (wrong-but-finite; the pool is zero-filled
    # at creation so even a never-written slot decodes finite bytes) and
    # the forward glue zeroes their weights from the same slot_table
    # in-graph — the replay after a fetch recomputes both from CURRENT
    # tensor contents, which is what makes it exact.

    def init_base_paged(self, btier, geom: dict, pairs) -> None:
        """Wire the paged base: `pairs` = [(layer_key, li), ...] mapping the
        base tier's slot_table rows to this tier's pack layers. Tiers are
        built ONE PER LAYER, so the pointer refresh is consolidated in a
        per-btier hub (_Exl3BasePtrHub): every tier's `_tables` become
        views of ONE [P, 9, E] buffer and a SINGLE mutate hook refreshes
        all layers with one index_select + 9 vectorized writes. The
        original per-tier hooks fired n_layers times per pool mutation
        (fetch/prefetch/eviction), ~15 small CUDA ops each, on the forward
        thread — profiled as the dominant CPU tax of the pager ([G]
        2026-08-22). Registration is strictly pre-capture (asserted), so
        captured graphs bake pointers to the hub buffer and refresh
        mutates it in place — the refresh_ptabs contract, unchanged."""
        assert self.paged, "init_base_paged on a resident-base tier"
        assert not torch.cuda.is_current_stream_capturing(), \
            "init_base_paged during capture"
        self._base_pairs = sorted(pairs)
        self._base_keys = torch.tensor([lk for lk, _ in self._base_pairs],
                                       dtype=torch.long, device=self.dev)
        self._base_pool_base = btier.pool.data_ptr()
        self._base_slot_bytes = int(btier.slot_bytes)
        self._base_offs = geom["offs"]
        hub = getattr(btier, "_exl3_base_ptr_hub", None)
        if hub is None:
            hub = _Exl3BasePtrHub(btier)
            btier._exl3_base_ptr_hub = hub
            btier.mutate_hooks.append(hub.refresh)
        for lk, li in self._base_pairs:
            hub.register(self, lk, li)

    @torch.inference_mode()
    def refresh_base_ptrs(self, btier) -> None:
        """Recompute THIS tier's base pointers from the CURRENT base
        slot_table. Kept for tools/tests; serving refreshes through the
        per-btier hub (one call for ALL layer-tiers), which owns the buffer
        these `_tables` view into."""
        st = btier.slot_table.index_select(0, self._base_keys)   # [L, E] i32
        sp = (st.clamp_min(0).to(torch.int64) * self._base_slot_bytes
              + self._base_pool_base)
        offs = self._base_offs
        for ix, (_lk, li) in enumerate(self._base_pairs):
            row = sp[ix]
            t = self._tables[li]
            for proj in ("g", "u", "d"):
                otr, osuh, osvh = offs[proj]
                tr, suh, svh = t[proj]
                tr.copy_(row + otr)
                suh.copy_(row + osuh)
                svh.copy_(row + osvh)
        self._write_base_ptab_rows(sp)

    def _write_base_ptab_rows(self, sp) -> None:
        """Write the decode-wave ptab BASE rows from `sp` = [n_pairs, E]
        pool byte addresses (Δ rows are refresh_ptabs'). No-op until the
        wave route builds a ptab stack for this tier."""
        if self._ptab is None:
            return
        offs = self._base_offs
        P = self._ptab_stack
        (g_tr, g_suh, g_svh) = offs["g"]
        (u_tr, u_suh, u_svh) = offs["u"]
        (d_tr, d_suh, d_svh) = offs["d"]
        P[:, 0] = sp + g_tr
        P[:, 1] = sp + u_tr
        P[:, 4] = sp + g_suh
        P[:, 5] = sp + u_suh
        P[:, 8] = sp + g_svh
        P[:, 9] = sp + u_svh
        P[:, 12] = sp + d_tr
        P[:, 14] = sp + d_suh
        P[:, 16] = sp + d_svh


class _Exl3BasePtrHub:
    """Consolidated base-pointer refresh for the paged base (D3 mode 5).

    One Exl3BaseTier exists PER LAYER (43 on DS4-Flash), and the original
    per-tier mutate hooks made every base-pool mutation (fetch, prefetch,
    eviction batch) fire n_layers refreshes x ~15 small CUDA ops each —
    ~800+ tiny launches per fetch on the forward thread, profiled as the
    dominant CPU-side tax of the pager (GPU idle 40-60%, [G] ledger
    2026-08-22). The hub owns ONE [P, 9, E] int64 pointer buffer; each
    registered tier's per-layer `_tables` are rebound to views of it
    (strictly pre-capture: registration happens at layer build), and a
    SINGLE mutate hook refreshes every layer with one index_select over
    the base slot_table + 9 vectorized column writes. Column order:
    g(tr, suh, svh), u(...), d(...) = rows 0..8 — matching the tuple
    layout consumers unpack. Decode-wave ptab base rows are delegated
    per tier from the same `sp` (only tiers that built a ptab)."""

    def __init__(self, btier):
        self.btier = btier
        self.slot_bytes = int(btier.slot_bytes)
        self.pool_base = btier.pool.data_ptr()
        self.entries: list = []          # (tier, layer_key, lidx)
        self.buf = None                  # [P, 9, E] int64 (device)
        self.keys = None                 # [P] long (device)

    def register(self, tier, layer_key: int, lidx: int) -> None:
        assert not torch.cuda.is_current_stream_capturing(), \
            "base-ptr hub registration during capture"
        self.entries.append((tier, layer_key, lidx))
        dev = tier._tables[lidx]["g"][0].device
        e = tier.E
        buf = torch.zeros((len(self.entries), 9, e), dtype=torch.int64,
                          device=dev)
        self.buf = buf
        self.keys = torch.tensor([lk for _t, lk, _l in self.entries],
                                 dtype=torch.long, device=dev)
        # rebind EVERY registered tier's tables into the (re)allocated
        # buffer; consumers re-read tier._tables[li] per call, and capture
        # has not happened yet, so old buffers die with their views.
        for ix, (t, _lk, li) in enumerate(self.entries):
            b = buf[ix]
            t._tables[li] = {
                "g": (b[0], b[1], b[2]),
                "u": (b[3], b[4], b[5]),
                "d": (b[6], b[7], b[8]),
            }
        self.refresh()

    @torch.inference_mode()
    def refresh(self) -> None:
        """Mutation-time, eager host code — the refresh_ptabs contract,
        executed ONCE for all layers instead of once per layer-tier."""
        st = self.btier.slot_table.index_select(0, self.keys)    # [P, E]
        sp = (st.clamp_min(0).to(torch.int64) * self.slot_bytes
              + self.pool_base)
        offs = self.entries[0][0]._base_offs
        b = self.buf
        (g_tr, g_suh, g_svh) = offs["g"]
        (u_tr, u_suh, u_svh) = offs["u"]
        (d_tr, d_suh, d_svh) = offs["d"]
        b[:, 0] = sp + g_tr
        b[:, 1] = sp + g_suh
        b[:, 2] = sp + g_svh
        b[:, 3] = sp + u_tr
        b[:, 4] = sp + u_suh
        b[:, 5] = sp + u_svh
        b[:, 6] = sp + d_tr
        b[:, 7] = sp + d_suh
        b[:, 8] = sp + d_svh
        for ix, (t, _lk, _li) in enumerate(self.entries):
            if t._ptab is not None:
                t._write_base_ptab_rows(sp[ix:ix + 1])
