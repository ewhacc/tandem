# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram tables read from the checkpoint files instead of pinned RAM.

VLLM_TANDEM_ENGRAM_FILES=1 (TP1, cpu_offload, no DP sharing): the two Engram
tables (97 GiB as e2m1, 2 x 192 GiB as FP8) are not loaded. A host fetcher
(engram_files.cpp) maps the tables' safetensors files and serves each step's
rows from the page cache, faulting cold pages in from disk, so a small-RAM
box keeps only the rows it uses in memory. Per step and Engram layer:

  publish  (prepare_embeddings, before the decoder layers) copy the layer's
           hash ids to its pinned request block, bump the device seq, ring
  host     master thread sees the seq, the pool copies rows (values +
           scales) into the pinned landing block, acks
  ready    (_ready_rows, at the layer) one GPU thread polls the ack, then
           the usual lookup kernel decodes the landing rows (row i of the
           request = landing row i) into the staged bf16 rows

Layer 1's lookup has layer 0 to hide behind, layer 14's has 13 layers.
Every graph replay issues a fresh seq, so the protocol is capture-safe.
VLLM_TANDEM_ENGRAM_THREADS: fetcher pool size (default 8).
"""

import ctypes
import hashlib
import json
import os
import re
import struct
import subprocess

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

_ON = os.getenv("VLLM_TANDEM_ENGRAM_FILES", "0") == "1"
_HDR = 16
_SPIN = 20_000_000      # ack polls before the GPU gives up (error counted)


def enabled() -> bool:
    return _ON


@triton.jit
def _publish_kernel(ids_ptr, stride_t, stride_h, req_addr, n,
                    HEADS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(ids_ptr + (i // HEADS) * stride_t + (i % HEADS) * stride_h,
                mask=m, other=-1)
    req = req_addr.to(tl.pointer_type(tl.int32))
    tl.store(req + i, v, mask=m)


@triton.jit
def _ring_kernel(seq_ptr, hdr_addr, n):
    s = tl.load(seq_ptr) + 1
    tl.store(seq_ptr, s)
    hd = hdr_addr.to(tl.pointer_type(tl.int32))
    tl.store(hd + 2, n)
    s = tl.inline_asm_elementwise("fence.sc.sys;\n\tmov.b32 $0, $1;", "=r,r",
                                  [s], dtype=tl.int32, is_pure=False, pack=1)
    tl.store(hd, s)


@triton.jit
def _wait_kernel(seq_ptr, hdr_addr, err_ptr, SPIN: tl.constexpr):
    s = tl.load(seq_ptr)
    ack = hdr_addr.to(tl.pointer_type(tl.int32)) + 1
    a = tl.load(ack, volatile=True)
    i = 0
    while (a != s) & (i < SPIN):
        a = tl.load(ack, volatile=True)
        i += 1
    if a != s:
        tl.atomic_add(err_ptr, 1)


def _table_file(model: str, name: str) -> tuple[str, int, list[int]]:
    """(file, data offset, shape) of tensor `name` in a local checkpoint."""
    with open(os.path.join(model, "model.safetensors.index.json")) as f:
        fname = json.load(f)["weight_map"][name]
    path = os.path.join(model, fname)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
    if meta["dtype"] not in ("U8", "F8_E4M3"):
        raise ValueError(f"{name}: dtype {meta['dtype']} (needs U8 / F8_E4M3)")
    return path, 8 + n + meta["data_offsets"][0], meta["shape"]


def _build():
    src = os.path.join(os.path.dirname(__file__), "engram_files.cpp")
    tag = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
    d = os.path.join(os.path.expanduser("~/.cache/vllm"), "engram_files")
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f"engram_files-{tag}.so")
    if not os.path.exists(so):
        tmp = f"{so}.{os.getpid()}.tmp"
        subprocess.run(["g++", "-O3", "-march=native", "-std=c++17", "-shared",
                        "-fPIC", "-o", tmp, src, "-lpthread"], check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.ef_create.restype = ctypes.c_void_p
    lib.ef_create.argtypes = ([ctypes.c_int] + [ctypes.c_void_p] * 9
                              + [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int])
    lib.ef_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.ef_destroy.argtypes = [ctypes.c_void_p]
    return lib


class _Layer:
    def __init__(self, layer_id, files, n_rows, max_rows, vb, sb, dev):
        self.layer_id = layer_id
        self.files = files                  # (wpath, woff, spath, soff)
        self.n_rows = n_rows
        pin = dict(device="cpu", pin_memory=True)   # model init runs under a cuda device context
        self.hdr = torch.zeros(_HDR, dtype=torch.int32, **pin)
        self.req = torch.full((max_rows,), -1, dtype=torch.int32, **pin)
        self.lv = torch.zeros(max_rows, vb, dtype=torch.uint8, **pin)
        self.ls = torch.zeros(max_rows, sb, dtype=torch.uint8, **pin)
        self.lv_d = get_accelerator_view_from_cpu_tensor(self.lv)
        self.ls_d = get_accelerator_view_from_cpu_tensor(self.ls)
        self.seq = torch.zeros(1, dtype=torch.int32, device=dev)


class _State:
    layers: dict = {}           # id(embedding) -> _Layer
    lib = None
    fetch = None
    err = None
    ids = None                  # [max_tokens, heads] i64: landing row of (t, h)


def register(emb, prefix: str, max_tokens: int) -> None:
    """Called by the embedding at construction: map its table files."""
    from vllm.config import get_current_vllm_config
    if emb.dp_size != 1:
        raise ValueError("VLLM_TANDEM_ENGRAM_FILES: no Engram DP")
    if emb.n_hash_cols % emb.tp_size:
        # a padded last shard would publish fewer heads than it stages
        raise ValueError(f"VLLM_TANDEM_ENGRAM_FILES: {emb.n_hash_cols} hash "
                         f"heads do not split evenly over TP={emb.tp_size}")
    m = re.search(r"layers\.(\d+)\.engram", prefix)
    if m is None:
        raise ValueError(f"VLLM_TANDEM_ENGRAM_FILES: no layer id in {prefix!r}")
    lid = int(m.group(1))
    model = get_current_vllm_config().model_config.model
    wpath, woff, wshape = _table_file(model, f"layers.{lid}.engram.embed.weight")
    spath, soff, sshape = _table_file(model, f"layers.{lid}.engram.embed.scale")
    # the files hold every head's rows; TP ranks read their own heads' rows
    # (global row ids, vocab_start..vocab_end) from the same files
    if (wshape[0] < emb.vocab_end_idx or sshape[0] != wshape[0]
            or wshape[1] != emb.row_bytes or sshape[1] != emb.dim // emb.block_size):
        raise ValueError(f"layer {lid} Engram table {wshape}/{sshape} does not "
                         f"match the model ({emb.vocab_end_idx} x "
                         f"{emb.row_bytes}, {emb.dim // emb.block_size})")
    heads = emb.part_n_hash_cols
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    _State.layers[id(emb)] = _Layer(lid, (wpath, woff, spath, soff), wshape[0],
                                    max_tokens * heads, wshape[1], sshape[1], dev)
    if _State.err is None or _State.ids is None or _State.ids.shape[0] < max_tokens:
        _State.err = torch.zeros(1, dtype=torch.int32, device=dev)
        _State.ids = torch.arange(max_tokens * heads, dtype=torch.int64,
                                  device=dev).view(max_tokens, heads)
    logger.info("Engram layer %d: rows read from %s (%.1f GiB, not loaded)", lid,
                os.path.basename(wpath),
                wshape[0] * (wshape[1] + sshape[1]) / 2**30)


def _start() -> None:
    if _State.fetch is not None:
        return
    _State.lib = _build()
    ls = list(_State.layers.values())
    n = len(ls)
    arr = lambda t, xs: (t * n)(*xs)                       # noqa: E731
    enc = [(l.files[0].encode(), l.files[2].encode()) for l in ls]
    first = next(iter(ls))
    thr = int(os.getenv("VLLM_TANDEM_ENGRAM_THREADS", "8"))
    _State.fetch = _State.lib.ef_create(
        n, arr(ctypes.c_char_p, [e[0] for e in enc]),
        arr(ctypes.c_int64, [l.files[1] for l in ls]),
        arr(ctypes.c_char_p, [e[1] for e in enc]),
        arr(ctypes.c_int64, [l.files[3] for l in ls]),
        arr(ctypes.c_int64, [l.n_rows for l in ls]),
        arr(ctypes.c_void_p, [l.hdr.data_ptr() for l in ls]),
        arr(ctypes.c_void_p, [l.req.data_ptr() for l in ls]),
        arr(ctypes.c_void_p, [l.lv.data_ptr() for l in ls]),
        arr(ctypes.c_void_p, [l.ls.data_ptr() for l in ls]),
        first.lv.shape[1], first.ls.shape[1], thr)
    if not _State.fetch:
        raise RuntimeError("Engram file fetcher: mapping the tables failed")
    _State.enc = enc            # keep the path bytes alive
    logger.info("Engram file fetcher started: %d layers, %d threads", n, thr)


def publish(emb, hash_ids: torch.Tensor) -> None:
    """Request this step's rows (hash_ids [T, heads])."""
    _start()
    L = _State.layers[id(emb)]
    n = hash_ids.shape[0] * emb.part_n_hash_cols
    if emb.tp_size > 1:      # this rank's heads (global row ids)
        hash_ids = hash_ids[:, emb.head_start:emb.head_start + emb.part_n_hash_cols]
    if n:
        _publish_kernel[(triton.cdiv(n, 1024),)](
            hash_ids, hash_ids.stride(0), hash_ids.stride(1), L.req.data_ptr(),
            n, HEADS=emb.part_n_hash_cols, BLOCK=1024)
    _ring_kernel[(1,)](L.seq, L.hdr.data_ptr(), n)


def ready(emb, rows: torch.Tensor) -> None:
    """Wait for the fetcher, decode the landing rows into rows [T, heads, dim]."""
    from vllm.models.deepseek_v41.common.engram import _engram_lookup_kernel
    L = _State.layers[id(emb)]
    _wait_kernel[(1,)](L.seq, L.hdr.data_ptr(), _State.err, SPIN=_SPIN)
    T = rows.shape[0]
    n = T * emb.part_n_hash_cols
    if not n:
        return
    grid = min(triton.cdiv(n, 16), emb._num_sms)
    ids = _State.ids
    w = L.lv_d if emb.packed else L.lv_d.view(torch.float8_e4m3fn)
    _engram_lookup_kernel[(grid,)](
        w, L.ls_d, ids, rows, 0, ids.numel(), n, ids.stride(0), ids.stride(1),
        HEAD_START=0, LOCAL_HEADS=emb.part_n_hash_cols,
        TOTAL_HEADS=emb.part_n_hash_cols, DIM=emb.dim, QUANT_BLOCK=emb.block_size,
        BLOCK_R=16, GRID=grid, PACKED=emb.packed)


def raw_stats() -> list | None:
    """Cumulative [requests, rows, busy ns, max ns], or None."""
    if _State.fetch is None:
        return None
    s = (ctypes.c_uint64 * 4)()
    _State.lib.ef_stats(_State.fetch, s)
    return list(s)


def kpi() -> str:
    s = raw_stats()
    if s is None:
        return ""
    req, rows, ns, mx = s
    return (f"Engram files: {req} requests, {rows} rows, "
            f"{ns / max(req, 1) / 1e3:.0f} us/request (max {mx / 1e3:.0f}), "
            f"ack timeouts {int(_State.err.item())}")
