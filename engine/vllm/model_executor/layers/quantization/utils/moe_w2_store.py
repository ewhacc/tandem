# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side expert stores for the 2-bit MoE tiers (moe_w2_delta.DeltaTier).

Three backends behind one tiny interface:

  - PinnedHostStore: today's behaviour — per-layer [E, slot_bytes] host
    tensors (pinned or pageable), rows handed to cudaMemcpyAsync directly.
    Default; byte-identical to the pre-store code path.
  - MmapPackStore (VLLM_MOE_W2_STORE_DIR=<dir>): rows live in a per-rank
    PACK FILE on disk; reads are buffered preads -> pinned stage -> H2D.
    The kernel page cache is the RAM tier (LRU for free), so host RAM holds
    only the hot part of the base instead of the whole 73-190 GiB store —
    and the pack doubles as a persistent quantization cache across boots (a
    layer already in the pack skips D2H staging entirely).
  - TieredPackStore (additionally VLLM_MOE_W2_BASE_RAM_GB=<GiB|auto>, base
    tier only): a PINNED arena of the N most-recently-used rows over the
    same pack. An arena hit is a zero-copy pinned view (H2D DMAs straight
    from it — the exact PinnedHostStore hot path, no syscall, no memcpy);
    a miss preadv's the row into the arena slot (the arena IS the bounce
    buffer), buffered by default so the page cache serves as an
    opportunistic L3 under the arena (VLLM_MOE_W2_TIER_DIRECT=1 for
    O_DIRECT misses). The arena itself can never be reclaimed under
    memory pressure — the hot fetch set stays RAM-fast even on hosts with
    zero spare page cache. Policy is recency (LRU): freq-pinning lost on
    live GLM traces (routing too flat — see GLM_RAMTIER_FINDINGS).

Pack layout (per tier tag, per TP rank):
    <dir>/<tag>.rank<r>of<w>.pack        raw rows, offset = (li*E+ei)*stride
    <dir>/<tag>.rank<r>of<w>.json        sidecar: shapes + layers written

`stride` is slot_bytes rounded up to 4 KiB so the SAME pack serves the
O_DIRECT reader without a repack (O_DIRECT needs 4K-aligned offset/length/
buffer; the pinned arena is page-aligned and stride-strided, so every row
satisfies all three). Rows of layers not listed in the sidecar are holes
(sparse file) and are never read.

Concurrency: every read/write caller already holds the owning DeltaTier's
lock (manager tick, force_promote, ensure_resident are serialized there),
so the shared pinned stage buffer / arena bookkeeping need no lock of
their own.
"""

import contextlib
import fcntl
import hashlib
import json
import mmap
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ALIGN = 4096
_PACK_VERSION = 2
_PACK_CHECK_FIELDS = (
    "version",
    "tag",
    "E",
    "n_layers",
    "slot_bytes",
    "stride",
    "ckpt_id",
    "zero_mode",
    "quantizer_abi",
)


def _ckpt_id():
    """Identity of the checkpoint the pack's rows derive from — the SAME
    function the planes cache uses (model path + sha1 of the safetensors
    index), so both persistent caches agree on what "same model" means.
    A pack is raw expert rows at (layer*E + expert)*stride: NOTHING in its
    geometry distinguishes two checkpoints with equal expert dimensions,
    so without this field a shared STORE_DIR silently serves the other
    model's weights (correct-looking logits, wrong model — the worst
    failure mode). None when no vllm config is current (offline tools,
    unit tests): None only ever matches a pack also written without a
    config, never a production pack."""
    try:
        from vllm.model_executor.layers.quantization.utils.moe_w2_planes_cache import (  # noqa: E501
            _ckpt_id as planes_ckpt_id,
        )
        return planes_ckpt_id()
    except Exception:  # noqa: BLE001 - uninitialized config/offline use
        return None


def _identity_fields() -> dict:
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        MOE_W2_QUANTIZER_ABI,
        zero_mode,
    )
    return {
        "ckpt_id": _ckpt_id(),
        "zero_mode": zero_mode(),
        "quantizer_abi": MOE_W2_QUANTIZER_ABI,
    }


def _rank_suffix() -> str:
    """Pack-file name suffix identifying this rank's shard. TP rank always;
    PP rank appended only under pipeline parallelism (PP ranks host disjoint
    layers but share TP rank numbers — same-name packs would race on the
    sidecar). Graceful fallback when torch.distributed is uninitialized
    (single-GPU, tests, offline tools)."""
    try:
        from vllm.distributed import (get_pp_group,
                                      get_tensor_model_parallel_rank,
                                      get_tensor_model_parallel_world_size)
        tp_rank = get_tensor_model_parallel_rank()
        tp_world = get_tensor_model_parallel_world_size()
        pp = get_pp_group()
        pp_rank, pp_world = pp.rank_in_group, pp.world_size
    except Exception:  # noqa: BLE001 - any uninitialized-state error
        tp_rank, tp_world, pp_rank, pp_world = 0, 1, 0, 1
    s = f"rank{tp_rank}of{tp_world}"
    if pp_world > 1:
        s += f".pp{pp_rank}of{pp_world}"
    return s


def _pack_paths(dir_: str, tag: str, n_layers: int, n_experts: int,
                slot_bytes: int, stride: int) -> tuple[str, str, str, dict]:
    """Identity-keyed v2 paths.

    Different checkpoints or quantizer ABIs must never replace a pack that a
    running process may still have open.  An unresolved checkpoint identity
    gets a process-local namespace and is therefore never reused across boots.
    """
    identity = _identity_fields()
    key_identity = {
        **identity,
        "tag": tag,
        "n_layers": n_layers,
        "E": n_experts,
        "slot_bytes": slot_bytes,
        "stride": stride,
    }
    if key_identity["ckpt_id"] is None:
        key_identity["ckpt_id"] = f"unverified-process-{os.getpid()}"
    key = hashlib.sha256(
        json.dumps(key_identity, sort_keys=True, separators=(",", ":"))
        .encode()).hexdigest()[:20]
    base = f"{tag}.v2.{key}.{_rank_suffix()}"
    return (
        os.path.join(dir_, base + ".pack"),
        os.path.join(dir_, base + ".json"),
        os.path.join(dir_, base + ".lock"),
        identity,
    )


def _pack_meta(tag: str, n_layers: int, n_experts: int, slot_bytes: int,
               stride: int, identity: dict) -> dict:
    return {
        "version": _PACK_VERSION,
        "tag": tag,
        "E": n_experts,
        "n_layers": n_layers,
        "slot_bytes": slot_bytes,
        "stride": stride,
        **identity,
        "layers": [],
    }


def _valid_layers(meta: dict, n_layers: int) -> list[int] | None:
    try:
        layers = [int(li) for li in meta.get("layers", [])]
    except (TypeError, ValueError):
        return None
    if len(layers) != len(set(layers)):
        return None
    if any(li < 0 or li >= n_layers for li in layers):
        return None
    return sorted(layers)


def _meta_matches(meta: dict, expected: dict) -> bool:
    return all(meta.get(k) == expected.get(k) for k in _PACK_CHECK_FIELDS)


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: str, value: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        with open(tmp, "x") as f:
            json.dump(value, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(os.path.dirname(path))
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def _exclusive_lock(fd: int):
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


_EXACT_PIN = os.getenv("VLLM_MOE_W2_EXACT_PIN", "1") == "1"
_PIN_THP = os.getenv("VLLM_MOE_W2_PIN_THP", "1") == "1"


class _RegisteredBuffer:
    """Exact-size, page-aligned pinned host memory: an anonymous mmap
    registered with CUDA (portable + mapped, so device kernels can read it
    through UVA). torch.empty(pin_memory=True) goes through the caching
    host allocator, which rounds every request up to a power of two - a
    3.19 GiB DS4 FP4 layer took 4 GiB, a 1.69 GiB 2-bit layer 2 GiB, a
    24 GiB arena 32 GiB (measured: 172 GiB pinned for a 137 GiB store).

    Private + MADV_HUGEPAGE (VLLM_MOE_W2_PIN_THP=1, default): 2 MiB pages.
    With the IOMMU in translation mode, DMA and UVA reads from 4 KiB-page
    pinned memory ran at half speed (14600KF + RTX 5090, PCIe 5.0 x16: H2D
    22 vs 41 GB/s), and 90 GiB of scattered unmovable 4 KiB pages kept
    kcompactd busy. A shared anonymous mapping is shmem, which gets no
    huge pages under the usual shmem_enabled=never, so the mapping is
    private (nothing forks with it)."""

    def __init__(self, nbytes: int):
        n = max(int(nbytes), 1)
        if _PIN_THP:
            self._mm = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            try:
                self._mm.madvise(mmap.MADV_HUGEPAGE)
            except (AttributeError, OSError):  # no THP: 4 KiB pages
                pass
        else:
            self._mm = mmap.mmap(-1, n)
        self.tensor = torch.frombuffer(self._mm, dtype=torch.uint8)
        self._ptr = self.tensor.data_ptr()
        rc = torch.cuda.cudart().cudaHostRegister(self._ptr, int(nbytes), 3)
        if int(rc) != 0:
            raise RuntimeError(f"cudaHostRegister({nbytes} B) failed: {rc}")
        self._registered = True

    def release(self) -> None:
        if getattr(self, "_registered", False):
            self._registered = False
            torch.cuda.cudart().cudaHostUnregister(self._ptr)

    def __del__(self):
        try:
            self.release()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


def pinned_empty(rows: int, row_bytes: int, pinned: bool = True,
                 keep: list | None = None) -> torch.Tensor:
    """[rows, row_bytes] u8 host tensor; pinned at exact size when
    VLLM_MOE_W2_EXACT_PIN=1 (default). `keep` receives the owning buffer,
    which must outlive every view of the tensor."""
    if not (pinned and _EXACT_PIN and torch.cuda.is_available()):
        return torch.empty(rows, row_bytes, dtype=torch.uint8,
                           pin_memory=pinned)
    buf = _RegisteredBuffer(rows * row_bytes)
    if keep is not None:
        keep.append(buf)
    return buf.tensor.view(rows, row_bytes)


class PinnedHostStore:
    """Per-layer host tensors, exactly the pre-store `_host` dict.

    EP / partial residency: `add_layer(..., owned=[global ids])` keeps a
    COMPACT [n_owned, slot_bytes] tensor plus a per-layer id->row map
    instead of the full [E, slot_bytes]. The full-E layout stored zeroed
    rows for foreign experts at full price — 65 GiB/rank regardless of
    ownership, which is what OOMed TP8 (8 ranks = 520 GiB of unreclaimable
    shm; measured 2026-08-19, 3x cgroup OOM). The ownership transform
    keeps foreign ids out of every candidate path, so `rows_for` maps
    global ids through the row map and FAIL-FASTS on a foreign id rather
    than silently serving zeros."""

    resident = True

    def __init__(self, slot_bytes: int, pinned: bool = True):
        self.slot_bytes = slot_bytes
        self._pinned = pinned
        self._layers: dict[int, torch.Tensor] = {}
        self._bufs: list = []           # exact-size pinned owners
        # layer_key -> [E] global-id -> compact row (-1 = foreign)
        self._rowmaps: dict[int, list[int]] = {}

    def __contains__(self, layer_key: int) -> bool:
        return layer_key in self._layers

    def __len__(self) -> int:
        return len(self._layers)

    def add_layer(self, layer_key: int, parts, owned=None, n_experts=None) -> None:
        """`n_experts` set: `parts` are already compact (one row per owned
        id, ascending), the layer has `n_experts` experts."""
        E = parts[0].shape[0] if n_experts is None else n_experts
        if owned is not None:
            idx = torch.as_tensor(sorted(int(e) for e in owned),
                                  dtype=torch.long, device=parts[0].device)
            if n_experts is None:
                parts = [t.index_select(0, idx) for t in parts]
            rmap = [-1] * E
            for row, ei in enumerate(idx.tolist()):
                rmap[ei] = row
            self._rowmaps[layer_key] = rmap
            E = int(idx.numel())
        host = pinned_empty(E, self.slot_bytes, self._pinned, self._bufs)
        off = 0
        for t in parts:
            host[:, off:off + t.shape[1]].copy_(t, non_blocking=False)
            off += t.shape[1]
        assert off == self.slot_bytes, (off, self.slot_bytes)
        self._layers[layer_key] = host

    def rows_for(self, pairs, scan: bool = False) -> list[torch.Tensor]:
        """Host rows for [(layer, expert), ...]; zero-copy pinned views.
        Global expert ids; compact layers map through their row map.
        `scan` (a prefill-sized one-shot batch) only matters for the
        tiered backend's arena policy — ignored here."""
        rows = []
        for li, ei in pairs:
            rmap = self._rowmaps.get(li)
            if rmap is not None:
                ei = rmap[ei]
                assert ei >= 0, (
                    f"moe_w2 store: foreign expert row requested from "
                    f"compact store (layer {li}) — ownership transform "
                    f"breached")
            rows.append(self._layers[li][ei])
        return rows

    def release(self) -> None:
        self._layers = {}
        self._rowmaps = {}
        for b in self._bufs:
            b.release()
        self._bufs = []


class MmapPackStore:
    """Rows in an on-disk pack file; reads staged through a pinned buffer.

    The store is append-only at load time and read-only afterwards. A layer
    present in the sidecar is trusted (shape-checked) and its staging is
    skipped on later boots — the persistent-quant-cache property.
    """

    resident = False

    def __init__(self, dir_: str, tag: str, n_layers: int, n_experts: int,
                 slot_bytes: int):
        self.tag = tag
        self.slot_bytes = slot_bytes
        self.E = n_experts
        self.n_layers = n_layers
        self.stride = (slot_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
        os.makedirs(dir_, exist_ok=True)
        (self.path, self._sidecar_path, self._lock_path,
         identity) = _pack_paths(
             dir_, tag, n_layers, n_experts, slot_bytes, self.stride)
        self._meta = _pack_meta(
            tag, n_layers, n_experts, slot_bytes, self.stride, identity)
        self._size = self.n_layers * self.E * self.stride
        self._lock_fd = os.open(
            self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        with _exclusive_lock(self._lock_fd):
            old = None
            try:
                with open(self._sidecar_path) as f:
                    old = json.load(f)
            except FileNotFoundError:
                pass
            except (OSError, ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    "moe_w2 store: unreadable sidecar %s (%s) — rebuilding",
                    self._sidecar_path, e)
            layers = _valid_layers(old, n_layers) if isinstance(old, dict) \
                else None
            valid_file = (
                os.path.isfile(self.path)
                and os.path.getsize(self.path) == self._size
            )
            valid = (
                self._meta["ckpt_id"] is not None
                and isinstance(old, dict)
                and _meta_matches(old, self._meta)
                and layers is not None
                and valid_file
            )
            if valid:
                self._meta["layers"] = layers
            else:
                if old is not None or os.path.exists(self.path):
                    logger.warning(
                        "moe_w2 store: incomplete/stale pack generation %s "
                        "— rebuilding without trusting cached layers",
                        self.path)
                # The pack is sparse until written; refuse a build the
                # disk cannot hold rather than fill it (PS8 hit ENOSPC at
                # 150 GiB and stopped everything else on the root disk).
                st = os.statvfs(dir_)
                free = st.f_bavail * st.f_frsize
                if os.path.exists(self.path):
                    free += os.stat(self.path).st_blocks * 512
                margin = int(float(os.getenv(
                    "VLLM_MOE_W2_PACK_MIN_FREE_GB", "32")) * (1 << 30))
                if free < self._size + margin:
                    raise RuntimeError(
                        f"moe_w2 store[{tag}]: building {self.path} needs "
                        f"{self._size / (1 << 30):.1f} GiB + "
                        f"{margin / (1 << 30):.0f} GiB margin but only "
                        f"{free / (1 << 30):.1f} GiB is free on {dir_} "
                        f"(VLLM_MOE_W2_PACK_MIN_FREE_GB sets the margin)")
                tmp = (f"{self.path}.tmp.{os.getpid()}."
                       f"{time.time_ns()}")
                tfd = os.open(
                    tmp, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
                try:
                    os.ftruncate(tfd, self._size)
                    os.fsync(tfd)
                finally:
                    os.close(tfd)
                os.replace(tmp, self.path)
                _fsync_dir(dir_)
                self._meta["layers"] = []
                _atomic_write_json(self._sidecar_path, self._meta)
            self._fd = os.open(self.path, os.O_RDWR)
            if os.fstat(self._fd).st_size != self._size:
                os.close(self._fd)
                raise RuntimeError(
                    f"moe_w2 pack changed size while opening {self.path}")
        self._present = set(self._meta["layers"])
        # reusable pinned stage for reads (grown on demand; callers hold the
        # tier lock, and the tier syncs its H2D copies before the next call,
        # so reuse is safe).
        self._stage = torch.empty(0, slot_bytes, dtype=torch.uint8,
                                  pin_memory=torch.cuda.is_available())
        # write-side staging reused across layers (pageable [E, stride])
        self._wbuf: torch.Tensor | None = None
        self._pool = ThreadPoolExecutor(
            max_workers=int(os.getenv("VLLM_MOE_W2_STORE_THREADS", "8")),
            thread_name_prefix="moe-w2-store")
        self._reads = 0
        self._read_bytes = 0
        self._read_s = 0.0
        if self._present:
            logger.info(
                "moe_w2 store[%s]: pack %s has %d/%d layers — staging for "
                "those layers will be SKIPPED (persistent quant cache)",
                tag, self.path, len(self._present), n_layers)

    # ---- staging (load time) ----------------------------------------

    def __contains__(self, layer_key: int) -> bool:
        return layer_key in self._present

    def __len__(self) -> int:
        return len(self._present)

    def add_layer(self, layer_key: int, parts, owned=None) -> None:
        # `owned` accepted for interface parity with PinnedHostStore; the
        # pack-file layout is fixed at (li*E + ei)*stride, so foreign rows
        # are written as zeros at full size (the pack lives on disk where
        # the page cache reclaims cold pages — RAM is not the constraint
        # this backend solves).
        if not 0 <= int(layer_key) < self.n_layers:
            raise ValueError(
                f"moe_w2 store layer {layer_key} outside "
                f"[0,{self.n_layers})")
        if layer_key in self._present:
            return          # already packed on a previous boot
        E = parts[0].shape[0]
        if E != self.E:
            raise ValueError(
                f"moe_w2 store expected {self.E} experts, got {E}")
        if self._wbuf is None:
            self._wbuf = torch.zeros(self.E, self.stride, dtype=torch.uint8)
        off = 0
        for t in parts:
            self._wbuf[:, off:off + t.shape[1]].copy_(t, non_blocking=False)
            off += t.shape[1]
        if off != self.slot_bytes:
            raise ValueError(
                f"moe_w2 store parts total {off} bytes, expected "
                f"{self.slot_bytes}")
        mv = memoryview(self._wbuf.numpy()).cast("B")
        base_off = layer_key * self.E * self.stride
        with _exclusive_lock(self._lock_fd):
            try:
                with open(self._sidecar_path) as f:
                    disk_meta = json.load(f)
            except (OSError, ValueError, json.JSONDecodeError) as e:
                raise RuntimeError(
                    f"moe_w2 pack manifest unavailable during write: {e}"
                ) from e
            disk_layers = _valid_layers(disk_meta, self.n_layers)
            if (not _meta_matches(disk_meta, self._meta)
                    or disk_layers is None
                    or not os.path.isfile(self.path)
                    or os.path.getsize(self.path) != self._size):
                raise RuntimeError(
                    "moe_w2 pack generation changed or became invalid "
                    f"while writing {self.path}")
            if layer_key in disk_layers:
                self._present = set(disk_layers)
                return
            # A concurrently starting writer may have atomically replaced a
            # generation before this process opened it.  Never write through
            # a stale inode.
            if os.fstat(self._fd).st_ino != os.stat(self.path).st_ino:
                os.close(self._fd)
                self._fd = os.open(self.path, os.O_RDWR)
            written = 0
            while written < len(mv):    # pwrite may be partial (>2 GiB rows)
                n = os.pwrite(
                    self._fd, mv[written:written + (1 << 30)],
                    base_off + written)
                if n <= 0:
                    raise IOError(
                        f"moe_w2 pack write made no progress @ "
                        f"{base_off + written} ({self.path})")
                written += n
            os.fdatasync(self._fd)
            disk_layers.append(layer_key)
            disk_meta["layers"] = sorted(set(disk_layers))
            _atomic_write_json(self._sidecar_path, disk_meta)
            self._meta = disk_meta
            self._present = set(disk_meta["layers"])
        if len(self._present) == self.n_layers:
            self._wbuf = None       # all layers packed; drop write staging

    # ---- reads (serve time) -----------------------------------------

    def rows_for(self, pairs, scan: bool = False) -> list[torch.Tensor]:
        """Pinned-stage rows for [(layer, expert), ...]. The returned views
        alias the shared stage buffer: consume (issue H2D + sync) before the
        next rows_for call — which every DeltaTier call site does.
        `scan` is the tiered backend's arena discipline — ignored here.

        Reads are buffered preads straight into the pinned stage: one
        syscall per row (GIL released, kernel readahead at full drive
        bandwidth) instead of mmap page-fault storms — measured 8x slower
        via mmap on the DS4 PoC, ~100 x 70 KB faults per 6.75 MiB row. An
        fadvise(WILLNEED) pass first lets the drive overlap the cold rows
        of a batch; warm rows are page-cache memcpys — the cache IS the
        RAM tier."""
        n = len(pairs)
        if self._stage.shape[0] < n:
            self._stage = torch.empty(max(n, 2 * self._stage.shape[0]),
                                      self.slot_bytes, dtype=torch.uint8,
                                      pin_memory=torch.cuda.is_available())
        t0 = time.perf_counter()
        offs = [(li * self.E + ei) * self.stride for li, ei in pairs]
        for off in offs:
            try:
                os.posix_fadvise(self._fd, off, self.slot_bytes,
                                 os.POSIX_FADV_WILLNEED)
            except OSError:
                break       # advisory only
        stage_mv = memoryview(self._stage.numpy()).cast("B")

        def _read_one(i_off):
            i, off = i_off
            row = stage_mv[i * self.slot_bytes:(i + 1) * self.slot_bytes]
            done = 0
            while done < self.slot_bytes:
                got = os.preadv(self._fd, [row[done:]], off + done)
                if got <= 0:
                    raise IOError(f"moe_w2 pack short read @ {off + done} "
                                  f"({self.path})")
                done += got

        # preadv releases the GIL: a pool turns both page-cache memcpys and
        # cold NVMe reads into parallel work (single-threaded memcpy at
        # ~6 GB/s made a 400 MB replay fetch cost ~70 ms — measured).
        if n > 2:
            futures = [
                self._pool.submit(_read_one, pair)
                for pair in enumerate(offs)
            ]
            first_error = None
            for future in futures:
                try:
                    future.result()
                except BaseException as e:
                    if first_error is None:
                        first_error = e
            if first_error is not None:
                raise first_error
        else:
            for pair in enumerate(offs):
                _read_one(pair)
        self._reads += n
        self._read_bytes += n * self.slot_bytes
        self._read_s += time.perf_counter() - t0
        return [self._stage[i] for i in range(n)]

    def release(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        try:
            os.close(self._fd)
        except OSError:
            pass
        try:
            os.close(self._lock_fd)
        except OSError:
            pass
        self._present = set()
        self._stage = torch.empty(0, self.slot_bytes, dtype=torch.uint8)
        self._wbuf = None

    def stats(self) -> dict:
        return dict(reads=self._reads, read_bytes=self._read_bytes,
                    read_s=self._read_s)


class TieredPackStore(MmapPackStore):
    """Pinned-arena RAM tier over the pack file, O_DIRECT NVMe underneath.

    The arena holds the `n_slots` most-recently-used rows in ONE pinned
    allocation ([n_slots, stride]; page-aligned base + 4K stride = every
    row O_DIRECT-legal). rows_for returns views INTO the arena:

      - hit: zero-copy pinned view, H2D DMAs straight from it — the exact
        PinnedHostStore hot path (no syscall, no memcpy — this is what
        recovers the pack backend's measured -9%);
      - miss: O_DIRECT preadv straight into the arena slot (the arena is
        its own bounce buffer), evicting the least-recently-used slot not
        referenced by the current batch. Host-only eviction is safe: the
        GPU reads its pool copy, never the arena.

    Miss reads are BUFFERED by default: the page cache then acts as an
    opportunistic L3 under the arena — on a RAM-rich host an arena miss is
    a page-cache memcpy, on a tight host the cache stays small and misses
    degrade gracefully to NVMe reads, while the pinned arena floor (the
    hot fetch set) can never be reclaimed either way. Measured on DS4
    1x5090 (base 11 GiB, arena 20 GiB, same-night A/B): pinned 33.0 tok/s,
    tiered-buffered 32.8 (PARITY — the pack backend without an arena sat
    at 25.5), while pure O_DIRECT misses on the box's Gen3-x4-linked drive
    (3.7 GB/s) cost -33%. VLLM_MOE_W2_TIER_DIRECT=1 forces O_DIRECT misses
    (no cache growth, fully deterministic latency = raw drive speed).

    SCAN RESISTANCE: a rows_for(..., scan=True) batch (ensure_resident,
    i.e. prefill layer working sets) may fill FREE arena slots but never
    evicts — a long-document prefill touching most experts once would
    otherwise wipe the decode hot set (measured on GLM needle runs: arena
    hit-rate halved). Scan misses beyond the free slots are served from
    the parent's buffered stage. Decode fetches (force_promote, manager
    _promote) insert/evict normally — the caller, not a batch-size
    heuristic, decides: on GLM the decode replay fetch is routinely
    100+ rows and a size threshold froze the arena (29->10 tok/s).
    VLLM_MOE_W2_TIER_SCAN=0 disables the discipline entirely.

    PREHEAT: the arena's key list (hot-first) is dumped to
    <pack>.heat.json every 1024 fetch calls (async, off the fetch path);
    on boot the previous hot set is read back into the arena
    (VLLM_MOE_W2_TIER_PREHEAT=0 to skip) so the first requests after a
    restart start from RAM instead of paying the NVMe warmup.

    View-lifetime contract (same as the parent's stage): consume the
    returned rows (issue H2D + sync) before the next rows_for call —
    every DeltaTier call site does, under the tier lock. Slots referenced
    by the CURRENT batch are never evicted within it; a batch larger than
    the whole arena overflows into the parent's buffered stage (correct,
    logged, expected only for absurdly small arenas).
    """

    def __init__(self, dir_: str, tag: str, n_layers: int, n_experts: int,
                 slot_bytes: int, ram_gb: float):
        super().__init__(dir_, tag, n_layers, n_experts, slot_bytes)
        self.n_arena = max(int(ram_gb * 2**30) // self.stride, 16)
        self._arena_bufs: list = []
        self._arena = pinned_empty(self.n_arena, self.stride, True,
                                   self._arena_bufs)
        assert self._arena.data_ptr() % _ALIGN == 0, "pinned base unaligned?"
        self._arena_mv = memoryview(self._arena.numpy()).cast("B")
        self._pos: dict[tuple[int, int], int] = {}     # (li,ei) -> slot
        self._owner_pair: list = [None] * self.n_arena
        self._last = [0] * self.n_arena                # recency clock stamps
        self._clock = 0
        self._free = list(range(self.n_arena))
        # Miss-read mode: buffered (default; page cache = opportunistic L3)
        # or O_DIRECT (deterministic, bypasses the cache). The O_DIRECT fd
        # is separate; the parent's buffered fd keeps serving writes
        # (add_layer) and stage-overflow reads.
        self.direct = os.getenv("VLLM_MOE_W2_TIER_DIRECT", "0") == "1"
        self._dfd = (os.open(self.path, os.O_RDONLY | os.O_DIRECT)
                     if self.direct else -1)
        self.scan_enabled = os.getenv("VLLM_MOE_W2_TIER_SCAN", "1") == "1"
        # fetch metrics (read by DeltaTier._log_summary/_dump)
        self._hit_rows = 0
        self._miss_rows = 0
        self._miss_bytes = 0
        self._lat_hit_ms = deque(maxlen=2048)    # pure arena-hit calls
        self._lat_miss_ms = deque(maxlen=2048)   # calls with >=1 NVMe row
        self._calls = 0
        self._heat_path = self.path + ".heat.json"
        if os.getenv("VLLM_MOE_W2_TIER_PREHEAT", "1") == "1":
            self._preheat()

    # -- internals ------------------------------------------------------

    def _read_row(self, slot: int, off: int) -> None:
        """One row read from the pack into arena slot `slot` (thread-pool
        body). O_DIRECT mode reads the full stride (offset/length/buffer
        all 4K-aligned by construction); buffered mode reads just
        slot_bytes through the page cache."""
        row = self._arena_mv[slot * self.stride:(slot + 1) * self.stride]
        fd, want = ((self._dfd, self.stride) if self.direct
                    else (self._fd, self.slot_bytes))
        done = 0
        while done < want:
            got = os.preadv(fd, [row[done:want]], off + done)
            if got <= 0:
                raise IOError(f"moe_w2 pack short read @ {off + done} "
                              f"({self.path})")
            done += got

    def _read_many(self, fills: list[tuple[int, int]]) -> None:
        """Read all rows and never return while a failed sibling still writes."""
        if len(fills) <= 2:
            for slot, off in fills:
                self._read_row(slot, off)
            return
        futures = [
            self._pool.submit(self._read_row, slot, off)
            for slot, off in fills
        ]
        first_error = None
        for future in futures:
            try:
                future.result()
            except BaseException as e:
                if first_error is None:
                    first_error = e
        if first_error is not None:
            raise first_error

    def _evict_order(self, busy: set) -> list:
        """Slot ids coldest-first, skipping the current batch's slots."""
        order = sorted(range(self.n_arena), key=self._last.__getitem__)
        return [s for s in order if s not in busy]

    def _dump_heat(self, keys: list) -> None:
        """Persist the arena's hot set (async, pool thread). Best-effort:
        a missed dump only costs preheat freshness."""
        try:
            tmp = self._heat_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(dict(version=1, keys=keys), f)
            os.replace(tmp, self._heat_path)
        except OSError as e:
            logger.warning_once("moe_w2 tiered store: heat dump failed: %s",
                                e)

    def _preheat(self) -> None:
        """Refill the arena with the previous run's hot set (boot time,
        before serving — no locking needed). Any failure leaves the arena
        empty and serving proceeds with a cold arena."""
        try:
            with open(self._heat_path) as f:
                keys = [tuple(k) for k in json.load(f).get("keys", [])]
        except (OSError, ValueError, json.JSONDecodeError):
            return
        keys = [k for k in dict.fromkeys(keys)          # dedupe, keep order
                if k[0] in self._present and 0 <= k[1] < self.E]
        keys = keys[:self.n_arena]
        if not keys:
            return
        t0 = time.perf_counter()
        try:
            fills = [(i, (li * self.E + ei) * self.stride)
                     for i, (li, ei) in enumerate(keys)]
            self._read_many(fills)
            for i, k in enumerate(keys):
                self._pos[k] = i
                self._owner_pair[i] = k
                self._last[i] = len(keys) - i      # heat order = recency
            self._free = list(range(len(keys), self.n_arena))
            self._clock = len(keys) + 1
            logger.info(
                "moe_w2 tiered store: arena PREHEATED — %d rows "
                "(%.1f GiB) from %s in %.1f s",
                len(keys), len(keys) * self.stride / 2**30,
                self._heat_path, time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001 - preheat must not kill boot
            logger.warning("moe_w2 tiered store: preheat failed (%s) — "
                           "starting cold", e)
            self._pos = {}
            self._owner_pair = [None] * self.n_arena
            self._last = [0] * self.n_arena
            self._free = list(range(self.n_arena))
            self._clock = 0

    # -- reads ----------------------------------------------------------

    def rows_for(self, pairs, scan: bool = False) -> list[torch.Tensor]:
        t0 = time.perf_counter()
        self._clock += 1
        self._calls += 1
        n = len(pairs)
        # Scan resistance: a prefill batch (caller-flagged) may consume
        # free slots but never evicts (see class docstring) — its overflow
        # reads go via the stage and leave the decode hot set alone.
        scan = scan and self.scan_enabled
        out: list = [None] * n
        busy: set = set()
        miss_idx: list[int] = []
        # pass 1: arena hits (and intra-batch duplicates via _pos updates)
        for i, (li, ei) in enumerate(pairs):
            s = self._pos.get((li, ei))
            if s is not None:
                out[i] = s
                self._last[s] = self._clock
                busy.add(s)
            else:
                miss_idx.append(i)
        # pass 2: place misses — free slots, then LRU eviction (non-scan)
        evict_order = None
        ev_i = 0
        overflow: list[int] = []
        # New keys stay call-local until every read succeeds.  Publishing
        # _pos before I/O made a failed read look like a later arena hit on
        # stale/partially overwritten bytes.
        placed: list[tuple[int, int, int, tuple[int, int]]] = []
        pending: dict[tuple[int, int], int] = {}
        for i in miss_idx:
            li, ei = pairs[i]
            pair = (li, ei)
            s = self._pos.get(pair)
            if s is not None:          # duplicate earlier in this batch
                out[i] = s
                busy.add(s)
                continue
            s = pending.get(pair)
            if s is not None:
                out[i] = s
                busy.add(s)
                continue
            if self._free:
                slot = self._free.pop()
            elif scan:
                overflow.append(i)         # scans never evict
                continue
            else:
                if evict_order is None:
                    evict_order = self._evict_order(busy)
                while ev_i < len(evict_order) \
                        and evict_order[ev_i] in busy:
                    ev_i += 1
                if ev_i >= len(evict_order):
                    overflow.append(i)     # batch > arena; stage fallback
                    continue
                slot = evict_order[ev_i]
                ev_i += 1
                old = self._owner_pair[slot]
                if old is not None:
                    del self._pos[old]
            # The old key is invalid once its bytes may be overwritten.  The
            # new key remains unpublished until the read transaction commits.
            self._owner_pair[slot] = None
            self._last[slot] = self._clock
            busy.add(slot)
            out[i] = slot
            pending[pair] = slot
            placed.append(
                (i, slot, (li * self.E + ei) * self.stride, pair))
        # pass 3: parallel fills (the link saturates at low QD for
        # multi-MiB rows; the pool mainly overlaps syscall latency and,
        # in buffered mode, parallelizes page-cache memcpys)
        stage_rows: dict[int, torch.Tensor] = {}
        try:
            if placed:
                if not self.direct:
                    for _, _, off, _ in placed:
                        try:
                            os.posix_fadvise(
                                self._fd, off, self.slot_bytes,
                                os.POSIX_FADV_WILLNEED)
                        except OSError:
                            break       # advisory only
                self._read_many([(p[1], p[2]) for p in placed])
            # overflow rows (scan discipline, or arena smaller than one
            # batch): parent's buffered stage.  Keep placed rows unpublished
            # until this part of the call succeeds as well.
            if overflow:
                if not scan:
                    logger.warning(
                        "moe_w2 tiered store: batch of %d rows exceeds the "
                        "arena (%d slots) — %d rows served via buffered "
                        "stage; raise VLLM_MOE_W2_BASE_RAM_GB",
                        n, self.n_arena, len(overflow))
                srows = MmapPackStore.rows_for(
                    self, [pairs[i] for i in overflow])
                stage_rows = dict(zip(overflow, srows))
        except BaseException:
            # Bytes in a failed slot are untrusted.  Leave evicted owners
            # absent and return the slot to the free list; a retry must read
            # it again rather than observe a false hit.
            for _, slot, _, _ in placed:
                self._owner_pair[slot] = None
                self._last[slot] = 0
                self._free.append(slot)
            raise
        for _, slot, _, pair in placed:
            self._pos[pair] = slot
            self._owner_pair[slot] = pair
        # metrics + result assembly
        n_miss = len(placed) + len(overflow)
        self._hit_rows += n - n_miss
        self._miss_rows += n_miss
        self._miss_bytes += n_miss * self.stride
        dt_ms = (time.perf_counter() - t0) * 1e3
        (self._lat_miss_ms if n_miss else self._lat_hit_ms).append(dt_ms)
        if placed and self._calls % 1024 == 0:
            keys = sorted(self._pos, key=lambda k: self._last[self._pos[k]],
                          reverse=True)
            self._pool.submit(self._dump_heat, [list(k) for k in keys])
        return [stage_rows[i] if out[i] is None
                else self._arena[out[i], :self.slot_bytes]
                for i in range(n)]

    def release(self) -> None:
        # Arena read tasks may still reference both _dfd and _arena_mv.
        self._pool.shutdown(wait=True, cancel_futures=True)
        if self._dfd >= 0:
            try:
                os.close(self._dfd)
            except OSError:
                pass
        self._pos = {}
        self._owner_pair = []
        self._free = []
        self._arena_mv = None
        self._arena = torch.empty(0, dtype=torch.uint8)
        super().release()

    def stats(self) -> dict:
        def pct(d, q):
            if not d:
                return 0.0
            v = sorted(d)
            return v[min(int(len(v) * q), len(v) - 1)]
        st = super().stats()
        st.update(
            arena_slots=self.n_arena,
            arena_used=self.n_arena - len(self._free),
            hit_rows=self._hit_rows,
            miss_rows=self._miss_rows,
            miss_bytes=self._miss_bytes,
            hit_p50_ms=pct(self._lat_hit_ms, 0.50),
            hit_p99_ms=pct(self._lat_hit_ms, 0.99),
            miss_p50_ms=pct(self._lat_miss_ms, 0.50),
            miss_p99_ms=pct(self._lat_miss_ms, 0.99),
        )
        return st


class PlanesFileStore:
    """Base tier over the planes-cache files: the hottest experts pinned in
    RAM, every other expert read from the files on demand (the NVMe tier).

    VLLM_MOE_W2_BASE_FILES=1 with VLLM_MOE_W2_PLANES_CACHE (a Tandem
    package's vq/). A slot row is [codes13 | sc13 | codes2 | sc2], and the
    planes cache keeps each section as its own file, expert-major (expert e
    at e * section bytes), so the files are the cold tier as shipped: no
    pack copy. VLLM_MOE_W2_BASE_RAM_GB pins the hottest experts, a global
    greedy prefix of VLLM_MOE_W2_BASE_RAM_RANK (tsv `layer rank expert
    count`), in compact rows (PinnedHostStore's EP layout: `_rowmaps`).
    Readers of a cold expert: rows_for (GPU cache fills, through a pinned
    stage; same contract as MmapPackStore) and the CPU experts
    (moe_w2_cpu: preads into its own bounce rows). The page cache sits
    under the pinned set as an opportunistic L3; VLLM_MOE_W2_COLD_DROP=1
    drops each cold read from it (the no-spare-RAM worst case, for A/B on
    a big-RAM box)."""

    _SECTIONS = 4

    def __init__(self, n_layers: int, n_experts: int, slot_bytes: int,
                 ram_gb: float, rank_path: str):
        self.slot_bytes = slot_bytes
        self.E = n_experts
        self.n_hot = max(0, int(ram_gb * 2**30 // slot_bytes))
        self.drop = os.getenv("VLLM_MOE_W2_COLD_DROP", "0") == "1"
        self._hot = self._plan(n_layers, n_experts, rank_path)
        self._layers: dict[int, torch.Tensor] = {}
        self._rowmaps: dict[int, list[int]] = {}
        self._rowmap_np: dict = {}
        self._fds: dict[int, list[int]] = {}
        self._secs: list[int] | None = None
        self._bufs: list = []
        self._stage = torch.empty(0, slot_bytes, dtype=torch.uint8)
        self._pool = ThreadPoolExecutor(
            max_workers=int(os.getenv("VLLM_MOE_W2_COLD_THREADS", "8")),
            thread_name_prefix="moe_w2_cold")
        self.last_cold = 0
        self._reads = self._read_bytes = 0
        self._read_s = 0.0

    def _plan(self, n_layers: int, n_experts: int,
              rank_path: str) -> dict[int, set[int]]:
        """Per-layer hot sets: the global top n_hot (layer, expert) pairs by
        count; unlisted experts count 0. Every layer keeps at least one."""
        cnt = {}
        with open(rank_path) as f:
            for ln in f:
                if ln.startswith("#") or not ln.strip():
                    continue
                li, _rk, ei, c = ln.split()[:4]
                cnt[(int(li), int(ei))] = int(c)
        from vllm.model_executor.layers.quantization.utils import moe_w2_ep
        own = [ei for ei in range(n_experts) if moe_w2_ep.local(ei)]
        pairs = sorted(((cnt.get((li, ei), 0), li, ei)
                        for li in range(n_layers) for ei in own),
                       key=lambda t: (-t[0], t[1], t[2]))
        ranked = {li for li, _ei in cnt}
        hot: dict[int, set[int]] = {li: set() for li in range(n_layers)}
        frac = min(1.0, self.n_hot / max(len(ranked) * len(own), 1))
        for li in range(n_layers):
            if li not in ranked:
                # unranked (the MTP layer): the global hot fraction by id
                hot[li] = set(own[:max(1, round(frac * len(own)))])
        budget = self.n_hot - sum(len(h) for h in hot.values())
        for _c, li, ei in pairs:
            if budget <= 0:
                break
            if li in ranked:
                hot[li].add(ei)
                budget -= 1
        for li in range(n_layers):
            if not hot[li]:
                hot[li].add(max(own, key=lambda e: cnt.get((li, e), 0)))
        return hot

    def __contains__(self, layer_key: int) -> bool:
        return layer_key in self._layers

    def __len__(self) -> int:
        return len(self._layers)

    def add_layer(self, layer_key: int, parts, owned=None, n_experts=None,
                  files=None, plan_key=None) -> None:
        """`parts`: the four sections [E, len] (CPU, may be file-backed);
        `files`: their four paths (same order); `plan_key`: the ranking's
        layer index (default layer_key)."""
        if files is None or len(files) != self._SECTIONS or owned is not None:
            raise RuntimeError(
                "VLLM_MOE_W2_BASE_FILES: the base layer needs its four "
                "planes-cache files (a planes-cache hit; no EP)")
        secs = [int(t.shape[1]) for t in parts]
        assert sum(secs) == self.slot_bytes, (secs, self.slot_bytes)
        for path, n in zip(files, secs):
            if os.path.getsize(path) != self.E * n:
                raise RuntimeError(f"{path}: size {os.path.getsize(path)} "
                                   f"!= {self.E} x {n}")
        self._secs = secs
        hot = sorted(self._hot.get(layer_key if plan_key is None
                                   else plan_key, ()))
        idx = torch.tensor(hot, dtype=torch.long)
        host = pinned_empty(len(hot), self.slot_bytes, True, self._bufs)
        off = 0
        for t in parts:
            host[:, off:off + t.shape[1]].copy_(t.index_select(0, idx))
            off += t.shape[1]
        rmap = [-1] * self.E
        for row, ei in enumerate(hot):
            rmap[ei] = row
        import numpy as np
        self._layers[layer_key] = host
        self._rowmaps[layer_key] = rmap
        self._rowmap_np[layer_key] = np.asarray(rmap, dtype=np.int32)
        self._fds[layer_key] = [os.open(p, os.O_RDONLY) for p in files]

    def cpu_layout(self, layer_key: int):
        """(compact host base pointer, int32 [E] rowmap, four section fds)
        for the CPU experts' cold reads."""
        return (self._layers[layer_key].data_ptr(),
                self._rowmap_np[layer_key], self._fds[layer_key])

    def hot_rows(self) -> int:
        return sum(t.shape[0] for t in self._layers.values())

    def rows_for(self, pairs, scan: bool = False) -> list[torch.Tensor]:
        """Hot experts: zero-copy pinned views. Cold ones: preads of the
        four sections into the pinned stage, which the returned views alias
        until the next call (`last_cold` > 0: the caller must finish its
        H2D copies before then)."""
        cold = []
        for i, (li, ei) in enumerate(pairs):
            if self._rowmaps[li][ei] < 0:
                cold.append(i)
        self.last_cold = len(cold)
        if cold and self._stage.shape[0] < len(cold):
            self._stage = torch.empty(max(len(cold), 2 * self._stage.shape[0]),
                                      self.slot_bytes, dtype=torch.uint8,
                                      pin_memory=torch.cuda.is_available())
        t0 = time.perf_counter()
        stage_mv = memoryview(self._stage.numpy()).cast("B") if cold else None
        secs = self._secs or []

        def _read(job):
            j, (li, ei) = job
            base = j * self.slot_bytes
            for fd, n in zip(self._fds[li], secs):
                done = 0
                while done < n:
                    got = os.preadv(fd, [stage_mv[base + done:base + n]],
                                    ei * n + done)
                    if got <= 0:
                        raise IOError(f"moe_w2 planes file short read "
                                      f"(layer {li} expert {ei})")
                    done += got
                if self.drop:
                    os.posix_fadvise(fd, ei * n, n, os.POSIX_FADV_DONTNEED)
                base += n

        jobs = [(j, pairs[i]) for j, i in enumerate(cold)]
        for f in [self._pool.submit(_read, jb) for jb in jobs]:
            f.result()
        if cold:
            self._reads += len(cold)
            self._read_bytes += len(cold) * self.slot_bytes
            self._read_s += time.perf_counter() - t0
        out, j = [], 0
        for li, ei in pairs:
            r = self._rowmaps[li][ei]
            if r >= 0:
                out.append(self._layers[li][r])
            else:
                out.append(self._stage[j])
                j += 1
        return out

    def release(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        for fds in self._fds.values():
            for fd in fds:
                with contextlib.suppress(OSError):
                    os.close(fd)
        self._fds = {}
        self._layers = {}
        self._rowmaps = {}
        for b in self._bufs:
            b.release()
        self._bufs = []

    def stats(self) -> dict:
        return dict(reads=self._reads, read_bytes=self._read_bytes,
                    read_s=self._read_s)


def pack_has_layer(tag: str, layer_key: int, n_layers: int, n_experts: int,
                   slot_bytes: int) -> bool:
    """Fail-closed loader-skip probe for one durable pack generation.

    A sidecar is not proof that its bytes still exist.  Validate the complete
    identity/geometry contract and the exact pack extent before checkpoint
    parameters may be replaced by stubs.
    """
    dir_ = os.getenv("VLLM_MOE_W2_STORE_DIR", "").strip()
    if not dir_:
        return False
    try:
        stride = (slot_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
        pack, sidecar, _lock_path, identity = _pack_paths(
            dir_, tag, n_layers, n_experts, slot_bytes, stride)
        # An unresolved model id is intentionally process-local and may serve
        # the current boot, but can never authorize checkpoint loader skip.
        if identity["ckpt_id"] is None:
            return False
        expected = _pack_meta(
            tag, n_layers, n_experts, slot_bytes, stride, identity)
        lock_fd = os.open(
            _lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            with _exclusive_lock(lock_fd):
                with open(sidecar) as f:
                    meta = json.load(f)
                layers = _valid_layers(meta, n_layers)
                if not _meta_matches(meta, expected) or layers is None:
                    return False
                fd = os.open(pack, os.O_RDONLY)
                try:
                    expected_size = n_layers * n_experts * stride
                    if os.fstat(fd).st_size != expected_size:
                        return False
                finally:
                    os.close(fd)
        finally:
            os.close(lock_fd)
        if int(layer_key) not in set(layers):
            return False
        return True
    except Exception:  # noqa: BLE001 - probe only, staging path still works
        return False


def make_store(tag: str, n_layers: int, n_experts: int, slot_bytes: int,
               pinned: bool):
    """Store factory: pack-file backends when VLLM_MOE_W2_STORE_DIR is set
    (plus a pinned arena for the BASE tier when VLLM_MOE_W2_BASE_RAM_GB
    is set, or for the FP4 tier when VLLM_MOE_W2_FP4_RAM_GB is set), else
    the classic pinned/pageable host store. Env read at call
    time so tests can toggle backends without reimporting the module."""
    if os.getenv("VLLM_MOE_W2_BASE_NVME_RATIO", "").strip():
        logger.error(
            "VLLM_MOE_W2_BASE_NVME_RATIO (the RAM:NVMe interleaved-split "
            "experiment, moe_w2_nvme) is superseded by the pack store and "
            "IGNORED. Equivalent config: VLLM_MOE_W2_STORE_DIR=<dir> + "
            "VLLM_MOE_W2_BASE_RAM_GB=<pinned GiB> (the arena fraction is "
            "the RAM share; it also persists quantization across boots).")
    if tag == "base" and os.getenv("VLLM_MOE_W2_BASE_FILES", "0") == "1":
        ram_raw = os.getenv("VLLM_MOE_W2_BASE_RAM_GB", "").strip()
        rank = os.getenv("VLLM_MOE_W2_BASE_RAM_RANK", "").strip()
        if not ram_raw or ram_raw.lower() == "auto" or not rank:
            # explicit only: inside a memory cgroup the engine still sees
            # the machine's RAM, so no size is derived from it
            raise RuntimeError(
                "VLLM_MOE_W2_BASE_FILES needs VLLM_MOE_W2_BASE_RAM_GB=<GiB> "
                "(explicit) and VLLM_MOE_W2_BASE_RAM_RANK=<ranking tsv>")
        store = PlanesFileStore(n_layers, n_experts, slot_bytes,
                                float(ram_raw), rank)
        logger.info(
            "moe_w2 store[base]: PLANES-FILE backend: %d of %d experts "
            "pinned (%.1f GiB), the rest read from the planes files%s",
            min(store.n_hot, n_layers * n_experts), n_layers * n_experts,
            min(store.n_hot, n_layers * n_experts) * slot_bytes / 2**30,
            " (page cache dropped after each read)" if store.drop else "")
        return store
    dir_ = os.getenv("VLLM_MOE_W2_STORE_DIR", "").strip()
    if not dir_:
        return PinnedHostStore(slot_bytes, pinned=pinned)
    # Which tiers take the pack-file backend (default: every tier, the
    # historical behaviour). A pack persists the tier's whole store on
    # disk — for the FP4/Δ need-pool that is the full expert set (GLM-5.2
    # TP4: 4 x 226 GB written in 8 minutes, 2026-09-05, filling the volume
    # that also holds the docker root). VLLM_MOE_W2_STORE_TAGS=base keeps
    # the base pack and leaves the need-pool store in pinned host RAM.
    pack_tags = {t.strip() for t in os.getenv(
        "VLLM_MOE_W2_STORE_TAGS", "base,fp4").split(",") if t.strip()}
    if tag not in pack_tags:
        logger.info(
            "moe_w2 store[%s]: %s host RAM (VLLM_MOE_W2_STORE_TAGS=%s "
            "keeps this tier off the pack dir %s)", tag,
            "pinned" if pinned else "pageable",
            ",".join(sorted(pack_tags)), dir_)
        return PinnedHostStore(slot_bytes, pinned=pinned)
    # Pinned MRU arena over the pack: VLLM_MOE_W2_BASE_RAM_GB for the base
    # tier, VLLM_MOE_W2_FP4_RAM_GB for the FP4 need-pool (low-RAM profile
    # R1: 2-bit base pinned in RAM, FP4 store on NVMe behind a small arena).
    ram_env = ("VLLM_MOE_W2_BASE_RAM_GB" if tag == "base"
               else "VLLM_MOE_W2_FP4_RAM_GB")
    ram_raw = os.getenv(ram_env, "").strip().lower()
    if ram_raw not in ("", "0", "0.0"):
        stride = (slot_bytes + _ALIGN - 1) // _ALIGN * _ALIGN
        pack_gib = n_layers * n_experts * stride / 2**30
        ram_gb = 0.25 * pack_gib if ram_raw == "auto" else float(ram_raw)
        store = TieredPackStore(dir_, tag, n_layers, n_experts, slot_bytes,
                                ram_gb)
        logger.info(
            "moe_w2 store[%s]: TIERED backend %s — pinned arena %.1f GiB "
            "(%d slots, %.0f%% of the %.1f GiB pack) + %s NVMe misses",
            tag, store.path, store.n_arena * store.stride / 2**30,
            store.n_arena, 100.0 * store.n_arena / (n_layers * n_experts),
            pack_gib, "O_DIRECT" if store.direct else "buffered")
        if store.n_arena < 2 * n_experts:
            logger.warning(
                "moe_w2 store[%s]: arena of %d slots is smaller than one "
                "prefill layer's worst case (2*E=%d) — expect stage "
                "overflows; raise VLLM_MOE_W2_BASE_RAM_GB",
                tag, store.n_arena, 2 * n_experts)
        return store
    store = MmapPackStore(dir_, tag, n_layers, n_experts, slot_bytes)
    logger.info(
        "moe_w2 store[%s]: PACK-FILE backend %s (slot %.2f MiB, "
        "stride %d, %d layers x %d experts; host RAM tier = page cache)",
        tag, store.path, slot_bytes / 2**20, store.stride, n_layers,
        n_experts)
    return store
