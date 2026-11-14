# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boot settings from a Tandem lite package (tools/e2e/tandem_package.py).

A package's config.json carries a "tandem" block naming its parts:

    "tandem": {"format": 1, "experts": "vq", "vq_layers": 40, "planes": "vq",
               "fp4_hot": "fp4hot", "ckpt_id": "v41-1017-g51", ...}

"format" is the package layout version (format 1: VQ planes with unpadded
scales, the FP4 hot-set store, e2m1 Engram files). The engine refuses a
format it does not know; a block without one is format 1. Packages are
re-uploaded only when the layout changes, not for every engine release.

apply_package_env(model) turns it into the moe_w2 environment before the
engine processes start (they inherit it), so a user passes only the model
path. Explicit env values always win (setdefault):

    VLLM_MOE_W2=1, VLLM_MOE_W2_PLANES_CACHE=<pkg>/<planes>,
    VLLM_MOE_W2_VQ=<pkg>/<planes>/vq-table.bin, VLLM_MOE_W2_CKPT_ID,
    VLLM_MOE_W2_FP4_HOT=<pkg>/<fp4_hot>, VLLM_MOE_W2_BACKEND=triton (VQ).

VLLM_TANDEM_FP4_TOP=K picks the K top-ranked experts of every layer from the
FP4 hot-set store as the static FP4 tier: VLLM_MOE_W2_DELTA_STATIC is the
shipped <fp4_hot>/top<K>.tsv, or written from the layer<L>.idx rank order
under ~/.cache/tandem when K has no shipped list, and VLLM_MOE_W2_DELTA_GB
defaults to the K * layers FP4 rows plus 0.5 GiB, rounded up to 0.5 GiB.
K=0 (default): no FP4 tier, every routed expert served from the VQ base.

VLLM_TANDEM_RAM_GB=N (small-RAM boxes): only the N GiB of most-routed VQ
experts are pinned in RAM (ranked by <fp4_hot>/ranking.tsv), every other
expert is read from the package's planes files when needed (CPU experts and
GPU cache fills; moe_w2_store.PlanesFileStore). Sets VLLM_MOE_W2_BASE_FILES,
VLLM_MOE_W2_BASE_RAM_GB, VLLM_MOE_W2_BASE_RAM_RANK (the CPU/PCIe split offers
the GPU's DIRECT reads pinned misses only; cold experts stay on the CPU) and
turns on VLLM_MOE_W2_COLD_HINT (moe_w2_looka: the next layer's
predicted cold experts are read ahead). Unset: all experts pinned.

Only local package directories are read (download the whole repo first: the
weights loader alone does not fetch vq/ or fp4hot/).
"""

import json
import math
import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_GIB = 1 << 30
FORMATS = (1,)   # package layouts this engine reads


def _read_block(model: str) -> dict | None:
    path = os.path.join(model, "config.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("tandem")


def _hot_layers(store: str) -> list[int]:
    return sorted(int(n[5:-4]) for n in os.listdir(store)
                  if n.startswith("layer") and n.endswith(".idx"))


def _static_list(store: str, k: int) -> str:
    shipped = os.path.join(store, f"top{k}.tsv")
    if os.path.isfile(shipped):
        return shipped
    import numpy as np
    out_dir = os.path.join(os.path.expanduser("~"), ".cache", "tandem")
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.dirname(os.path.abspath(store)))
    out = os.path.join(out_dir, f"{tag}-top{k}.tsv")
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        f.write(f"# top {k} experts per layer from {store}/layer<L>.idx (rank order)\n")
        for li in _hot_layers(store):
            ids = np.fromfile(os.path.join(store, f"layer{li}.idx"), dtype=np.int32)
            if k > len(ids):
                raise ValueError(f"VLLM_TANDEM_FP4_TOP={k}: the hot-set store holds "
                                 f"{len(ids)} experts of layer {li}")
            for e in sorted(ids[:k].tolist()):
                f.write(f"{li}\t{e}\n")
    os.replace(out + ".tmp", out)
    return out


def _fp4_gb(store: str, k: int) -> float:
    rows = 0
    for li in _hot_layers(store):
        n = os.path.getsize(os.path.join(store, f"layer{li}.idx")) // 4
        rows += k * os.path.getsize(os.path.join(store, f"layer{li}.fp4.bin")) // n
    return math.ceil((rows / _GIB + 0.5) * 2) / 2


def apply_package_env(model: str) -> bool:
    """Set the moe_w2 env of the Tandem package at `model`; False if it is none."""
    t = _read_block(model)
    if not t:
        return False
    fmt = int(t.get("format", 1))
    if fmt not in FORMATS:
        raise ValueError(f"Tandem package {model} has format {fmt}; this engine reads "
                         f"format {', '.join(map(str, FORMATS))}. Upgrade the engine.")
    env = os.environ
    env.setdefault("VLLM_MOE_W2", "1")
    planes = os.path.join(model, t.get("planes", "vq"))
    env.setdefault("VLLM_MOE_W2_PLANES_CACHE", planes)
    if t.get("experts") == "vq":
        env.setdefault("VLLM_MOE_W2_VQ", os.path.join(planes, "vq-table.bin"))
        env.setdefault("VLLM_MOE_W2_BACKEND", "triton")
    if t.get("ckpt_id"):
        env.setdefault("VLLM_MOE_W2_CKPT_ID", t["ckpt_id"])
    # the GPU cache's warm-start file (pool-heat) is per user, not shipped
    env.setdefault("VLLM_MOE_W2_POOL_HEAT_DIR", os.path.join(
        os.path.expanduser("~"), ".cache", "tandem",
        os.path.basename(os.path.abspath(model))))
    k = int(env.get("VLLM_TANDEM_FP4_TOP", "0") or 0)
    if t.get("fp4_hot"):
        store = os.path.join(model, t["fp4_hot"])
        env.setdefault("VLLM_MOE_W2_FP4_HOT", store)
        if k > 0:
            env.setdefault("VLLM_MOE_W2_DELTA_STATIC", _static_list(store, k))
            env.setdefault("VLLM_MOE_W2_DELTA_GB", f"{_fp4_gb(store, k):g}")
    elif k > 0:
        raise ValueError("VLLM_TANDEM_FP4_TOP: this package has no FP4 hot-set store")
    if k == 0:
        env.setdefault("VLLM_MOE_W2_DELTA_GB", "0")
    if t.get("fp4_hot"):
        # the package ranking: the RAM tier's pin order, and also the helper
        # GPU's start set and the tensor-parallel fallback expert (needed
        # with all experts pinned too, i.e. without RAM_GB)
        env.setdefault("VLLM_MOE_W2_BASE_RAM_RANK",
                       os.path.join(model, t["fp4_hot"], "ranking.tsv"))
    ram = env.get("VLLM_TANDEM_RAM_GB", "").strip()
    if ram:
        if not t.get("fp4_hot"):
            raise ValueError("VLLM_TANDEM_RAM_GB: the package has no ranking "
                             "(<fp4_hot>/ranking.tsv)")
        env.setdefault("VLLM_MOE_W2_BASE_FILES", "1")
        env.setdefault("VLLM_MOE_W2_BASE_RAM_GB", ram)
        # pinned misses: a share over PCIe (DIRECT), the rest on the CPU. Both
        # read the same DRAM, so past ~0.3 the link starves the CPU (PC8:
        # 14600KF DDR4 + 5090: 0.3 +12%, 0.5+ slower; EX1 epyc: 0.5 +19% c1)
        env.setdefault("VLLM_MOE_W2_CPU_PCIE_FRAC", "0.3")
        # router-lookahead read-ahead of the next layer's cold experts
        env.setdefault("VLLM_MOE_W2_COLD_HINT", "1")
    logger.info("Tandem package %s: %s", model, ", ".join(
        f"{n}={env[n]}" for n in ("VLLM_MOE_W2_PLANES_CACHE", "VLLM_MOE_W2_VQ",
                                  "VLLM_MOE_W2_CKPT_ID", "VLLM_MOE_W2_FP4_HOT",
                                  "VLLM_MOE_W2_DELTA_STATIC", "VLLM_MOE_W2_DELTA_GB",
                                  "VLLM_MOE_W2_BASE_RAM_GB")
        if env.get(n)))
    return True
