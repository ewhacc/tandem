# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tandem panel snapshot: with VLLM_TANDEM_STATS=<path> the engine process
rewrites <path> (atomically) about once a second with cumulative counters;
the API server's /tandem/stats serves it (entrypoints/serve/middleware/
tandem_panel.py), the page turns them into rates. Counters only.

  routes: routed expert computations by source, cumulative. gpu = served from
          the GPU cache (base tier, token-weighted, sampled by its manager
          thread); ram / nvme = CPU-engine pairs on pinned / cold rows.
  cpu:    CPU-engine jobs and times, cold reads, read-ahead hints.
  engram: Engram-from-files requests.
"""
import json
import os
import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

_started = False


def _snapshot(btier, cfg: dict) -> dict:
    from vllm.model_executor.layers.quantization.utils import moe_w2_cpu
    from vllm.models.deepseek_v41.common import engram_files
    d: dict = {"t": time.time(), "cfg": cfg}
    s = moe_w2_cpu.raw_stats()
    if s is not None:
        cold_pairs = s[32]
        d["cpu"] = {"jobs": s[0], "pairs": s[1], "busy_ns": s[2],
                    "cold_experts": s[8], "cold_ns": s[9], "io_err": s[10],
                    "blocked_ns": s[11], "hints": s[12], "pf_drop": s[13],
                    "hints_used": s[14]}
        d["routes"] = {"gpu": float(getattr(btier, "_m_hits", 0.0)),
                       "ram": s[1] - cold_pairs, "nvme": cold_pairs}
    e = engram_files.raw_stats()
    if e is not None:
        d["engram"] = {"req": e[0], "rows": e[1], "busy_ns": e[2]}
    return d


def start(btier, cfg: dict) -> None:
    """Start the writer thread once (no-op without VLLM_TANDEM_STATS)."""
    global _started
    path = os.getenv("VLLM_TANDEM_STATS", "")
    if not path or _started:
        return
    _started = True
    tmp = f"{path}.{os.getpid()}.tmp"

    def loop() -> None:
        while True:
            try:
                with open(tmp, "w") as f:
                    json.dump(_snapshot(btier, cfg), f)
                os.replace(tmp, path)
            except Exception as e:  # noqa: BLE001 - panel only, never kill serving
                logger.warning("Tandem stats writer stopped: %s", e)
                return
            time.sleep(1.0)

    threading.Thread(target=loop, name="tandem-stats", daemon=True).start()
    logger.info("Tandem stats snapshot -> %s (every 1 s)", path)
