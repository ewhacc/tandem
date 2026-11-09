# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tandem status panel: GET /tandem/ (one static page, no CDN) and
GET /tandem/stats (JSON). Counters only, never prompts.

Load with `--middleware vllm.entrypoints.serve.middleware.tandem_panel.TandemPanelMiddleware`
(tools/tandem_serve.sh does this for PANEL=1). /tandem/stats merges

  - the engine's snapshot file (VLLM_TANDEM_STATS, written about once a
    second by the engine process: where the routed experts were computed,
    CPU-expert job times, cold reads, read-ahead hints, Engram, gate);
  - host counters read here: CPU busy, RAM / page cache, disk bytes, and per
    GPU memory / utilization / power / temperature (NVML).

The page polls /tandem/stats and /metrics and keeps 10 minutes of history in
the browser. With VLLM_API_KEY set, /tandem/stats needs the key (Bearer); the
page takes it from its URL fragment (/tandem/#key=...), which never reaches
the server logs. Put ALLOW= in front for anything beyond localhost.
"""
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

from starlette.datastructures import Headers
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm.logger import init_logger

logger = init_logger(__name__)

_PAGE = Path(__file__).with_name("tandem_panel.html")


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _cpu() -> dict:
    # aggregate jiffies; the page turns two samples into busy %
    f = _read("/proc/stat").split("\n", 1)[0].split()[1:]
    v = [int(x) for x in f[:8]] if f else [0] * 8
    return {"busy": sum(v) - v[3] - v[4], "total": sum(v),
            "n": os.cpu_count() or 1}


def _mem() -> dict:
    m = {}
    for line in _read("/proc/meminfo").splitlines():
        k, _, rest = line.partition(":")
        if k in ("MemTotal", "MemAvailable", "Cached", "Mlocked", "Unevictable"):
            m[k] = int(rest.split()[0]) * 1024
    return m


def _disk() -> dict:
    # whole NVMe / SATA devices only (no partitions): bytes read / written
    out = {}
    for line in _read("/proc/diskstats").splitlines():
        f = line.split()
        if len(f) < 10:
            continue
        name = f[2]
        if (name.startswith("nvme") and "p" not in name[4:]) or (
                name.startswith("sd") and name[-1].isalpha()):
            out[name] = {"r": int(f[5]) * 512, "w": int(f[9]) * 512}
    return out


class _Gpus:
    def __init__(self) -> None:
        self.nv = None
        try:
            from vllm.third_party import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = [pynvml.nvmlDeviceGetHandleByIndex(i)
                      for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception as e:  # noqa: BLE001 - panel only
            logger.info("Tandem panel: no NVML (%s)", e)

    def read(self) -> list:
        if self.nv is None:
            return []
        nv, out = self.nv, []
        for i, h in enumerate(self.h):
            try:
                m = nv.nvmlDeviceGetMemoryInfo(h)
                u = nv.nvmlDeviceGetUtilizationRates(h)
                out.append({
                    "i": i, "name": nv.nvmlDeviceGetName(h),
                    "mem_used": m.used, "mem_total": m.total, "util": u.gpu,
                    "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
                    "temp_c": nv.nvmlDeviceGetTemperature(h, 0)})
            except Exception:  # noqa: BLE001 - one bad query must not 500
                out.append({"i": i})
        return out


class TandemPanelMiddleware:
    """Pure ASGI: answers /tandem/ and /tandem/stats, passes the rest on."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.stats_path = os.environ.get("VLLM_TANDEM_STATS", "")
        key = os.environ.get("VLLM_API_KEY", "")
        self.key = hashlib.sha256(key.encode()).digest() if key else None
        self.gpus = _Gpus()
        logger.info("Tandem panel at /tandem/ (engine stats: %s%s)",
                    self.stats_path or "none",
                    ", API key required" if self.key else "")

    def _authorized(self, scope: Scope) -> bool:
        if self.key is None:
            return True
        scheme, _, tok = Headers(scope=scope).get("Authorization", "").partition(" ")
        return scheme.lower() == "bearer" and secrets.compare_digest(
            hashlib.sha256(tok.encode()).digest(), self.key)

    def stats(self) -> dict:
        eng = None
        if self.stats_path:
            try:
                eng = json.loads(_read(self.stats_path) or "null")
            except ValueError:  # caught mid-write: the page keeps the last one
                eng = None
        return {"t": time.time(), "engine": eng, "cpu": _cpu(), "mem": _mem(),
                "disk": _disk(), "gpus": self.gpus.read()}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("method") in ("GET", "HEAD"):
            path = scope["path"].removeprefix(scope.get("root_path", ""))
            if path == "/tandem":   # the page's relative fetches need the slash
                await RedirectResponse(scope.get("root_path", "") + "/tandem/")(
                    scope, receive, send)
                return
            if path == "/tandem/":
                await HTMLResponse(_read(str(_PAGE)) or "panel page missing",
                                   headers={"Cache-Control": "no-store"})(
                    scope, receive, send)
                return
            if path == "/tandem/stats":
                if not self._authorized(scope):
                    await JSONResponse({"error": "Unauthorized"}, 401)(
                        scope, receive, send)
                    return
                await JSONResponse(self.stats(),
                                   headers={"Cache-Control": "no-store"})(
                    scope, receive, send)
                return
        await self.app(scope, receive, send)
