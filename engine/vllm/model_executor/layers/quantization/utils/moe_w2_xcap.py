# SPDX-License-Identifier: Apache-2.0
"""Env-gated MoE-input capture for Hessian building (offline EXL3 quant).

Purpose ([G] 2026-08-22): the DSpark draft head's MoE layers
(layers.{43..45}) only execute during DECODE drafting (prefill builds KV
and returns before the ffn), so their w1/w3 input distribution cannot be
captured by the corpus-prefill rig (fd/tools/rigs/capture_hidden.py).
This module captures it from a LIVE serving cell instead: hook the
routed-expert input in the quant method's apply(), accumulate
H = sum x x^T + fp16 samples per layer, and periodically write files in
exactly the format fd/tools/build_base_pack.py consumes
(H_layer{K}.pt: {"H": fp32 [D, D], "count": int}; x_layer{K}.pt: fp16
[N, D]).

Env:
  VLLM_MOE_W2_XCAP_DIR       enable + output dir (mount rw)
  VLLM_MOE_W2_XCAP_LAYERS    comma layer indices (default: 43,44,45)
  VLLM_MOE_W2_XCAP_SAMPLE_CAP  stored x rows per layer (default 24000)
  VLLM_MOE_W2_XCAP_SAVE_EVERY  save each N record calls (default 500)

Arming: records only while the marker file ``<DIR>/ARM`` exists — create
it AFTER startup so profiling/warmup dummy batches never pollute the
capture. No-ops during cudagraph capture (a D2H inside capture would
fault) and adds nothing to the step unless enabled.
"""
import os
import re
import threading

import torch

_DIR = os.environ.get("VLLM_MOE_W2_XCAP_DIR", "")
_LAYERS = ({int(x) for x in
            os.environ.get("VLLM_MOE_W2_XCAP_LAYERS", "43,44,45").split(",")
            if x.strip()}
           if _DIR else set())
_CAP = int(os.environ.get("VLLM_MOE_W2_XCAP_SAMPLE_CAP", "24000"))
_SAVE_EVERY = int(os.environ.get("VLLM_MOE_W2_XCAP_SAVE_EVERY", "500"))
_LOCK = threading.Lock()
_STATE: dict = {}
_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def enabled() -> bool:
    return bool(_DIR)


def _armed() -> bool:
    return os.path.exists(os.path.join(_DIR, "ARM"))


def record(layer_name: str, x: torch.Tensor) -> None:
    """Accumulate this call's routed-expert input rows for `layer_name`."""
    if not _DIR or torch.cuda.is_current_stream_capturing():
        return
    m = _IDX_RE.search(layer_name or "")
    if m is None or int(m.group(1)) not in _LAYERS:
        return
    if not _armed():
        return
    li = int(m.group(1))
    xf = x.detach().reshape(-1, x.shape[-1]).float()
    with _LOCK:
        st = _STATE.get(li)
        if st is None:
            st = _STATE[li] = dict(
                H=torch.zeros(xf.shape[-1], xf.shape[-1],
                              dtype=torch.float32, device=xf.device),
                count=0, xs=[], sampled=0, calls=0)
        st["H"] += xf.T @ xf
        st["count"] += xf.shape[0]
        if st["sampled"] < _CAP:
            st["xs"].append(xf.half().cpu())
            st["sampled"] += xf.shape[0]
        st["calls"] += 1
        if st["calls"] % _SAVE_EVERY == 0:
            _save(li, st)


def _save(li: int, st: dict) -> None:
    os.makedirs(_DIR, exist_ok=True)
    hp = os.path.join(_DIR, f"H_layer{li:02d}.pt")
    torch.save({"H": st["H"].cpu(), "count": st["count"]}, hp + ".tmp")
    os.replace(hp + ".tmp", hp)
    if st["xs"]:
        xp = os.path.join(_DIR, f"x_layer{li:02d}.pt")
        torch.save(torch.cat(st["xs"])[:_CAP], xp + ".tmp")
        os.replace(xp + ".tmp", xp)


def flush() -> None:
    """Write every layer's current accumulators (call-site optional)."""
    with _LOCK:
        for li, st in _STATE.items():
            _save(li, st)
