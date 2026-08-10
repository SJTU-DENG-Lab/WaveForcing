"""Unified experiment logging — writes JSON to outputs/ after each run."""
import json
import os
import time
from datetime import datetime
from pathlib import Path


def log_run(tag: str, args: dict, metrics: dict, outputs_dir: str = "outputs"):
    """Save a structured JSON log of one run.

    Args:
        tag: short identifier, e.g. "serve_nstep4_nf252"
        args: CLI args dict (nstep, steps, gen_ckpt, num_output_frames, ...)
        metrics: measured numbers (tick_ms_list, diffusion_wall_ms, vae_wall_ms, fps, ...)
        outputs_dir: directory to write into (default "outputs/")
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(outputs_dir) / f"{ts}_{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "timestamp": ts,
        "args": _sanitize(args),
        "metrics": _sanitize(metrics),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return str(path)


def _sanitize(d):
    """Make dict JSON-serializable (convert non-serializable types to str)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (int, float, str, bool, type(None))) else str(x) for x in v]
        elif isinstance(v, dict):
            out[k] = _sanitize(v)
        else:
            out[k] = str(v)
    return out
