"""Small helpers shared across denoiser components."""

from __future__ import annotations

import os

import torch

from wave_rt.denoiser import settings


def record_cuda_event() -> torch.cuda.Event:
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def debug_log(rank: int, message: str) -> None:
    if not settings.DEBUG:
        return
    os.makedirs(settings.DEBUG_DIR, exist_ok=True)
    path = os.path.join(settings.DEBUG_DIR, f"r{rank}.log")
    with open(path, "a") as output:
        output.write(message + "\n")
