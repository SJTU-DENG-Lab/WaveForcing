"""Opt-in per-rank resource measurements; no inherited environment is dumped."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import time

import torch
import torch.distributed as dist


def process_memory():
    # Linux ru_maxrss is KiB. Current RSS is separate from the lifetime peak.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    try:
        current = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        current = None
    return {"cpu_rss_bytes": current, "cpu_peak_rss_bytes": peak}


@contextmanager
def memory_scope(config, label, *, step=None, device=None):
    if not getattr(config, "profile_memory", False):
        yield
        return
    cuda = device is not None and torch.cuda.is_available()
    rank = dist.get_rank() if dist.is_initialized() else 0
    path = Path(config.logdir) / f"memory_rank{rank:02d}.jsonl"

    def record(event, elapsed=None, failed=False):
        payload = dict(event=event, section=label, rank=rank, step=step,
                       time_unix=time.time(), elapsed_seconds=elapsed, failed=failed,
                       **process_memory())
        if cuda:
            payload.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        with path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")

    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    record("begin")
    started = time.monotonic()
    failed = True
    try:
        yield
        if cuda:
            torch.cuda.synchronize(device)
        failed = False
    finally:
        record("end", time.monotonic() - started, failed)

