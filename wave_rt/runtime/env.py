"""Environment parsing shared by WaveRT runtime modules."""

from __future__ import annotations

import os

_FALSE_VALUES = {"", "0", "false"}


def env_flag(name: str, *, default: bool = False) -> bool:
    """Read a boolean flag while preserving WaveRT's legacy parsing rules."""
    fallback = "1" if default else ""
    return os.environ.get(name, fallback) not in _FALSE_VALUES
