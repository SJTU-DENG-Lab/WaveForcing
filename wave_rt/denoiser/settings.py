"""Environment-controlled settings for the denoiser runtime."""

from __future__ import annotations

import os

import torch

from wave_rt.runtime.env import env_flag

DEBUG = env_flag("WAVE_DEBUG")
DEBUG_DIR = os.environ.get("WAVE_DEBUG_DIR", "/tmp/wrt_dbg")
PREFETCH = env_flag("WAVE_PREFETCH", default=True)
DISABLE_ALL_GATHER = env_flag("WAVE_NOAG")
FULL_BROADCAST = env_flag("WAVE_FULL_BCAST")
ORIGINAL_SEND_ORDER = env_flag("WAVE_ORIG_SEND_ORDER")
FORCE_TICK_SYNC = env_flag("WAVE_FORCE_TICK_SYNC")
DISABLE_CAT = env_flag("WAVE_NO_CAT")

FP8_KV = env_flag("WAVE_FP8_KV")
FP8_KV_ONESIDED = env_flag("WAVE_FP8KV_ONESIDED")
FP8_E4M3_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn
GENERATION_SLOTS = int(os.environ.get("WAVE_OS_GT", "2"))

TICK_PROFILE = env_flag("WAVE_TICKPROF")
TIMELINE = env_flag("WAVE_TIMELINE")
TIMELINE_DIR = os.environ.get("WAVE_TIMELINE_DIR", "/tmp/wrt_timeline")
TIMELINE_PROFILE = env_flag("WAVE_TL_PROF")
BROADCAST_CHECK = env_flag("WAVE_BCAST_CHECK")
BROADCAST_CHECK_DIR = os.environ.get("WAVE_BCAST_CHECK_DIR", "/tmp/wrt_bcast_check")
EDGE_TIMING = env_flag("WAVE_EDGE_TS")
EDGE_TIMING_DIR = os.environ.get("WAVE_EDGE_TS_DIR", "/tmp/wrt_edge_ts")
CAT_TIMING = env_flag("WAVE_CAT_TS")
ALL_GATHER_TIMING = env_flag("WAVE_AG_TS")
MACHINE_PROFILE = env_flag("WAVE_MACHPROF")
MACHINE_PROFILE_DIR = os.environ.get("WAVE_MACHPROF_DIR", "/tmp/wrt_machprof")
LAYER_PROFILE = env_flag("WAVE_LAYER_PROF")
LAYER_PROFILE_DIR = os.environ.get("WAVE_LAYER_PROF_DIR", "/tmp/wrt_layer_os")
PROFILE_TICKS = {
    int(value)
    for value in os.environ.get("WAVE_PROF_TICKS", "6,7").split(",")
    if value.strip()
}
