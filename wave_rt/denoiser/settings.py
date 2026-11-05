"""Environment-controlled Wave scheduling and diagnostic settings."""
import os
import torch

DEBUG = os.environ.get("WAVE_DEBUG", "") not in ("", "0", "false")

DEBUG_DIR = os.environ.get("WAVE_DEBUG_DIR", "/tmp/wrt_dbg")

TICK_PROFILE = os.environ.get("WAVE_TICKPROF", "") not in ("", "0", "false")

TIMELINE = os.environ.get("WAVE_TIMELINE", "") not in ("", "0", "false")

TIMELINE_DIR = os.environ.get("WAVE_TIMELINE_DIR", "/tmp/wrt_timeline")

TIMELINE_PROFILE = os.environ.get("WAVE_TL_PROF", "") not in ("", "0", "false")

PREFETCH = os.environ.get("WAVE_PREFETCH", "1") not in ("", "0", "false")

DISABLE_ALL_GATHER = os.environ.get("WAVE_NOAG", "") not in ("", "0", "false")

FP8_KV = os.environ.get("WAVE_FP8_KV", "") not in ("", "0", "false")

FP8_KV_ONESIDED = os.environ.get("WAVE_FP8KV_ONESIDED", "") not in ("", "0", "false")

FP8_E4M3_MAX = 448.0

FP8_DTYPE = torch.float8_e4m3fn

BROADCAST_CHECK = os.environ.get("WAVE_BCAST_CHECK", "") not in ("", "0", "false")

BROADCAST_CHECK_DIR = os.environ.get("WAVE_BCAST_CHECK_DIR", "/tmp/wrt_bcast_check")

FULL_BROADCAST = os.environ.get("WAVE_FULL_BCAST", "") not in ("", "0", "false")

ORIGINAL_SEND_ORDER = os.environ.get("WAVE_ORIG_SEND_ORDER", "") not in ("", "0", "false")

FORCE_TICK_SYNC = os.environ.get("WAVE_FORCE_TICK_SYNC", "") not in ("", "0", "false")

EDGE_TIMING = os.environ.get("WAVE_EDGE_TS", "") not in ("", "0", "false")

EDGE_TIMING_DIR = os.environ.get("WAVE_EDGE_TS_DIR", "/tmp/wrt_edge_ts")

CAT_TIMING = os.environ.get("WAVE_CAT_TS", "") not in ("", "0", "false")

DISABLE_CAT = os.environ.get("WAVE_NO_CAT", "") not in ("", "0", "false")

ALL_GATHER_TIMING = os.environ.get("WAVE_AG_TS", "") not in ("", "0", "false")

MACHINE_PROFILE = os.environ.get("WAVE_MACHPROF", "") not in ("", "0", "false")

MACHINE_PROFILE_DIR = os.environ.get("WAVE_MACHPROF_DIR", "/tmp/wrt_machprof")

GENERATION_SLOTS = int(os.environ.get("WAVE_OS_GT", "2"))

LAYER_PROFILE = os.environ.get("WAVE_LAYER_PROF", "") not in ("", "0", "false")

LAYER_PROFILE_DIR = os.environ.get("WAVE_LAYER_PROF_DIR", "/tmp/wrt_layer_os")

PROFILE_TICKS = {
    int(x) for x in os.environ.get("WAVE_PROF_TICKS", "6,7").split(",") if x.strip()
}
