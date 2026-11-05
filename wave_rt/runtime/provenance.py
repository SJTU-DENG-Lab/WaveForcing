"""Small, non-secret runtime provenance for comparable performance results."""
import os
from pathlib import Path
import subprocess


ENVIRONMENT_FIELDS = ('WAVE_THREADS', 'WAVE_FORCE_TICK_SYNC', 'WAVE_DIST_MODE',
    'WAVE_MAX_ATTN_FRAMES', 'WAVE_OS_GT', 'WAVE_FP8_SCOPE', 'WAVE_FP8_COMPILE',
    'WAVE_FP8KV', 'WAVE_FP8KV_ONESIDED', 'WAVE_NO_CAT', 'WAVE_FULL_BCAST',
    'WAVE_PHYSICAL_SKEW_MS', 'WAVE_TIMELINE', 'WAVE_TICKPROF', 'WAVE_LAYER_PROF', 'WAVE_OUTPUT_TRACE',
    'SGLANG_USE_RUNAI_MODEL_STREAMER', 'TORCH_NCCL_ASYNC_ERROR_HANDLING',
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES',
    'CUDA_MODULE_LOADING', 'CUDA_MODULE_DATA_LOADING')


def runtime_provenance():
    root = Path(__file__).resolve().parents[2]
    def revision(path):
        try:
            return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return dict(git_revision=revision(root), sglang_revision=revision(root / 'sglang'),
                effective_environment={key: os.environ[key] for key in ENVIRONMENT_FIELDS if key in os.environ})
