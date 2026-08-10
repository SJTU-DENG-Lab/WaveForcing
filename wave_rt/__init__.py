"""WaveRT -- a standalone wavefront runtime for Rolling Forcing long-video
generation that reuses the sglang (sgl-dlab) diffusion backend as a library.

Design (see /root/.claude/plans/ticklish-toasting-steele.md):
  * launcher : multi-process / NCCL bring-up (owns a world of size ``wp_size``)
  * backend  : the ONLY layer that imports sglang internals; monkeypatches
               ``CausalWanSelfAttention.forward`` to inject a per-layer KV
               exchange hook, and wraps build_pipeline / text encode / decode.
  * wavefront: the systolic diffusion loop (ported from naive_rt s2).
  * vae/bench: serial VAE decode + metrics/video output.

WaveRT does NOT modify sglang source; all wave-specific behaviour is a runtime
monkeypatch installed by ``backend``.
"""

__all__ = ["WaveConfig"]

from wave_rt.config import WaveConfig
