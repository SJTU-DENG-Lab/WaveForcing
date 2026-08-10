"""WaveRT bench: structured metrics.json + video (imageio) + latent PSNR.

Mirrors naive_rt/bench/harness.py so WaveRT results are directly comparable
with the naive_rt s0/s2/s3 experiments. All I/O here must run AFTER timing.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime

import numpy as np
import torch

PIXEL_PER_LATENT = 4  # Wan causal VAE temporal upsampling


def run_dir(out_root: str, task: str, run_tag: str) -> str:
    d = os.path.join(out_root, task, run_tag)
    os.makedirs(d, exist_ok=True)
    return d


def save_video(video, path: str, fps: int = 16) -> str:
    """Save [B,F,C,H,W] / [F,C,H,W] / [F,H,W,C] (float [0,1] or uint8) as mp4."""
    import imageio

    v = video
    if isinstance(v, torch.Tensor):
        v = v.detach().float().cpu()
        if v.dim() == 5:
            v = v[0]
        if v.dim() != 4:
            raise ValueError(f"expected 4D/5D video, got {tuple(v.shape)}")
        if v.shape[1] in (1, 3) and v.shape[-1] not in (1, 3):
            v = v.permute(0, 2, 3, 1)
        if v.max() <= 1.5:
            v = v * 255.0
        v = v.clamp(0, 255).to(torch.uint8).numpy()
    else:
        v = np.asarray(v)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimsave(path, list(v), fps=fps, codec="libx264",
                    macro_block_size=None, quality=8)
    return path


def latent_psnr(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float().cpu()
    b = b.float().cpu()
    if a.shape != b.shape and a.dim() == 5 and b.dim() == 5:
        # Tolerate (B,C,T,H,W) vs (B,T,C,H,W) ordering: naive_rt's
        # return_latents comes out as (B,T,C,H,W), wave_rt stores (B,C,T,H,W).
        a5, b5 = a.shape, b.shape
        if a5[0] == b5[0] and a5[1] == b5[2] and a5[2] == b5[1] \
                and a5[3] == b5[3] and a5[4] == b5[4]:
            a = a.permute(0, 2, 1, 3, 4)
        elif b5[0] == a5[0] and b5[1] == a5[2] and b5[2] == a5[1] \
                and b5[3] == a5[3] and b5[4] == a5[4]:
            b = b.permute(0, 2, 1, 3, 4)
    if a.shape != b.shape:
        return {"psnr_db": None,
                "note": f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"}
    diff = (a - b).abs()
    mse = ((a - b) ** 2).mean().item()
    rng = (a.max() - a.min()).item() + 1e-8
    psnr = 20.0 * math.log10(rng / (mse ** 0.5 + 1e-12))
    return {"psnr_db": psnr, "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(), "mse": mse}


def write_metrics(out_dir: str, *, task, run_tag, method, nstep,
                  num_output_frames, num_gpus, diffusion_s, vae_s=None,
                  end_to_end_s=None, seed=0, per_tick_ms=None,
                  psnr_db=None, speedup_vs_baseline=None, extra=None) -> str:
    pf = num_output_frames * PIXEL_PER_LATENT
    if end_to_end_s is None:
        end_to_end_s = (diffusion_s or 0.0) + (vae_s or 0.0)
    fps_without_vae = pf / diffusion_s if diffusion_s else None
    fps_with_vae = pf / end_to_end_s if end_to_end_s else None
    payload = {
        "task": task, "run_tag": run_tag, "method": method,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "nstep": nstep, "num_output_frames": num_output_frames,
        "pixel_frames": pf, "num_gpus": num_gpus, "seed": seed,
        "diffusion_s": round(diffusion_s, 4) if diffusion_s is not None else None,
        "vae_s": round(vae_s, 4) if vae_s is not None else None,
        "end_to_end_s": round(end_to_end_s, 4) if end_to_end_s is not None else None,
        "fps_with_vae": round(fps_with_vae, 3) if fps_with_vae else None,
        "fps_without_vae": round(fps_without_vae, 3) if fps_without_vae else None,
        "per_tick_ms": per_tick_ms,
        "psnr_db": round(psnr_db, 3) if isinstance(psnr_db, (int, float)) else psnr_db,
        "speedup_vs_baseline": round(speedup_vs_baseline, 3)
        if speedup_vs_baseline else None,
    }
    if extra:
        payload.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def finalize_run(out_dir, *, task, run_tag, method, nstep, num_output_frames,
                 num_gpus, diffusion_ms, vae_ms, e2e_ms, latents=None,
                 video=None, ref_latents="", baseline_e2e=0.0,
                 per_tick_ms=None, seed=0) -> float | None:
    os.makedirs(out_dir, exist_ok=True)
    if video is not None:
        try:
            save_video(video, os.path.join(out_dir, "video.mp4"))
        except Exception as e:
            print(f"[bench] save_video failed: {e!r}", flush=True)
    psnr = None
    if latents is not None:
        lat = latents.detach().cpu()
        torch.save(lat, os.path.join(out_dir, "latents.pt"))
        if ref_latents and os.path.isfile(ref_latents):
            psnr = latent_psnr(
                lat, torch.load(ref_latents, map_location="cpu")
            ).get("psnr_db")
    e2e_s = (e2e_ms / 1000.0) if e2e_ms else None
    speedup = (baseline_e2e / e2e_s) if (baseline_e2e and e2e_s) else None
    write_metrics(
        out_dir, task=task, run_tag=run_tag, method=method, nstep=nstep,
        num_output_frames=num_output_frames, num_gpus=num_gpus, seed=seed,
        diffusion_s=(diffusion_ms / 1000.0) if diffusion_ms else None,
        vae_s=(vae_ms / 1000.0) if vae_ms else None,
        end_to_end_s=e2e_s, per_tick_ms=per_tick_ms,
        psnr_db=psnr, speedup_vs_baseline=speedup,
    )
    return psnr
