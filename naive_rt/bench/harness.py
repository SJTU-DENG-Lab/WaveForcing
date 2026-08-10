"""Shared benchmark harness: structured metrics.json, video saving (imageio),
and latent PSNR — used by every experiment script (s0/s2/s3/serve/tp_naive).

Output layout (see plan):
    outputs/experiments/<task>/<run_tag>/{video.mp4, metrics.json, latents.pt?}
"""
import json
import math
import os
from datetime import datetime

import numpy as np
import torch

# Wan causal VAE temporal upsampling: 1 latent frame -> 4 pixel frames.
PIXEL_PER_LATENT = 4


def pixel_frames(num_latent_frames: int) -> int:
    return num_latent_frames * PIXEL_PER_LATENT


def run_dir(out_root: str, task: str, run_tag: str) -> str:
    d = os.path.join(out_root, task, run_tag)
    os.makedirs(d, exist_ok=True)
    return d


def save_video(video, path: str, fps: int = 16):
    """Save a video tensor to mp4 via imageio (torchvision.io.write_video was
    removed in torchvision 0.26).

    Accepts either [B,F,C,H,W] or [F,C,H,W] or [F,H,W,C] float in [0,1]
    (or uint8). Writes H.264.
    """
    import imageio

    v = video
    if isinstance(v, torch.Tensor):
        v = v.detach().float().cpu()
        if v.dim() == 5:
            v = v[0]
        if v.dim() != 4:
            raise ValueError(f"expected 4D/5D video, got {tuple(v.shape)}")
        # to [F,H,W,C]
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
    """PSNR/stats between two latent tensors (same formula as s1_wave_sim)."""
    a = a.float().cpu()
    b = b.float().cpu()
    if a.shape != b.shape:
        return {"psnr_db": None, "note": f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"}
    diff = (a - b).abs()
    mse = ((a - b) ** 2).mean().item()
    rng = (a.max() - a.min()).item() + 1e-8
    psnr = 20.0 * math.log10(rng / (mse ** 0.5 + 1e-12))
    return {
        "psnr_db": psnr,
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "mse": mse,
    }


def write_metrics(out_dir: str, *, task, run_tag, method, nstep, num_output_frames,
                  num_gpus, diffusion_s, vae_s=None, end_to_end_s=None,
                  tp_size=1, seed=1, per_tick_ms=None, per_window_ms=None,
                  psnr_db=None, speedup_vs_baseline=None, extra=None):
    """Write the per-run metrics.json (see plan schema). Times in seconds.

    fps_without_vae = diffusion-only throughput; fps_with_vae = end-to-end.
    """
    pf = pixel_frames(num_output_frames)
    if end_to_end_s is None:
        end_to_end_s = diffusion_s + (vae_s or 0.0)
    fps_without_vae = pf / diffusion_s if diffusion_s else None
    fps_with_vae = pf / end_to_end_s if end_to_end_s else None
    payload = {
        "task": task,
        "run_tag": run_tag,
        "method": method,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "nstep": nstep,
        "num_output_frames": num_output_frames,
        "pixel_frames": pf,
        "num_gpus": num_gpus,
        "tp_size": tp_size,
        "seed": seed,
        "diffusion_s": round(diffusion_s, 4) if diffusion_s is not None else None,
        "vae_s": round(vae_s, 4) if vae_s is not None else None,
        "end_to_end_s": round(end_to_end_s, 4) if end_to_end_s is not None else None,
        "fps_with_vae": round(fps_with_vae, 3) if fps_with_vae else None,
        "fps_without_vae": round(fps_without_vae, 3) if fps_without_vae else None,
        "per_tick_ms": per_tick_ms,
        "per_window_ms": per_window_ms,
        "psnr_db": round(psnr_db, 3) if isinstance(psnr_db, (int, float)) else psnr_db,
        "speedup_vs_baseline": round(speedup_vs_baseline, 3) if speedup_vs_baseline else None,
    }
    if extra:
        payload.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def finalize_wave_run(out_dir, *, task, run_tag, method, nstep, num_output_frames,
                      num_gpus, diffusion_ms, vae_ms, e2e_ms, latents=None,
                      video=None, ref_latents="", baseline_e2e=0.0,
                      per_tick_ms=None, seed=1):
    """Save video + latents + metrics for a (multi-process) wave run.

    IMPORTANT: call this ONLY after all timing has been captured — video/latents
    I/O and PSNR must never fall inside a timed region.
    """
    os.makedirs(out_dir, exist_ok=True)
    if video is not None:
        try:
            save_video(video, os.path.join(out_dir, "video.mp4"))
        except Exception as e:  # never let saving break a completed run
            print(f"[harness] save_video failed: {e!r}")
    psnr = None
    if latents is not None:
        lat = latents.detach().cpu()
        torch.save(lat, os.path.join(out_dir, "latents.pt"))
        if ref_latents and os.path.isfile(ref_latents):
            psnr = latent_psnr(lat, torch.load(ref_latents, map_location="cpu")).get("psnr_db")
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

