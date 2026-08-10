"""
s0_naive_baseline.py -- Milestone 0: Single-GPU naive rolling forcing baseline.

Wraps CausalInferencePipeline to run inference end-to-end on a single GPU
and reports timing. This is the unoptimized reference that all later
milestones measure against.

Expected numbers (252 latent frames, H200):
  - Diffusion: ~55 s
  - VAE decode: ~19 s
  - Total: ~74 s  ->  ~13.6 fps (1008 pixel frames / 74 s)

Run:
  cd .
  python -m naive_rt.wave.history.s0_naive_baseline --num_output_frames 252
"""
import argparse
import os
import time

import torch
from collections import OrderedDict
import naive_rt.rolling_forcing.configs as _rf_configs_pkg
from omegaconf import OmegaConf

from naive_rt.rolling_forcing.pipeline.rolling_forcing_inference import (
    CausalInferencePipeline,
)

_CFG_DIR = os.path.dirname(_rf_configs_pkg.__file__)

DEFAULT_CKPT = "./ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt"
DEFAULT_PROMPT = (
    "A cinematic shot of a fluffy corgi running on a sunny beach, "
    "waves in the background."
)


def load_pipeline(gen_ckpt, device):
    """Load config, build pipeline, load checkpoint."""
    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(_CFG_DIR, "default_config.yaml")),
        OmegaConf.load(os.path.join(_CFG_DIR, "rolling_forcing_dmd.yaml")),
    )
    pipe = CausalInferencePipeline(cfg, device=device)

    sd = torch.load(gen_ckpt, map_location="cpu", weights_only=False)
    gen_sd = sd["generator_ema"]
    # Strip FSDP wrapper prefix if present
    fixed = OrderedDict()
    for k, v in gen_sd.items():
        if k.startswith("model._fsdp_wrapped_module."):
            k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[k] = v

    pipe.generator.load_state_dict(fixed, strict=False)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device)
    pipe.generator.to(device)
    pipe.vae.to(device)
    return pipe


def main():
    ap = argparse.ArgumentParser(
        description="s0: single-GPU naive rolling forcing baseline"
    )
    ap.add_argument("--num_output_frames", type=int, default=252)
    ap.add_argument("--gen_ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_video", default="",
                    help="If set, save the output video to this path")
    args = ap.parse_args()

    device = "cuda"
    torch.set_grad_enabled(False)

    print(f"[s0] Loading pipeline from {args.gen_ckpt} ...")
    pipe = load_pipeline(args.gen_ckpt, device)

    npb = pipe.num_frame_per_block
    assert args.num_output_frames % npb == 0, (
        f"num_output_frames must be a multiple of {npb}"
    )

    prompts = [args.prompt]
    shape = [1, args.num_output_frames, 16, 60, 104]

    g = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(shape, generator=g, device=device, dtype=torch.bfloat16)

    # -- warmup --
    print("[s0] Warmup pass (21 frames) ...")
    warmup_noise = torch.randn(
        [1, 21, 16, 60, 104], device=device, dtype=torch.bfloat16
    )
    _ = pipe.inference_rolling_forcing(
        noise=warmup_noise, text_prompts=prompts
    )
    pipe.vae.model.clear_cache()
    pipe.kv_cache_clean = None
    torch.cuda.synchronize()

    # -- measured pass --
    print(f"[s0] Generating {args.num_output_frames} latent frames ...")
    torch.manual_seed(args.seed + 1)

    diff_start = torch.cuda.Event(enable_timing=True)
    diff_end = torch.cuda.Event(enable_timing=True)
    vae_start = torch.cuda.Event(enable_timing=True)
    vae_end = torch.cuda.Event(enable_timing=True)

    diff_start.record()
    video, latents = pipe.inference_rolling_forcing(
        noise=noise, text_prompts=prompts, return_latents=True, profile=True,
    )
    diff_end.record()
    torch.cuda.synchronize()

    diff_ms = diff_start.elapsed_time(diff_end)

    # VAE is already called inside inference_rolling_forcing when profile=True,
    # but let's time it separately for clarity
    pipe.vae.model.clear_cache()
    vae_start.record()
    video2 = pipe.vae.decode_to_pixel(latents, use_cache=False)
    video2 = (video2 * 0.5 + 0.5).clamp(0, 1)
    vae_end.record()
    torch.cuda.synchronize()
    vae_ms = vae_start.elapsed_time(vae_end)

    total_ms = diff_ms + vae_ms
    # Each latent frame decodes to 4 pixel frames (temporal upsampling)
    pixel_frames = args.num_output_frames * 4
    fps = pixel_frames / (total_ms / 1000.0)

    print(f"\n{'='*50}")
    print(f"[s0] Results ({args.num_output_frames} latent frames):")
    print(f"  Diffusion time : {diff_ms/1000:.2f} s")
    print(f"  VAE decode time: {vae_ms/1000:.2f} s")
    print(f"  Total time     : {total_ms/1000:.2f} s")
    print(f"  Pixel frames   : {pixel_frames}")
    print(f"  FPS            : {fps:.1f}")
    print(f"{'='*50}")

    if args.save_video:
        from torchvision.io import write_video
        from einops import rearrange
        vid = rearrange(video2, "b t c h w -> b t h w c")
        vid = (vid * 255).byte().cpu()[0]
        write_video(args.save_video, vid, fps=16)
        print(f"[s0] Saved video -> {args.save_video}")


if __name__ == "__main__":
    main()
