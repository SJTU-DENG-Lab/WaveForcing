"""
s1_wave_sim.py -- Milestone 1: Single-GPU wavefront simulator (oracle correctness check).

Uses WaveLayerPipeline on a single GPU to execute the wavefront schedule and
optionally compares the output against the naive rolling forcing baseline.
This proves the wavefront decomposition is mathematically equivalent (or very
close) to the sequential baseline.

Expected numbers (21 latent frames, H200):
  - PSNR vs naive: ~41 dB  (nearly identical latents)
  - Wall time is slower than naive (single GPU does all stages serially);
    the point is correctness, not speed.

Run:
  cd .
  python -m naive_rt.wave.history.s1_wave_sim --num_output_frames 21 --compare_naive
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
from naive_rt.wave.layer_pipeline import WaveLayerPipeline

_CFG_DIR = os.path.dirname(_rf_configs_pkg.__file__)

DEFAULT_CKPT = "./ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt"
DEFAULT_PROMPT = (
    "A cinematic shot of a fluffy corgi running on a sunny beach, "
    "waves in the background."
)


def compare_latents(a, b, name_a="naive", name_b="sim"):
    """Latent comparison statistics between two tensors [B,F,C,H,W]."""
    a = a.float()
    b = b.float()
    assert a.shape == b.shape, (a.shape, b.shape)
    diff = (a - b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = a.abs().mean().item() + 1e-8
    rel = mean_abs / denom
    mse = ((a - b) ** 2).mean().item()
    rng = (a.max() - a.min()).item() + 1e-8
    psnr = 20.0 * torch.log10(torch.tensor(rng / (mse ** 0.5 + 1e-12))).item()
    print(f"--- latent compare [{name_a} vs {name_b}] shape={tuple(a.shape)} ---")
    print(f"  max|d|   = {max_abs:.5f}")
    print(f"  mean|d|  = {mean_abs:.6f}   (rel {rel*100:.3f}% of mean|a|)")
    print(f"  MSE      = {mse:.6f}")
    print(f"  PSNR     = {psnr:.2f} dB")
    npb = 3
    per = []
    for i in range(0, a.shape[1], npb):
        per.append((a[:, i:i+npb] - b[:, i:i+npb]).abs().mean().item())
    print("  per-chunk mean|d|: " + " ".join(f"{x:.4f}" for x in per))
    return dict(max_abs=max_abs, mean_abs=mean_abs, rel=rel, mse=mse, psnr=psnr)


def load_pipeline(gen_ckpt, device):
    """Load config, build pipeline, load checkpoint."""
    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(_CFG_DIR, "default_config.yaml")),
        OmegaConf.load(os.path.join(_CFG_DIR, "rolling_forcing_dmd.yaml")),
    )
    pipe = CausalInferencePipeline(cfg, device=device)
    sd = torch.load(gen_ckpt, map_location="cpu", weights_only=False)
    gen_sd = sd["generator_ema"]
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
    return pipe, cfg


def main():
    ap = argparse.ArgumentParser(
        description="s1: single-GPU wavefront simulator (correctness oracle)"
    )
    ap.add_argument("--num_output_frames", type=int, default=21)
    ap.add_argument("--gen_ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare_naive", action="store_true",
                    help="Also run naive baseline and compare latents")
    ap.add_argument("--save_video", default="",
                    help="If set, save the wavefront video to this path")
    args = ap.parse_args()

    device = "cuda"
    torch.set_grad_enabled(False)

    print(f"[s1] Loading pipeline from {args.gen_ckpt} ...")
    pipe, cfg = load_pipeline(args.gen_ckpt, device)

    npb = pipe.num_frame_per_block
    assert args.num_output_frames % npb == 0, (
        f"num_output_frames must be a multiple of {npb}"
    )

    prompts = [args.prompt]
    shape = [1, args.num_output_frames, 16, 60, 104]

    def make_noise():
        g = torch.Generator(device=device).manual_seed(args.seed)
        return torch.randn(shape, generator=g, device=device, dtype=torch.bfloat16)

    # --- optionally run naive baseline first ---
    naive_out = None
    if args.compare_naive:
        print("[s1] Running naive baseline for comparison ...")
        noise = make_noise()
        torch.manual_seed(args.seed + 1)
        t0 = time.perf_counter()
        _, naive_out = pipe.inference_rolling_forcing(
            noise=noise, text_prompts=prompts, return_latents=True,
        )
        torch.cuda.synchronize()
        naive_ms = (time.perf_counter() - t0) * 1000
        print(f"[s1/naive] latents {tuple(naive_out.shape)}  wall {naive_ms:.0f} ms")
        pipe.vae.model.clear_cache()
        pipe.kv_cache_clean = None

    # --- wavefront sim ---
    print("[s1] Running wavefront sim ...")
    noise = make_noise()
    ctx_noise = float(getattr(cfg, "context_noise", 0.0))
    wl = WaveLayerPipeline(pipe, context_noise=ctx_noise, verbose=True)

    torch.manual_seed(args.seed + 1)
    t0 = time.perf_counter()
    sim_out = wl.generate(noise, prompts)
    torch.cuda.synchronize()
    sim_ms = (time.perf_counter() - t0) * 1000
    print(
        f"[s1/wave] latents {tuple(sim_out.shape)}  diffusion wall {sim_ms:.0f} ms  "
        f"(sum tick {sum(wl.tick_times):.0f} ms, "
        f"mean {sum(wl.tick_times)/len(wl.tick_times):.1f} ms)"
    )

    # --- compare ---
    if naive_out is not None:
        compare_latents(naive_out, sim_out, "naive", "wavefront-sim")

    # --- optionally save video ---
    if args.save_video:
        from torchvision.io import write_video
        from einops import rearrange
        vid = pipe.vae.decode_to_pixel(sim_out, use_cache=False)
        vid = ((vid * 0.5 + 0.5).clamp(0, 1) * 255)
        vid = rearrange(vid, "b t c h w -> b t h w c").byte().cpu()[0]
        write_video(args.save_video, vid, fps=16)
        print(f"[s1] Saved video -> {args.save_video}")


if __name__ == "__main__":
    main()
