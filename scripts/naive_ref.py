"""Naive single-GPU RF baseline latent dumper (for cross-framework quality check).

Runs the vendored naive CausalInferencePipeline (the trusted reference) and
saves the denoised latents as naive_rt returns them ((B, T, C, H, W), see
inference_rolling_forcing) for latent-PSNR comparison against wrt / sgl --
bench.latent_psnr aligns the (B,C,T,H,W) vs (B,T,C,H,W) ordering.  Also
records the single-GPU end-to-end wall time (diffusion + VAE
decode, the README 13.6 fps baseline) as e2e.json, so the same artifacts serve
both halves of the sweep baseline:  PSNR reference (latents.pt) and single-GPU
speed reference (e2e.json -> wave_rt --baseline-e2e).

The step line is parameterized (--nstep/--steps), so one baseline per topology
is possible:  nstep 4 [1000,750,500,250] for 4+3, nstep 5 [1000,800,600,400,200]
for 5+2.  NOTE:  with the default 4-step ckpt a 5-step baseline is
OFF-DISTRIBUTION (system timing only) unless --gen-ckpt points at TencentARC's
5-step rolling_forcing_dmd.pt -- same status as a 5+2 wave_rt run.

Initial noise parity with wave_rt:  noise is drawn from a cuda generator seeded
--seed, exactly what sgl's LatentPreparation does for the same seed
(Generator("cuda").manual_seed(seed)), so a sweep run at --seed 0 consumes
bit-identical initial noise and the PSNR gate is honest.  The seed is not
enough -- the LAYOUT must match too:  a seeded torch.randn stream is mapped
into the tensor in linear order, so drawing (B, T, C, H, W) where sgl draws
(B, C, T, H, W) permutes the noise voxels and destroys PSNR (observed ~19 dB
"noise mismatch").  We therefore draw in sgl's (B, C, T, H, W) layout and
permute to the (B, T, C, H, W) naive_rt expects.  noise.pt is saved too
for auditing / future feeding.

Run:
  cd /path/to/WaveParallel
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/naive_ref.py \
      --num_output_frames 48 --seed 0 --out ./outputs/ref/naive_s48_seed0
  # 5-step line (5+2 topology baseline):
  .venv/bin/python scripts/naive_ref.py --nstep 5 \
      --steps 1000,800,600,400,200 --out ./outputs/sweep/baseline/5+2
"""
import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_output_frames", type=int, default=48)
    ap.add_argument("--gen_ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nstep", type=int, default=4,
                    help="denoising steps; len(--steps) must equal this")
    ap.add_argument("--steps", default="1000,750,500,250",
                    help="denoising_step_list, comma-separated, length=--nstep "
                         "(5+2 topo: 1000,800,600,400,200)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda"
    torch.set_grad_enabled(False)

    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(_CFG_DIR, "default_config.yaml")),
        OmegaConf.load(os.path.join(_CFG_DIR, "rolling_forcing_dmd.yaml")),
    )
    dsl = [int(x) for x in args.steps.split(",")]
    if len(dsl) != args.nstep:
        raise ValueError(f"--steps has {len(dsl)} entries but --nstep={args.nstep}")
    cfg.denoising_step_list = dsl   # same override naive serve.py does
    print(f"[naive] denoising_step_list={list(cfg.denoising_step_list)} "
          f"warp={cfg.warp_denoising_step}", flush=True)
    pipe = CausalInferencePipeline(cfg, device=device)
    sd = torch.load(args.gen_ckpt, map_location="cpu", weights_only=False)["generator_ema"]
    fixed = OrderedDict()
    for k, v in sd.items():
        if k.startswith("model._fsdp_wrapped_module."):
            k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[k] = v
    pipe.generator.load_state_dict(fixed, strict=False)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device)
    pipe.generator.to(device)
    pipe.vae.to(device)

    # Draw the noise in the SAME layout sgl's LatentPreparation uses
    # ((B, C, T, H, W), see wan.py prepare_latent_shape), so the seeded randn
    # stream maps to the same (c, t, h, w) voxel as wave_rt's full noise -- the
    # PSNR gate only holds if BOTH the layout and the seed match.  naive_rt's
    # inference_rolling_forcing expects (B, T, C, H, W), hence the permute.
    shape_bcthw = [1, 16, args.num_output_frames, 60, 104]
    g = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(shape_bcthw, generator=g, device=device,
                        dtype=torch.bfloat16).permute(0, 2, 1, 3, 4)

    t0 = time.perf_counter()
    out = pipe.inference_rolling_forcing(
        noise=noise, text_prompts=[args.prompt], return_latents=True
    )
    e2e_s = time.perf_counter() - t0     # diffusion + VAE decode (README 13.6fps basis)
    latents = out[1] if isinstance(out, (tuple, list)) else out
    latents = latents.detach().float().cpu()
    os.makedirs(args.out, exist_ok=True)
    torch.save(latents, os.path.join(args.out, "latents.pt"))
    # also save the initial noise so wrt could be fed the SAME noise later
    torch.save(noise.detach().cpu(), os.path.join(args.out, "noise.pt"))
    with open(os.path.join(args.out, "e2e.json"), "w") as f:
        json.dump({"e2e_s": round(e2e_s, 3), "nstep": args.nstep,
                   "steps": dsl, "num_frames": args.num_output_frames,
                   "seed": args.seed}, f, indent=2)
    print(f"[naive] latents {tuple(latents.shape)} -> {args.out}/latents.pt "
          f"e2e={e2e_s:.1f}s ({args.num_output_frames * 4 / e2e_s:.1f} fps)",
          flush=True)


if __name__ == "__main__":
    main()
