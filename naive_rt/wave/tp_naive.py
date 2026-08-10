"""tp_naive.py -- naive Rolling-Forcing baseline with head-parallel Tensor
Parallelism (Group C). tp=1 reproduces the s0 naive baseline; tp>1 shards the
DiT attention/FFN across `tp` GPUs (12 heads -> tp in {2,3,4,6}).

The VAE is NOT sharded (runs full on rank0), so TP accelerates diffusion only —
which is the honest same-model comparison against the wave-parallel pipeline.

Run:
  python -m naive_rt.wave.tp_naive --tp 4 --num_output_frames 252 \
      --out_root outputs/experiments --task 21_naive_tp4 --run_tag tp4_nf252 \
      --ref_latents outputs/experiments/00_naive_baseline/<run>/latents.pt \
      --baseline_e2e 74.0
"""
import argparse
import os
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

import naive_rt.rolling_forcing.configs as _cfgpkg
from naive_rt.rolling_forcing.pipeline.rolling_forcing_inference import (
    CausalInferencePipeline,
)
from naive_rt.tp.shard import (
    init_distributed,
    patch_cache_init,
    shard_model_for_tp,
)
from naive_rt.bench import harness

_CFG = os.path.dirname(_cfgpkg.__file__)
DEFAULT_CKPT = "./ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt"
DEFAULT_PROMPT = (
    "A cinematic shot of a fluffy corgi running on a sunny beach, "
    "waves in the background."
)
LAT_SHAPE = (16, 60, 104)  # C,H,W per latent frame


def load_pipeline(steps, gen_ckpt, device):
    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(_CFG, "default_config.yaml")),
        OmegaConf.load(os.path.join(_CFG, "rolling_forcing_dmd.yaml")),
    )
    if steps:
        cfg.denoising_step_list = list(steps)
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
    return pipe


def _make_noise(nframes, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn([1, nframes, *LAT_SHAPE], generator=g,
                       device=device, dtype=torch.bfloat16)


def worker(rank, world, a):
    init_distributed(rank, world, a.port)
    device = torch.device(f"cuda:{rank}")
    torch.set_grad_enabled(False)
    steps = [int(x) for x in a.steps.split(",")] if a.steps else None
    nstep = len(steps) if steps else 4

    pipe = load_pipeline(steps, a.gen_ckpt, device)
    shard_model_for_tp(pipe.generator.model, rank, world)
    patch_cache_init(pipe, world)

    npb = pipe.num_frame_per_block
    assert a.num_output_frames % npb == 0, f"frames must be multiple of {npb}"
    prompts = [a.prompt]

    # --- warmup (identical noise on all ranks via broadcast) ---
    warm = _make_noise(21, a.seed, device)
    if world > 1:
        dist.broadcast(warm, src=0)
    torch.manual_seed(a.seed)  # sync any internal RNG across ranks
    _ = pipe.inference_rolling_forcing(noise=warm, text_prompts=prompts)
    pipe.vae.model.clear_cache()
    pipe.kv_cache_clean = None
    torch.cuda.synchronize()

    # --- measured pass ---
    noise = _make_noise(a.num_output_frames, a.seed + 1, device)
    if world > 1:
        dist.broadcast(noise, src=0)
        dist.barrier()
    torch.manual_seed(a.seed + 1)
    t0 = time.perf_counter()
    video, latents = pipe.inference_rolling_forcing(
        noise=noise, text_prompts=prompts, return_latents=True, profile=True,
    )
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier()
    e2e_wall = time.perf_counter() - t0

    if rank == 0:
        tm = pipe.last_timing
        diffusion_s = tm["diffusion_ms"] / 1000.0
        vae_s = tm["vae_ms"] / 1000.0
        out = harness.run_dir(a.out_root, a.task, a.run_tag)
        harness.save_video(video, os.path.join(out, "video.mp4"))
        torch.save(latents.detach().cpu(), os.path.join(out, "latents.pt"))
        psnr = None
        if a.ref_latents and os.path.isfile(a.ref_latents):
            ref = torch.load(a.ref_latents, map_location="cpu")
            psnr = harness.latent_psnr(latents.detach().cpu(), ref)["psnr_db"]
        speedup = (a.baseline_e2e / (diffusion_s + vae_s)) if a.baseline_e2e > 0 else None
        harness.write_metrics(
            out, task=a.task, run_tag=a.run_tag,
            method=("naive_tp" if world > 1 else "naive"),
            nstep=nstep, num_output_frames=a.num_output_frames, num_gpus=world,
            diffusion_s=diffusion_s, vae_s=vae_s, end_to_end_s=diffusion_s + vae_s,
            tp_size=world, seed=a.seed, psnr_db=psnr,
            speedup_vs_baseline=speedup, per_window_ms=tm.get("block_ms"),
            extra={"e2e_wall_s": round(e2e_wall, 4)},
        )
        pf = harness.pixel_frames(a.num_output_frames)
        print(f"[tp{world}] diffusion={diffusion_s:.2f}s vae={vae_s:.2f}s "
              f"e2e={diffusion_s + vae_s:.2f}s fps_with_vae="
              f"{pf / (diffusion_s + vae_s):.2f} fps_no_vae={pf / diffusion_s:.2f}"
              + (f" PSNR={psnr:.2f}dB" if psnr else ""))

    if world > 1:
        dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description="naive Rolling-Forcing + Tensor Parallel")
    ap.add_argument("--tp", type=int, default=1, help="TP degree (must divide 12)")
    ap.add_argument("--gpus", default="", help="CUDA_VISIBLE_DEVICES, default 0..tp-1")
    ap.add_argument("--num_output_frames", type=int, default=252)
    ap.add_argument("--steps", default="", help="denoising_step_list csv (default 4-step yaml)")
    ap.add_argument("--gen_ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_root", default="outputs/experiments")
    ap.add_argument("--task", default="20_naive_tp")
    ap.add_argument("--run_tag", default="tp")
    ap.add_argument("--ref_latents", default="", help="reference latents.pt for PSNR")
    ap.add_argument("--baseline_e2e", type=float, default=0.0, help="baseline e2e s for speedup")
    ap.add_argument("--port", type=int, default=29711)
    a = ap.parse_args()

    if a.tp > 1 and 12 % a.tp != 0:
        raise SystemExit(f"tp={a.tp} must divide 12 (use 2,3,4,6)")
    if a.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(a.tp))

    if a.tp == 1:
        worker(0, 1, a)
    else:
        mp.spawn(worker, args=(a.tp, a), nprocs=a.tp)


if __name__ == "__main__":
    main()
