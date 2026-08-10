"""Single-GPU Rolling Forcing reference (canonical sglang pipeline).

Runs the stock WanRollingForcingPipeline end-to-end on ONE GPU with the same
prompt/seed/num_frames as a WaveRT run and dumps the denoised latents to
latents.pt.  WaveRT's wavefront latents are PSNR-gated against this file.

Run:
  cd .
  LD_LIBRARY_PATH=".venv/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH" \
    CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/wrt_ref_single.py \
      --num-frames 6 --out ./outputs/ref/s6
"""

import argparse
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-frames", type=int, default=6)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--prompt",
        default=(
            "A cinematic shot of a fluffy corgi running on a sunny beach, "
            "waves in the background."
        ),
    )
    ap.add_argument("--model-path", default="./ckpts/Wan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--gen-ckpt",
                    default="./ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt")
    ap.add_argument("--out", required=True, help="output dir for latents.pt + video.mp4")
    ap.add_argument("--save-video", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29712")

    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.pipelines_core import build_pipeline
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

    sa = ServerArgs.from_dict({
        "model_path": args.model_path,
        "pipeline_class_name": "WanRollingForcingPipeline",
        "num_gpus": 1, "tp_size": 1, "ulysses_degree": 1, "ring_degree": 1,
        "cfg_parallel_degree": 1, "attention_backend": "torch_sdpa",
        "dit_cpu_offload": False, "use_fsdp_inference": False,
    })
    sa.hsdp_shard_dim = None
    sa.pipeline_config.rolling_forcing_checkpoint_path = args.gen_ckpt
    sa.pipeline_config.rolling_forcing_use_ema = True
    # longvideo.pt is a 4-step DMD [1000,750,500,250]; sgl's config default is a
    # WRONG 5-step [1000,800,600,400,200].  Force the correct schedule so this
    # reference matches the real Causal-Forcing long_video output (and wrt).
    sa.pipeline_config.dmd_denoising_steps = [1000, 750, 500, 250]
    sa.pipeline_config.warp_denoising_step = True
    set_global_server_args(sa)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=1, cfg_degree=1, ulysses_degree=1, ring_degree=1, dp_size=1,
    )
    pipe = build_pipeline(sa)
    sp_cls = type(pipe).sampling_params_cls
    sampling = sp_cls(prompt=args.prompt, num_frames=args.num_frames,
                      height=args.height, width=args.width, seed=args.seed)
    batch = Req(sampling_params=sampling)

    t0 = time.perf_counter()
    # run prefix + RF denoise (stages 0..3) to get canonical latents
    for stage in pipe.stages[:4]:
        batch = stage(batch, sa)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    latents = batch.latents.detach().cpu()  # (1, C, nf, H, W)
    os.makedirs(args.out, exist_ok=True)
    torch.save(latents, os.path.join(args.out, "latents.pt"))
    print(f"[ref] single-GPU RF latents {tuple(latents.shape)} in {dt:.1f}s "
          f"-> {args.out}/latents.pt", flush=True)

    if args.save_video:
        batch = pipe.stages[4](batch, sa)  # decode
        from wave_rt import bench
        bench.save_video(batch.output, os.path.join(args.out, "video.mp4"))
        print(f"[ref] video -> {args.out}/video.mp4", flush=True)


if __name__ == "__main__":
    main()
