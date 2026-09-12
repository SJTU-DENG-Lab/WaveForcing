"""Generate and verify original-DMD video ``(z_ref, y_ref)`` pairs.

Each target is a deterministic 50-step Wan2.1-T2V-14B UniPC endpoint from
the exact stored initial Gaussian latent and prompt. The script is resumable
and can be stride-sharded over independent GPUs.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf

from importlib.resources import files
from wf_training.assets import configure_model_root

SHAPE = (21, 16, 60, 104)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", help="Optional override of the packaged teacher config")
    parser.add_argument("--model-root", help="Wan model root; required for generation/replay")
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=1504386)
    parser.add_argument("--prompt-seed", type=int, default=1504386)
    parser.add_argument("--teacher-name", default="Wan2.1-T2V-14B")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--replay-index", type=int)
    return parser.parse_args()


def selected_prompts(path, count, seed):
    with open(path, encoding="utf-8") as handle:
        prompts = [line.rstrip() for line in handle if line.rstrip()]
    if count > len(prompts):
        raise ValueError(f"requested {count} prompts from a file with {len(prompts)}")
    indices = list(range(len(prompts)))
    random.Random(seed).shuffle(indices)
    return [(indices[i], prompts[indices[i]]) for i in range(count)]


def build_teacher(args, config):
    from wf_training.pipeline import BidirectionalDiffusionInferencePipeline
    from wf_training.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder

    device = torch.device("cuda")
    generator = WanDiffusionWrapper(
        model_name=args.teacher_name,
        timestep_shift=args.shift,
        is_causal=False,
    ).to(device=device, dtype=torch.bfloat16)
    text_encoder = WanTextEncoder(model_name=args.teacher_name).to(device=device)
    pipeline = BidirectionalDiffusionInferencePipeline(
        config,
        device=device,
        generator=generator,
        text_encoder=text_encoder,
        vae=torch.nn.Identity(),
    )
    pipeline.sampling_steps = args.sampling_steps
    pipeline.shift = args.shift
    config.guidance_scale = args.guidance_scale
    pipeline.eval()
    return pipeline


def run_teacher(pipeline, z_ref, prompt):
    teacher_input = z_ref.unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, endpoint = pipeline.inference(
            noise=teacher_input,
            text_prompts=[prompt],
            return_latents=True,
            decode=False,
        )
    return endpoint[0].half().cpu()


def item_path(output_dir, pair_index):
    return output_dir / f"{pair_index:06d}.pt"


def verify_and_write_manifests(args, prompts):
    output_dir = Path(args.output_dir).resolve()
    train_records = []
    validation_records = []
    for pair_index, (prompt_index, prompt) in enumerate(prompts):
        path = item_path(output_dir, pair_index)
        if not path.is_file():
            raise FileNotFoundError(f"missing pair {pair_index}: {path}")
        item = torch.load(path, map_location="cpu", weights_only=True)
        z_ref = item["z_ref"]
        y_ref = item["y_ref"]
        expected_seed = args.base_seed + pair_index
        checks = {
            "pair_index": int(item["pair_index"]) == pair_index,
            "prompt_index": int(item["prompt_index"]) == prompt_index,
            "prompt": item["prompt"] == prompt,
            "seed": int(item["seed"]) == expected_seed,
            "z_shape": tuple(z_ref.shape) == SHAPE,
            "y_shape": tuple(y_ref.shape) == SHAPE,
            "z_finite": bool(torch.isfinite(z_ref).all()),
            "y_finite": bool(torch.isfinite(y_ref).all()),
            "teacher": item["teacher_name"] == args.teacher_name,
            "sampling_steps": int(item["sampling_steps"]) == args.sampling_steps,
            "guidance_scale": float(item["guidance_scale"]) == args.guidance_scale,
            "shift": float(item["shift"]) == args.shift,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"pair {pair_index} failed checks: {failed}")
        record = {
            "pair_index": pair_index,
            "prompt_index": prompt_index,
            "seed": expected_seed,
            "path": path.name,
        }
        if pair_index < args.train_count:
            train_records.append(record)
        else:
            validation_records.append(record)

    for name, records in (
        ("manifest_train.jsonl", train_records),
        ("manifest_val.jsonl", validation_records),
    ):
        path = output_dir / name
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(tmp_path, path)
    summary = {
        "status": "verified",
        "count": len(prompts),
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "shape": list(SHAPE),
        "teacher": args.teacher_name,
        "sampling_steps": args.sampling_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
    }
    summary_path = output_dir / "verification.json"
    tmp_summary = summary_path.with_suffix(".json.tmp")
    with tmp_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_summary, summary_path)
    print("PAIR_VERIFY_OK " + json.dumps(summary, sort_keys=True), flush=True)


def main():
    args = parse_args()
    if args.stride < 1 or args.start < 0 or args.start >= args.stride:
        raise ValueError("Require stride >= 1 and 0 <= start < stride")
    if args.replay_index is not None and not 0 <= args.replay_index < args.count:
        raise ValueError("replay-index must lie in [0, count)")
    if not 0 <= args.train_count <= args.count:
        raise ValueError("train-count must lie in [0, count]")
    prompts = selected_prompts(args.prompt_path, args.count, args.prompt_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(str(files("wf_training").joinpath("configs/base.yaml")))
    if args.config_path:
        config = OmegaConf.merge(config, OmegaConf.load(args.config_path))

    if args.verify_only:
        verify_and_write_manifests(args, prompts)
        return

    requested = list(range(args.start, args.count, args.stride))
    pending = [index for index in requested if not item_path(output_dir, index).is_file()]
    if not pending and args.replay_index is None:
        print(f"PAIR_SHARD_ALREADY_COMPLETE start={args.start} stride={args.stride}", flush=True)
        return

    if not args.model_root:
        raise ValueError("--model-root is required for generation or replay")
    configure_model_root(args.model_root)
    pipeline = build_teacher(args, config)
    if args.replay_index is not None:
        pair_index = args.replay_index
        path = item_path(output_dir, pair_index)
        item = torch.load(path, map_location="cpu", weights_only=True)
        replay = run_teacher(pipeline, item["z_ref"], item["prompt"])
        stored = item["y_ref"]
        difference = (replay.float() - stored.float()).abs()
        result = {
            "pair_index": pair_index,
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
        }
        if result["max_abs"] > 0.015625 or result["mean_abs"] > 0.0005:
            raise RuntimeError(f"deterministic teacher replay mismatch: {result}")
        replay_path = output_dir / f"replay_{pair_index:06d}.json"
        with replay_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("PAIR_REPLAY_OK " + json.dumps(result, sort_keys=True), flush=True)
        return

    for pair_index in pending:
        prompt_index, prompt = prompts[pair_index]
        seed = args.base_seed + pair_index
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        z_ref = torch.randn(SHAPE, generator=generator, dtype=torch.float32)
        y_ref = run_teacher(pipeline, z_ref, prompt)
        item = {
            "version": 1,
            "pair_index": pair_index,
            "prompt_index": prompt_index,
            "prompt": prompt,
            "seed": seed,
            "z_ref": z_ref,
            "y_ref": y_ref,
            "teacher_name": args.teacher_name,
            "sampling_steps": args.sampling_steps,
            "guidance_scale": args.guidance_scale,
            "shift": args.shift,
            "solver": "unipc",
            "teacher_input_dtype": "bfloat16",
        }
        path = item_path(output_dir, pair_index)
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save(item, tmp_path)
        os.replace(tmp_path, path)
        print(
            f"PAIR_DONE index={pair_index} prompt_index={prompt_index} seed={seed}",
            flush=True,
        )


if __name__ == "__main__":
    main()
