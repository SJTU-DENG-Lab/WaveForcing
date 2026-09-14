"""WF reference training entry point; help and dry-run never import torch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

from wf_training.config import (
    ConfigError, RECIPES, STAGES, checkpoint_complete,
    load_assets, plain, preflight, recipe_defaults, resolve_config, validate_config,
)
from wf_training.utils.source import source_provenance

_ENV_WHITELIST = (
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "WANDB_MODE",
    "RF_BLOCK_CAUSAL", "RF_FLEX_ATTN", "RF_FLEX_COMPILE", "TORCH_HOME",
    "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "PYTORCH_CUDA_ALLOC_CONF", "CC", "CXX",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WaveForcing S1/S2/S3 training")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("show-config", "preflight", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--assets", required=True, help="YAML of explicit local asset paths")
        sub.add_argument("--output", required=True, help="run root; each stage gets its own subdirectory")
        sub.add_argument("--world-size", type=int, default=8, help="local torchrun GPU count")
        sub.add_argument("--recipe", choices=RECIPES, default="reference",
                         help="reference (default) or experimental single-node 14B FSDP8 recipe")
        stages = sub.add_mutually_exclusive_group()
        stages.add_argument("--stage", choices=STAGES)
        stages.add_argument("--stages", help="ordered contiguous list; default: reference s2,s3; 14B s1,s2,s3")
        sub.add_argument("--init-key", help="asset key for the first stage checkpoint; never means resume")
        sub.add_argument("--set", action="append", default=[], dest="overrides",
                         help="explicit key=value or stage.key=value recipe override")
        sub.add_argument("--dry-run", action="store_true", help="print the plan without writes or GPU imports")
    resume = commands.add_parser("resume", help="restore a stage's complete model/optimizer/RNG state")
    resume.add_argument("--run", required=True, help="stage directory containing resolved_config.yaml")
    resume.add_argument("--checkpoint", required=True, help="complete checkpoint directory for this stage")
    resume.add_argument("--dry-run", action="store_true")
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", required=True)
    worker.add_argument("--resume-from", default="")
    return parser


def selected_stages(args) -> list[str]:
    defaults = recipe_defaults(getattr(args, "recipe", "reference"))
    stages = ([args.stage] if args.stage else args.stages.split(",") if args.stages
              else list(defaults.default_stages))
    if not stages or any(stage not in STAGES for stage in stages):
        raise ConfigError("--stages must contain s1, s2, and/or s3")
    offsets = [STAGES.index(stage) for stage in stages]
    if offsets != list(range(offsets[0], offsets[0] + len(offsets))):
        raise ConfigError("stages must be ordered, contiguous, and contain no duplicates")
    return stages


def build_plan(args) -> list:
    assets = load_assets(args.assets)
    configs = []
    previous = None
    for stage in selected_stages(args):
        config = resolve_config(stage, assets, args.output, args.world_size,
                                init_key=args.init_key if previous is None else None,
                                init_checkpoint=previous, overrides=args.overrides,
                                recipe=getattr(args, "recipe", "reference"))
        configs.append(config)
        previous = str(Path(config.logdir) / f"checkpoint_model_{config.max_steps:06d}" / "model.pt")
    return configs


def worker_command(config, config_path: Path, resume_from: str = "") -> list[str]:
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={config.world_size}",
               "--log-dir", str(Path(config.logdir) / "torchrun_logs"), "--tee", "3",
               "-m", "wf_training",
               "_worker", "--config", str(config_path)]
    if resume_from:
        command.extend(["--resume-from", resume_from])
    return command


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _display_plan(configs: list) -> None:
    print(json.dumps({"mode": "plan_only", "stages": [
        {"config": plain(config), "command": worker_command(config, Path(config.logdir) / "resolved_config.yaml")}
        for config in configs]}, indent=2, ensure_ascii=False))


def _execute(config, config_path: Path, *, resume_from: str = "") -> None:
    run_dir = Path(config.logdir)
    command = worker_command(config, config_path, resume_from)
    status = {"status": "running", "started_at": _now(), "command": command,
              "resume_from": resume_from}
    _write_json(run_dir / "status.json", status)
    environment = os.environ.copy()
    environment.setdefault("WANDB_MODE", "offline")
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    environment.setdefault("OMP_NUM_THREADS", "1")
    try:
        subprocess.run(command, check=True, env=environment)
        endpoint = run_dir / f"checkpoint_model_{config.max_steps:06d}"
        checkpoint_complete(endpoint, config, expected_step=config.max_steps)
    except (Exception, KeyboardInterrupt) as error:
        status.update(status="failed", finished_at=_now(), error=str(error))
        _write_json(run_dir / "status.json", status)
        raise
    status.update(status="completed", finished_at=_now(), checkpoint=str(endpoint))
    _write_json(run_dir / "status.json", status)


def _run(configs: list) -> None:
    # Check the entire plan before starting stage one. Later init files are
    # expected outputs; all other model/data assets must already be available.
    for index, config in enumerate(configs):
        if Path(config.logdir).exists():
            raise ConfigError(f"refusing existing stage directory; use resume: {config.logdir}")
        preflight(config, allow_pending_init=index > 0)
    for config in configs:
        report = preflight(config)
        directory = Path(config.logdir)
        directory.mkdir(parents=True, exist_ok=False)
        config_path = directory / "resolved_config.yaml"
        OmegaConf.save(config, config_path)
        _write_json(directory / "preflight.json", report)
        _write_json(directory / "run_manifest.json", {
            "schema_version": 1, "created_at": _now(), "stage": config.stage,
            "source_path": str(Path(__file__).resolve().parent),
            "source_provenance": source_provenance(),
            "python": sys.executable, "initial_checkpoint": report["assets"]["initial_checkpoint"],
            "environment": {key: os.environ[key] for key in _ENV_WHITELIST if key in os.environ},
        })
        _execute(config, config_path)


def load_resume(run: str, checkpoint: str):
    directory = Path(run).expanduser().resolve()
    config_path = directory / "resolved_config.yaml"
    config = OmegaConf.load(config_path)
    validate_config(config)
    if not (directory / "run_manifest.json").is_file():
        raise ConfigError(f"missing run_manifest.json: {directory}")
    if str(directory) != config.logdir:
        raise ConfigError("resume run directory differs from the recorded stage directory")
    path = Path(checkpoint).expanduser().resolve()
    if path.parent != directory:
        raise ConfigError("resume checkpoint must belong to this stage run directory")
    marker = checkpoint_complete(path, config)
    if marker.get("step", -1) >= config.max_steps:
        raise ConfigError("checkpoint is already at the configured final step")
    previous = json.loads((directory / "preflight.json").read_text())
    current = preflight(config)
    if (previous["assets"] != current["assets"]
            or previous["package_versions"] != current["package_versions"]
            or previous["python_version"] != current["python_version"]):
        raise ConfigError("asset identities or dependency versions changed since the run started")
    return config, config_path, str(path)


def _worker(args) -> None:
    config = OmegaConf.load(Path(args.config).resolve())
    validate_config(config)
    if args.resume_from:
        checkpoint_complete(args.resume_from, config)
        config.resume_from = str(Path(args.resume_from).resolve())
    # No model module is imported until root paths and algorithm switches are explicit.
    from wf_training.assets import configure_reference_runtime
    configure_reference_runtime(config)
    import torch
    from wf_training.trainer import distillation
    trainer = None
    try:
        trainer = distillation.Trainer(config)
        rank = torch.distributed.get_rank()
        properties = torch.cuda.get_device_properties(trainer.device)
        _write_json(Path(config.logdir) / f"worker_rank{rank:02d}.json", {
            "created_at": _now(), "rank": rank, "world_size": trainer.world_size,
            "sequence_parallel_size": getattr(config, "sequence_parallel_size", 1),
            "data_parallel_size": getattr(config, "data_parallel_size", trainer.world_size),
            "gradient_accumulation_steps": getattr(config, "gradient_accumulation_steps", 1),
            "effective_batch_size": config.effective_batch_size,
            "python": sys.executable, "trainer_source": str(Path(distillation.__file__).resolve()),
            "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "gpu": properties.name, "gpu_memory_bytes": properties.total_memory,
            "denoising_timesteps": trainer.model.denoising_step_list.cpu().tolist(),
            "resume_from": args.resume_from,
            "environment": {key: os.environ[key] for key in _ENV_WHITELIST if key in os.environ},
        })
        trainer.train()
    finally:
        if trainer is not None and hasattr(trainer, "writer"):
            trainer.writer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "_worker":
            _worker(args)
        elif args.command == "resume":
            config, path, resume_from = load_resume(args.run, args.checkpoint)
            if args.dry_run:
                print(json.dumps({"mode": "resume_plan_only", "config": plain(config),
                                  "command": worker_command(config, path, resume_from)}, indent=2))
            else:
                _execute(config, path, resume_from=resume_from)
        else:
            configs = build_plan(args)
            if args.command == "show-config" or args.dry_run:
                _display_plan(configs)
            elif args.command == "preflight":
                print(json.dumps([preflight(config, allow_pending_init=index > 0)
                                  for index, config in enumerate(configs)], indent=2))
            else:
                _run(configs)
    except (ConfigError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"wf-training: {error}", file=sys.stderr)
        return 2
    return 0
