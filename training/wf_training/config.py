"""CPU-only configuration for reference and experimental 14B training."""
from __future__ import annotations

import importlib.metadata
from importlib import resources
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

from omegaconf import DictConfig, OmegaConf

STAGES = ("s1", "s2", "s3")
RECIPES = ("reference", "14b-fsdp8", "14b-fsdp8-smoke")
_PATH_FIELDS = {
    "model_root", "data_path", "generator_ckpt", "paired_manifest",
    "paired_validation_manifest", "logdir", "wandb_save_dir", "resume_from",
}


class ConfigError(ValueError):
    """An invalid recipe or configuration."""


def plain(config: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(config, resolve=True)


def load_assets(path: str | Path) -> dict[str, str]:
    asset_file = Path(path).expanduser().resolve()
    data = OmegaConf.to_container(OmegaConf.load(asset_file), resolve=True)
    if not isinstance(data, dict):
        raise ConfigError("assets YAML must be a mapping of names to paths")
    result = {}
    for key, value in data.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"asset {key!r} must be a nonempty path string")
        item = Path(value).expanduser()
        if not item.is_absolute():
            item = asset_file.parent / item
        result[key] = str(item.resolve())
    return result


def _resource(name: str) -> DictConfig:
    content = resources.files("wf_training").joinpath("configs", name + ".yaml").read_text()
    return OmegaConf.create(content)


def recipe_defaults(recipe: str = "reference") -> DictConfig:
    if recipe not in RECIPES:
        raise ConfigError(f"unsupported recipe: {recipe}")
    if recipe == "reference":
        return OmegaConf.create({
            "default_stages": ["s2", "s3"],
            "init_keys": {"s1": "ode_init", "s2": "rf_init", "s3": "s3_init"},
        })
    return _resource(recipe)


def parse_denoising_step_list(value: Any) -> list[int]:
    try:
        raw = list(value)
        steps = [int(item) for item in raw]
    except (TypeError, ValueError) as error:
        raise ConfigError("denoising_step_list must be a list of integers") from error
    if not steps:
        raise ConfigError("denoising_step_list must not be empty")
    if any(int(item) != item or item <= 0 or item > 1000 for item in raw):
        raise ConfigError("denoising_step_list values must be integers in [1, 1000]")
    if len(steps) != len(set(steps)):
        raise ConfigError("denoising_step_list must not contain duplicates")
    if any(steps[index] <= steps[index + 1] for index in range(len(steps) - 1)):
        raise ConfigError("denoising_step_list must be strictly decreasing")
    return steps


def resolve_config(stage: str, assets: dict[str, str], output: str | Path,
                   world_size: int = 8,
                   init_key: str | None = None, init_checkpoint: str | None = None,
                   overrides: list[str] | None = None,
                   recipe: str = "reference",
                   sequence_parallel_size: int | None = None) -> DictConfig:
    if stage not in STAGES:
        raise ConfigError(f"unsupported stage: {stage}")
    if world_size < 1:
        raise ConfigError("world_size must be positive")
    defaults = recipe_defaults(recipe)
    if recipe == "reference":
        config = OmegaConf.merge(_resource("base"), _resource(stage))
    else:
        config = OmegaConf.merge(defaults.common, defaults.stages[stage])
    selected_overrides = []
    for override in overrides or []:
        prefix, dot, tail = override.partition(".")
        if dot and prefix in STAGES:
            if prefix != stage:
                continue
            override = tail
        selected_overrides.append(override)
        name = override.partition("=")[0]
        if "=" not in override or name not in config or name in _PATH_FIELDS:
            raise ConfigError(f"unknown or non-overridable configuration field: {name}")
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(selected_overrides))
    if sequence_parallel_size is not None:
        if (any(item.partition("=")[0] == "sequence_parallel_size"
                for item in selected_overrides)
                and config.sequence_parallel_size != sequence_parallel_size):
            raise ConfigError("--sp conflicts with --set sequence_parallel_size for " + stage)
        config.sequence_parallel_size = sequence_parallel_size
    # An automatic default preserves the recipe's effective batch when SP
    # reduces its independent data replicas. Explicit accumulation is separate.
    if config.gradient_accumulation_steps == "auto":
        config.gradient_accumulation_steps = config.sequence_parallel_size
    required = ["model_root", "prompts"]
    if stage == "s2":
        required += ["paired_train", "paired_val"]
    if not init_checkpoint:
        init_key = init_key or defaults.init_keys[stage]
        required += [init_key]
    missing = [key for key in required if key not in assets]
    if missing:
        raise ConfigError("missing assets: " + ", ".join(missing))
    run_dir = Path(output).expanduser().resolve() / stage
    config.stage = stage
    config.denoising_step_list = parse_denoising_step_list(config.denoising_step_list)
    config.recipe = recipe
    config.world_size = world_size
    sp_size, accumulation = _parallel_sizes(config)
    config.sequence_parallel_size = sp_size
    config.gradient_accumulation_steps = accumulation
    config.data_parallel_size = world_size // sp_size
    config.effective_batch_size = config.data_parallel_size * config.batch_size * accumulation
    sp_suffix = f"_sp{sp_size}" if sp_size > 1 else ""
    config.recipe_id = (f"wf{sp_suffix}_{stage}" if recipe == "reference" else
                        f"wf_14b_fsdp8{sp_suffix}_experimental_"
                        + ("smoke_" if recipe.endswith("-smoke") else "") + stage)
    config.model_root = assets["model_root"]
    config.data_path = assets["prompts"]
    config.generator_ckpt = init_checkpoint or assets[init_key]
    config.init_checkpoint_key = "generator"
    config.logdir = str(run_dir)
    config.wandb_save_dir = str(run_dir)
    config.config_name = config.recipe_id
    config.resume_from = ""
    if stage == "s2":
        config.paired_manifest = assets["paired_train"]
        config.paired_validation_manifest = assets["paired_val"]
    validate_config(config)
    return config


def _parallel_sizes(config: DictConfig) -> tuple[int, int]:
    values = []
    for key in ("sequence_parallel_size", "gradient_accumulation_steps"):
        value = getattr(config, key, 1)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 1 or int(value) != value):
            raise ConfigError(f"{key} must be a positive integer")
        values.append(int(value))
    if config.world_size % values[0]:
        raise ConfigError("world_size must be divisible by sequence_parallel_size")
    return values[0], values[1]


def validate_config(config: DictConfig) -> None:
    if config.stage not in STAGES:
        raise ConfigError("invalid stage")
    steps = parse_denoising_step_list(config.denoising_step_list)
    recipe = getattr(config, "recipe", "reference")
    recipe_defaults(recipe)
    if int(config.world_size) != config.world_size or config.world_size < 1:
        raise ConfigError("world_size must be a positive integer")
    sp_size, accumulation = _parallel_sizes(config)
    dp_size = config.world_size // sp_size
    if getattr(config, "data_parallel_size", dp_size) != dp_size:
        raise ConfigError("data_parallel_size must equal world_size / sequence_parallel_size")
    if getattr(config, "fsdp_init_mode", "replicated") not in ("replicated", "rank0"):
        raise ConfigError("fsdp_init_mode must be replicated or rank0")
    if getattr(config, "ema_mode", "full") not in ("full", "sharded"):
        raise ConfigError("ema_mode must be full or sharded")
    if not isinstance(getattr(config, "profile_memory", False), bool):
        raise ConfigError("profile_memory must be boolean")
    ema_start = config.ema_start_step
    if (isinstance(ema_start, bool) or not isinstance(ema_start, (int, float))
            or not math.isfinite(ema_start) or int(ema_start) != ema_start or ema_start < 0):
        raise ConfigError("ema_start_step must be a nonnegative integer")
    ema_weight = config.ema_weight
    if ema_weight is not None and (
        isinstance(ema_weight, bool) or not isinstance(ema_weight, (int, float))
        or not math.isfinite(ema_weight) or not 0 <= ema_weight < 1
    ):
        raise ConfigError("ema_weight must be null or a finite number in [0, 1); 0 disables EMA")
    if recipe != "reference":
        if config.world_size != 8:
            raise ConfigError("14b-fsdp8 recipes require one node with exactly 8 ranks")
        if any(config[name] != "Wan2.1-T2V-14B"
               for name in ("generator_name", "real_name", "fake_name")):
            raise ConfigError("14b-fsdp8 generator, real score, and fake score must all be 14B")
        if steps != [1000, 750, 500, 250] or not config.warp_denoising_step:
            raise ConfigError("14b-fsdp8 requires the native four-step table with warping")
        if (config.timestep_shift != 5.0 or config.model_kwargs.timestep_shift != 5.0
                or config.num_train_timestep != 1000 or config.ts_schedule):
            raise ConfigError("14b-fsdp8 requires shift 5, 1000 timesteps, and ts_schedule=false")
        if (config.sharding_strategy != "full" or config.fsdp_init_mode != "rank0"
                or config.ema_mode != "sharded"):
            raise ConfigError("14b-fsdp8 requires full FSDP, rank0 initialization, and sharded EMA")
        if config.mixed_precision is not True or config.gradient_checkpointing is not True:
            raise ConfigError("14b-fsdp8 requires mixed precision and gradient checkpointing")
    if config.seed <= 0 or int(config.seed) != config.seed:
        raise ConfigError("seed must be a fixed positive integer (seed=0 is random)")
    if (config.batch_size != 1
            or config.effective_batch_size != dp_size * config.batch_size * accumulation):
        raise ConfigError("recipes require batch_size=1, effective batch=DP * batch * accumulation")
    if "total_batch_size" in config:
        raise ConfigError("total_batch_size was unused; use the explicit effective batch")
    if config.attention_mode not in ("full", "strict"):
        raise ConfigError("attention_mode must be full or strict")
    if config.mix_schedule != "fixed":
        raise ConfigError("this reference release supports fixed mix probabilities")
    probs = list(config.mix_final_probs)
    if len(probs) != 3 or any(not math.isfinite(p) or p < 0 for p in probs) or not math.isclose(sum(probs), 1.0):
        raise ConfigError("mix_final_probs must be three nonnegative probabilities summing to one")
    if config.num_frame_per_block != 3 or list(config.image_or_video_shape) != [1, 21, 16, 60, 104]:
        raise ConfigError("reference recipes require 3-frame blocks and [1,21,16,60,104] input shape")
    if sp_size > 1:
        # Both supported native Wan architectures use spatial 2x2 patches.
        height, width = config.image_or_video_shape[-2:]
        spatial_tokens = (height // 2) * (width // 2)
        if spatial_tokens % sp_size:
            raise ConfigError("spatial tokens per frame must be divisible by sequence_parallel_size")
        model_heads = {"Wan2.1-T2V-1.3B": 12, "Wan2.1-T2V-14B": 40}
        for field in ("generator_name", "fake_name", "real_name"):
            name = config[field]
            heads = model_heads.get(name)
            if heads is None:
                raise ConfigError(f"cannot validate SP attention heads for {field}={name}")
            if heads % sp_size:
                raise ConfigError(
                    f"{field}={name} has {heads} attention heads, which must be "
                    "divisible by sequence_parallel_size"
                )
    if config.no_save:
        raise ConfigError("the reproducible launcher requires checkpoint saving")
    for name in ("max_steps", "log_iters", "resume_save_iters", "dfake_gen_update_ratio"):
        if int(config[name]) != config[name] or config[name] <= 0:
            raise ConfigError(f"{name} must be a positive integer")
    # The trainer always saves a complete final checkpoint, including short
    # smoke runs that end between the periodic save boundaries.
    if config.resume_save_iters % config.log_iters:
        raise ConfigError("resume_save_iters must be a multiple of log_iters")
    if not config.no_visualize:
        raise ConfigError("training previews are not included in this migration")
    if config.i2v or config.load_raw_video or config.independent_first_frame:
        raise ConfigError("this migration supports the reference text-to-video recipes only")
    if config.num_training_frames < 21 or config.num_training_frames % 3:
        raise ConfigError("num_training_frames must be a multiple of 3 and at least 21")
    if not 0 < config.lpips_subset_ratio <= 1 or config.max_context_blocks < 0:
        raise ConfigError("invalid LPIPS subset or clean context size")
    if config.stage == "s2":
        if config.distribution_loss != "paired_coupled_dmd" or not config.paired_only:
            raise ConfigError("S2 requires paired_coupled_dmd and paired_only=true")
        if config.lpips_weight < 0:
            raise ConfigError("LPIPS weight cannot be negative")
        if config.paired_expected_train < config.world_size or config.paired_expected_val < 1:
            raise ConfigError("paired training must supply at least one sample per rank and a validation split")
        available_blocks = int(config.num_training_frames) // int(config.num_frame_per_block)
        if len(steps) >= available_blocks:
            raise ConfigError(
                f"S2 window needs {len(steps)} blocks but the pair only has {available_blocks}"
            )
    elif config.distribution_loss != "dmd" or config.lpips_weight != 0:
        raise ConfigError("S1/S3 require pure DMD with lpips_weight=0")


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing asset file: {path}")
    stat = path.stat()
    if stat.st_size == 0:
        raise ConfigError(f"empty asset file: {path}")
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _paired_record(path: Path, expected: int) -> dict[str, Any]:
    record = _file_record(path)
    entries = []
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            item = Path(row["path"])
        except (ValueError, KeyError, TypeError) as error:
            raise ConfigError(f"invalid pair manifest {path}:{number}") from error
        if not item.is_absolute():
            item = path.parent / item
        item = item.resolve()
        if str(item) in seen:
            raise ConfigError(f"duplicate pair asset: {item}")
        seen.add(str(item))
        if "shape" in row and list(row["shape"]) != [21, 16, 60, 104]:
            raise ConfigError(f"unexpected pair shape in {path}:{number}")
        entries.append(_file_record(item))
    if len(entries) != expected:
        raise ConfigError(f"{path}: expected {expected} pairs, found {len(entries)}")
    record.update(count=len(entries), assets=entries,
                  tensor_shape_contract=[21, 16, 60, 104],
                  tensor_values_checked=False)
    return record


def _native_14b_config(path: Path) -> dict[str, Any]:
    """Check the native architecture without importing torch or reading weights."""
    try:
        native = json.loads(path.read_text())
    except (ValueError, OSError) as error:
        raise ConfigError(f"invalid native Wan model configuration: {path}") from error
    if not isinstance(native, dict):
        raise ConfigError(f"native Wan configuration must be a mapping: {path}")
    expected = {"model_type": "t2v", "dim": 5120, "ffn_dim": 13824,
                "num_heads": 40, "num_layers": 40, "in_dim": 16, "out_dim": 16}
    for name, value in expected.items():
        if native.get(name) != value:
            raise ConfigError(
                f"14B native config {path}: {name} must be {value!r}, got {native.get(name)!r}"
            )
    # Native Wan omits these entries via ignore_for_config; the constructor
    # defaults are part of the checked latent/text contract.
    patch_size = native.get("patch_size", [1, 2, 2])
    if patch_size != [1, 2, 2] or native.get("text_dim", 4096) != 4096:
        raise ConfigError(f"14B native config has incompatible latent patches or text features: {path}")
    return {**expected, "patch_size": patch_size, "text_dim": native.get("text_dim", 4096),
            "latent_shape_contract": [21, 16, 60, 104], "weights_loaded": False}


def preflight(config: DictConfig, *, allow_pending_init: bool = False) -> dict[str, Any]:
    """Static checks only: no CUDA, model loading, or tensor-value verification."""
    validate_config(config)
    records: dict[str, Any] = {}
    root = Path(config.model_root)
    for name in sorted({config.generator_name, config.real_name, config.fake_name}):
        model = root / name
        weights = sorted(model.glob("diffusion_pytorch_model*.safetensors"))
        if not weights:
            raise ConfigError(f"no safetensors model shards in {model}")
        records[name] = {"config": _file_record(model / "config.json"),
                         "weights": [_file_record(p) for p in weights]}
        if getattr(config, "recipe", "reference") != "reference":
            records[name]["architecture"] = _native_14b_config(model / "config.json")
    student = root / config.generator_name
    for name in ("Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"):
        records[name] = _file_record(student / name)
    tokenizer = student / "google/umt5-xxl"
    records["tokenizer_config"] = _file_record(tokenizer / "tokenizer_config.json")
    tokenizer_files = [p for p in (tokenizer / "tokenizer.json", tokenizer / "spiece.model") if p.is_file()]
    if not tokenizer_files:
        raise ConfigError(f"missing tokenizer.json or spiece.model in {tokenizer}")
    records["tokenizer"] = [_file_record(p) for p in tokenizer_files]
    for name in (config.generator_name, config.real_name):
        index = root / name / "diffusion_pytorch_model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            if not weight_map:
                raise ConfigError(f"empty DiT weight map in {index}")
            for shard in set(weight_map.values()):
                if Path(shard).name != shard:
                    raise ConfigError(f"unsafe shard name in {index}: {shard}")
                _file_record(index.parent / shard)
            records[name]["index"] = _file_record(index)
    records["prompts"] = _file_record(Path(config.data_path))
    prompts = [line for line in Path(config.data_path).read_text().splitlines() if line.strip()]
    if len(prompts) < config.world_size:
        raise ConfigError("prompt dataset must have at least world_size nonempty prompts")
    if allow_pending_init and not Path(config.generator_ckpt).is_file():
        records["initial_checkpoint"] = {"path": config.generator_ckpt, "pending_previous_stage": True}
    else:
        records["initial_checkpoint"] = _file_record(Path(config.generator_ckpt))
    if config.stage == "s2":
        records["paired_train"] = _paired_record(Path(config.paired_manifest), config.paired_expected_train)
        records["paired_val"] = _paired_record(Path(config.paired_validation_manifest), config.paired_expected_val)
    versions = dict(sorted(
        (distribution.metadata["Name"].lower().replace("_", "-"), distribution.version)
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    ))
    return {"status": "static_checks_passed", "python": sys.executable,
            "python_version": platform.python_version(), "package_versions": versions,
            "assets": records,
            "limitations": "Tensor shapes/values and GPU imports require the training loader/smoke; large weights are recorded by size and mtime."}


def checkpoint_complete(path: str | Path, config: DictConfig,
                        expected_step: int | None = None) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    marker_path = checkpoint / "resume_complete.json"
    if not marker_path.is_file():
        raise ConfigError(f"incomplete checkpoint (no resume_complete.json): {checkpoint}")
    marker = json.loads(marker_path.read_text())
    if marker.get("world_size") != config.world_size:
        raise ConfigError("resume checkpoint world_size differs from the saved configuration")
    for name in ("sequence_parallel_size", "gradient_accumulation_steps"):
        if marker.get(name, 1) != getattr(config, name, 1):
            raise ConfigError(f"resume checkpoint {name} differs from the saved configuration")
    if expected_step is not None and marker.get("step") != expected_step:
        raise ConfigError("stage endpoint checkpoint has the wrong iteration")
    expected_files = [f"trainer_state_rank{rank:02d}.pt" for rank in range(config.world_size)]
    if marker.get("rank_state_files") != expected_files:
        raise ConfigError("resume checkpoint does not contain the expected rank set")
    _file_record(checkpoint / "model.pt")
    for name in expected_files:
        _file_record(checkpoint / name)
    return marker
