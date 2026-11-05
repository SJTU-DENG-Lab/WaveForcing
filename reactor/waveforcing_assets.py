"""Prepare pinned inference assets outside the model worker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from waveforcing_model import Settings

SOURCE_REVISION = "fde159c60173be612a447ca1062ceb98fc0bb092"
BASE_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
WAVE_REVISION = "ce9dddcb00484b906dac118b991aa188c389afb0"
BASE_FILES = ("config.json", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth")
PROVENANCE = {
    "source": SOURCE_REVISION,
    "base": BASE_REVISION,
    "wave": WAVE_REVISION,
    "variant": "4step",
}


def prepare(config_path: Path, weights: Path) -> Settings:
    config = yaml.safe_load(config_path.read_text()) or {}
    source = Path(config["source_path"]).expanduser()
    if not source.is_absolute():
        source = config_path.parent / source
    source = source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"Expected WaveForcing source {SOURCE_REVISION}, got {revision}"
        )
    prepare_weights(weights.resolve())
    return Settings(source=source, weights=weights.resolve())


def prepare_weights(weights: Path) -> None:
    """Reuse a complete read-only layout; fetch only the served EMA and encoders."""
    marker = weights / "waveforcing-assets.json"
    model = weights / "wan_models/Wan2.1-T2V-1.3B"
    required = [
        model / n
        for n in (
            *BASE_FILES,
            "diffusion_pytorch_model.safetensors",
            "google/umt5-xxl/tokenizer_config.json",
            "google/umt5-xxl/spiece.model",
        )
    ]
    if (
        marker.is_file()
        and json.loads(marker.read_text()) == PROVENANCE
        and all(p.is_file() for p in required)
    ):
        return
    from huggingface_hub import snapshot_download

    base = Path(
        snapshot_download(
            "Wan-AI/Wan2.1-T2V-1.3B",
            revision=BASE_REVISION,
            allow_patterns=[*BASE_FILES, "google/umt5-xxl/*"],
            cache_dir=weights / "hub",
        )
    )
    wave = Path(
        snapshot_download(
            "SJTU-DENG-Lab/WaveForcing-1.3B",
            revision=WAVE_REVISION,
            allow_patterns=["4step/model.safetensors", "4step/configuration.json"],
            cache_dir=weights / "hub",
        )
    )
    model.mkdir(parents=True, exist_ok=True)
    links = {n: base / n for n in BASE_FILES}
    links["google"] = base / "google"
    links["diffusion_pytorch_model.safetensors"] = wave / "4step/model.safetensors"
    for name, target in links.items():
        path = model / name
        if path.is_symlink() and path.resolve() == target.resolve():
            continue
        if path.exists() or path.is_symlink():
            raise ValueError(f"Asset destination already exists: {path}")
        path.symlink_to(target.resolve(), target_is_directory=target.is_dir())
    marker.write_text(json.dumps(PROVENANCE, indent=2) + "\n")
