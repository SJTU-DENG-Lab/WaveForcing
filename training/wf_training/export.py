"""Export explicitly selected generator weights, without importing a runtime."""

import argparse
import json
from pathlib import Path


def generator_state(checkpoint, weights):
    """Return native Wan keys; never silently substitute raw for EMA."""
    import torch

    key = {"raw": "generator", "ema": "generator_ema"}[weights]
    state = checkpoint.get(key)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint has no nonempty {key} state")
    exported = {}
    for original, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Non-tensor generator entry: {original}")
        name = original.replace("_fsdp_wrapped_module.", "")
        if name.startswith("model."):
            name = name[len("model."):]
        if name in exported:
            raise ValueError(f"Duplicate key after stripping wrapper: {name}")
        exported[name] = tensor
    return exported


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32", "preserve"), default="bfloat16")
    parser.add_argument("--config", help="Resolved stage configuration; auto-detected beside the checkpoint")
    args = parser.parse_args(argv)

    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import save_file

    source = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Export directory already exists: {output}")
    config_path = Path(args.config) if args.config else source.parent.parent / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError("Pass --config with the checkpoint's resolved training configuration")
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    checkpoint = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    state = generator_state(checkpoint, args.weights)
    dtype = getattr(torch, args.dtype) if args.dtype != "preserve" else None
    state = {key: (value.to(dtype) if dtype is not None else value).contiguous().clone()
             for key, value in state.items()}
    metadata = {
        "format": "native_wan",
        "weights": args.weights,
        "dtype": args.dtype,
        "source_checkpoint": str(source),
        "source_configuration": config,
        "tensor_count": len(state),
        "parameters": sum(t.numel() for t in state.values()),
        "note": "DiT only; text encoder, tokenizer and VAE remain external base-model assets.",
    }
    output.mkdir(parents=True)
    temporary = output / "model.safetensors.tmp"
    save_file(state, str(temporary), metadata={"format": "pt", "weights": args.weights})
    temporary.replace(output / "model.safetensors")
    (output / "model_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), "weights": args.weights, "tensor_count": len(state)}))


if __name__ == "__main__":
    main()
