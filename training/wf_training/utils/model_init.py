"""Construct one model at a time, with checkpoint I/O on rank zero only."""
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import init_empty_weights

from wf_training.utils.checkpoint import (
    generator_state, load_model_checkpoint, validate_load,
)
from wf_training.utils.distributed import fsdp_wrap
from wf_training.utils.memory import memory_scope


def build_rank0_model(factory, *, state_loader=None, trainable=False, **wrap_kwargs):
    """Build CPU weights on rank 0 and parameter-only meta replicas elsewhere.

    ``factory(meta)`` must leave ordinary tensors (notably complex RoPE
    frequencies) initialized. File/config errors are exchanged before any rank
    enters FSDP's parameter broadcasts, so peers do not wait for missing weights.
    """
    rank = dist.get_rank()
    module = None
    error = None
    try:
        with init_empty_weights(include_buffers=False) if rank else nullcontext():
            module = factory(rank != 0)
        if rank == 0 and state_loader is not None:
            state = state_loader()
            validate_load(module, state)
            module.load_state_dict(state, strict=True)
            del state
        module.requires_grad_(trainable)
        # FP32 master parameters are retained. FSDP MixedPrecision controls
        # compute dtype, not the precision of Adam's parameters and moments.
        for name, parameter in module.named_parameters():
            if parameter.dtype != torch.float32:
                raise ValueError(f"Expected FP32 master parameter {name}: {parameter.dtype}")
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError("Model initialization failed: " + "; ".join(e for e in errors if e))
    return fsdp_wrap(module, sync_module_states=True, **wrap_kwargs)


def initialize_rank0_models(owner, config, device):
    # Import lazily to keep the construction helper testable without GPU kernels.
    from wf_training.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

    resume = getattr(config, "resume_from", "") or ""
    checkpoint_path = (str(Path(resume) / "model.pt") if resume
                       else getattr(config, "generator_ckpt", ""))

    def load_generator():
        return generator_state(load_model_checkpoint(checkpoint_path))

    def load_critic():
        checkpoint = load_model_checkpoint(checkpoint_path)
        state = checkpoint.get("critic")
        if not isinstance(state, dict) or not state:
            raise ValueError(f"Resume checkpoint has no critic: {checkpoint_path}")
        return state

    for role, name, causal, trainable, strategy, loader in (
        ("generator", owner.generator_name, True, True,
         config.generator_fsdp_wrap_strategy, load_generator if checkpoint_path else None),
        ("real_score", owner.real_model_name, False, False,
         config.real_score_fsdp_wrap_strategy, None),
        ("fake_score", owner.fake_model_name, False, True,
         config.fake_score_fsdp_wrap_strategy, load_critic if resume else None),
    ):
        def factory(meta, name=name, causal=causal, loader=loader, trainable=trainable):
            kwargs = dict(getattr(config, "model_kwargs", {})) if causal else {}
            module = WanDiffusionWrapper(
                model_name=name, is_causal=causal,
                load_pretrained=not meta and loader is None, **kwargs,
            )
            if causal:
                module.model.num_frame_per_block = config.num_frame_per_block
                module.model.independent_first_frame = config.independent_first_frame
            if trainable and config.gradient_checkpointing:
                module.enable_gradient_checkpointing()
            return module

        with memory_scope(config, f"initialize/{role}", device=device):
            module = build_rank0_model(
                factory, state_loader=loader, trainable=trainable,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision, wrap_strategy=strategy,
            )
            setattr(owner, role, module)

    with memory_scope(config, "initialize/text_encoder", device=device):
        owner.text_encoder = build_rank0_model(
            lambda meta: WanTextEncoder(
                model_name=owner.generator_name, load_pretrained=not meta,
                init_device="meta" if meta else "cpu",
            ),
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
        )
    with memory_scope(config, "initialize/vae", device=device):
        owner.vae = WanVAEWrapper(model_name=owner.generator_name).requires_grad_(False)
    owner.fsdp_initialized = True
