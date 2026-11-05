"""Layer partitions of the actual sgldiff Wan2.1 causal decoder.

Each stage owns its cache and processes one *latent* frame per message. Temporal
upsampling can make that message contain several feature frames downstream.
No native-RF VAE implementation or checkpoint is used here.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


def wan_api():
    from sglang.multimodal_gen.runtime.models.vaes import wanvae
    return wanvae


def load_vae(cfg, device):
    """Load only the sgldiff VAE component from the selected Diffusers model."""
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
    from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
    from wave_rt.distributed.compat import _install_sgl_shim

    component_path = Path(cfg.model_path) / "vae"
    if not (component_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing Diffusers VAE component: {component_path}")
    args = ServerArgs.from_dict(dict(
        model_path=cfg.model_path, pipeline_class_name=cfg.pipeline_class_name,
        num_gpus=1, tp_size=1, ulysses_degree=1, ring_degree=1,
        cfg_parallel_degree=1, attention_backend="torch_sdpa",
        use_fsdp_inference=False, vae_cpu_offload=False,
    ))
    args.hsdp_shard_dim = None
    args.pipeline_config.vae_precision = "bf16"
    config = args.pipeline_config.vae_config
    config.use_parallel_encode = False
    config.use_parallel_decode = False
    config.use_parallel_tiling = False
    config.use_feature_cache = True
    set_global_server_args(args)
    _install_sgl_shim()
    vae = VAELoader(device=torch.device(device)).load_customized(str(component_path), args, "vae")
    return vae.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


class CachedBoundaryConv(nn.Module):
    """WanDecoder3d handles its boundary conv caches outside the conv module."""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        api = wan_api()
        cache, index = api.feat_cache.get(), api.feat_idx.get()
        saved = x[:, :, -api.CACHE_T:].clone()
        if saved.shape[2] < 2 and cache[index] is not None:
            saved = torch.cat((cache[index][:, :, -1:].to(saved.device), saved), dim=2)
        result = self.conv(x, cache[index])
        cache[index] = saved
        api.feat_idx.set(index + 1)
        return result


def decoder_units(vae):
    """Flatten only sequential Wan2.1 containers; keep residual units intact."""
    api = wan_api()
    if not vae.use_feature_cache or vae.config.patch_size is not None:
        raise ValueError("Streaming comparison requires Wan2.1 causal feature cache")
    decoder = vae.decoder
    if decoder.use_parallel_decode:
        raise ValueError("VAE pipeline stages require local, unsharded decoder units")
    units = [("conv_in", CachedBoundaryConv(decoder.conv_in))]
    mid = decoder.mid_block
    units.append(("mid.resnets.0", mid.resnets[0]))
    for i, attention in enumerate(mid.attentions):
        units.extend(((f"mid.attentions.{i}", attention),
                      (f"mid.resnets.{i + 1}", mid.resnets[i + 1])))
    for i, block in enumerate(decoder.up_blocks):
        if not isinstance(block, api.WanUpBlock):
            raise ValueError("Only Wan2.1 sequential up blocks are supported")
        units.extend((f"up.{i}.resnets.{j}", layer) for j, layer in enumerate(block.resnets))
        units.extend((f"up.{i}.upsamplers.{j}", layer)
                     for j, layer in enumerate(block.upsamplers or ()))
    units.extend((("norm_out", decoder.norm_out), ("activation", decoder.nonlinearity),
                  ("conv_out", CachedBoundaryConv(decoder.conv_out))))
    return units


class DecoderStage(nn.Module):
    def __init__(self, units):
        super().__init__()
        if not units:
            raise ValueError("Decoder stage must own at least one unit")
        self.layers = nn.ModuleList([layer for _, layer in units])
        # Counting convolutions is an upper bound: shortcuts do not consume slots.
        api = wan_api()
        self.cache_slots = sum(isinstance(m, api.WanCausalConv3d) for m in self.layers.modules())
        self.reset()

    def reset(self):
        self.cache = [None] * self.cache_slots
        self.frame_index = 0

    def forward(self, x):
        api = wan_api()
        # Restore context variables even on failure; multiple stages can be tested
        # in one process without stealing each other's first-frame/cache state.
        variables = (api.feat_cache, api.feat_idx, api.first_chunk, api.is_first_frame)
        tokens = [v.set(value) for v, value in zip(
            variables, (self.cache, 0, self.frame_index == 0, False))]
        try:
            for layer in self.layers:
                x = layer(x)
        finally:
            for variable, token in reversed(list(zip(variables, tokens))):
                variable.reset(token)
        self.frame_index += 1
        return x


def unscale_latents(vae, latents):
    """Same normalization as sgldiff DecodingStage, before post_quant_conv."""
    arch = vae.config.arch_config
    # Preserve upstream division and cast order: multiplying by a BF16 std
    # instead changes rounding and would invalidate decoder parity checks.
    scale = arch.scaling_factor.to(device=latents.device, dtype=latents.dtype)
    shift = arch.shift_factor.to(device=latents.device, dtype=latents.dtype)
    return latents / scale + shift


def balanced_partition(costs, stages):
    """Exact contiguous minimax partition, with no empty stage."""
    n = len(costs)
    if not 1 <= stages <= n or any(not math.isfinite(c) or c < 0 for c in costs):
        raise ValueError("Invalid stage count or decoder cost")
    prefix = [0.0]
    for cost in costs:
        prefix.append(prefix[-1] + cost)
    dp = [[math.inf] * (n + 1) for _ in range(stages + 1)]
    split = [[0] * (n + 1) for _ in range(stages + 1)]
    dp[0][0] = 0.0
    for count in range(1, stages + 1):
        for end in range(count, n + 1):
            best, start = min((max(dp[count - 1][k], prefix[end] - prefix[k]), k)
                              for k in range(count - 1, end))
            dp[count][end], split[count][end] = best, start
    cuts = [n]
    for count in range(stages, 0, -1):
        cuts.append(split[count][cuts[-1]])
    return list(reversed(cuts))


@torch.no_grad()
def unit_flops(vae, latent_hw):
    """Profile steady causal frames, including temporal upsampling and attention.

    The first latent frame is exceptional (one decoded frame instead of four).
    Count the second frame so the partition reflects the long-video workload.
    Hooks and feature caches are scoped to this probe, outside measurement.
    """
    api = wan_api()
    units = decoder_units(vae)
    stage = DecoderStage(units)
    costs = [0.0] * len(units)
    handles = []

    def hook_for(index):
        def hook(module, inputs, output):
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                costs[index] += 2 * output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            elif isinstance(module, nn.Linear):
                costs[index] += 2 * output.numel() * module.in_features
            elif isinstance(module, api.WanAttentionBlock):
                b, c, t, h, w = inputs[0].shape
                costs[index] += 4 * b * t * (h * w) ** 2 * c
        return hook

    parameter = next(vae.parameters())
    z = torch.zeros((1, vae.z_dim, 1, *latent_hw), device=parameter.device, dtype=parameter.dtype)
    try:
        stage(vae.post_quant_conv(unscale_latents(vae, z)))
        for i, (_, layer) in enumerate(units):
            for module in layer.modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, api.WanAttentionBlock)):
                    handles.append(module.register_forward_hook(hook_for(i)))
        stage(vae.post_quant_conv(unscale_latents(vae, z)))
    finally:
        for handle in handles:
            handle.remove()
        stage.reset()
    return units, costs


@torch.no_grad()
def unit_times(vae, latent_hw, warmup=4, repeats=8):
    """CUDA-event costs per steady latent frame; call on one rank then broadcast."""
    units = decoder_units(vae)
    stage = DecoderStage(units)
    parameter = next(vae.parameters())
    z = torch.zeros((1, vae.z_dim, 1, *latent_hw), device=parameter.device, dtype=parameter.dtype)
    x = vae.post_quant_conv(unscale_latents(vae, z))
    costs = [0.0] * len(units)
    events = []
    handles = []
    try:
        for _ in range(warmup):
            stage(x)
        # Hooks bracket each atomic unit while DecoderStage owns upstream context.
        for _, layer in units:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            events.append((begin, end))
            handles.append(layer.register_forward_pre_hook(lambda m, a, e=begin: e.record()))
            handles.append(layer.register_forward_hook(lambda m, a, o, e=end: e.record()))
        for _ in range(repeats):
            stage(x)
            torch.cuda.synchronize(parameter.device)
            for i, (begin, end) in enumerate(events):
                costs[i] += begin.elapsed_time(end) / repeats
    finally:
        for handle in handles:
            handle.remove()
        stage.reset()
    return units, costs
