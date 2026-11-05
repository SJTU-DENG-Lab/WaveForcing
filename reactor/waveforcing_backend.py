"""Incremental execution of WaveForcing's released single-GPU RF baseline.

The mixed-noise window, warped schedule, per-block renoising and clean-KV
commit follow naive_rt/rolling_forcing/pipeline/rolling_forcing_inference.py.
Only finalized blocks are decoded; the native causal VAE keeps its context.
"""

from __future__ import annotations

from contextlib import chdir
from dataclasses import dataclass, field

import numpy as np

from waveforcing_model import BLOCK_FRAMES, Settings


@dataclass
class Run:
    prompt: str
    conditioning: dict
    initial_rng: object
    page: object | None = None
    page_offset: int = 0
    pending: list = field(default_factory=list)
    tick: int = 0


class WaveForcingBackend:
    def __init__(self, settings: Settings) -> None:
        import torch
        from naive_rt.rolling_forcing.pipeline.rolling_forcing_inference import (
            CausalInferencePipeline,
        )
        from omegaconf import OmegaConf

        from waveforcing_rope import install

        if settings.noise_page_frames < 12 or settings.noise_page_frames % BLOCK_FRAMES:
            raise ValueError("Noise pages must contain complete native blocks")
        torch.cuda.set_device(0)
        self.settings = settings
        self.device = torch.device("cuda:0")
        cfg_root = settings.source / "naive_rt/rolling_forcing/configs"
        config = OmegaConf.merge(
            OmegaConf.load(cfg_root / "default_config.yaml"),
            OmegaConf.load(cfg_root / "rolling_forcing_dmd.yaml"),
        )
        config.denoising_step_list = [1000, 750, 500, 250]
        with chdir(settings.weights):
            self.pipeline = CausalInferencePipeline(config, device=self.device)
        self.pipeline.to(
            dtype=torch.bfloat16, device=self.device
        ).eval().requires_grad_(False)
        # The base config and released EMA share native Wan keys. No teacher or
        # training checkpoint is loaded: the prepared model path points to EMA.
        from safetensors import safe_open

        checkpoint = (
            settings.weights
            / "wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
        )
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            if set(weights.keys()) != set(self.pipeline.generator.model.state_dict()):
                raise ValueError("Released EMA keys do not match the native model")
        install()
        self.schedule = self.pipeline.denoising_step_list.to(self.device)
        self.shared_timestep = (
            self.schedule.flip(0).repeat_interleave(BLOCK_FRAMES)[None].float()
        )
        self.run: Run | None = None

    def reset(self) -> None:
        self.run = None
        self.pipeline.kv_cache_clean = None
        self.pipeline.crossattn_cache = None
        self.pipeline.vae.model.clear_cache()

    def start(self, prompt: str, *, seed: int) -> None:
        import torch

        with torch.no_grad():
            self.reset()
            self.pipeline._initialize_kv_cache(1, torch.bfloat16, self.device)
            self.pipeline._initialize_crossattn_cache(1, torch.bfloat16, self.device)
            self.run = Run(
                prompt=prompt,
                conditioning=self.pipeline.text_encoder(text_prompts=[prompt]),
                initial_rng=torch.Generator(device=self.device).manual_seed(seed),
            )
            # Initial noise has a separate stream, matching the released naive
            # reference's explicit generator. Renoising uses its own seed stream.
            torch.cuda.manual_seed(seed + 1)

    def _noise(self):
        import torch

        run = self.run
        assert run is not None
        if run.page is None or run.page_offset == self.settings.noise_page_frames:
            run.page = torch.randn(
                (1, 16, self.settings.noise_page_frames, 60, 104),
                generator=run.initial_rng,
                device=self.device,
                dtype=torch.bfloat16,
            ).permute(0, 2, 1, 3, 4)
            run.page_offset = 0
        block = run.page[:, run.page_offset : run.page_offset + BLOCK_FRAMES]
        run.page_offset += BLOCK_FRAMES
        return block

    def _update_prompt(self, prompt: str) -> None:
        run = self.run
        assert run is not None
        if prompt == run.prompt:
            return
        conditioning = self.pipeline.text_encoder(text_prompts=[prompt])
        for block in self.pipeline.crossattn_cache:
            block["is_init"] = False
        run.conditioning, run.prompt = conditioning, prompt

    def _advance(self, prompt: str):
        """Return one finalized native latent block, retaining the noisy frontier."""
        import torch

        run, pipe = self.run, self.pipeline
        assert run is not None
        self._update_prompt(prompt)
        with torch.no_grad():
            while True:
                start_block = max(0, run.tick - len(self.schedule) + 1)
                noisy = torch.cat([*run.pending, self._noise()], dim=1)
                frames = noisy.shape[1]
                timestep = self.shared_timestep[:, -frames:]
                _, prediction = pipe.generator(
                    noisy_image_or_video=noisy,
                    conditional_dict=run.conditioning,
                    timestep=timestep,
                    kv_cache=pipe.kv_cache_clean,
                    crossattn_cache=pipe.crossattn_cache,
                    current_start=start_block * BLOCK_FRAMES * pipe.frame_seq_length,
                )
                pending = []
                for index in range(frames // BLOCK_FRAMES):
                    step = (
                        timestep[:, index * BLOCK_FRAMES : (index + 1) * BLOCK_FRAMES]
                        .mean()
                        .item()
                    )
                    stage = int(
                        torch.nonzero(
                            torch.abs(self.schedule - step) < 1e-4, as_tuple=True
                        )[0].item()
                    )
                    if stage == len(self.schedule) - 1:
                        continue
                    # Preserve the reference's full-window random draw for EACH
                    # block before slicing; drawing only a block changes RNG.
                    renoised = pipe.scheduler.add_noise(
                        prediction.flatten(0, 1),
                        torch.randn_like(prediction.flatten(0, 1)),
                        self.schedule[stage + 1]
                        * torch.ones((frames,), device=self.device, dtype=torch.long),
                    ).unflatten(0, prediction.shape[:2])
                    pending.append(
                        renoised[
                            :, index * BLOCK_FRAMES : (index + 1) * BLOCK_FRAMES
                        ].clone()
                    )
                stable = prediction[:, :BLOCK_FRAMES].clone()
                pipe.generator(
                    noisy_image_or_video=stable,
                    conditional_dict=run.conditioning,
                    timestep=torch.zeros_like(timestep[:, :BLOCK_FRAMES]),
                    kv_cache=pipe.kv_cache_clean,
                    crossattn_cache=pipe.crossattn_cache,
                    current_start=start_block * BLOCK_FRAMES * pipe.frame_seq_length,
                    updating_cache=True,
                )
                run.pending = pending
                run.tick += 1
                if run.tick >= len(self.schedule):
                    return stable

    def _decode(self, latents) -> np.ndarray:
        import torch

        with torch.no_grad():
            pixels = self.pipeline.vae.decode_to_pixel(latents, use_cache=True)
            pixels = (pixels * 0.5 + 0.5).clamp(0, 1)
            return (
                (pixels[0].permute(0, 2, 3, 1) * 255)
                .to(torch.uint8)
                .contiguous()
                .cpu()
                .numpy()
            )

    def step(self, prompt: str) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self._decode(self._advance(prompt))
