from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from wf_training.pipeline import RollingForcingTrainingPipeline
from wf_training.utils.loss import get_denoising_loss
from wf_training.utils.sequence_parallel import sp_broadcast
from wf_training.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
        self.generator_name = getattr(args, "generator_name", "Wan2.1-T2V-1.3B")

        if getattr(args, "fsdp_init_mode", "replicated") == "rank0":
            from wf_training.utils.model_init import initialize_rank0_models
            initialize_rank0_models(self, args, device)
            self.scheduler = self.generator.get_scheduler()
            self.scheduler.timesteps = self.scheduler.timesteps.to(device)
            return

        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}),
            model_name=self.generator_name,
            is_causal=True,
        )
        self.generator.model.requires_grad_(True)

        self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
        self.real_score.model.requires_grad_(False)
        self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, is_causal=False)
        self.fake_score.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder(model_name=self.generator_name)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(model_name=self.generator_name)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return sp_broadcast(timestep)
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return sp_broadcast(timestep)


class RollingForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=sp_broadcast(torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype)),
            **conditional_dict,
        )
        # Slice last 21 frames
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                # Deccode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        import wf_training.wan.modules.causal_model as causal_model
        if not hasattr(self, "_mask_default_flag"):
            # Snapshot import/env default once; trainer restores to this after backward.
            self._mask_default_flag = causal_model._RF_BLOCK_CAUSAL

        # Mix RF / CRF / SF. Default uniform (1/3 each); optional cosine
        # annealing from mix_initial_probs (uniform when omitted) to
        # mix_final_probs over mix_total_steps.
        #   0: RF  — full-window rolling forcing
        #   1: CRF — block-causal rolling forcing
        #   2: SF  — self forcing (full-window)
        probs = self._current_mix_probs()
        path = torch.multinomial(probs, 1)
        dist.broadcast(path, src=0)
        path = int(path.item())
        # 0=RF, 1=CRF, 2=SF — trainer logs this to wandb/TB
        self._last_train_path = path
        self._last_mix_probs = [float(p) for p in probs.tolist()]

        # Hold the per-step mask through backward: grad-checkpoint recomputes
        # branch on _RF_BLOCK_CAUSAL. Trainer restores via _mask_restore_flag.
        use_causal = path == 1
        causal_model._RF_BLOCK_CAUSAL = use_causal
        causal_model._RF_USE_FLEX = (
            use_causal
            and causal_model._RF_FLEX_ATTN
            and causal_model.FLEX_ATTENTION_AVAILABLE
        )
        self._mask_restore_flag = self._mask_default_flag

        if path == 0:
            return self.inference_pipeline.inference_with_rolling_forcing(
                noise=noise, **conditional_dict
            )
        elif path == 1:
            return self.inference_pipeline.inference_with_causal_rolling_forcing(
                noise=noise, **conditional_dict
            )
        else:
            return self.inference_pipeline.inference_with_self_forcing(
                noise=noise, **conditional_dict
            )

    def _current_mix_probs(self) -> torch.Tensor:
        """RF/CRF/SF sampling probabilities for the current train step.

        schedule "uniform" (default): constant 1/3 each.
        schedule "fixed": constant `mix_final_probs`.
        schedule "cosine": anneal from `mix_initial_probs` (defaulting to
        1/3 each for backward compatibility) to `mix_final_probs` over
        `mix_total_steps` steps with a half-cosine, then hold final.
        Deterministic in the step so all ranks agree; the actual path draw
        is done on rank 0 and broadcast by the caller.
        """
        import math

        start = getattr(self.args, "mix_initial_probs", None)
        if start is None:
            start = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        start = tuple(float(p) for p in start)
        schedule = getattr(self.args, "mix_schedule", "uniform") or "uniform"
        final = getattr(self.args, "mix_final_probs", None)
        if final is None:
            final = (0.65, 0.15, 0.20)  # RF, CRF, SF
        final = tuple(float(p) for p in final)
        for label, probs in (("mix_initial_probs", start), ("mix_final_probs", final)):
            if len(probs) != 3 or any(p < 0.0 for p in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-6):
                raise ValueError(f"{label} must contain three non-negative probabilities summing to 1, got {probs}")
        total = int(getattr(self.args, "mix_total_steps", 2000) or 2000)
        if schedule == "uniform":
            probs = start
        elif schedule == "fixed":
            probs = final
        elif schedule == "cosine":
            step = int(getattr(self, "_global_train_step", 0))
            t = min(max(step, 0), total) / max(total, 1)
            w = 0.5 * (1.0 + math.cos(math.pi * t))
            probs = tuple(f + (s - f) * w for s, f in zip(start, final))
        else:
            raise ValueError(f"unknown mix_schedule: {schedule}")
        return torch.tensor(probs, device=self.device, dtype=torch.float32)

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = RollingForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise
        )
