"""Stage 2: teacher-state staircase prediction with coupled DMD and LPIPS.

The configured attention mask applies to the student window. Historical KV
comes from clean teacher blocks; the critic samples another window and noise.
LPIPS uses a half-resolution decode of the window and every third pixel frame.
"""
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from wf_training.model.dmd import DMD
from wf_training.pipeline import RollingForcingTrainingPipeline


class OffPolicyDMD(DMD):
    def __init__(self, args, device):
        super().__init__(args, device)
        assert not getattr(args, "i2v", False), "OffPolicyDMD is t2v-only"
        # DMD v1 用 λ=0.25（class-conditional）/0.5（unconditional）。
        # 旧默认 1.0 太大，会压过 DMD loss 导致 consistency 崩。
        self.lpips_weight = float(getattr(args, "lpips_weight", 0.25))
        # LPIPS 只作用在 batch 的一个子集上（DMD v1 做法），
        # 既省显存又防止过正则化。1.0 = 全 batch。
        self.lpips_subset_ratio = float(getattr(args, "lpips_subset_ratio", 0.25))
        self.lpips_net = None
        # Maximum number of clean teacher blocks retained as context.
        self.max_context_blocks = int(getattr(args, "max_context_blocks", 26))
        self._offp_pipeline = None

    # ------------------------------------------------------------------ util

    def _get_lpips(self):
        if self.lpips_net is None:
            import lpips as lpips_pkg
            self.lpips_net = lpips_pkg.LPIPS(net="vgg", verbose=False)
            self.lpips_net = self.lpips_net.to(self.device)
            for p in self.lpips_net.parameters():
                p.requires_grad_(False)
            self.lpips_net.eval()
        return self.lpips_net

    def _get_pipeline(self):
        if self._offp_pipeline is None:
            num_frames_needed = (self.max_context_blocks +
                                 len(self.denoising_step_list)) * self.num_frame_per_block
            self._offp_pipeline = RollingForcingTrainingPipeline(
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                num_frame_per_block=self.num_frame_per_block,
                independent_first_frame=self.independent_first_frame,
                same_step_across_blocks=self.same_step_across_blocks,
                last_step_only=False,
                num_max_frames=num_frames_needed,
                context_noise=self.args.context_noise,
            )
        return self._offp_pipeline

    def _window_timesteps(self, batch_size, num_window_blocks, device):
        """Deployment staircase: oldest block at the lowest noise level."""
        ts = torch.ones(
            [batch_size, num_window_blocks * self.num_frame_per_block],
            device=device, dtype=torch.float32)
        for index, current_timestep in enumerate(reversed(self.denoising_step_list)):
            ts[:, index * self.num_frame_per_block:
               (index + 1) * self.num_frame_per_block] *= current_timestep
        return ts.long()

    # ------------------------------------------------------- student forward

    def _offpolicy_student_pred(
        self,
        teacher_latent: torch.Tensor,
        conditional_dict: dict,
        require_grad: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build an off-policy window from teacher latents and run the student.

        Returns (pred, teacher_window_clean):
          pred                 [B, W*fpb, C, H, W] student x0 prediction
          teacher_window_clean [B, W*fpb, C, H, W] the clean latents it saw re-noised
        """
        batch_size, total_frames = teacher_latent.shape[:2]
        fpb = self.num_frame_per_block
        num_window_blocks = len(self.denoising_step_list)
        total_blocks = total_frames // fpb
        assert total_blocks > num_window_blocks

        pipe = self._get_pipeline()
        pipe._initialize_kv_cache(
            batch_size=batch_size, dtype=self.dtype, device=self.device)
        pipe._initialize_crossattn_cache(
            batch_size=batch_size, dtype=self.dtype, device=self.device)

        # same window start on all ranks
        max_start = total_blocks - num_window_blocks
        start_block = torch.randint(0, max_start + 1, (1,), device=self.device)
        dist.broadcast(start_block, src=0)
        start_block = start_block.item()

        # KV context: clean teacher blocks at their absolute positions
        ctx_lo = max(0, start_block - self.max_context_blocks)
        with torch.no_grad():
            for b in range(ctx_lo, start_block):
                block = teacher_latent[:, b * fpb:(b + 1) * fpb].to(
                    device=self.device, dtype=self.dtype)
                ctx_ts = torch.zeros(
                    [batch_size, fpb], device=self.device, dtype=torch.long)
                self.generator(
                    noisy_image_or_video=block,
                    conditional_dict=conditional_dict,
                    timestep=ctx_ts,
                    kv_cache=pipe.kv_cache_clean,
                    crossattn_cache=pipe.crossattn_cache,
                    current_start=b * fpb * pipe.frame_seq_length,
                    updating_cache=True,
                )

        # re-noise the window under the deployment staircase
        window_clean = teacher_latent[
            :, start_block * fpb:(start_block + num_window_blocks) * fpb
        ].to(device=self.device, dtype=self.dtype)
        timestep = self._window_timesteps(batch_size, num_window_blocks, self.device)
        noise = torch.randn_like(window_clean)
        noisy_window = self.scheduler.add_noise(
            window_clean.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_window_blocks * fpb))

        ctx = torch.enable_grad() if require_grad else torch.no_grad()
        with ctx:
            _, pred = self.generator(
                noisy_image_or_video=noisy_window,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=pipe.kv_cache_clean,
                crossattn_cache=pipe.crossattn_cache,
                current_start=start_block * fpb * pipe.frame_seq_length,
            )
        return pred, window_clean

    # ------------------------------------------------------------- LPIPS

    def _lpips_anchor(self, pred, window_clean):
        """LPIPS between decoded student pred and decoded teacher clean latents,
        computed on ALL blocks of the window, on a subset of the batch.

        DMD v1 做法：只取 batch 的一个子集（默认 25%）做 LPIPS，
        既省显存又防止过正则化。全窗口 decode 会吃显存，OOM 是正常的，
        说明锚定覆盖了整个窗口而不是只锚开头两帧。
        """
        lp = self._get_lpips()

        # batch subset: 只取部分样本做 LPIPS（DMD v1 用法）
        bs = pred.shape[0]
        n_sub = max(1, int(bs * self.lpips_subset_ratio))
        idx = torch.randperm(bs, device=pred.device)[:n_sub]
        pred = pred[idx]
        window_clean = window_clean[idx]

        if not hasattr(self, "_vae_ready"):
            self.vae = self.vae.to(
                device=self.device,
                dtype=torch.bfloat16 if self.args.mixed_precision else torch.float32)
            self._vae_ready = True

        # E1 修复：latent 先降采样（60x104 -> 30x52）再 decode，
        # VAE decode 输出 240x416，显存减 4 倍（全分辨率 decode 会 OOM，
        # 442MiB 分配失败）。LPIPS 对半分辨率不敏感。
        torch.cuda.empty_cache()
        B, F_, C, H, W = pred.shape
        pred = F.interpolate(
            pred.flatten(0, 1), scale_factor=0.5, mode="bilinear",
            align_corners=False).unflatten(0, (B, F_))
        window_clean = F.interpolate(
            window_clean.flatten(0, 1), scale_factor=0.5, mode="bilinear",
            align_corners=False).unflatten(0, (B, F_))

        # decode 全窗口所有 block（不是只取最旧两块）
        pix_s = self.vae.decode_to_pixel(pred)
        with torch.no_grad():
            pix_t = self.vae.decode_to_pixel(window_clean)

        # [B, T, C, H, W] -> subsample frames, flatten batch（已是半分辨率）
        pix_s = pix_s[:, ::3].flatten(0, 1).float().clamp(-1, 1)
        pix_t = pix_t[:, ::3].flatten(0, 1).float().clamp(-1, 1)
        loss = lp(pix_s, pix_t).mean()
        del pix_s, pix_t
        torch.cuda.empty_cache()
        return loss

    # --------------------------------------------------------------- losses

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        assert clean_latent is not None, \
            "OffPolicyDMD needs teacher latents from the paired manifest"
        pred, window_clean = self._offpolicy_student_pred(
            clean_latent, conditional_dict, require_grad=True)

        dmd_loss, dmd_log = self.compute_distribution_matching_loss(
            image_or_video=pred,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
        )

        if self.lpips_weight > 0:
            lpips_loss = self._lpips_anchor(pred, window_clean)
        else:
            lpips_loss = torch.zeros((), device=self.device)

        loss = dmd_loss + self.lpips_weight * lpips_loss
        dmd_log.update({
            "dmd_loss": dmd_loss.detach(),
            "lpips_loss": lpips_loss.detach()
            if torch.is_tensor(lpips_loss) else torch.tensor(lpips_loss),
        })
        return loss, dmd_log

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        assert clean_latent is not None
        with torch.no_grad():
            generated_image, _ = self._offpolicy_student_pred(
                clean_latent, conditional_dict, require_grad=False)

        batch_size, num_frame = generated_image.shape[:2]
        critic_timestep = self._get_timestep(
            self.min_score_timestep,
            self.num_train_timestep,
            batch_size,
            num_frame,
            self.num_frame_per_block,
            uniform_timestep=True,
        )
        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * (
                critic_timestep / 1000) / (
                1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000
        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frame))

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
        )

        if self.args.denoising_loss_type == "flow":
            from wf_training.utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frame))

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
        )
        return denoising_loss, {"critic_timestep": critic_timestep.detach()}
