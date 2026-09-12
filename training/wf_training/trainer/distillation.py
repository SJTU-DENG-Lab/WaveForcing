import gc
import json
import logging
import random

from wf_training.utils.dataset import cycle
from wf_training.utils.dataset import TextDataset
from wf_training.utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from wf_training.utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from wf_training.model import DMD, OffPolicyDMD
import torch
import numpy as np
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self.data_batches_seen = 0
        self.paired_batches_seen = 0
        self.resume_from = getattr(config, "resume_from", "") or ""

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = getattr(config, "disable_wandb", False)
        self._wandb = None

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(
                log_dir=os.path.join(config.logdir, "tensorboard"),
                flush_secs=10
            )
            self._wandb = None
            if not self.disable_wandb:
                import wandb

                # GPU nodes have no internet → default offline (sync later from CPU).
                # Override with WANDB_MODE=online when the cluster can reach wandb.
                wandb_mode = os.environ.get("WANDB_MODE", "offline")
                wandb_dir = getattr(config, "wandb_save_dir", "") or config.logdir
                project = (
                    getattr(config, "wandb_project", None)
                    or os.environ.get("WANDB_PROJECT", "rolling-forcing")
                )
                entity = (
                    getattr(config, "wandb_entity", None)
                    or os.environ.get("WANDB_ENTITY")
                )
                key = getattr(config, "wandb_key", None)
                host = getattr(config, "wandb_host", None)
                if key and host:
                    wandb.login(host=host, key=key)
                self._wandb = wandb
                init_kwargs = dict(
                    config=OmegaConf.to_container(config, resolve=True),
                    name=getattr(config, "config_name", None) or os.path.basename(config.logdir),
                    mode=wandb_mode,
                    project=project,
                    dir=wandb_dir,
                )
                if entity:
                    init_kwargs["entity"] = entity
                run_id = os.environ.get("WANDB_RUN_ID")
                if run_id:
                    init_kwargs["id"] = run_id
                    init_kwargs["resume"] = os.environ.get(
                        "WANDB_RESUME", "allow")
                wandb.init(**init_kwargs)

        self.output_path = config.logdir

        # The public stages share DMD and differ in their source of student states.
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "paired_coupled_dmd" and config.paired_only:
            self.model = OffPolicyDMD(config, device=self.device)
        else:
            raise ValueError("Supported objectives: dmd or paired_coupled_dmd with paired_only=true")

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        if self.model.real_score is not None:
            self.model.real_score = fsdp_wrap(
                self.model.real_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.real_score_fsdp_wrap_strategy
            )

        if self.model.fake_score is not None:
            self.model.fake_score = fsdp_wrap(
                self.model.fake_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.fake_score_fsdp_wrap_strategy
            )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        needs_training_vae = (
            int(getattr(config, "num_training_frames", 21)) > 21
            or config.distribution_loss == "paired_coupled_dmd"
        )
        if needs_training_vae:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=self.dtype)

        if needs_training_vae:
            vae_param = next(self.model.vae.model.parameters())
            expected_device = torch.device("cuda", self.device)
            if vae_param.device != expected_device or vae_param.dtype != self.dtype:
                raise RuntimeError(
                    "Training VAE placement mismatch: "
                    f"got device={vae_param.device} dtype={vae_param.dtype}, "
                    f"expected device={expected_device} dtype={self.dtype}"
                )

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = None
        if self.model.fake_score is not None:
            self.critic_optimizer = torch.optim.AdamW(
                [param for param in self.model.fake_score.parameters()
                 if param.requires_grad],
                lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
                betas=(config.beta1_critic, config.beta2_critic),
                weight_decay=config.weight_decay
            )

        # Step 3: Initialize the dataloader
        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self._dataloader_len = len(dataloader)
        self.dataloader = cycle(dataloader)

        self.paired_dataloader = None
        self._paired_dataloader_len = 0
        if config.distribution_loss == "paired_coupled_dmd":
            from wf_training.utils.paired_dmd_dataset import PairedDMDLatentDataset

            paired_dataset = PairedDMDLatentDataset(config.paired_manifest)
            paired_sampler = torch.utils.data.distributed.DistributedSampler(
                paired_dataset, shuffle=True, drop_last=True)
            paired_dataloader = torch.utils.data.DataLoader(
                paired_dataset,
                batch_size=config.batch_size,
                sampler=paired_sampler,
                num_workers=int(getattr(config, "paired_num_workers", 2)),
                pin_memory=True,
            )
            self._paired_dataloader_len = len(paired_dataloader)
            self.paired_dataloader = cycle(paired_dataloader)
            if dist.get_rank() == 0:
                print("PAIRED DATASET SIZE %d" % len(paired_dataset))

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # Stage initialization uses raw G; resume additionally restores all training state.
        checkpoint_path = (
            os.path.join(self.resume_from, "model.pt")
            if self.resume_from
            else getattr(config, "generator_ckpt", "")
        )
        if checkpoint_path:
            print(f"Loading pretrained generator from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "generator" in checkpoint:
                state_dict = checkpoint["generator"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            state_dict = {
                key.replace("_fsdp_wrapped_module.", "")
                .replace("_checkpoint_wrapped_module.", "")
                .replace("_orig_mod.", ""): value
                for key, value in state_dict.items()
            }
            native_wan_prefixes = (
                "patch_embedding.",
                "text_embedding.",
                "time_embedding.",
                "time_projection.",
                "blocks.",
                "head.",
            )
            if (
                state_dict
                and not any(key.startswith("model.") for key in state_dict)
                and any(key.startswith(native_wan_prefixes) for key in state_dict)
            ):
                state_dict = {
                    f"model.{key}": value for key, value in state_dict.items()
                }
                print(
                    "Restored WanDiffusionWrapper 'model.' prefix for exported native Wan weights"
                )
            self.model.generator.load_state_dict(state_dict, strict=True)
            if self.resume_from:
                if self.model.fake_score is not None:
                    critic_state = checkpoint.get("critic")
                    if critic_state is None:
                        raise KeyError(
                            f"resume checkpoint has no critic state: {checkpoint_path}")
                    self.model.fake_score.load_state_dict(
                        critic_state, strict=True)
                self._load_resume_state(self.resume_from)
                self._restore_ema(checkpoint.get("generator_ema"))

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def _restore_ema(self, ema_state):
        """Restore EMA, including legacy snapshots at or before EMA initialization."""
        enabled = self.config.ema_weight is not None and self.config.ema_weight > 0
        if ema_state is not None:
            if not enabled:
                raise ValueError("resume contains EMA but ema_weight is disabled")
            self.generator_ema.load_state_dict(ema_state)
        elif not enabled or self.step < self.config.ema_start_step:
            self.generator_ema = None
        elif self.step == self.config.ema_start_step:
            # The reference creates EMA from current G at this boundary, but
            # omitted it from model.pt until the next checkpoint.
            self.generator_ema = EMA_FSDP(
                self.model.generator, decay=self.config.ema_weight)
        else:
            raise KeyError("resume checkpoint has no generator_ema after ema_start_step")

    def save(self, *, force_resume=False):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = (
            fsdp_state_dict(self.model.fake_score)
            if self.model.fake_score is not None else None
        )

        state_dict = {"generator": generator_state_dict, "critic": critic_state_dict}
        if self.generator_ema is not None:
            state_dict["generator_ema"] = self.generator_ema.state_dict()

        checkpoint_dir = os.path.join(
            self.output_path, f"checkpoint_model_{self.step:06d}")
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = os.path.join(checkpoint_dir, "model.pt")
            tmp_model_path = model_path + ".tmp"
            torch.save(state_dict, tmp_model_path)
            os.replace(tmp_model_path, model_path)
            print("Model saved to", model_path)
        dist.barrier()

        resume_save_iters = int(
            getattr(self.config, "resume_save_iters", 0) or 0)
        if force_resume or (resume_save_iters > 0 and self.step % resume_save_iters == 0):
            self._save_resume_state(checkpoint_dir)
            self._last_full_saved_step = self.step

    def _next_batch(self):
        batch = next(self.dataloader)
        self.data_batches_seen += 1
        return batch

    def _next_paired_batch(self):
        if self.paired_dataloader is None:
            raise RuntimeError("paired dataloader requested outside Stage 2")
        batch = next(self.paired_dataloader)
        self.paired_batches_seen += 1
        return batch

    def _save_resume_state(self, checkpoint_dir):
        """Save a same-world-size, per-rank exact training snapshot."""
        rank = dist.get_rank()
        state = {
            "version": 1,
            "step": self.step,
            "world_size": self.world_size,
            "rank": rank,
            "data_batches_seen": self.data_batches_seen,
            "paired_batches_seen": self.paired_batches_seen,
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "critic_optimizer": (
                self.critic_optimizer.state_dict()
                if self.critic_optimizer is not None else None
            ),
            "nan_skip_count": getattr(self, "nan_skip_count", 0),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state(self.device),
            },
        }
        rank_path = os.path.join(
            checkpoint_dir, f"trainer_state_rank{rank:02d}.pt")
        tmp_rank_path = rank_path + ".tmp"
        torch.save(state, tmp_rank_path)
        os.replace(tmp_rank_path, rank_path)
        dist.barrier()

        if self.is_main_process:
            marker = {
                "version": 1,
                "step": self.step,
                "world_size": self.world_size,
                "data_batches_seen": self.data_batches_seen,
                "rank_state_files": [
                    f"trainer_state_rank{r:02d}.pt"
                    for r in range(self.world_size)
                ],
            }
            marker_path = os.path.join(
                checkpoint_dir, "resume_complete.json")
            tmp_marker_path = marker_path + ".tmp"
            with open(tmp_marker_path, "w", encoding="utf-8") as f:
                json.dump(marker, f, indent=2)
                f.write("\n")
            os.replace(tmp_marker_path, marker_path)
            print("Resume state saved to", checkpoint_dir)
        dist.barrier()

    def _load_resume_state(self, checkpoint_dir):
        """Restore exact local optimizer/RNG state for the same rank topology."""
        rank = dist.get_rank()
        marker_path = os.path.join(
            checkpoint_dir, "resume_complete.json")
        with open(marker_path, encoding="utf-8") as f:
            marker = json.load(f)
        if int(marker["world_size"]) != self.world_size:
            raise ValueError(
                "resume world_size mismatch: "
                f"checkpoint={marker['world_size']} current={self.world_size}")

        rank_path = os.path.join(
            checkpoint_dir, f"trainer_state_rank{rank:02d}.pt")
        state = torch.load(rank_path, map_location="cpu", weights_only=False)
        if int(state["rank"]) != rank:
            raise ValueError(
                f"resume rank mismatch: file={state['rank']} current={rank}")
        if int(state["world_size"]) != self.world_size:
            raise ValueError(
                "rank-state world_size mismatch: "
                f"checkpoint={state['world_size']} current={self.world_size}")

        self.generator_optimizer.load_state_dict(
            state["generator_optimizer"])
        critic_optimizer_state = state.get("critic_optimizer")
        if self.critic_optimizer is not None:
            if critic_optimizer_state is None:
                raise KeyError(f"resume state has no critic optimizer: {rank_path}")
            self.critic_optimizer.load_state_dict(critic_optimizer_state)

        self.step = int(state["step"])
        if self.step != int(marker["step"]):
            raise ValueError(
                f"resume step mismatch: rank={self.step} marker={marker['step']}")
        self.data_batches_seen = int(state["data_batches_seen"])
        self.paired_batches_seen = int(state.get("paired_batches_seen", 0))
        self.nan_skip_count = int(state.get("nan_skip_count", 0))

        # The text dataset and DistributedSampler are deterministic and repeat
        # the same order each cycle. Advance to the exact saved cursor before
        # restoring RNG so iterator construction cannot perturb model RNG.
        skip_batches = self.data_batches_seen % self._dataloader_len
        for _ in range(skip_batches):
            next(self.dataloader)
        if self.paired_dataloader is not None:
            paired_skip = self.paired_batches_seen % self._paired_dataloader_len
            for _ in range(paired_skip):
                next(self.paired_dataloader)

        # A completed Stage 2 step has already initialized LPIPS. Recreate it
        # before restoring RNG: VGG/LPIPS construction consumes CPU randomness,
        # whereas the next uninterrupted step reuses the existing network.
        # Keep fresh runs (and a hypothetical step-zero snapshot) lazy.
        if (self.step > 0
                and getattr(self.config, "distribution_loss", "") == "paired_coupled_dmd"
                and float(getattr(self.model, "lpips_weight", 0.0)) > 0):
            self.model._get_lpips()

        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        torch.cuda.set_rng_state(rng["torch_cuda"], self.device)
        print(
            f"Rank {rank} resumed at step={self.step}, "
            f"data_batches_seen={self.data_batches_seen}",
            flush=True,
        )

    def _restore_generator_mask(self):
        restore = getattr(self.model, "_mask_restore_flag", None)
        if restore is not None:
            import wf_training.wan.modules.causal_model as causal_model

            causal_model._RF_BLOCK_CAUSAL = restore
            self.model._mask_restore_flag = None

    def _encode_paired_dmd_prompts(self, text_prompts):
        batch_size = len(text_prompts)
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            unconditional_dict = self.model.text_encoder(
                text_prompts=[self.config.negative_prompt] * batch_size)
            unconditional_dict = {
                key: value.detach() for key, value in unconditional_dict.items()
            }
        return conditional_dict, unconditional_dict


    def fwdbwd_paired_coupled_generator(self, batch, paired_batch):
        """Stage 2: DMD and LPIPS share a single teacher-state prediction."""
        self.model.eval()
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        paired_prompts = paired_batch["prompts"]
        paired_cond, paired_uncond = self._encode_paired_dmd_prompts(paired_prompts)

        y_ref = paired_batch["y_ref"].to(
            device=self.device, dtype=self.dtype, non_blocking=True)
        coupled_loss, coupled_log = self.model.generator_loss(
            image_or_video_shape=None,
            conditional_dict=paired_cond,
            unconditional_dict=paired_uncond,
            clean_latent=y_ref,
        )
        try:
            coupled_loss.backward()
        finally:
            self._restore_generator_mask()

        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm_generator)
        nonfinite = False
        if torch.is_tensor(generator_grad_norm):
            nonfinite = not bool(torch.isfinite(generator_grad_norm).all().item())
        if nonfinite:
            self.generator_optimizer.zero_grad(set_to_none=True)

        lpips_loss = coupled_log.get("lpips_loss", torch.zeros(()))
        lpips_w = float(getattr(self.model, "lpips_weight", 0.0))
        paired_dmd_loss = coupled_loss.detach() - lpips_w * lpips_loss.detach()
        return {
            "generator_loss": coupled_loss.detach(),
            "dmd_loss": paired_dmd_loss,
            "paired_dmd_loss": paired_dmd_loss,
            "lpips_loss": lpips_loss.detach(),
            "dmdtrain_gradient_norm": coupled_log.get("dmdtrain_gradient_norm", 0.0),
            "generator_grad_norm": generator_grad_norm,
            "grad_nonfinite": float(nonfinite),
        }

    def fwdbwd_paired_offpolicy_critic(self, paired_batch):
        """Stage 2 纯 off-policy：fake score 在 off-policy 状态上训练。"""
        paired_prompts = paired_batch["prompts"]
        cond, uncond = self._encode_paired_dmd_prompts(paired_prompts)
        y_ref = paired_batch["y_ref"].to(
            device=self.device, dtype=self.dtype, non_blocking=True)
        loss, log_dict = self.model.critic_loss(
            image_or_video_shape=None,
            conditional_dict=cond,
            unconditional_dict=uncond,
            clean_latent=y_ref,
        )
        loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)
        return {
            "critic_loss": loss.detach(),
            "critic_timestep": log_dict.get("critic_timestep", 0.0),
            "critic_grad_norm": critic_grad_norm,
        }


    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        clean_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=None
            )

            try:
                generator_loss.backward()
            finally:
                # Preserve the chosen RF/CRF/SF mask through checkpoint
                # recomputation, then restore the configured default.
                restore = getattr(self.model, "_mask_restore_flag", None)
                if restore is not None:
                    import wf_training.wan.modules.causal_model as causal_model
                    causal_model._RF_BLOCK_CAUSAL = restore
                    self.model._mask_restore_flag = None
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            # NaN/Inf guard: a single non-finite gradient would permanently
            # poison Adam state (inf grad * clip_coef=0 -> NaN -> m/v NaN).
            # Zero the grads and let train() skip this update instead.
            nonfinite = False
            if torch.is_tensor(generator_grad_norm):
                nonfinite = not bool(torch.isfinite(generator_grad_norm).all().item())
            if nonfinite:
                self.generator_optimizer.zero_grad(set_to_none=True)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm,
                                       "grad_nonfinite": float(nonfinite)})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        if self.model.fake_score is None:
            return {}

        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict


    def train(self):
        start_step = self.step
        max_steps = int(getattr(self.config, "max_steps", 0) or 0)
        if max_steps > 0 and self.step >= max_steps:
            if self.is_main_process:
                print(
                    f"Training already reached configured max_steps={max_steps}; exiting.",
                    flush=True,
                )
            return

        while max_steps <= 0 or self.step < max_steps:
            paired_coupled = self.config.distribution_loss == "paired_coupled_dmd"
            TRAIN_GENERATOR = paired_coupled or (
                self.step % self.config.dfake_gen_update_ratio == 0)

            # Expose global step to the model (mix schedule annealing etc.)
            try:
                self.model._global_train_step = self.step
            except Exception:
                pass

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = self._next_batch()
                if paired_coupled:
                    paired_batch = self._next_paired_batch()
                    extra = self.fwdbwd_paired_coupled_generator(batch, paired_batch)
                else:
                    extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                if paired_coupled and self.is_main_process:
                    def _paired_scalar(value):
                        return value.mean().item() if torch.is_tensor(value) else float(value)

                    print(
                        "DMD_LPIPS_STEP "
                        f"step={self.step} "
                        f"paired_dmd={_paired_scalar(generator_log_dict['paired_dmd_loss']):.6f} "
                        f"lpips={_paired_scalar(generator_log_dict['lpips_loss']):.6f} "
                        f"grad_nonfinite={int(generator_log_dict['grad_nonfinite'])}",
                        flush=True,
                    )
                if generator_log_dict.get("grad_nonfinite", 0.0):
                    # Skip the poisoned update entirely (see guard in
                    # fwdbwd_one_step); count so we can monitor frequency.
                    self.nan_skip_count = getattr(self, "nan_skip_count", 0) + 1
                    generator_log_dict["nan_skips"] = float(self.nan_skip_count)
                    if self.is_main_process:
                        print(f"[train] NON-FINITE grad norm at step {self.step}; "
                              f"update skipped (total skips={self.nan_skip_count})",
                              flush=True)
                else:
                    self.generator_optimizer.step()
                    if self.generator_ema is not None:
                        self.generator_ema.update(self.model.generator)

            # Train the critic
            critic_log_dict = {}
            if self.critic_optimizer is not None:
                self.critic_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                if paired_coupled:
                    paired_critic_batch = self._next_paired_batch()
                    extra = self.fwdbwd_paired_offpolicy_critic(paired_critic_batch)
                else:
                    batch = self._next_batch()
                    extra = self.fwdbwd_one_step(batch, False)
                extras_list.append(extra)
                critic_log_dict = merge_dict_list(extras_list)
                self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging (TB + wandb_loss_dict)
            wandb_loss_dict = {}
            if self.is_main_process:

                if TRAIN_GENERATOR:
                    def _scalar(v):
                        return v.mean().item() if torch.is_tensor(v) else float(v)

                    wandb_loss_dict["generator_loss"] = _scalar(
                        generator_log_dict["generator_loss"])
                    wandb_loss_dict["generator_grad_norm"] = _scalar(
                        generator_log_dict["generator_grad_norm"])
                    if "dmdtrain_gradient_norm" in generator_log_dict:
                        wandb_loss_dict["dmdtrain_gradient_norm"] = _scalar(
                            generator_log_dict["dmdtrain_gradient_norm"])
                    for kd_key in ("nan_skips", "dmd_loss", "paired_dmd_loss", "lpips_loss",
                                   "grad_nonfinite"):
                        if kd_key in generator_log_dict:
                            wandb_loss_dict[kd_key] = _scalar(generator_log_dict[kd_key])
                    # RF/CRF/SF mix path: 0=RF, 1=CRF, 2=SF
                    train_path = getattr(self.model, "_last_train_path", None)
                    if train_path is not None:
                        wandb_loss_dict["train_path"] = float(train_path)
                        wandb_loss_dict["path_rf"] = float(train_path == 0)
                        wandb_loss_dict["path_crf"] = float(train_path == 1)
                        wandb_loss_dict["path_sf"] = float(train_path == 2)
                    mix_probs = getattr(self.model, "_last_mix_probs", None)
                    if mix_probs is not None:
                        wandb_loss_dict["mix/prob_rf"] = float(mix_probs[0])
                        wandb_loss_dict["mix/prob_crf"] = float(mix_probs[1])
                        wandb_loss_dict["mix/prob_sf"] = float(mix_probs[2])

                    for k, v in wandb_loss_dict.items():
                        self.writer.add_scalar(k, v, self.step)

                if self.critic_optimizer is not None and "critic_loss" in critic_log_dict:
                    critic_loss_v = critic_log_dict["critic_loss"].mean().item()
                    critic_gn_v = critic_log_dict["critic_grad_norm"].mean().item()
                    wandb_loss_dict["critic_loss"] = critic_loss_v
                    wandb_loss_dict["critic_grad_norm"] = critic_gn_v
                    self.writer.add_scalar("critic_loss", critic_loss_v, self.step)
                    self.writer.add_scalar("critic_grad_norm", critic_gn_v, self.step)

                if wandb_loss_dict and self._wandb is not None:
                    self._wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    iter_time = current_time - self.previous_time
                    self.writer.add_scalar("per iteration time", iter_time, self.step)
                    if self._wandb is not None:
                        self._wandb.log({"per iteration time": iter_time}, step=self.step)
                    path = getattr(self.model, "_last_train_path", None)
                    path_name = {0: "RF", 1: "CRF", 2: "SF"}.get(path, "?")
                    print(
                        f"Step {self.step} | path={path_name} | "
                        f"Iteration time: {iter_time:.2f} seconds | ",
                        flush=True,
                    )
                    self.previous_time = current_time

        if not self.config.no_save and getattr(self, "_last_full_saved_step", None) != self.step:
            self.save(force_resume=True)
        dist.barrier()
        if self.is_main_process:
            self.writer.flush()
            if self._wandb is not None:
                self._wandb.finish()
            print(
                f"Training reached configured max_steps={max_steps}; all ranks exiting cleanly.",
                flush=True,
            )
