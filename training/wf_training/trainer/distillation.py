import gc
import json
import logging
import random

from wf_training.utils.dataset import cycle
from wf_training.utils.dataset import TextDataset
from wf_training.utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from wf_training.utils.ema import ShardedEMA
from wf_training.utils.checkpoint import load_model_checkpoint, generator_state
from wf_training.utils.memory import memory_scope
from wf_training.utils.misc import set_seed
from wf_training.utils.sequence_parallel import (
    initialize_sequence_parallel,
    get_data_parallel_rank,
    get_data_parallel_world_size,
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
        self._resume_ema_shard = None
        self.resume_from = getattr(config, "resume_from", "") or ""

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if self.world_size != config.world_size:
            raise ValueError("torchrun world size differs from the resolved configuration")
        self.sequence_parallel_size = int(getattr(config, "sequence_parallel_size", 1))
        self.gradient_accumulation_steps = int(
            getattr(config, "gradient_accumulation_steps", 1))
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        initialize_sequence_parallel(self.sequence_parallel_size)

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

        # Every rank in an SP group executes the same example and random choices.
        set_seed(config.seed + get_data_parallel_rank())

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

        if not getattr(self.model, "fsdp_initialized", False):
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
            dataset, num_replicas=get_data_parallel_world_size(),
            rank=get_data_parallel_rank(), shuffle=True, drop_last=True)
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
                paired_dataset, num_replicas=get_data_parallel_world_size(),
                rank=get_data_parallel_rank(), shuffle=True, drop_last=True)
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

        # EMA is created only after the correct raw stage weights are loaded,
        # and only at its configured start. Sharded EMA never gathers here.
        self.generator_ema = None
        checkpoint = None
        checkpoint_path = (
            os.path.join(self.resume_from, "model.pt")
            if self.resume_from else getattr(config, "generator_ckpt", "")
        )
        if checkpoint_path and not getattr(self.model, "fsdp_initialized", False):
            with memory_scope(config, "initialize/stage_weights", device=self.device):
                checkpoint = load_model_checkpoint(checkpoint_path)
                self.model.generator.load_state_dict(generator_state(checkpoint), strict=True)
                if self.resume_from and self.model.fake_score is not None:
                    if "critic" not in checkpoint:
                        raise KeyError(f"resume checkpoint has no critic: {checkpoint_path}")
                    self.model.fake_score.load_state_dict(checkpoint["critic"], strict=True)
        if self.resume_from:
            with memory_scope(config, "resume/trainer_state", device=self.device):
                self._load_resume_state(self.resume_from)
                if getattr(config, "ema_mode", "full") == "sharded":
                    self._restore_ema(self._resume_ema_shard)
                    self._resume_ema_shard = None
                else:
                    if checkpoint is None:
                        checkpoint = load_model_checkpoint(checkpoint_path)
                    self._restore_ema(checkpoint.get("generator_ema"))
        elif self._ema_enabled() and self.step >= config.ema_start_step:
            self.generator_ema = self._new_ema()
        del checkpoint

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def _ema_enabled(self):
        return self.config.ema_weight is not None and self.config.ema_weight > 0

    def _new_ema(self, *, initialize=True):
        ema_class = ShardedEMA if getattr(self.config, "ema_mode", "full") == "sharded" else EMA_FSDP
        return ema_class(self.model.generator, decay=self.config.ema_weight, initialize=initialize)

    def _restore_ema(self, ema_state):
        """Restore either rank-local EMA or the reference full EMA format."""
        if ema_state is not None:
            if not self._ema_enabled():
                raise ValueError("resume contains EMA but EMA is disabled")
            if self.step < self.config.ema_start_step:
                raise ValueError("resume contains EMA before its configured start")
            self.generator_ema = self._new_ema(initialize=False)
            if isinstance(self.generator_ema, ShardedEMA):
                self.generator_ema.load_state_dict(ema_state, self.model.generator)
            else:
                self.generator_ema.load_state_dict(ema_state)
        elif not self._ema_enabled() or self.step < self.config.ema_start_step:
            self.generator_ema = None
        elif self.step == self.config.ema_start_step and getattr(self.config, "ema_mode", "full") == "full":
            # Legacy full-EMA checkpoints omitted EMA exactly at this boundary.
            self.generator_ema = self._new_ema()
        else:
            raise KeyError("resume checkpoint has no EMA at/after its configured start")

    def save(self, *, force_resume=False):
        with memory_scope(self.config, "checkpoint/save", step=self.step, device=self.device):
            self._save_checkpoint(force_resume=force_resume)

    def _save_checkpoint(self, *, force_resume=False):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = (
            fsdp_state_dict(self.model.fake_score)
            if self.model.fake_score is not None else None
        )

        state_dict = {"generator": generator_state_dict, "critic": critic_state_dict}
        if isinstance(self.generator_ema, ShardedEMA):
            with self.generator_ema.applied_to(self.model.generator):
                state_dict["generator_ema"] = fsdp_state_dict(self.model.generator)
        elif self.generator_ema is not None:
            state_dict["generator_ema"] = self.generator_ema.state_dict()
        state_dict["ema_mode"] = getattr(self.config, "ema_mode", "full")

        checkpoint_dir = os.path.join(
            self.output_path, f"checkpoint_model_{self.step:06d}")
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            complete_path = os.path.join(checkpoint_dir, "resume_complete.json")
            if os.path.exists(complete_path):
                os.unlink(complete_path)
            model_path = os.path.join(checkpoint_dir, "model.pt")
            tmp_model_path = model_path + ".tmp"
            torch.save(state_dict, tmp_model_path)
            os.replace(tmp_model_path, model_path)
            print("Model saved to", model_path)
        dist.barrier()
        # Full CPU exports are no longer needed while each rank serializes its
        # optimizer/EMA shards. The peak during export itself is still large.
        del state_dict, generator_state_dict, critic_state_dict

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
            **self._training_topology(rank),
            "data_batches_seen": self.data_batches_seen,
            "paired_batches_seen": self.paired_batches_seen,
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "critic_optimizer": (
                self.critic_optimizer.state_dict()
                if self.critic_optimizer is not None else None
            ),
            "nan_skip_count": getattr(self, "nan_skip_count", 0),
            "critic_nan_skip_count": getattr(self, "critic_nan_skip_count", 0),
            "ema_mode": getattr(self.config, "ema_mode", "full"),
            "generator_ema_shard": (self.generator_ema.state_dict()
                                    if isinstance(self.generator_ema, ShardedEMA) else None),
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
                **self._training_topology(),
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

    def _training_topology(self, rank=None):
        sp_size = int(getattr(self.config, "sequence_parallel_size", 1))
        topology = {
            "sequence_parallel_size": sp_size,
            "gradient_accumulation_steps": int(
                getattr(self.config, "gradient_accumulation_steps", 1)),
            "data_parallel_world_size": self.world_size // sp_size,
        }
        if rank is not None:
            topology.update(sequence_parallel_rank=rank % sp_size,
                            data_parallel_rank=rank // sp_size)
        return topology

    def _validate_resume_topology(self, saved, *, rank=None):
        # Snapshots written before SP/accumulation describe SP1 and one microbatch.
        legacy = {
            "sequence_parallel_size": 1,
            "gradient_accumulation_steps": 1,
            "data_parallel_world_size": int(saved["world_size"]),
            "sequence_parallel_rank": 0,
            "data_parallel_rank": rank,
        }
        for key, expected in self._training_topology(rank).items():
            value = int(saved.get(key, legacy[key]))
            if value != expected:
                raise ValueError(
                    f"resume {key} mismatch: checkpoint={value} current={expected}")

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
        self._validate_resume_topology(marker)

        rank_path = os.path.join(
            checkpoint_dir, f"trainer_state_rank{rank:02d}.pt")
        state = load_model_checkpoint(rank_path)
        if int(state["rank"]) != rank:
            raise ValueError(
                f"resume rank mismatch: file={state['rank']} current={rank}")
        if int(state["world_size"]) != self.world_size:
            raise ValueError(
                "rank-state world_size mismatch: "
                f"checkpoint={state['world_size']} current={self.world_size}")
        self._validate_resume_topology(state, rank=rank)

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
        self.critic_nan_skip_count = int(state.get("critic_nan_skip_count", 0))
        if state.get("ema_mode", "full") != getattr(self.config, "ema_mode", "full"):
            raise ValueError("EMA storage mode differs from the saved rank state")
        self._resume_ema_shard = state.get("generator_ema_shard")

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

    def _release_generator_caches(self):
        for name in ("inference_pipeline", "_offp_pipeline"):
            pipeline = getattr(self.model, name, None)
            if pipeline is not None:
                pipeline.release_caches()

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


    @staticmethod
    def _detach_logs(log_dict):
        return {key: value.detach() if torch.is_tensor(value) else value
                for key, value in log_dict.items()}

    def _clip_phase_gradients(self, train_generator):
        model = self.model.generator if train_generator else self.model.fake_score
        max_norm = (self.max_grad_norm_generator if train_generator
                    else self.max_grad_norm_critic)
        return model.clip_grad_norm_(max_norm)

    def _finish_microbatch(self, loss, log_dict, train_generator, defer_clip):
        prefix = "generator" if train_generator else "critic"
        result = self._detach_logs(log_dict)
        result[f"{prefix}_loss"] = loss.detach()
        nonfinite = not bool(torch.isfinite(loss.detach()).all().item())
        if not defer_clip:
            norm = self._clip_phase_gradients(train_generator)
            result[f"{prefix}_grad_norm"] = norm.detach() if torch.is_tensor(norm) else norm
            nonfinite |= not bool(torch.isfinite(torch.as_tensor(norm)).all().item())
        result["grad_nonfinite"] = float(nonfinite)
        return result

    def fwdbwd_paired_coupled_generator(
            self, batch, paired_batch, *, defer_clip=False, loss_divisor=1):
        """Stage 2: DMD and LPIPS share a single teacher-state prediction."""
        self.model.eval()
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        paired_prompts = paired_batch["prompts"]
        paired_cond, paired_uncond = self._encode_paired_dmd_prompts(paired_prompts)

        y_ref = paired_batch["y_ref"].to(
            device=self.device, dtype=self.dtype, non_blocking=True)
        try:
            coupled_loss, coupled_log = self.model.generator_loss(
                image_or_video_shape=None,
                conditional_dict=paired_cond,
                unconditional_dict=paired_uncond,
                clean_latent=y_ref,
            )
            (coupled_loss / loss_divisor).backward()
        finally:
            self._restore_generator_mask()
            self._release_generator_caches()

        lpips_loss = coupled_log.get("lpips_loss", torch.zeros(()))
        lpips_w = float(getattr(self.model, "lpips_weight", 0.0))
        paired_dmd_loss = coupled_loss.detach() - lpips_w * lpips_loss.detach()
        return self._finish_microbatch(coupled_loss, {
            "dmd_loss": paired_dmd_loss,
            "paired_dmd_loss": paired_dmd_loss,
            "lpips_loss": lpips_loss.detach(),
            "dmdtrain_gradient_norm": coupled_log.get("dmdtrain_gradient_norm", 0.0),
        }, True, defer_clip)

    def fwdbwd_paired_offpolicy_critic(
            self, paired_batch, *, defer_clip=False, loss_divisor=1):
        """Stage 2 纯 off-policy：fake score 在 off-policy 状态上训练。"""
        paired_prompts = paired_batch["prompts"]
        cond, uncond = self._encode_paired_dmd_prompts(paired_prompts)
        y_ref = paired_batch["y_ref"].to(
            device=self.device, dtype=self.dtype, non_blocking=True)
        try:
            loss, log_dict = self.model.critic_loss(
                image_or_video_shape=None,
                conditional_dict=cond,
                unconditional_dict=uncond,
                clean_latent=y_ref,
            )
            (loss / loss_divisor).backward()
        finally:
            self._restore_generator_mask()
            self._release_generator_caches()
        return self._finish_microbatch(loss, log_dict, False, defer_clip)


    def fwdbwd_one_step(
            self, batch, train_generator, *, defer_clip=False, loss_divisor=1):
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
            try:
                generator_loss, generator_log_dict = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent,
                    initial_latent=None
                )
                (generator_loss / loss_divisor).backward()
            finally:
                self._restore_generator_mask()
                self._release_generator_caches()
            return self._finish_microbatch(
                generator_loss, generator_log_dict, True, defer_clip)

        # Step 4: Store gradients for the critic (if training the critic)
        if self.model.fake_score is None:
            return {}

        try:
            critic_loss, critic_log_dict = self.model.critic_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=None
            )
            (critic_loss / loss_divisor).backward()
        finally:
            self._restore_generator_mask()
            self._release_generator_caches()
        return self._finish_microbatch(
            critic_loss, critic_log_dict, False, defer_clip)

    def _accumulate_phase(self, train_generator, paired_coupled):
        """Accumulate sharded gradients with a normal FSDP backward per microbatch."""
        microbatches = int(getattr(self.config, "gradient_accumulation_steps", 1))
        if microbatches < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        # Retain the legacy single-batch call signature for integrations that
        # override fwdbwd methods. Those methods already clip their one batch.
        kwargs = ({"defer_clip": True, "loss_divisor": microbatches}
                  if microbatches > 1 else {})
        logs = []
        train_paths = []
        nonfinite = False
        for _ in range(microbatches):
            if train_generator:
                batch = self._next_batch()
                if paired_coupled:
                    extra = self.fwdbwd_paired_coupled_generator(
                        batch, self._next_paired_batch(), **kwargs)
                else:
                    extra = self.fwdbwd_one_step(batch, True, **kwargs)
            elif paired_coupled:
                extra = self.fwdbwd_paired_offpolicy_critic(
                    self._next_paired_batch(), **kwargs)
            else:
                extra = self.fwdbwd_one_step(self._next_batch(), False, **kwargs)
            if train_generator:
                path = getattr(self.model, "_last_train_path", None)
                if path is not None:
                    train_paths.append(int(path))
            nonfinite |= bool(extra.get("grad_nonfinite", 0.0))
            # Reduce each diagnostic immediately, retaining neither a graph nor
            # large timestep/latent tensors across the remaining microbatches.
            logs.append({key: value.detach().float().mean() if torch.is_tensor(value) else value
                         for key, value in extra.items()})

        merged = {}
        for key, first in logs[0].items():
            values = [log[key] for log in logs]
            if torch.is_tensor(first):
                merged[key] = torch.stack(values).mean()
            elif isinstance(first, (int, float, np.number)):
                merged[key] = sum(values) / microbatches
            else:
                merged[key] = values[-1]
        if train_paths:
            merged["train_path"] = train_paths[0] if len(set(train_paths)) == 1 else -1
            for path, name in enumerate(("rf", "crf", "sf")):
                merged[f"path_{name}"] = train_paths.count(path) / microbatches

        prefix = "generator" if train_generator else "critic"
        norm = (self._clip_phase_gradients(train_generator) if microbatches > 1
                else merged[f"{prefix}_grad_norm"])
        merged[f"{prefix}_grad_norm"] = norm.detach() if torch.is_tensor(norm) else norm
        nonfinite |= not bool(torch.isfinite(torch.as_tensor(norm)).all().item())
        # FSDP spans the full world, so every DP/SP rank must make the same
        # optimizer/EMA decision, including a non-finite loss with finite grads.
        skip = torch.tensor(int(nonfinite), dtype=torch.int32, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(skip, op=dist.ReduceOp.MAX)
        merged["grad_nonfinite"] = float(skip.item())
        return merged


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
                with memory_scope(self.config, "train/generator", step=self.step, device=self.device):
                    self.generator_optimizer.zero_grad(set_to_none=True)
                    generator_log_dict = self._accumulate_phase(True, paired_coupled)
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
                        # Skip the entire phase, leaving Adam and EMA unchanged.
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
                    self.generator_optimizer.zero_grad(set_to_none=True)

            # Train the critic
            critic_log_dict = {}
            if self.critic_optimizer is not None:
                with memory_scope(self.config, "train/critic", step=self.step, device=self.device):
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_log_dict = self._accumulate_phase(False, paired_coupled)
                    critic_finite = not bool(critic_log_dict["grad_nonfinite"])
                    if critic_finite:
                        self.critic_optimizer.step()
                    else:
                        self.critic_nan_skip_count = getattr(self, "critic_nan_skip_count", 0) + 1
                        if self.is_main_process:
                            print(f"[train] NON-FINITE critic gradient at step {self.step}; update skipped", flush=True)
                    critic_log_dict["grad_nonfinite"] = float(not critic_finite)
                    self.critic_optimizer.zero_grad(set_to_none=True)

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if self._ema_enabled() and self.step >= self.config.ema_start_step and self.generator_ema is None:
                self.generator_ema = self._new_ema()

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
                    # -1 marks a phase mixing paths; coverage counts every microbatch.
                    train_path = generator_log_dict.get("train_path")
                    if train_path is not None:
                        wandb_loss_dict["train_path"] = float(train_path)
                        for name in ("rf", "crf", "sf"):
                            wandb_loss_dict[f"path_{name}"] = generator_log_dict[f"path_{name}"]
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
                    wandb_loss_dict["critic_grad_nonfinite"] = critic_log_dict["grad_nonfinite"]
                    self.writer.add_scalar("critic_loss", critic_loss_v, self.step)
                    self.writer.add_scalar("critic_grad_norm", critic_gn_v, self.step)
                    self.writer.add_scalar("critic_grad_nonfinite", critic_log_dict["grad_nonfinite"], self.step)

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
                    path = (generator_log_dict.get("train_path") if TRAIN_GENERATOR
                            else getattr(self.model, "_last_train_path", None))
                    path_name = {-1: "mixed", 0: "RF", 1: "CRF", 2: "SF"}.get(path, "?")
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
