"""WaveRT configuration and command-line wiring."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

DEFAULT_MODEL_PATH = "./ckpts/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_GEN_CKPT = "./ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt"
DEFAULT_PROMPT = (
    "A cinematic shot of a fluffy corgi running on a sunny beach, "
    "waves in the background."
)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_ROOT = os.path.join(_REPO_ROOT, "outputs")

ATTENTION_BACKENDS = ("torch_sdpa", "fa_custom", "sa", "fa", "fa2")
KV_CONTEXTS = ("joint", "causal")
EXCHANGE_MODES = ("sync", "overlap", "onesided", "relay", "staggered", "paged")
CAUSAL_EXCHANGE_MODES = frozenset(EXCHANGE_MODES[1:])
VAE_PARTITIONS = ("time", "flops")


@dataclass
class WaveConfig:
    # Wavefront topology: one denoise rank per DMD step plus one KV store rank.
    rf_step: int = 4
    wp_size: int = 5
    denoising_step_list: tuple = (1000, 750, 500, 250)

    # Generation shape. num_frames counts latent frames.
    num_frames: int = 24
    height: int = 480
    width: int = 832
    seed: int = 0
    prompt: str = DEFAULT_PROMPT

    model_path: str = DEFAULT_MODEL_PATH
    gen_ckpt: str = DEFAULT_GEN_CKPT
    pipeline_class_name: str = "WanRollingForcingPipeline"
    attention_backend: str = "torch_sdpa"
    # Post-load DiT linear quantization. None keeps bf16 weights.
    quantization: str | None = None
    # Optional random-weight DiT used only for system scaling measurements.
    dit_scale: str | None = None
    # "joint" sees every in-flight chunk; "causal" sees ranks n >= r.
    kv_context: str = "joint"
    # KV movement strategy; every non-sync mode requires causal context.
    exchange_mode: str = "sync"
    stagger_lead: int = 1
    kv_transport_dtype: str = "bf16"

    master_addr: str = "127.0.0.1"
    master_port: int = 29677
    cuda_visible_devices: str | None = None

    # Zero VAE stages decodes on the store rank; positive values stream stages.
    vae_stages: int = 3
    vae_port: int = 29688
    vae_partition: str = "time"

    serve: bool = False
    serve_host: str = "127.0.0.1"
    serve_port: int = 8890
    warmup_frames: int = 24

    out_root: str = DEFAULT_OUT_ROOT
    task: str = "wave_rt_phase1"
    run_tag: str = "wrt"
    save_video: bool = True
    ref_latents: str = ""
    baseline_e2e: float = 0.0

    num_frames_per_block: int = 3
    compile: bool = False

    def __post_init__(self) -> None:
        if self.rf_step + 1 != self.wp_size:
            raise ValueError(
                f"WaveRT requires rf_step + 1 == wp_size, got "
                f"rf_step={self.rf_step}, wp_size={self.wp_size}"
            )
        if self.wp_size > 1 and len(self.denoising_step_list) != self.rf_step:
            raise ValueError(
                f"denoising_step_list has {len(self.denoising_step_list)} steps but "
                f"rf_step={self.rf_step}; they must match (one denoise rank per step)."
            )
        if self.num_frames % self.num_frames_per_block != 0:
            raise ValueError(
                f"num_frames ({self.num_frames}) must be divisible by "
                f"num_frames_per_block ({self.num_frames_per_block})"
            )
        if (
            self.exchange_mode in CAUSAL_EXCHANGE_MODES
            and self.kv_context != "causal"
        ):
            raise ValueError(
                f"--exchange-mode {self.exchange_mode} requires --kv-context causal "
                "(only the causal context creates the per-stage slack / the n>=r "
                "in-flight semantics the cascade relies on)."
            )
        if self.exchange_mode == "staggered" and self.stagger_lead < 1:
            raise ValueError(
                f"--stagger-lead must be >= 1 (got {self.stagger_lead})"
            )

    @property
    def num_blocks(self) -> int:
        return self.num_frames // self.num_frames_per_block

    @property
    def denoise_ranks(self) -> int:
        return self.rf_step

    @property
    def store_rank(self) -> int:
        return self.wp_size - 1

    @property
    def n_total_gpus(self) -> int:
        """Number of diffusion and VAE ranks combined."""
        return self.wp_size + max(0, self.vae_stages)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        g = parser
        g.add_argument("--rf-step", type=int, default=WaveConfig.rf_step)
        g.add_argument("--wp-size", type=int, default=WaveConfig.wp_size)
        g.add_argument("--denoising-step-list", type=str, default=None,
                       help="comma-separated raw DMD steps (pre-warp), e.g. "
                            "'1000,750,500,250'; len must equal --rf-step. "
                            "Default is the 4-step longvideo.pt schedule.")
        g.add_argument("--num-frames", type=int, default=WaveConfig.num_frames,
                       help="LATENT frames (divisible by 3)")
        g.add_argument(
            "--vae-stages",
            dest="vae_stages",
            type=int,
            default=WaveConfig.vae_stages,
            help="number of streaming VAE stages; 0 decodes on the store rank",
        )
        g.add_argument("--vae-port", type=int, default=WaveConfig.vae_port)
        g.add_argument("--vae-partition", type=str, default=WaveConfig.vae_partition,
                       choices=VAE_PARTITIONS,
                       help="split decoder across VAE stages by measured time "
                            "(default) or analytic FLOPs")
        g.add_argument("--serve", action="store_true",
                       help="resident serving: load once, accept HTTP /generate requests")
        g.add_argument("--serve-host", type=str, default=WaveConfig.serve_host)
        g.add_argument("--serve-port", type=int, default=WaveConfig.serve_port)
        g.add_argument("--warmup-frames", type=int, default=WaveConfig.warmup_frames,
                       help="serve startup warmup: latent frames for the one-time dummy "
                            "full-window generation (divisible by 3; ~24 reaches steady state)")
        g.add_argument("--height", type=int, default=WaveConfig.height)
        g.add_argument("--width", type=int, default=WaveConfig.width)
        g.add_argument("--seed", type=int, default=WaveConfig.seed)
        g.add_argument("--prompt", type=str, default=WaveConfig.prompt)
        g.add_argument("--model-path", type=str, default=WaveConfig.model_path)
        g.add_argument("--gen-ckpt", type=str, default=WaveConfig.gen_ckpt)
        g.add_argument("--attention-backend", type=str,
                       default=WaveConfig.attention_backend,
                       choices=ATTENTION_BACKENDS)
        g.add_argument("--quantization", type=str, default=WaveConfig.quantization,
                       help="post-load DiT linear quantization; None keeps bf16")
        g.add_argument("--dit-scale", type=str, default=WaveConfig.dit_scale,
                       choices=["1.3b", "5b", "14b"],
                       help="system-level scaling test: build a larger random-init "
                            "DiT (no RF ckpt; output meaningless). Default = real 1.3b.")
        g.add_argument("--kv-context", type=str, default=WaveConfig.kv_context,
                       choices=KV_CONTEXTS,
                       help="'joint' sees all in-flight ranks; 'causal' sees n>=r")
        g.add_argument("--exchange-mode", type=str, default=WaveConfig.exchange_mode,
                       choices=EXCHANGE_MODES,
                       help="KV transport strategy; non-sync modes require causal context")
        g.add_argument("--stagger-lead", type=int, default=WaveConfig.stagger_lead,
                       help="layers of acquire prefetch for staggered exchange")
        g.add_argument("--master-port", type=int, default=WaveConfig.master_port)
        g.add_argument("--cuda-visible-devices", type=str, default=None)
        g.add_argument("--out-root", type=str, default=WaveConfig.out_root)
        g.add_argument("--task", type=str, default=WaveConfig.task)
        g.add_argument("--run-tag", type=str, default=WaveConfig.run_tag)
        g.add_argument("--no-save-video", dest="save_video", action="store_false")
        g.add_argument("--compile", action="store_true",
                       help="torch.compile the DiT block to fuse fp32 norms/FFN")
        g.add_argument("--ref-latents", type=str, default=WaveConfig.ref_latents)
        g.add_argument("--baseline-e2e", type=float, default=WaveConfig.baseline_e2e)

    @classmethod
    def from_args(cls, ns: argparse.Namespace) -> "WaveConfig":
        kw = {}
        if getattr(ns, "denoising_step_list", None):
            kw["denoising_step_list"] = tuple(
                int(x) for x in ns.denoising_step_list.split(",") if x.strip()
            )
        return cls(
            rf_step=ns.rf_step,
            wp_size=ns.wp_size,
            num_frames=ns.num_frames,
            vae_stages=ns.vae_stages,
            vae_port=ns.vae_port,
            vae_partition=ns.vae_partition,
            serve=ns.serve,
            serve_host=ns.serve_host,
            serve_port=ns.serve_port,
            warmup_frames=ns.warmup_frames,
            height=ns.height,
            width=ns.width,
            seed=ns.seed,
            prompt=ns.prompt,
            model_path=ns.model_path,
            gen_ckpt=ns.gen_ckpt,
            attention_backend=ns.attention_backend,
            quantization=ns.quantization,
            dit_scale=ns.dit_scale,
            kv_context=ns.kv_context,
            exchange_mode=ns.exchange_mode,
            stagger_lead=ns.stagger_lead,
            master_port=ns.master_port,
            cuda_visible_devices=ns.cuda_visible_devices,
            out_root=ns.out_root,
            task=ns.task,
            run_tag=ns.run_tag,
            save_video=ns.save_video,
            compile=ns.compile,
            ref_latents=ns.ref_latents,
            baseline_e2e=ns.baseline_e2e,
            **kw,
        )
