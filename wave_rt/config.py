"""WaveConfig -- all knobs for a WaveRT run, plus argparse wiring.

Kept as a plain dataclass (WaveRT is a standalone runtime outside the sglang
source tree, so the sgl ``msgspec.Struct`` rule does not apply here).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

# --- default asset paths (verified present on this box) ---------------------
DEFAULT_MODEL_PATH = "./ckpts/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_GEN_CKPT = (
    "./ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt"
)
DEFAULT_PROMPT = (
    "A cinematic shot of a fluffy corgi running on a sunny beach, "
    "waves in the background."
)
# Outputs live inside the repo (gitignored via ``outputs/*``).
# Repo root = two levels up from this file.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_ROOT = os.path.join(_REPO_ROOT, "outputs")


@dataclass
class WaveConfig:
    # --- wavefront topology ---
    # The real Causal-Forcing long_video RF (chunkwise longvideo.pt) is a 4-step
    # DMD: denoising_step_list = [1000,750,500,250] warped (see
    # Causal-Forcing/long_video/configs/rolling_forcing_dmd.yaml).  So 4 denoise
    # ranks + 1 clean-KV store rank = 5.  NOTE: sglang's RollingForcingWanT2V480PConfig
    # ships a WRONG 5-step [1000,800,600,400,200] default -- we override it in
    # backend.init from denoising_step_list below (never trust the sgl config here).
    rf_step: int = 4  # number of denoising steps (== number of denoise ranks)
    wp_size: int = 5  # total ranks = rf_step + 1 (last rank = clean-KV store)
    # DMD schedule for the loaded checkpoint (raw, pre-warp).  len == rf_step.
    denoising_step_list: tuple = (1000, 750, 500, 250)

    # --- generation shape ---
    num_frames: int = 24  # LATENT frames (must be divisible by num_frames_per_block=3)
    height: int = 480
    width: int = 832
    seed: int = 0
    prompt: str = DEFAULT_PROMPT

    # --- model / checkpoints ---
    model_path: str = DEFAULT_MODEL_PATH
    gen_ckpt: str = DEFAULT_GEN_CKPT
    pipeline_class_name: str = "WanRollingForcingPipeline"
    attention_backend: str = "torch_sdpa"  # driver-safe default; fa2 later
    # Online DiT quantization method passed straight through to sglang's
    # ServerArgs.quantization (e.g. "fp8" = W8A8 fp8 weights + dynamic per-token
    # fp8 activations).  None = bf16 (default).  Quantizes the linear GEMMs
    # (qkv/out proj + FFN); complementary to --attention-backend sa (attention).
    quantization: str | None = None
    # System-level scaling test: build a LARGER Wan DiT (e.g. "14b") from config
    # with random weights (no RF checkpoint exists for it), skipping weight load.
    # None / "1.3b" = the real model.  Output is meaningless; measures the
    # wavefront system's tick/memory/comm at scale.
    dit_scale: str | None = None
    # KV attention context assembly for the denoise ranks:
    #   "joint"  = every in-flight chunk attends to ALL in-flight chunks (current
    #              default; uniform load, no slack; slightly non-causal).
    #   "causal" = chunk-causal / "overlap" semantics: rank r attends only to
    #              in-flight ranks n >= r (chunks at position <= its own) -> matches
    #              naive block-causal, and gives the per-stage load imbalance (slack)
    #              the overlap design needs.  all_gather is unchanged (NCCL-aligned).
    kv_context: str = "joint"
    # KV exchange comm form:
    #   "sync"    = per-layer blocking all_gather (current; works with joint/causal).
    #   "overlap" = async P2P cascade (older/faster ranks -> newer/slower ranks)
    #               overlapped with compute; hides the KV writeback in the per-stage
    #               slack.  REQUIRES kv_context="causal" (only causal gives slack).
    #               Numerically identical to causal+sync (movement-only).
    #   "onesided"= one-sided current-KV publication (Gate C-final-1): producer
    #               writes each consumer's remote_kv slab directly the instant its
    #               layer-L K/V is ready (cuMemcpyDtoDAsync on a copy stream + a
    #               device-side generation flag), no matching recv.  Eliminates the
    #               depth-1 two-sided inbound-KV wait AND the cold-start cascade.
    #               REQUIRES kv_context="causal".  Numerically identical to causal+sync.
    #   "relay"   = neighbor-accumulating relay: instead of the all-pairs one-sided
    #               fan-out (producer r writes every slower consumer -> heavy
    #               copy-engine contention on the middle ranks), each rank talks to
    #               exactly ONE neighbor.  Layer-wise the KV bundle cascades down the
    #               chain store -> rf_step-1 -> ... -> 1 -> 0, each hop MERGING the
    #               upstream bundle with the local KV into one contiguous slice and
    #               forwarding it in a single cuMemcpyDtoDAsync.  NVLink bytes are
    #               unchanged (1+2+..+rf_step chunk-hops == all-pairs) but the fan-out
    #               drops from rf_step to 1, eliminating the all-pairs CE burst.
    #               REQUIRES kv_context="causal".  Numerically identical to causal+sync.
    #   "staggered"= onesided transport + PREFETCHED (staggered) inbound acquire:
    #               the device-side generation-flag acquire for a consumer's layer L
    #               is issued `stagger_lead` layers EARLY on a dedicated prefetch
    #               stream (recorded as a CUDA event), so the producer's layer-L KV
    #               transfer overlaps the consumer's (L-lead..L) compute -- hiding the
    #               tight same-layer producer->consumer edge (esp. store->highest
    #               denoise rank) that onesided gates at the point of use.  Reuses the
    #               one-sided slab/IPC machinery; REQUIRES kv_context="causal".
    #               Numerically identical to causal+sync (movement/timing only).
    exchange_mode: str = "sync"
    # staggered lead: how many layers earlier the inbound KV acquire is prefetched
    # (== how many layers the tight-edge producer leads the consumer).  Only used by
    # exchange_mode="staggered"; 1 is the validated default (deeper leads just kick
    # the same device-side waits earlier and cost a few more in-flight events).
    stagger_lead: int = 1
    # KV transport precision for the ONESIDED exchange only.  "bf16" (default) sends
    # the exchanged KV verbatim; "fp8" quantizes each published (2,T,H,D) KV slot to
    # fp8 e4m3 (per-tensor absmax scale) before the NVLink copy -> ~half the bytes /
    # copy time, dequantized back to bf16 on the consumer before attention.  No effect
    # on sync/overlap/relay/staggered (they stay bf16).
    kv_transport_dtype: str = "bf16"

    # --- distributed ---
    master_addr: str = "127.0.0.1"
    master_port: int = 29677
    cuda_visible_devices: str | None = None  # e.g. "0,1,2,3,4"; None = leave as-is

    # --- VAE decode: n-stage FLOPs-balanced streaming pipeline (naive VAE, bf16) ---
    # 0 = serial VAE on the store rank (wave_rt/vae.py::finalize_on_store, backward-compat).
    # n>0 = n independent VAE-stage procs (GPUs wp_size..wp_size+n-1) fed by an mp.Queue
    # from the store rank; decoder split by per-layer FLOPs (naive_rt.wave.vae_pipe).
    vae_stages: int = 3
    vae_port: int = 29688  # separate NCCL world (disjoint from the diffusion world)
    # How to split the decoder across VAE stages.  "time" = measure each unit's
    # real ms at init and min-max partition (the high-res upsample blocks have
    # equal FLOPs but ~6x the time, so FLOPs mis-balances by ~1.8x at n=3; timing
    # cuts the slowest stage ~1.6x).  "flops" = the old analytic FLOPs split.
    vae_partition: str = "time"

    # --- serving (model resident; HTTP + FIFO job queue) ---
    serve: bool = False
    serve_host: str = "127.0.0.1"
    serve_port: int = 8890
    # one-time startup warmup runs a full dummy generation through the real path so
    # steady-state full-window kernels/NCCL/VAE (+ torch.compile if on) are all hot.
    # Per-tick shapes depend on the sliding window, not total frames, so ~8 chunks
    # (24 frames) warms the same kernels as a long video -> keep it small.
    warmup_frames: int = 24

    # --- output / bench ---
    out_root: str = DEFAULT_OUT_ROOT
    task: str = "wave_rt_phase1"
    run_tag: str = "wrt"
    save_video: bool = True
    ref_latents: str = ""  # single-GPU RF latents.pt for the PSNR gate
    baseline_e2e: float = 0.0

    # --- derived / constants (Wan2.1-T2V-1.3B causal RF) ---
    num_frames_per_block: int = 3
    compile: bool = False  # torch.compile the DiT block (fuse fp32 norms/FFN)

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
        if self.exchange_mode in ("overlap", "onesided", "relay", "staggered", "paged") and self.kv_context != "causal":
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
        """diffusion ranks + VAE-stage ranks (VAE on GPUs wp_size..wp_size+vae_stages-1)."""
        return self.wp_size + max(0, self.vae_stages)

    # ------------------------------------------------------------------ CLI
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
        g.add_argument("--vae-stages", dest="vae_stages", type=int, default=WaveConfig.vae_stages,
                       help="n-stage FLOPs-balanced streaming VAE (0 = serial on store rank); "
                            "total GPUs = wp_size + vae_stages")
        g.add_argument("--vae-port", type=int, default=WaveConfig.vae_port)
        g.add_argument("--vae-partition", type=str, default=WaveConfig.vae_partition,
                       choices=["time", "flops"],
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
                       choices=["torch_sdpa", "fa_custom", "sa", "fa", "fa2"])
        g.add_argument("--quantization", type=str, default=WaveConfig.quantization,
                       help="online DiT quantization passed to sglang ServerArgs "
                            "(e.g. 'fp8' = W8A8 fp8 weights + dynamic fp8 acts). "
                            "None = bf16.")
        g.add_argument("--dit-scale", type=str, default=WaveConfig.dit_scale,
                       choices=["1.3b", "5b", "14b"],
                       help="system-level scaling test: build a larger random-init "
                            "DiT (no RF ckpt; output meaningless). Default = real 1.3b.")
        g.add_argument("--kv-context", type=str, default=WaveConfig.kv_context,
                       choices=["joint", "causal"],
                       help="denoise attention context: 'joint' (all in-flight, current "
                            "default) or 'causal' (chunk-causal / overlap semantics: "
                            "rank r sees only in-flight ranks n>=r).")
        g.add_argument("--exchange-mode", type=str, default=WaveConfig.exchange_mode,
                       choices=["sync", "overlap", "onesided", "relay", "staggered", "paged"],
                       help="KV exchange comm: 'sync' (per-layer all_gather), "
                            "'overlap' (async P2P cascade hidden in compute), "
                            "'onesided' (Gate C-final-1 direct remote-slab publication "
                            "via copy-engine + generation flag), 'relay' "
                            "(neighbor-accumulating one-sided relay: fan-out 1, merged "
                            "single memcpy per hop), 'staggered' (onesided transport "
                            "+ prefetched inbound acquire: producer leads consumer by "
                            "--stagger-lead layers, hiding the tight same-layer edge), or "
                            "'paged' (onesided transport, but producers DMA KV directly "
                            "into fixed slots of the consumer's contiguous attention "
                            "buffer so attention reads in place -- no per-layer torch.cat; "
                            "steady-state ticks only, falls back to onesided during "
                            "fill/drain). "
                            "overlap/onesided/relay/staggered/paged require --kv-context causal.")
        g.add_argument("--stagger-lead", type=int, default=WaveConfig.stagger_lead,
                       help="exchange-mode=staggered: how many layers early to prefetch "
                            "the inbound KV acquire (== producer lead). Default 1.")
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
