# WaveRT package layout

- `config.py` and `launcher.py` define the public configuration and process entrypoint.
- `runtime/` integrates SGLang, model variants, environment flags, and FP8 linears.
- `runtime/model.py` and `model_base.py` implement the optimized **unsharded** Wan
  forward; `compute.py`, `kernels.py`, `projection.py`, and `conditioning.py` own
  its local computation and request caches. There is no SP/TP runtime dependency.
- `denoiser/engine.py` owns lifecycle and the systolic schedule.
- `denoiser/exchange/` contains independent KV transport strategies.
- `denoiser/diagnostics.py` contains optional profiling and report generation.
- `distributed/` contains process-group compatibility and CUDA IPC primitives.
- `pipelines/` contains the streaming VAE pipeline.

Runtime code should import implementations from these package paths directly.

## WaveParallel optimization migration

The Wave execution path now includes these optimizations from WaveParallel:

| Area | Implementation |
|---|---|
| Attention | Explicit FA3 (`flash_attn_interface`) / FA2 (`flash_attn`) dispatch, fused Sage preparation and versioned SM90 statistics |
| Local DiT | Packed QKV GEMM, text K/V cache, fixed-stage timestep/modulation caches, axis RoPE, fused QK norm/RoPE/final-address packing, fused residual/next-block normalization |
| Clean store | Last block computes K/V only; skip its attention/FFN and the output head |
| Scheduling | Cached sigma lookup preserving upstream integer/fractional timestep semantics; eligible Sage pointwise kernels |
| KV/history | Reusable allocations and clean pages, fused paged context assembly, one payload slot per layer, generation credits, direct final-address publication and Sage KV promotion |
| Loading | Rank-0 text conditioning broadcast, role-specific text/VAE loading, tiny native KV placeholders without allocating a full unused cache |
| Output/VAE | Bounded asynchronous BF16 D2H staging, sgldiff causal decoder partitioning, measured-time balancing, retained partition weights only, reusable frame channels, chunkwise video encoding |
| Lifecycle | Supervised workers and output metadata; retain WaveForcing's coordinated IPC close and noncollective exception cleanup |

Local DiT optimizations apply to `--kv-context causal` with `sync`, `overlap`,
`onesided`, `staggered`, or `paged`. The legacy joint and relay paths remain
available but do not use the optimized local model. No SP, TP, Ulysses, Ring,
or multi-method comparison runtime is included.

FA3 is now the default attention backend. Install the official FlashAttention
Hopper package (`flash-attn-3`, importing `flash_attn_interface`) into the existing
CUDA/SGLang environment. FA2 uses the separate `flash-attn` package. Missing
selected backends and kernel failures propagate without automatic substitution.
Historical `fa_custom` and `fa` CLI aliases resolve to FA2.

The optimized local model uses FA3 cross-attention even when self-attention is
`sa` or `torch_sdpa`; selecting `fa2` uses FA2 for both. It also needs Triton.
Sage's fused preparation is enabled only for the supported SM90/SageAttention
2.2.0/head-dimension contract; other shapes use the public Sage implementation.
`--quantization fp8` retains post-checkpoint W8A8 quantization; use
`WAVE_FP8_SCOPE=all` to include packed QKV, and `WAVE_FP8_COMPILE=1` to opt into
compiled activation quantization. `--kv-transport-dtype fp8` is an independent,
experimental, unscaled E4M3 payload cast for `onesided` only; BF16 remains default.

Example with the public four-step checkpoint (eight GPUs):

```bash
python -m wave_rt \
  --model-path ckpts/Wan2.1-T2V-1.3B-Diffusers \
  --gen-ckpt ckpts/WaveForcing-1.3B/4step/model.safetensors \
  --rf-step 4 --wp-size 5 --denoising-step-list 1000,750,500,250 \
  --num-frames 42 --vae-stages 3 --vae-partition time \
  --kv-context causal --exchange-mode paged --attention-backend fa3
```

## Validation

FPS benchmarking uses one full resident warmup request before each measured
request (default: 399 latent frames for both). The same diffusion and VAE workers
remain loaded across those requests, and phase barriers keep warmup decode and
cache cleanup out of the measured request. Each repetition starts fresh workers
and performs its own full warmup:

```bash
python -m scripts.bench_wave_fps --model-path MODEL --dummy \
  --frames 399 --warmup-frames 399 --warmup-runs 1 --repeats 3
```

Warmup frames default to the measured frame count. `--warmup-runs 0` explicitly
restores the old kernel-only preflight, which is not the same benchmark protocol.
For direct launches, use `--full-warmup-runs 1 --warmup-frames 399`.

CPU checks exercise an independent dense model reference versus packed/cached
computation across requests, clean-tail pruning, schedule precision, FA dispatch,
page retention/credits, and IPC lifecycle:

```bash
python -m unittest discover -s tests -v
```

Run the following in the CUDA/SGLang environment before accepting GPU numerical
parity or performance. These checks require real kernels and fail if dependencies
are missing; the CPU checks do not establish CUDA/NCCL correctness.

```bash
python -m scripts.check_wave_optimized_kernels
python -m scripts.check_wave_quantization
python -m scripts.check_wave_vae_stream
torchrun --standalone --nproc-per-node=3 -m scripts.check_wave_paged_slots
python -m scripts.check_wave_wp_model --model-path MODEL --gen-ckpt CHECKPOINT
python -m scripts.check_wave_transports --model-path MODEL --gen-ckpt CHECKPOINT
```

The model oracle runs on one GPU with two model copies and checks actual native
Wan against the optimized adapter, including repeated requests and history
eviction. The transport gate uses at least 42 latent frames and compares real
generated latents against causal+sync. Use `--steps 1000,800,600,400,200
--vae-stages 2` for a five-step checkpoint. Re-measure performance under identical
checkpoint, shape, topology and precision settings; historical README results
are not measurements of this migration.
