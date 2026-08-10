# wave-parallel

Wavefront pipeline parallelism for long-video generation (Rolling Forcing on
Wan2.1-T2V-1.3B). Two runtimes:

- **`naive_rt/`** — self-contained wavefront runtime (vendored Wan/RF model, bare
  NCCL). The reference that proved the approach (4.4× / 60 fps).
- **`wave_rt/`** — the same wavefront but reusing the **sglang (sgl-dlab) diffusion
  backend as a library** (sgl model/attention/VAE), driven by a runtime monkeypatch;
  our current integration target.

## Results (252 latent = 1005 pixel frames, 8× H200, BF16, 4-step)

| Runtime | GPUs | diffusion | e2e | FPS |
|---|---:|---:|---:|---:|
| naive single-GPU | 1 | — | 74.0s | 13.6 |
| naive wave-parallel (3-stage VAE) | 8 | 16.7s | 16.7s | 60 |
| **wrt (sgl backend) + n-stage streaming VAE** | 8 | 13.7s | 14.3s | **70.5** |

wrt: diffusion tick ~150ms (matches/beats naive); VAE is an **n-stage
streaming pipeline** (naive Wan VAE, bf16) fully hidden behind diffusion.

### Attention + VAE-split optimizations (SageAttention · time-balanced VAE)

Two knobs push wrt well past the table above (96 latent / 384 px frames, 8 GPUs):

| attention | VAE split | diffusion | e2e | FPS |
|---|---|---:|---:|---:|
| `torch_sdpa` | `flops` (old) | 5.36s | 6.56s | 58.5 |
| `sa` (Sage) | `flops` | 4.37s | 6.69s | 57.4 |
| `torch_sdpa` | `time` | 5.35s | 5.49s | 69.9 |
| **`sa` (Sage)** | **`time`** | **3.95s** | **4.09s** | **93.9** |

60s long video (240 latent / 957 px frames) at the best config: **9.62s, 99.8 fps
(6.2× realtime)**.

- **`--vae-partition time`** (default): split the VAE decoder across stages by
  *measured* per-unit time, not FLOPs. FLOPs is a poor proxy here — the high-res
  upsample blocks have ~equal FLOPs but ~6× the time (memory-bound at 480×832), so
  the FLOPs split is ~1.8× time-imbalanced at n=3. Timing → VAE 146→99 ms/chunk,
  bit-exact vs the FLOPs split. `--vae-partition flops` reverts.
- **`--attention-backend sa`** (SageAttention, int8): cuts the diffusion tick
  159→115ms (1.23×). **Only pays off together with `time`** — alone it's wasted
  because the pipeline stays VAE-bound. Default stays `torch_sdpa`; see the build
  step in Setup. `torch.compile` on the VAE is a no-op here (compute-bound, not
  launch-bound), so it is left off.
- **Quality**: `time` is bit-exact. Sage's int8 is near-lossless per step; long
  videos drift (oversaturation/texture artifacts after ~20–30s) but that is
  **inherent to the Wan2.1-1.3B rolling-forcing checkpoint** — `sdpa` and `sa`
  drift near-identically, so Sage is safe. Real usable window ≈ 15–20s.


## Setup

This repo vendors sglang as a **git submodule** (`sglang/` →
`SJTU-DENG-Lab/sglang` @ `wm2/rolling-forcing`).

```bash
# clone with the submodule
git clone --recursive <this-repo>            # or, if already cloned:
git submodule update --init                  #   (fetches sglang/ from GitHub)
```

The environment is **hand-assembled** (Python 3.12 + torch 2.11.0+cu128 + editable
sglang + source-built flash-attn + specific pins). **Do NOT run `uv sync`** — it will
re-resolve torch to the wrong CUDA build and clobber the pins. Full step-by-step recipe:
**[`docs/10_env_py312_setup.md`](docs/10_env_py312_setup.md)**. It is already built at
`.venv/`; the notes below are the gotchas that cost real time:

- **Python 3.12** (3.10 dropped; sgl-kernel/torchcodec/flash-attn have reliable cp312 wheels).
- **torch must be the `+cu128` build** — plain `torch==2.11.0` on PyPI is the **cu130**
  build → "NVIDIA driver too old" / `cuda.is_available==False` on the 12.9 driver.
  Install `torch==2.11.0+cu128` explicitly from the aliyun cu128 mirror.
- **The sgl editable install re-pulls cu130 torch** (its `torch==2.11.0` pin) →
  reinstall the `+cu128` trio **last**.
- **`nvidia-cutlass-dsl==4.5.1`** — 4.5.2 breaks flashinfer's rmsnorm CUTE kernel
  (`TypeError: incompatible function arguments` in the DiT qk-norm forward).
- **flash-attn**: no cp312/torch2.11 wheel exists — build from source (mirror has
  2.8.3.post1); the built wheel is cached in `fa-dists/`.
- **`SGLANG_USE_RUNAI_MODEL_STREAMER=0`** — the RunAI streamer does a distributed
  all_gather during weight load that crashes the multi-proc launch (the launcher sets this).
- **venv on local SSD, not network storage** — the latter is NFS and makes imports 5–15× slower.
  Put the uv cache on network storage (`UV_CACHE_DIR`) so downloads don't fill local SSD.
- **cu13 libs**: torch 2.11 pulls `nvidia/*-cu13`; `deep_gemm` needs
  `LD_LIBRARY_PATH` to include `.venv/.../nvidia/cu13/lib`. `python -m wave_rt`
  auto-injects this (re-exec); bare scripts must set it.
- Building the sgl grpc extension needs a Rust toolchain + `protoc` (see docs/08, docs/10).

### SageAttention (optional, for `--attention-backend sa`)

Enables the int8 attention path (1.23× on the diffusion tick). **Optional** — the
default `torch_sdpa` needs nothing extra. The build is source-only on Hopper:

```bash
# PyPI only ships sageattention 1.0.6 (old Ampere API); Hopper sm90 needs 2.x.
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=48 CUDA_HOME=/usr/local/cuda \
  /path/to/.venv/bin/pip install --no-build-isolation --no-deps -e .
# ~10 min compile -> installs sageattention 2.2.0
/path/to/.venv/bin/python -c "from sageattention import sageattn; print('ok')"
```

- **nvcc ≥ 12.4** required for the sm90 kernels (this box has 12.9); set `CUDA_HOME`.
- `--no-build-isolation --no-deps` so it reuses the installed `+cu128` torch instead
  of re-resolving it (which would pull cu130, as elsewhere).
- Not tracked by this repo (it lives outside the tree / in `.venv`); rebuild per env.
- Enable at runtime with `--attention-backend sa` (see the optimization table above
  — pair it with `--vae-partition time` or it is wasted).


## Quick Start

```bash
cd /path/to/WaveParallel

# wrt: 5 diffusion + 3 streaming VAE = 8 GPUs, 4-step, long video
.venv/bin/python -m wave_rt --wp-size 5 --rf-step 4 --num-frames 252 --vae-stages 3 \
    --cuda-visible-devices 0,1,2,3,4,5,6,7

# fastest config: SageAttention + time-balanced VAE split (needs the Sage build)
.venv/bin/python -m wave_rt --wp-size 5 --rf-step 4 --num-frames 240 --vae-stages 3 \
    --attention-backend sa --vae-partition time \
    --cuda-visible-devices 0,1,2,3,4,5,6,7

# resident serving: load once, HTTP /generate with a FIFO queue (concurrent submit,
# sequential generation). One startup warmup runs a full dummy generation first.
.venv/bin/python -m wave_rt --serve --wp-size 5 --rf-step 4 --vae-stages 3 \
    --attention-backend sa --vae-partition time \
    --cuda-visible-devices 0,1,2,3,4,5,6,7 &
curl -s -H "Content-Type: application/json" localhost:8890/generate \
    -d '{"prompt":"a red fox in snow","num_frames":48,"seed":7}'   # -> video path + fps
curl -s localhost:8890/status        # {"pending": N}
curl -s -X POST localhost:8890/shutdown

# naive wave-parallel (5 diffusion + 3 VAE), 252 latent frames
.venv/bin/python -m naive_rt.wave.serve --nstep 4 --num_output_frames 252 --dump outputs/out.mp4
```

Key wrt flags: `--wp-size N+1 --rf-step N` (N denoise + 1 store), `--vae-stages n`
(0 = serial VAE; n = streaming pipeline, needs `wp_size+n` GPUs), `--vae-partition
time|flops` (time = measured-time split, default), `--attention-backend sa`
(SageAttention int8; needs the optional build), `--num-frames` (latent frames, ×3),
`--serve [--serve-port 8890 --warmup-frames 24]` (resident HTTP serving),
`--ref-latents` (PSNR gate).

## Package Structure

```
wave-parallel/
├── sglang/                     # git submodule -> SJTU-DENG-Lab/sglang @ wm2/rolling-forcing
├── wave_rt/                    # wrt: wavefront on the sgl backend
│   ├── backend.py              #   the ONLY module touching sgl internals (attn monkeypatch)
│   ├── dist_compat.py          #   bare-NCCL + degree-1 shim (no sgl GroupCoordinators)
│   ├── wavefront.py            #   systolic diffusion loop + store-rank streaming to VAE
│   ├── vae.py                  #   n-stage streaming VAE (naive Wan VAE, bf16) + serial fallback
│   ├── launcher.py / config.py / bench.py
├── naive_rt/                   # naive runtime (reference)
│   ├── wave/
│   │   ├── serve.py            #   (nstep+1) diffusion + n-stage VAE pipeline
│   │   ├── vae_pipe.py         #   reusable: layer FLOPs + measured-time split + apply_unit
│   │   ├── layer_pipeline.py   #   WaveLayerPipeline oracle (_qkv/_finish/_embed_chunk)
│   │   └── history/            #   s0..s3 milestone scripts (read in order)
│   └── rolling_forcing/        #   vendored minimal Causal-Forcing/long_video subset
├── scripts/                    # smoke / ref / probe / cmp helpers
└── docs/                       # devlog + experiments + feasibility + env setup (00..10)
```

## Architecture

**Diffusion** (both runtimes): systolic wavefront — `rf_step` denoise ranks + 1 store
rank. Per-layer all_gather exchanges in-flight KV; the store rank contributes clean KV
in real time. Deterministic re-noise (seed `1000*chunk+stage`) for reproducibility.
wrt owns distribution the naive way (bare NCCL + degree-1 shim, zero sgl GroupCoordinators).

**VAE** (`naive_rt/wave/vae_pipe.py`): the Wan causal 3D decoder is a sequential chain
of decoupled layers, split at inter-layer boundaries into **n stages** (DP min-max).
The split is balanced by **measured per-unit time** (`--vae-partition time`, default;
`measure_unit_times` at init on rank 0, cuts broadcast over the VAE NCCL world) rather
than analytic FLOPs, which mis-estimates the memory-bound high-res upsample blocks
(`scripts/vae_balance_probe.py` quantifies this). Each stage owns its feat_cache;
per-frame p2p pipeline, bit-exact vs single-GPU. Streamed from the store rank via an
mp.Queue as chunks finalize, overlapped with diffusion. Two NCCL worlds (diffusion /
VAE) are disjoint (different ports, non-overlapping ranks) bridged only by the CPU
queue — avoids in-job NCCL deadlock.

## Docs

`docs/` (read in order): `04` main devlog · `05` experiments · `07` SP/USP analysis ·
`08` sgl RF port · `09` wrt-on-sgl feasibility · `10` py3.12 env setup. History of the
naive milestones: `naive_rt/wave/history/README.md`.

## Environment

- Python 3.12, PyTorch 2.11.0+cu128, flash-attn 2.8.3.post1 (cp312 sm_90, in `fa-dists/`),
  sglang editable (submodule), nvidia-cutlass-dsl 4.5.1.
- Hardware: 8× H200, driver 570 / CUDA 12.9.
- Model: Wan2.1-T2V-1.3B, 4-step DMD `[1000,750,500,250]` (Rolling Forcing chunkwise, longvideo.pt).
```
