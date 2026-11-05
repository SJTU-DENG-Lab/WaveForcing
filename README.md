<div align="center">
  <img src="page/static/images/wf_v2.svg" width="132" alt="Wave Forcing logo">
  <h1>Wave Forcing</h1>
  <h3>Towards Speed-of-Light Streaming Video Generation</h3>

  <a href="https://arxiv.org/abs/2611.00859"><img src="https://img.shields.io/badge/arXiv-2611.00859-b31b1b" alt="arXiv paper"></a>
  <a href="https://sjtu-deng-lab.github.io/WaveForcing/"><img src="https://img.shields.io/badge/Project-Page-20C7C7" alt="Project Page"></a>
  <a href="https://github.com/SJTU-DENG-Lab/WaveForcing"><img src="https://img.shields.io/badge/GitHub-Code-111827?logo=github" alt="GitHub Code"></a>
  <a href="https://huggingface.co/SJTU-DENG-Lab/WaveForcing-1.3B"><img src="https://img.shields.io/badge/%F0%9F%A4%97-WaveForcing--1.3B-FFD21E" alt="Hugging Face WaveForcing-1.3B"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-6B5BFF" alt="Apache 2.0 License"></a>
</div>

Wave Forcing is a block-causal streaming video diffusion method, paired with
**Wave Parallelism** and the **WaveRT** runtime. It preserves a mixed-noise
frontier and fresh cleaner-to-noisier context while removing the reverse
dependencies that prevent efficient pipelining. Across GPUs, full-model replicas
process different chunk-stage tasks and publish layer-wise K/V states in one
direction. Communication is prefetched and hidden under the longer attention
computation on the critical rank.

The inference/runtime code in this repository is now open source.

## Release status

| Asset | Status |
|---|---|
| Inference code and WaveRT | ✅ Available in this repository |
| Interactive project page and full configuration matrix | ✅ [Available here](https://sjtu-deng-lab.github.io/WaveForcing/) |
| WaveForcing-1.3B, 5-step | ✅ [`5step/` on Hugging Face](https://huggingface.co/SJTU-DENG-Lab/WaveForcing-1.3B) |
| WaveForcing-1.3B, 4-step | ✅ [`4step/` on Hugging Face](https://huggingface.co/SJTU-DENG-Lab/WaveForcing-1.3B) |
| Additional checkpoints | ⏳ More checkpoints coming soon |
| Reference training code | [S1/S2/S3 source and recipes](training/README.md) |
| Paper / arXiv | ✅ [arXiv:2611.00859](https://arxiv.org/abs/2611.00859) |

> [!NOTE]
> Both public checkpoints live in one repository,
> [SJTU-DENG-Lab/WaveForcing-1.3B](https://huggingface.co/SJTU-DENG-Lab/WaveForcing-1.3B):
> `5step/` and `4step/`, each a native Wan `model.safetensors` (BF16, EMA weights).
> A mirror is on [ModelScope](https://modelscope.cn/models/Osc7YuxuanLiu/WaveForcing-1.3B).

## Quick start: WaveForcing-1.3B

The example below uses the 5-step checkpoint with an eight-GPU **5+2**
pipeline: five denoising stages, one clean-KV store rank, and two streaming VAE
stages. The runtime is currently validated with Python 3.12, PyTorch
2.11.0+cu128, the SGLang diffusion runtime, and Hopper GPUs. See
[`pyproject.toml`](pyproject.toml) for the pinned Python stack.

### 1. Download the base model and WaveForcing-1.3B

```bash
python -m pip install -U "huggingface_hub[cli]"

hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir ckpts/Wan2.1-T2V-1.3B-Diffusers

hf download SJTU-DENG-Lab/WaveForcing-1.3B \
  --local-dir ckpts/WaveForcing-1.3B
```

WaveForcing-1.3B supplies the distilled DiT weights. Text encoder,
tokenizer, VAE, and scheduler components are loaded from the Wan2.1 base model.

### 2. Launch WaveRT

WaveRT now includes the WaveParallel local-compute and KV transport
optimizations, with explicit FlashAttention 3 support. The optimized causal path
requires the official Hopper `flash-attn-3` package (`flash_attn_interface`) and
Triton in the CUDA/SGLang environment. See the [runtime guide](wave_rt/README.md)
for backend choices, the optimization inventory, and CPU/GPU validation commands.

```bash
python -m wave_rt \
  --model-path ckpts/Wan2.1-T2V-1.3B-Diffusers \
  --gen-ckpt ckpts/WaveForcing-1.3B/5step/model.safetensors \
  --rf-step 5 \
  --wp-size 6 \
  --denoising-step-list 1000,800,600,400,200 \
  --num-frames 24 \
  --height 480 \
  --width 832 \
  --vae-stages 2 \
  --vae-partition time \
  --kv-context causal \
  --exchange-mode paged \
  --attention-backend fa3 \
  --cuda-visible-devices 0,1,2,3,4,5,6,7 \
  --prompt "A cinematic shot of a fluffy corgi running on a sunny beach, waves in the background." \
  --task quickstart \
  --run-tag corgi
```

The generated video and metrics are written to:

```text
outputs/quickstart/corgi/video.mp4
outputs/quickstart/corgi/metrics.json
```

`--num-frames` counts latent frames and must be divisible by three. Use
`--num-frames 399` to match the long-video benchmark protocol below; this
produces 1593 output frames (approximately 100 seconds at 16 FPS).

### 3. Run the 4-step checkpoint

Point `--gen-ckpt` at `ckpts/WaveForcing-1.3B/4step/model.safetensors` and set
`--rf-step 4 --wp-size 5 --denoising-step-list 1000,750,500,250 --vae-stages 3`.
This is the **4+3** layout used for the headline numbers below: four denoising
stages, one store rank, and three streaming VAE stages on the same eight GPUs.
The quick start above is the **5+2** layout, which the results table also lists.

## Reference training

The [training guide](training/README.md) documents the S1/S2/S3 recipe, asset
preparation, launch, and resume commands. Training uses the same Python 3.12 /
PyTorch 2.11.0+cu128 environment as WaveRT, under a separate
[academic-use license](training/LICENSE).

## Current results

The following numbers are measured on **8× NVIDIA H200 GPUs** at 832×480 with
399 latent frames (1593 decoded frames). Steady-state values pool per-tick
timings across three repetitions and report p50; DiT FPS includes pipeline
fill/drain, and E2E FPS additionally includes the streaming VAE. Loading and
file encoding are excluded.

| Model | Steps | Topology | Runtime configuration | Steady FPS (p50) | DiT FPS | E2E FPS |
|---|---:|---:|---|---:|---:|---:|
| 1.3B | 4 | 4+3 | BF16/FA3 · causal · paged | 152.1 | 147.6 | **134.8** |
| 1.3B | 4 | 4+3 | Sage · causal · paged | 153.2 | **147.8** | 134.8 |
| 1.3B | 4 | 4+3 | Sage + W8A8 · causal · paged | **157.8** | 145.5 | 132.9 |
| 1.3B | 5 | 5+2 | BF16/FA3 · causal · paged | 99.6 | 104.8 | 96.5 |
| 14B system prototype¹ | 4 | 4+3 | Sage + W8A8 · causal · paged | 33.9 | 33.6 | **33.5** |

The best 1.3B 4+3 run decodes 1593 frames in approximately **11.8 seconds**, or
**134.8 E2E FPS**, versus the matched 19.06 FPS single-GPU baseline
(**7.07× speedup**). The maximum steady-state throughput is **157.8 FPS**.

¹ The 14B row uses shape-accurate random-initialized weights and measures system
scaling only; it is not a generation-quality result. The 1.3B rows use the
released 4-step and 5-step checkpoints.

For every configuration, precise metric definitions, and the exchange-mode
ablation, see the
[interactive results matrix](https://sjtu-deng-lab.github.io/WaveForcing/#results).
The machine-readable summary is available at
[`page/static/data/results_summary.json`](page/static/data/results_summary.json).

## Runtime layout

```text
wave_rt/
├── denoiser/              # wavefront schedule, diagnostics, and KV exchange
│   └── exchange/          # sync, overlap, one-sided, staggered, relay, paged
├── runtime/               # SGLang integration, attention backends, FP8
├── distributed/           # process-group compatibility and CUDA IPC
├── pipelines/             # streaming VAE pipeline
├── config.py              # public configuration and CLI arguments
└── launcher.py            # one-shot and resident serving launchers
```

## Citation

Please cite our [arXiv paper](https://arxiv.org/abs/2611.00859):

```bibtex
@misc{denglab2026waveforcing,
  title        = {Wave Forcing: Towards Speed-of-Light Streaming Video Generation},
  author       = {Jin, Yijie and Liu, Yuxuan and Xu, Jiajun and Huang, Ye and
                  Wang, Chuzhi and Li, Jiayang and Chen, Ruiyang and Hong, Chenling and
                  Wang, Xu and Deng, Zhijie},
  year         = {2026},
  eprint       = {2611.00859},
  archivePrefix = {arXiv},
  url          = {https://arxiv.org/abs/2611.00859}
}
```

## License

The WaveRT code is released under the [Apache License 2.0](LICENSE).
The RollingForcing-derived reference trainer retains its separate
[academic-use license](training/LICENSE).
