<div align="center">
  <img src="page/static/images/wf_v1.svg" width="128" alt="Wave Forcing logo">
  <h1>Wave Forcing</h1>
  <h3>Towards Speed-of-Light Streaming Video Generation</h3>

  <a href="https://sjtu-deng-lab.github.io/WaveForcing/"><img src="https://img.shields.io/badge/Project-Page-20C7C7" alt="Project Page"></a>
  <a href="https://github.com/SJTU-DENG-Lab/WaveForcing"><img src="https://img.shields.io/badge/GitHub-Code-111827?logo=github" alt="GitHub Code"></a>
  <a href="https://huggingface.co/SJTU-DENG-Lab/WaveForcing-T2V-1.3B-5step-Preview"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Preview_Checkpoint-FFD21E" alt="Hugging Face Preview Checkpoint"></a>
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
| WaveForcing-T2V-1.3B-5step | ✅ [Preview checkpoint on Hugging Face](https://huggingface.co/SJTU-DENG-Lab/WaveForcing-T2V-1.3B-5step-Preview) |
| Additional checkpoints | ⏳ More checkpoints coming soon |
| Training code | ⏳ Coming soon |
| Paper / arXiv | ⏳ Coming soon |

> [!NOTE]
> The Hugging Face repository currently contains the **only public checkpoint**:
> a preview 1.3B model trained for five denoising steps. The 4-step and larger
> checkpoints used in parts of the system study are not public yet.

## Quick start: 1.3B 5-step Preview

The example below uses the released checkpoint with an eight-GPU **5+2**
pipeline: five denoising stages, one clean-KV store rank, and two streaming VAE
stages. The runtime is currently validated with Python 3.12, PyTorch
2.11.0+cu128, the SGLang diffusion runtime, and Hopper GPUs. See
[`pyproject.toml`](pyproject.toml) for the pinned Python stack.

### 1. Download the base model and Preview checkpoint

```bash
python -m pip install -U "huggingface_hub[cli]"

hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir ckpts/Wan2.1-T2V-1.3B-Diffusers

hf download SJTU-DENG-Lab/WaveForcing-T2V-1.3B-5step-Preview \
  --local-dir ckpts/WaveForcing-T2V-1.3B-5step-Preview
```

The Preview repository supplies the distilled DiT weights. Text encoder,
tokenizer, VAE, and scheduler components are loaded from the Wan2.1 base model.

### 2. Launch WaveRT

```bash
python -m wave_rt \
  --model-path ckpts/Wan2.1-T2V-1.3B-Diffusers \
  --gen-ckpt ckpts/WaveForcing-T2V-1.3B-5step-Preview/model.safetensors \
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
  --attention-backend torch_sdpa \
  --cuda-visible-devices 0,1,2,3,4,5,6,7 \
  --prompt "A cinematic shot of a fluffy corgi running on a sunny beach, waves in the background." \
  --task preview \
  --run-tag corgi
```

The generated video and metrics are written to:

```text
outputs/preview/corgi/video.mp4
outputs/preview/corgi/metrics.json
```

`--num-frames` counts latent frames and must be divisible by three. Use
`--num-frames 399` to match the long-video benchmark protocol below; this
produces 1596 output frames (approximately 100 seconds at 16 FPS).

## Current results

The following numbers are measured on **8× NVIDIA H200 GPUs** with 399 latent
frames / 1596 output frames. Steady-state numbers pool
`per_tick_ms[12:-4]` across three repetitions and report p50. DiT FPS includes
pipeline fill/drain; E2E FPS additionally includes the streaming VAE.

| Model | Topology | Runtime configuration | Steady FPS (p50) | DiT FPS | E2E FPS |
|---|---:|---|---:|---:|---:|
| 1.3B | 4+3 | Sage · causal · paged | 125.7 | 124.1 | **117.7** |
| 1.3B | 4+3 | Sage + W8A8 · causal · paged | **126.5** | 121.5 | 115.9 |
| 1.3B Preview | 5+2 | BF16 / SDPA · causal · paged | 86.5 | 84.9 | **84.2** |
| 14B system prototype¹ | 4+3 | Sage + W8A8 · causal · paged | 28.5 | 28.1 | **28.1** |

The best 1.3B 4+3 run generates 1596 frames in approximately **13.56 seconds**,
or **117.7 E2E FPS**, versus the matched 15.0 FPS single-GPU baseline
(**7.9× speedup**). The maximum valid steady-state throughput is **126.5 FPS**.

The released five-step Preview uses the 5+2 topology. Its two-stage VAE is too
coarse to hide all decoding work: the best reliable BF16 configuration reaches
**84.2 E2E FPS**, even though faster DiT configurations exceed 120 FPS. The full
three-stage VAE in 4+3 follows the DiT cadence more closely and avoids this
decoder tail.

¹ The 14B row uses shape-accurate random-initialized weights and measures system
scaling only; it is not a generation-quality result.

For every configuration, precise metric definitions, and the synchronized vs.
overlapped comparison, see the
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

The arXiv entry is not public yet. For now, please cite the project page:

```bibtex
@misc{denglab2026waveforcing,
  title        = {Wave Forcing: Towards Speed-of-Light Streaming Video Generation},
  author       = {{DENG Lab MLSys Team}},
  year         = {2026},
  howpublished = {Project blog},
  organization = {Shanghai Jiao Tong University},
  url          = {https://sjtu-deng-lab.github.io/WaveForcing/}
}
```

## License

Released under the [Apache License 2.0](LICENSE).
