# WaveRT package layout

- `config.py` and `launcher.py` define the public configuration and process entrypoint.
- `runtime/` integrates SGLang, model variants, environment flags, and FP8 linears.
- `denoiser/engine.py` owns lifecycle and the systolic schedule.
- `denoiser/exchange/` contains independent KV transport strategies.
- `denoiser/diagnostics.py` contains optional profiling and report generation.
- `distributed/` contains process-group compatibility and CUDA IPC primitives.
- `pipelines/` contains the streaming VAE pipeline.
- `serving/` contains the typed request protocol, FIFO admission, resident
  worker supervision, bounded full-set recovery, and HTTP adapter.

Runtime code should import implementations from these package paths directly.

Serving keeps model weights and process groups resident, but reconstructs prompt
conditioning, KV/cross-attention caches, noise, latent state, and the VAE feature
cache for every request. Requests are tagged end to end so stale completion
events cannot cross session boundaries. If a rank exits or a request times out,
the complete distributed worker set is poisoned and replaced within the
configured `--serve-max-restarts` budget.
