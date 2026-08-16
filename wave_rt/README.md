# WaveRT package layout

- `config.py` and `launcher.py` define the public configuration and process entrypoint.
- `runtime/` integrates SGLang, model variants, environment flags, and FP8 linears.
- `denoiser/engine.py` owns lifecycle and the systolic schedule.
- `denoiser/exchange/` contains independent KV transport strategies.
- `denoiser/diagnostics.py` contains optional profiling and report generation.
- `distributed/` contains process-group compatibility and CUDA IPC primitives.
- `pipelines/` contains the streaming VAE pipeline.

Runtime code should import implementations from these package paths directly.
