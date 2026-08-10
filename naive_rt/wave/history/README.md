# Wave-Parallel History: Milestone Scripts

Runnable snapshots showing the iterative optimization from naive single-GPU
baseline to pipelined multi-GPU wavefront inference.

## Progression

### s0 -- Naive Baseline (single GPU)
Sequential rolling-forcing inference on one GPU. No parallelism.
- Diffusion: ~55 s, VAE: ~19 s, Total: ~74 s, **~13.6 fps**
```
python -m naive_rt.wave.history.s0_naive_baseline --num_output_frames 252
```

### s1 -- Wavefront Simulator (single GPU, correctness oracle)
Runs the wavefront schedule on a single GPU and compares against naive to
verify the decomposition is mathematically correct (~41 dB PSNR).
```
python -m naive_rt.wave.history.s1_wave_sim --num_output_frames 21 --compare_naive
```

### s2 -- 5-Rank NCCL Wavefront (serial VAE)
5 GPUs form a systolic wavefront (4 denoise + 1 store), but VAE decode
still runs serially after all diffusion is done.
- Diffusion: ~17.8 s, Total with serial VAE: ~37 s, **~2x speedup**
```
python -m naive_rt.wave.history.s2_wave_dist --num_output_frames 252
```

### s3 -- Decoupled VAE Decode (mp.Queue bridge)
Same 5-rank diffusion, but VAE runs as an independent process on GPU 5,
receiving chunks via mp.Queue as they are finalized. VAE latency is hidden.
- End-to-end: ~22.4 s, **~45 fps, 3.3x speedup**
```
python -m naive_rt.wave.history.s3_wave_serve --num_output_frames 252
```

### Current: serve.py (3-stage VAE pipeline)
Production version at `naive_rt/wave/serve.py`. Adds 3-stage pipelined VAE
decode across 3 additional GPUs (8 GPUs total), further reducing latency.

## Common arguments
- `--gen_ckpt PATH`: Generator checkpoint (default: longvideo.pt)
- `--prompt TEXT`: Text prompt for generation
- `--seed INT`: Random seed (default: 0)
- `--num_output_frames INT`: Number of latent frames (must be multiple of 3)
- `--save_video PATH` / `--dump PATH`: Save output video
