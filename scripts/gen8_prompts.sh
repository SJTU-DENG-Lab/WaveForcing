#!/usr/bin/env bash
# 8 prompts x 6 configs = 48 generation runs (399 latent frames = 100 s video).
#
#   baseline-rf5   naive 5+2  5-step [1000,800,600,400,200]   RF-5step (lyx)
#   baseline-cf4   naive 4+3  4-step [1000,750,500,250]       longvideo (zhuhz22)
#   sync-full-rf5  wave_rt joint+sync    5+2  5-step           RF-5step
#   sync-full-cf4  wave_rt joint+sync    4+3  4-step           longvideo
#   overlap-wf4    wave_rt causal+overlap 4+3  4-step          RF-5step (wf 4step = rf weights + causal mask)
#   overlap-wf5    wave_rt causal+overlap 5+2  5-step          RF-5step
#
# Outputs: outputs/gen8/<config>/p<1..8>/{video.mp4, metrics.json, latents.pt}
# Prompts recorded to outputs/gen8/prompts.txt for run_tag lookup.
#
# Run on the 8-GPU box from the repo root (NOT on the CPU-only local box):
#   nohup bash scripts/gen8_prompts.sh > logs/gen8_run.log 2>&1 &
set -u

cd "$(dirname "$0")/.."

OUT=outputs/gen8
SEED=0
RF_CKPT=ckpts/lyx/RF-5step-1.3B/model.pt
CF_CKPT=ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt
FAILS=""

PROMPTS=(
  "A cinematic shot of a fluffy corgi running on a sunny beach, waves in the background."
  "A cinematic shot of a fluffy corgi walking through a neon-lit city street at night, rain reflecting the lights."
  "A cinematic shot of a fluffy corgi playing in fresh snow in a pine forest, snowflakes falling."
  "A cinematic shot of a fluffy corgi sitting under blooming cherry blossom trees, petals drifting in the wind."
  "A cinematic shot of a fluffy corgi trotting along a forest trail in golden autumn light."
  "A cinematic shot of a fluffy corgi watching a sunrise over a calm lake, mist on the water."
  "A cinematic shot of a fluffy corgi on a mountain summit overlooking clouds at sunset."
  "A cinematic shot of a fluffy corgi chasing a frisbee in a sunny meadow with wildflowers."
)

mkdir -p "$OUT"
: > "$OUT/prompts.txt"
for i in "${!PROMPTS[@]}"; do
  printf "p%d\t%s\n" "$((i + 1))" "${PROMPTS[$i]}" >> "$OUT/prompts.txt"
done

# naive baseline (serve.py, 8 GPUs: nstep+1 diffusion + vae_stages)
run_naive() { # config nstep steps vae_stages ckpt
  local cfg=$1 nstep=$2 steps=$3 vae=$4 ckpt=$5
  for i in "${!PROMPTS[@]}"; do
    local tag="p$((i + 1))"
    echo "[$(date +%H:%M:%S)] $cfg $tag (naive ${nstep}-step) ..." | tee -a "$OUT/run.log"
    if ! python -m naive_rt.wave.serve \
        --num_output_frames 399 --nstep "$nstep" --steps "$steps" \
        --vae-stages "$vae" --gen_ckpt "$ckpt" \
        --prompt "${PROMPTS[$i]}" --seed $SEED \
        --out_root "$OUT" --task "$cfg" --run_tag "$tag" \
        >> "$OUT/run.log" 2>&1; then
      echo "FAIL $cfg $tag" >> "$OUT/run.log"; FAILS="$FAILS $cfg/$tag"
    fi
  done
}

# wave_rt (8 GPUs, full pipeline + KV exchange)
run_wave() { # config rf_step wp_size steps vae_stages kv exchange ckpt
  local cfg=$1 rf=$2 wp=$3 steps=$4 vae=$5 kv=$6 exch=$7 ckpt=$8
  for i in "${!PROMPTS[@]}"; do
    local tag="p$((i + 1))"
    echo "[$(date +%H:%M:%S)] $cfg $tag (wave ${rf}-step $kv/$exch) ..." | tee -a "$OUT/run.log"
    if ! python -m wave_rt \
        --rf-step "$rf" --wp-size "$wp" --denoising-step-list "$steps" \
        --num-frames 399 --vae-stages "$vae" \
        --kv-context "$kv" --exchange-mode "$exch" \
        --gen-ckpt "$ckpt" --prompt "${PROMPTS[$i]}" --seed $SEED \
        --task "$cfg" --run-tag "$tag" --out-root "$OUT" \
        --cuda-visible-devices 0,1,2,3,4,5,6,7 \
        >> "$OUT/run.log" 2>&1; then
      echo "FAIL $cfg $tag" >> "$OUT/run.log"; FAILS="$FAILS $cfg/$tag"
    fi
  done
}

run_naive baseline-rf5 5 "1000,800,600,400,200" 2 "$RF_CKPT"
run_naive baseline-cf4 4 "1000,750,500,250"     3 "$CF_CKPT"
run_wave  sync-full-rf5 5 6 "1000,800,600,400,200" 2 joint sync "$RF_CKPT"
run_wave  sync-full-cf4 4 5 "1000,750,500,250"     3 joint sync "$CF_CKPT"
run_wave  overlap-wf4   4 5 "1000,750,500,250"     3 causal overlap "$RF_CKPT"
run_wave  overlap-wf5   5 6 "1000,800,600,400,200" 2 causal overlap "$RF_CKPT"

echo "[$(date +%H:%M:%S)] all done. failures:$FAILS" | tee -a "$OUT/run.log"
