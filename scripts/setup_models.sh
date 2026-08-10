#!/usr/bin/env bash
# Idempotent model-layout setup for WaveParallel (both runtimes).
#
# wave_rt  needs  ./ckpts/Wan2.1-T2V-1.3B-Diffusers      (--model-path default)
#                  ./ckpts/zhuhz22/.../longvideo.pt        (--gen-ckpt default)
# naive_rt  needs  wan_models/Wan2.1-T2V-1.3B/            (HARD-CODED in
#                  naive_rt/rolling_forcing/utils/wan_wrapper.py:128, Wan-AI
#                  FLAT layout (config.json + safetensors at repo root) -- the
#                  two runtimes need DIFFERENT layouts!  sglang verifies a
#                  standard Diffusers pipeline (model_index.json + transformer/
#                  + vae/ + text_encoder/ + tokenizer/ + scheduler/), see
#                  _verify_diffusers_model_complete in hf_diffusers_utils.py;
#                  a flat-layout dir fails that check and falls back to an HF
#                  download attempt.
#
# All targets live on the shared mount (same filesystem as this repo checkout);
# symlinks here are visible to any host that mounts it.  Run as-is on the GPU
# host before a sweep; safe to re-run (ln -sfn).
#
# NOTE:  the TencentARC 5-step checkpoint (rolling_forcing_dmd.pt, needed for a
# REAL-quality 5+2 / 1.3b topo) is NOT on the shared mount yet -- drop it at
# data/ckpts/TencentARC/RollingForcing/checkpoints/rolling_forcing_dmd.pt and
# point --gen-ckpt there, or 5+2 stays off-distribution (system timing only).

set -euo pipefail

# shared mount root (edit if the mount path changes on your host)
BASE=/inspire/hdd/global_user/yangyi-253108120173/inspire_shared/mount/advanced-machine-learning-and-deep-learning-applications

# naive_rt: Wan-AI FLAT layout (config.json + safetensors + T5 + VAE + tokenizer)
WAN_FLAT=$BASE/cyy/ckpts/Wan-AI/Wan2.1-T2V-1.3B
# wave_rt/sglang: standard Diffusers pipeline layout (model_index.json + ...)
WAN_SGL=$BASE/data/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b
# 4-step long-video RF weights (zhuhz22 Causal-Forcing line)
GEN_CKPT=$BASE/data/ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

fail() { echo "ERROR: $1" >&2; exit 1; }
[ -f "$WAN_FLAT/config.json" ] || fail "naive base not found at $WAN_FLAT"
[ -f "$WAN_SGL/model_index.json" ] || fail "sgl base not found at $WAN_SGL"
[ -f "$GEN_CKPT" ] || fail "gen ckpt not found at $GEN_CKPT"

mkdir -p ckpts/zhuhz22/Causal-Forcing/chunkwise wan_models
ln -sfn "$WAN_SGL"  ckpts/Wan2.1-T2V-1.3B-Diffusers
ln -sfn "$WAN_FLAT" wan_models/Wan2.1-T2V-1.3B
ln -sfn "$GEN_CKPT"  ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt

# verify the symlink chains resolve
for p in ckpts/Wan2.1-T2V-1.3B-Diffusers/model_index.json \
         ckpts/Wan2.1-T2V-1.3B-Diffusers/transformer/config.json \
         wan_models/Wan2.1-T2V-1.3B/config.json \
         ckpts/zhuhz22/Causal-Forcing/chunkwise/longvideo.pt; do
    [ -e "$ROOT/$p" ] || fail "symlink does not resolve: $p"
    echo "  ok  $p"
done
echo "[setup_models] layout ready:"
echo "    ckpts/Wan2.1-T2V-1.3B-Diffusers  -> wave_rt --model-path default (sgl Diffusers layout)"
echo "    wan_models/Wan2.1-T2V-1.3B       -> naive_rt hard-coded base (flat layout)"
echo "    ckpts/zhuhz22/.../longvideo.pt   -> --gen-ckpt default (4-step)"
