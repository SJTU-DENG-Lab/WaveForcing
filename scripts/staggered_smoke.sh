#!/bin/zsh
# Smoke + correctness for exchange_mode=staggered.
# Real 1.3B, causal, 12 latent frames (4 chunks), vae-stages 0 (5 GPUs 0-4).
# staggered is movement/timing-only -> latents must match causal-sync closely.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOGD=logs/stag
OUT=/tmp/wrt_stag_out
mkdir -p $LOGD
rm -rf $OUT

COMMON=(--wp-size 5 --rf-step 4 --num-frames 12 \
        --kv-context causal --vae-stages 0 --no-save-video \
        --out-root $OUT --task stag \
        --cuda-visible-devices 0,1,2,3,4)

killall_wave() { pkill -9 -f "wave_rt" 2>/dev/null; pgrep -f spawn_main | xargs -r kill -9 2>/dev/null; sleep 4; }

run() {
  local tag=$1 marker=$2; shift 2
  echo "=== $tag ($(date +%H:%M:%S)) ==="
  killall_wave
  timeout 1200 $PY -m wave_rt "${COMMON[@]}" "$@" --run-tag $tag \
      > $LOGD/$tag.log 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 220); do
    if grep -qE "$marker" $LOGD/$tag.log 2>/dev/null; then ok=1; break; fi
    kill -0 $pid 2>/dev/null || { echo "  proc exited"; break; }
    sleep 5
  done
  sleep 3
  killall_wave
  if [[ $ok == 1 ]]; then
    echo "  $tag OK"; grep -E "mean tick|NaN|inf|FAILED" $LOGD/$tag.log | head
  else
    echo "  $tag FAILED (marker '$marker')"; tail -40 $LOGD/$tag.log
  fi
}

run sync      "mean tick" --exchange-mode sync
run staggered "mean tick" --exchange-mode staggered --stagger-lead 1

echo "=== compare latents ==="
$PY - <<PYEOF
import torch, os
a=os.path.join("$OUT","stag","sync","latents.pt")
b=os.path.join("$OUT","stag","staggered","latents.pt")
if not (os.path.isfile(a) and os.path.isfile(b)):
    print("MISSING latents:", os.path.isfile(a), os.path.isfile(b)); raise SystemExit(1)
x=torch.load(a,map_location="cpu").float(); y=torch.load(b,map_location="cpu").float()
print("shapes", tuple(x.shape), tuple(y.shape))
d=(x-y).abs()
print(f"maxabs={d.max().item():.3e} meanabs={d.mean().item():.3e} "
      f"exact={torch.equal(x,y)}")
den=x.pow(2).mean().sqrt().item()
import math
mse=(d.pow(2).mean().item())
psnr=float('inf') if mse==0 else 20*math.log10(x.abs().max().item()/math.sqrt(mse))
print(f"PSNR={psnr:.2f}dB (inf==bit-identical)")
PYEOF
echo "DONE ($(date +%H:%M:%S))"
