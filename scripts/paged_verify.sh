#!/bin/zsh
# paged mode byte-exact verification: sync (reference) vs onesided vs paged at
# 1.3B / 60 latent frames (steady-state paged ticks t in [7,19]).  All three must
# produce bit-identical latents (max|diff|=0.0) -- paged only changes KV transport
# layout, not numerics.  8 GPUs (5 diffusion + 3 VAE).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOGD=logs/paged
mkdir -p $LOGD
killall_wave() { pkill -9 -f "wave_rt" 2>/dev/null; pgrep -f spawn_main | xargs -r kill -9 2>/dev/null; sleep 4; }

run() {  # $1=tag $2=mode  $3..=extra env is inline
  local tag=$1 mode=$2; shift 2
  echo "=== $tag ($mode) $(date +%H:%M:%S) ==="
  killall_wave
  timeout 600 $PY -m wave_rt \
    --wp-size 5 --rf-step 4 --num-frames 60 --vae-stages 3 \
    --cuda-visible-devices 0,1,2,3,4,5,6,7 --kv-context causal \
    --exchange-mode $mode --task pgverify --run-tag $tag --no-save-video "$@" \
    > $LOGD/$tag.log 2>&1 &
  local pid=$!
  for i in $(seq 1 120); do
    grep -qE "latents ->|metrics ->" $LOGD/$tag.log 2>/dev/null && break
    kill -0 $pid 2>/dev/null || { echo "  proc exited"; break; }
    sleep 5
  done
  sleep 3
  killall_wave
  grep -E "mean tick|latents ->|RuntimeError|Traceback|paged r|NaN" $LOGD/$tag.log | tail -8
}

run pg_sync     sync
run pg_onesided onesided
run pg_paged    paged

killall_wave
echo "=== byte-exact gate ==="
$PY - <<'PY'
import torch, os
d = "outputs/pgverify"
def load(t):
    p = os.path.join(d, t, "latents.pt")
    return torch.load(p, map_location="cpu") if os.path.isfile(p) else None
s, o, p = load("pg_sync"), load("pg_onesided"), load("pg_paged")
def cmp(a, b, na, nb):
    if a is None or b is None:
        print(f"  {na} vs {nb}: MISSING ({na}={a is not None}, {nb}={b is not None})"); return
    eq = torch.equal(a, b)
    md = (a.float()-b.float()).abs().max().item()
    print(f"  {na} vs {nb}: shapes {tuple(a.shape)}/{tuple(b.shape)}  torch.equal={eq}  max|diff|={md:.3e}")
cmp(s, o, "sync", "onesided")
cmp(s, p, "sync", "paged")
cmp(o, p, "onesided", "paged")
PY
echo "ALL DONE $(date +%H:%M:%S)"
