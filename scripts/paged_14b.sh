#!/bin/zsh
# paged vs onesided on 14B (dummy): (a) rank0 cat_ms via WAVE_LAYER_PROF, (b) mean
# tick A/B for exact + sage.  14B random-init loads fast (no ckpt).  5 diffusion GPUs.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOGD=logs/paged14
mkdir -p $LOGD
killall_wave() { pkill -9 -f "wave_rt" 2>/dev/null; pgrep -f spawn_main | xargs -r kill -9 2>/dev/null; sleep 4; }
COMMON=(--dit-scale 14b --wp-size 5 --rf-step 4 --num-frames 60 --kv-context causal \
        --vae-stages 0 --no-save-video --cuda-visible-devices 0,1,2,3,4)

run() {  # $1=tag  $2=env-prefix  $3..=extra args
  local tag=$1 envp=$2; shift 2
  echo "=== $tag $(date +%H:%M:%S) ==="
  killall_wave
  eval "$envp" timeout 700 $PY -m wave_rt "${COMMON[@]}" "$@" \
    --task paged14 --run-tag $tag > $LOGD/$tag.log 2>&1 &
  local pid=$!
  for i in $(seq 1 130); do
    grep -qE "mean tick" $LOGD/$tag.log 2>/dev/null && break
    kill -0 $pid 2>/dev/null || { echo "  proc exited"; break; }
    sleep 5
  done
  sleep 3
  killall_wave
  grep -E "mean tick|RuntimeError|Traceback|paged r|NaN" $LOGD/$tag.log | tail -4
}

# ---- (a) layer-prof: rank0 cat_ms, exact ----
rm -rf /tmp/pg_lp_os /tmp/pg_lp_pg
run lp_os   "WAVE_LAYER_PROF=1 WAVE_LAYER_PROF_DIR=/tmp/pg_lp_os" --exchange-mode onesided
run lp_pg   "WAVE_LAYER_PROF=1 WAVE_LAYER_PROF_DIR=/tmp/pg_lp_pg" --exchange-mode paged

# ---- (b) mean tick A/B, exact (bf16) ----
for k in 1 2 3; do run ex_os_$k "" --exchange-mode onesided; done
for k in 1 2 3; do run ex_pg_$k "" --exchange-mode paged;    done

# ---- (c) mean tick A/B, sage ----
for k in 1 2 3; do run sa_os_$k "" --attention-backend sa --exchange-mode onesided; done
for k in 1 2 3; do run sa_pg_$k "" --attention-backend sa --exchange-mode paged;    done

killall_wave
echo "=== rank0 cat_ms (exact) ==="
$PY - <<'PY'
import json, os
def r0(d):
    f=os.path.join(d,"layer_summary_r0.json")
    if not os.path.isfile(f): return None
    s=json.load(open(f)); L=s["layers"]
    return sum(x["cat_ms"] for x in L), s["sum_total_ms"]
for name,d in [("onesided","/tmp/pg_lp_os"),("paged","/tmp/pg_lp_pg")]:
    v=r0(d)
    print(f"  {name:9}: cat_ms={v[0]:.2f}  sum_total={v[1]:.2f}" if v else f"  {name}: MISSING")
PY
echo "=== mean tick summary ==="
$PY - <<'PY'
import re, glob, statistics
def ticks(pat):
    out=[]
    for f in sorted(glob.glob(f"logs/paged14/{pat}.log")):
        t=open(f,errors="ignore").read()
        m=re.search(r"mean tick\s+([\d.]+)",t)
        if m: out.append(float(m.group(1)))
    return out
for lab,pat in [("exact onesided","ex_os_*"),("exact paged","ex_pg_*"),
                ("sage  onesided","sa_os_*"),("sage  paged","sa_pg_*")]:
    v=ticks(pat)
    if v:
        m=statistics.mean(v); sd=statistics.stdev(v) if len(v)>1 else 0
        print(f"  {lab}: {m:.1f} ± {sd:.1f} ms  (n={len(v)}, {[round(x,1) for x in v]})")
    else:
        print(f"  {lab}: no data")
PY
echo "ALL DONE $(date +%H:%M:%S)"
