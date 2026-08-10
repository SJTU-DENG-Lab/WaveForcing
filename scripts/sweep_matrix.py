#!/usr/bin/env python3
"""
Full config-matrix sweep for wave_rt (14B + 1.3B) -- steady-state edition.

Grid (per topo x scale):  {bf16, sage, sage+fp8} x {sync, overlap, onesided, paged} = 12
configs (default), extensible with --exch (relay / staggered) and --tiers (bf16fp8 =
exact+fp8).  Scales: 1.3b (real longvideo.pt ckpt) + 14b (random-init dummy).
PRUNED: 5+2 x 14b never runs (no 5-step 14B ckpt line; the dummy would be
off-distribution twice over) -- so the default full matrix is 4+3 x {14b,1.3b}
(24 configs) + 5+2 x 1.3b (12 configs) = 36 configs x 3 reps.

Topology dimension (--topo), both = 8 GPUs:
  * "4+3" (default):  rf_step=4, 5 diffusion ranks + 3 streaming VAE, DMD
    [1000,750,500,250]  (zhuhz22 Causal-Forcing longvideo.pt, the repo baseline).
  * "5+2":           rf_step=5, 6 diffusion ranks + 2 streaming VAE, DMD
    [1000,800,600,400,200]  (sglang RollingForcingWanT2V480PConfig official
    default, TencentARC/RollingForcing checkpoint line).  NOTE: no 5-step
    checkpoint exists on this box -- with the default --gen-ckpt (4-step
    longvideo.pt) a 5+2 / 1.3b run is OFF-DISTRIBUTION: system-level timing only
    (same status as 14b dummy).  Point --gen-ckpt at TencentARC's
    rolling_forcing_dmd.pt to make it a real-quality config.

Steady-state protocol (mirrors scripts/l400_fps.sh):
  * --num-frames 399 (133 chunks -> 137 ticks), 8 GPUs = 5 diffusion + 3 streaming VAE.
    (Streaming VAE is REQUIRED: per_tick_ms only lands in metrics.json via the
    launcher's meta_q aggregation; vae_stages=0 never writes it.)
  * per run: poll outputs/sweep/<tag>/metrics.json (written by the launcher after
    ALL procs exit), extract per_tick_ms, drop the fill ramp + drain tail
    (STEADY_SKIP_HEAD / STEADY_SKIP_TAIL), keep the steady window.
  * --reps runs per config (default 3): every rep's steady ticks are POOLED and
    reported as p50 / p75 / p90 / mean+-std -- quantiles are the headline (robust
    to a one-off outlier rep), mean/std for reference.
  * fps(p50) = 12 * 1000 / p50 pixel frames/s (each tick emits 3 latent frames,
    Wan VAE upscales 4x temporally).

The old "mean tick" log marker is still parsed as a fallback when metrics.json is
missing (e.g. a run that died inside the VAE tail) -- recorded as mean_tick, not
steady, and NOT pooled into the quantile table.

Per-topology single-GPU baseline (--no-baseline to skip):
  * One naive single-GPU run per topo (scripts/naive_ref.py, same 399 frames,
    same seed 0, same default prompt; step line = the topo's DMD steps).
    Artifacts in outputs/sweep/baseline/<topo>/:  latents.pt (PSNR reference),
    noise.pt (audit), e2e.json (single-GPU e2e seconds incl. VAE decode -- the
    README 13.6fps basis).
  * HOST PREREQ:  naive loads its base model from the HARD-CODED relative path
    wan_models/Wan2.1-T2V-1.3B/ (naive_rt/.../wan_wrapper.py:128, Diffusers
    format incl. config.json) plus the gen ckpt ./ckpts/zhuhz22/.../longvideo.pt
    -- both must exist on the GPU host (typically a symlink into /data/ckpts/,
    see docs/03 and docs/08).  Without them naive_ref.py dies in
    CausalWanModel.from_pretrained trying to download from hf-mirror; a failed
    baseline is non-fatal for the sweep (1.3b rows just lose the gate columns).
  * Noise parity:  both sides draw from Generator("cuda").manual_seed(0), so a
    1.3b sweep run consumes bit-identical initial noise and the PSNR gate is
    honest (zero pre-warm latency for a 5+2 topo's 5-step baseline note: it is
    OFF-DISTRIBUTION with the 4-step ckpt, system timing only -- same status as
    the 5+2/1.3b runs themselves).
  * Every 1.3b run gets --ref-latents + --baseline-e2e:  the launcher then
    writes psnr_db and speedup_vs_baseline into metrics.json, rendered as the
    PSNR (dB) and x-single-GPU columns (14b dummy runs have no quality meaning
    and get "---").

Results stream to  logs/sweep/results.json  (incremental, crash-safe) and a
rendered  logs/sweep/summary.md  after every run.  Per-run logs in logs/sweep/.

Usage:
    .venv/bin/python scripts/sweep_matrix.py                        # 4+3, 12x3 per scale
    .venv/bin/python scripts/sweep_matrix.py --topos 4+3 5+2 --scales 1.3b
    .venv/bin/python scripts/sweep_matrix.py --scales 14b --reps 1
    .venv/bin/python scripts/sweep_matrix.py --tiers bf16 sage sagefp8 bf16fp8
    .venv/bin/python scripts/sweep_matrix.py --exch sync onesided paged relay staggered
    .venv/bin/python scripts/sweep_matrix.py --smoke                # single config sanity
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv/bin/python")
LOGD = os.path.join(ROOT, "logs/sweep")

# per-topology single-GPU baseline artifacts (see module docstring)
BASELINE_ROOT = os.path.join(ROOT, "outputs", "sweep", "baseline")
BASELINE_TIMEOUT = 1200   # naive single-GPU 399 frames + VAE decode + load

# topology name -> (rf_step, vae_stages, dmd steps); wp_size = rf_step + 1.
# Both fit 8 GPUs: diffusion ranks + vae stages == 8.
TOPOS = {
    "4+3": (4, 3, [1000, 750, 500, 250]),
    "5+2": (5, 2, [1000, 800, 600, 400, 200]),   # TencentARC official DMD line
}
DEFAULT_TOPOS = ["4+3"]


def build_common(topo: str, kv: str) -> list:
    """wave_rt CLI args shared by every run of a topology (5 diffusion + VAE stages
    on GPUs 0..7; num_frames 399 -> 137 ticks at 4+3, 138 at 5+2)."""
    rf_step, vae_stages, steps = TOPOS[topo]
    return [
        "--wp-size", str(rf_step + 1), "--rf-step", str(rf_step),
        "--denoising-step-list", ",".join(str(s) for s in steps),
        "--num-frames", "399",
        "--kv-context", kv, "--vae-stages", str(vae_stages),
        "--no-save-video",
        "--seed", "0",   # must match the naive baseline (noise parity -> PSNR gate)
        "--cuda-visible-devices", "0,1,2,3,4,5,6,7",
    ]

# tier name -> (extra cli args, extra env dict)
TIERS = {
    "bf16":    (["--attention-backend", "torch_sdpa"], {}),
    "bf16fp8": (["--attention-backend", "torch_sdpa", "--quantization", "fp8"],
                {"WAVE_FP8_COMPILE": "1", "WAVE_FP8_SCOPE": "all"}),
    "sage":    (["--attention-backend", "sa"], {}),
    "sagefp8": (["--attention-backend", "sa", "--quantization", "fp8"],
                {"WAVE_FP8_COMPILE": "1", "WAVE_FP8_SCOPE": "all"}),
}
DEFAULT_TIERS = ["bf16", "sage", "sagefp8"]   # exact+fp8 (bf16fp8) is optional
DEFAULT_EXCH = ["sync", "overlap", "onesided", "paged"]
EXTRA_EXCH = ["relay", "staggered"]
SCALES = ["14b", "1.3b"]   # 14b first: dummy loads fast, validates pipeline early
# topo x scale pruning:  5+2 is 1.3b-only -- there is no 5-step 14B ckpt line,
# the 14b dummy would be off-distribution twice over (uninterpretable speed).
SKIP = {("5+2", "14b")}

# Fallback marker (per-tick means nothing in a plain run; only whole-run mean tick
# is honest without WAVE_TICKPROF -- see wavefront.run()).
MARKER = re.compile(r"mean tick\s+([\d.]+)\s*ms")
NAN_RE = re.compile(r"\bNaN\b|\binf\b", re.IGNORECASE)

# Steady-state window: 399 frames -> 137 ticks; drop the fill ramp (t < world-1 is
# partially-active anyway; 12 keeps a safety margin) and the drain tail.
STEADY_SKIP_HEAD = 12
STEADY_SKIP_TAIL = 4


def metrics_path(tag: str) -> str:
    """outputs/sweep/<tag>/metrics.json (task=sweep, run_tag=tag -> bench.run_dir)."""
    return os.path.join(ROOT, "outputs", "sweep", tag, "metrics.json")


# ---------------------------------------------------------------------------
# per-topology single-GPU baseline (PSNR reference + single-GPU speed reference)
# ---------------------------------------------------------------------------
def baseline_dir(topo: str) -> str:
    return os.path.join(BASELINE_ROOT, topo)


def baseline_ready(topo: str) -> bool:
    """latents.pt (PSNR ref) + e2e.json (single-GPU e2e s) both present."""
    d = baseline_dir(topo)
    return (os.path.isfile(os.path.join(d, "latents.pt"))
            and os.path.isfile(os.path.join(d, "e2e.json")))


def baseline_e2e_s(topo: str) -> float | None:
    try:
        return json.load(open(os.path.join(baseline_dir(topo), "e2e.json")))["e2e_s"]
    except Exception:
        return None


def run_baseline(topo: str, timeout=BASELINE_TIMEOUT, force=False) -> bool:
    """Generate (or reuse) the naive single-GPU baseline for a topology:  399
    frames, seed 0, the topo's DMD step line.  Returns True if artifacts are
    usable afterwards.  force=True rebuilds even if artifacts exist (e.g. after
    the noise-layout fix in naive_ref.py)."""
    if baseline_ready(topo) and not force:
        print(f"  [baseline] {topo} already exists: {baseline_dir(topo)}", flush=True)
        return True
    if force and os.path.isdir(baseline_dir(topo)):
        import shutil
        shutil.rmtree(baseline_dir(topo))
        print(f"  [baseline] {topo}: removed stale artifacts, rebuilding", flush=True)
    rf_step, _vae_stages, steps = TOPOS[topo]
    out = baseline_dir(topo)
    log = os.path.join(LOGD, f"baseline_{topo.replace('+', 'p')}.log")
    cmd = [PY, "scripts/naive_ref.py", "--num_output_frames", "399",
           "--nstep", str(rf_step), "--steps", ",".join(str(s) for s in steps),
           "--seed", "0", "--out", out]
    print(f"  [baseline] {topo}: naive single-GPU ({rf_step}-step) ...",
          end="", flush=True)
    with open(log, "w") as f:
        f.write(f"# CMD: {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                env=dict(os.environ), cwd=ROOT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f" TIMEOUT ({timeout}s)", flush=True)
        return False
    ok = proc.returncode == 0 and baseline_ready(topo)
    if ok:
        print(f" e2e={baseline_e2e_s(topo):.1f}s  -> {out}", flush=True)
    else:
        print(f" FAILED rc={proc.returncode} (log {log})", flush=True)
    return ok


def quantile(sorted_vals: list, p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("empty sample")
    return sorted_vals[min(n - 1, max(0, round(p * (n - 1))))]


def steady_stats(per_tick_ms: list) -> dict | None:
    """Steady-window stats of a per-run per_tick_ms array; None if the window is empty.
    Keeps the raw steady ticks under 'ticks' so cross-rep quantiles can be pooled
    exactly (not approximated from rep-level quantiles)."""
    win = per_tick_ms[STEADY_SKIP_HEAD:]
    if STEADY_SKIP_TAIL:
        win = win[:-STEADY_SKIP_TAIL]
    if not win:
        return None
    s = sorted(win)
    return {
        "ticks": [round(x, 3) for x in win],
        "p50": quantile(s, 0.50), "p75": quantile(s, 0.75), "p90": quantile(s, 0.90),
        "mean": statistics.mean(s), "std": statistics.stdev(s) if len(s) > 1 else 0.0,
        "n": len(s), "min": s[0], "max": s[-1],
    }


def killall():
    subprocess.run("pkill -9 -f wave_rt", shell=True,
                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    subprocess.run("pgrep -f spawn_main | xargs -r kill -9", shell=True,
                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    time.sleep(5)


def read_metrics(tag: str) -> list | None:
    """per_tick_ms list from outputs/sweep/<tag>/metrics.json (None if absent/invalid)."""
    p = metrics_path(tag)
    if not os.path.isfile(p):
        return None
    try:
        m = json.load(open(p))
    except Exception:
        return None
    pt = m.get("per_tick_ms")
    return pt if isinstance(pt, list) and len(pt) > 0 else None


def run_one(topo, scale, tier_name, tier_args, tier_env, exch, rep, port,
            timeout=1800, kv="causal"):
    tag = f"{topo}_{scale}_{tier_name}_{kv}_{exch}_r{rep}"
    log = os.path.join(LOGD, f"{tag}.log")
    env = dict(os.environ)
    env.update(tier_env)
    cmd = [PY, "-m", "wave_rt", *build_common(topo, kv), "--dit-scale", scale,
           *tier_args, "--exchange-mode", exch,
           "--master-port", str(port), "--task", "sweep", "--run-tag", tag]
    # PSNR gate + single-GPU speed reference, only meaningful for real 1.3b runs
    # with the SAME causal semantics as the naive baseline.  joint (=full) has no
    # naive counterpart -> PSNR would be meaningless; keep only the speed ref.
    ref, b_e2e = "", 0.0
    if scale == "1.3b" and baseline_ready(topo) and kv == "causal":
        ref = os.path.join(baseline_dir(topo), "latents.pt")
        b_e2e = baseline_e2e_s(topo) or 0.0
        cmd += ["--ref-latents", ref, "--baseline-e2e", f"{b_e2e:.3f}"]
    elif scale == "1.3b" and baseline_ready(topo):
        b_e2e = baseline_e2e_s(topo) or 0.0
        cmd += ["--baseline-e2e", f"{b_e2e:.3f}"]

    killall()
    # A stale metrics.json from a previous run of the same tag is a skeleton
    # (per_tick_ms=null) that would read as "invalid" forever and force every run
    # onto the log-marker fallback; drop it so this run's launcher owns the file.
    stale = metrics_path(tag)
    if os.path.exists(stale):
        os.remove(stale)
    print(f"  [{time.strftime('%H:%M:%S')}] {tag} (port {port}) ...",
          end="", flush=True)
    with open(log, "w") as f:
        f.write(f"# CMD: {' '.join(cmd)}\n# ENV: {tier_env}\n\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                cwd=ROOT)
    steady, mean_tick, status = None, None, "TIMEOUT"
    psnr, speedup, e2e_s, diff_s = None, None, None, None
    t0 = time.time()
    while time.time() - t0 < timeout:
        pt = read_metrics(tag)          # steady-state source of truth
        if pt is not None:
            steady = steady_stats(pt)
            status = "OK"
            # same metrics.json already carries the launcher's PSNR/speedup
            try:
                m = json.load(open(metrics_path(tag)))
                psnr = m.get("psnr_db")
                speedup = m.get("speedup_vs_baseline")
                e2e_s = m.get("end_to_end_s")
                diff_s = m.get("diffusion_s")
            except Exception:
                pass
            break
        if os.path.exists(log):
            txt = open(log, errors="ignore").read()
            m = MARKER.search(txt)
            if m:
                # Diffusion stage is done, but the launcher still has to join the
                # VAE stages and finalize metrics.json (PSNR/speedup live there).
                # Record the marker as a fallback; it is NOT a completion signal.
                mean_tick = float(m.group(1))
        if proc.poll() is not None:
            # Launcher exited -> metrics.json is final (written before exit).
            # Prefer it; the log marker only covers a launcher crash-before-write.
            pt = read_metrics(tag)
            if pt is not None:
                steady = steady_stats(pt)
                status = "OK"
                try:
                    mj = json.load(open(metrics_path(tag)))
                    psnr = mj.get("psnr_db")
                    speedup = mj.get("speedup_vs_baseline")
                    e2e_s = mj.get("end_to_end_s")
                    diff_s = mj.get("diffusion_s")
                except Exception:
                    pass
            else:
                txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
                m = MARKER.search(txt)
                if m:
                    mean_tick = float(m.group(1))
                    status = "NAN" if NAN_RE.search(txt) else "OK"
                else:
                    status = f"EXIT{proc.returncode}"
            break
        time.sleep(5)
    time.sleep(3)
    killall()
    dur = time.time() - t0
    if steady:
        extra = ""
        if psnr is not None:
            extra += f"  PSNR={psnr:.1f}dB"
        if speedup:
            extra += f"  x{speedup:.1f}"
        print(f" {status}  steady_p50={steady['p50']:.1f}ms "
              f"p75={steady['p75']:.1f} (n={steady['n']} ticks)"
              f"  ({dur:.0f}s){extra}", flush=True)
    else:
        print(f" {status}"
              + (f"  mean_tick={mean_tick:.1f}ms" if mean_tick else "")
              + f"  ({dur:.0f}s)", flush=True)
    return {"tag": tag, "topo": topo, "scale": scale, "tier": tier_name,
            "kv": kv, "exch": exch, "rep": rep, "steady": steady,
            "mean_tick": mean_tick, "status": status, "wall_s": round(dur, 1),
            "psnr_db": psnr, "speedup": speedup, "e2e_s": e2e_s,
            "diff_s": diff_s,
            "baseline_e2e": b_e2e or None}


def _e2e_s(r) -> float:
    """End-to-end seconds for a run record: stored e2e_s, else speedup x
    baseline inverse, else the run's metrics.json (legacy 14b records carry
    neither).  0.0 when nothing is available."""
    if r.get("e2e_s"):
        return r["e2e_s"]
    if r.get("speedup") and r.get("baseline_e2e"):
        return r["baseline_e2e"] / r["speedup"]
    p = metrics_path(r["tag"])
    if os.path.isfile(p):
        try:
            return json.load(open(p)).get("end_to_end_s") or 0.0
        except Exception:
            return 0.0
    return 0.0


def _diff_s(r) -> float:
    """Diffusion-stage seconds (VAE excluded): stored diff_s, else the run's
    metrics.json (legacy records carry neither).  0.0 when nothing is
    available."""
    if r.get("diff_s"):
        return r["diff_s"]
    p = metrics_path(r["tag"])
    if os.path.isfile(p):
        try:
            return json.load(open(p)).get("diffusion_s") or 0.0
        except Exception:
            return 0.0
    return 0.0


def render(results, path, topo_order, tiers_order, exch_order, kv_order):
    """Pool every OK rep's steady ticks per (topo, scale, tier, kv, exch);
    quantiles headline.  Only (topo, tier, kv, exch) combos that actually ran
    get rows."""
    agg = {}
    for r in results:
        if r["status"] != "OK" or r.get("steady") is None:
            continue
        agg.setdefault((r.get("topo"), r["scale"], r["tier"], r.get("kv"),
                        r["exch"]), []).append(r)   # full record: ticks + psnr
    # "joint" is wave_rt's all-pairs attention -- what the user calls "full".
    kv_label = {"joint": "full", "causal": "causal"}
    lines = ["# wave_rt config-matrix sweep (399-frame steady state)", "",
             f"_generated {time.strftime('%Y-%m-%d %H:%M:%S')}; "
             f"{len(results)} runs recorded_",
             f"_steady window = per_tick_ms[{STEADY_SKIP_HEAD}:-{STEADY_SKIP_TAIL}] "
             "pooled across reps; fps(p50) = 12×1000/p50 (3 latent frames ×4 per tick)_",
             ""]
    for topo in topo_order:
        rf_step, vae_stages, _ = TOPOS[topo]
        bl = (f" (single-GPU baseline {baseline_e2e_s(topo):.1f}s "
              f"-> {399 * 4 / baseline_e2e_s(topo):.1f} fps)"
              if baseline_ready(topo) else " (no single-GPU baseline)")
        lines += [f"## topo {topo} ({rf_step}+1 diffusion, {vae_stages} VAE){bl}", ""]
        for scale in SCALES:
            if not any(agg.get((topo, scale, t, k, e))
                       for t in tiers_order for k in kv_order
                       for e in exch_order):
                continue  # pruned (5+2 x 14b) or nothing ran
            lines += [f"### {scale}", "",
                      "| tier | attn | exch | p50 (ms) | p75 | p90 | mean±std | "
                      "n | fps(p50) | PSNR (dB) | ×single-GPU | dit fps | e2e fps |",
                      "|------|------|------|---------:|----:|----:|---------:|"
                      "--:|---------:|----------:|------------:|-------:|-------:|"]
            for tname in tiers_order:
                for kv in kv_order:
                    for exch in exch_order:
                        stats = agg.get((topo, scale, tname, kv, exch))
                        if not stats:
                            continue  # only rows that actually ran
                        pooled = sorted(v for s in stats
                                        for v in s["steady"]["ticks"])
                        n_tot = len(pooled)
                        p50, p75, p90 = (quantile(pooled, p)
                                         for p in (0.50, 0.75, 0.90))
                        mean = statistics.mean(pooled)
                        std = statistics.stdev(pooled) if n_tot > 1 else 0.0
                        fps = 12.0 * 1000.0 / p50
                        # PSNR / speedup: pooled mean across reps (quality gate,
                        # not a distribution) -- "---" when the run had no
                        # reference (14b, or joint/full vs causal baseline).
                        psnrs = [r["psnr_db"] for r in stats
                                 if r.get("psnr_db") is not None]
                        sups = [r["speedup"] for r in stats if r.get("speedup")]
                        psnr = (f"{statistics.mean(psnrs):.1f}" if psnrs else "---")
                        sup = (f"{statistics.mean(sups):.1f}×" if sups else "---")
                        # e2e fps: 1596 pixel frames / end-to-end wall clock
                        # (no single-GPU baseline needed -- 14b rows have it).
                        # tick fps alone hides VAE-stage stalls (5+2 onesided
                        # sage: tick 96ms but e2e 4.5x because vae_s ~29s).
                        e2es = [e for e in (_e2e_s(r) for r in stats) if e > 0]
                        e2e_fps = (f"{1596.0 / statistics.mean(e2es):.1f}"
                                   if e2es else "---")
                        # dit fps: 1596 pixel frames / diffusion-stage wall
                        # clock (VAE excluded) -- same pixel-frame basis as
                        # e2e fps, so diffusion-only throughput is directly
                        # comparable.  tick fps is steady-state; this includes
                        # pipeline warmup/drain.
                        diffs = [d for d in (_diff_s(r) for r in stats) if d > 0]
                        dit_fps = (f"{1596.0 / statistics.mean(diffs):.1f}"
                                   if diffs else "---")
                        lines.append(
                            f"| {tname} | {kv_label.get(kv, kv)} | {exch} | "
                            f"{p50:.1f} | {p75:.1f} | {p90:.1f} | "
                            f"{mean:.1f}±{std:.1f} | {n_tot} | "
                            f"{fps:.1f} | {psnr} | {sup} | {dit_fps} | {e2e_fps} |")
            lines.append("")
    open(path, "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topos", nargs="+", default=DEFAULT_TOPOS, choices=list(TOPOS),
                    help="wavefront topology: '4+3' (rf_step=4, 5 diffusion + 3 VAE, "
                         "zhuhz22 4-step line) or '5+2' (rf_step=5, 6 diffusion + 2 VAE, "
                         "TencentARC 5-step line; 1.3b needs a 5-step ckpt to be "
                         "real-quality, else system-level timing only)")
    ap.add_argument("--scales", nargs="+", default=SCALES, choices=SCALES)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tiers", nargs="+", default=DEFAULT_TIERS, choices=list(TIERS),
                    help="attention/quant tiers (add bf16fp8 for exact+fp8)")
    ap.add_argument("--exch", nargs="+", default=DEFAULT_EXCH,
                    choices=DEFAULT_EXCH + EXTRA_EXCH,
                    help="KV exchange modes (add relay/staggered to extend)")
    ap.add_argument("--kv-contexts", nargs="+", default=("joint", "causal"),
                    choices=("joint", "causal"),
                    help="attention context: 'joint' (=full, all in-flight "
                         "chunks attend all; sync exchange only) and/or "
                         "'causal' (chunk-causal; all exchanges)")
    ap.add_argument("--base-port", type=int, default=28800)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the per-topology single-GPU naive baseline "
                         "(PSNR + speed reference) generation; 1.3b runs then "
                         "have no PSNR gate / speedup columns")
    ap.add_argument("--rebuild-baseline", action="store_true",
                    help="regenerate the naive baselines even if artifacts "
                         "already exist (noise-layout changed in naive_ref.py)")
    ap.add_argument("--smoke", action="store_true",
                    help="single 14b/bf16/onesided/1rep sanity run")
    ap.add_argument("--render-only", action="store_true",
                    help="re-render summary.md from results.json and exit "
                         "(no runs)")
    args = ap.parse_args()

    os.makedirs(LOGD, exist_ok=True)
    res_path = os.path.join(LOGD, "results.json")
    sum_path = os.path.join(LOGD, "summary.md")
    results = []
    if os.path.exists(res_path):
        try:
            results = json.load(open(res_path))
            print(f"[resume] loaded {len(results)} prior results")
        except Exception:
            results = []
    # a run counts as done only with a steady window (old mean-tick-only records
    # lack it -> re-run under the new protocol) AND a baseline fingerprint that
    # matches the current one:  PSNR/speedup are computed against the baseline
    # latents + e2e_s, so a rebuilt baseline (--rebuild-baseline) invalidates
    # every prior 1.3b result regardless of its steady window.
    done = {r["tag"]: r.get("baseline_e2e")
            for r in results if r["status"] == "OK" and r.get("steady")}

    if args.render_only:
        # render every dimension ever swept (defaults would drop e.g. staggered)
        render(results, sum_path, list(TOPOS), list(TIERS),
               DEFAULT_EXCH + EXTRA_EXCH, ("joint", "causal"))
        print(open(sum_path).read())
        return

    if args.smoke:
        # 14b / bf16 / causal / onesided / 1 rep sanity run
        matrix = [("4+3", "14b", "bf16", TIERS["bf16"], "causal", "onesided", 1)]
    else:
        matrix = []
        for topo in args.topos:
            for scale in args.scales:
                if (topo, scale) in SKIP:
                    print(f"[sweep] skip {topo} x {scale} (pruned: no 5-step 14B line)",
                          flush=True)
                    continue
                for tname in args.tiers:
                    targs, tenv = TIERS[tname]
                    for kv in args.kv_contexts:
                        for exch in args.exch:
                            # joint (=full) attends to ALL in-flight chunks ->
                            # uniform load, no per-stage slack -> only the
                            # blocking sync exchange is valid (async modes
                            # REQUIRE causal; wave_rt's config validation would
                            # reject them anyway).
                            if kv != "causal" and exch != "sync":
                                print(f"[sweep] skip {topo} {scale} "
                                      f"{kv}x{exch} (joint/full: no slack -> "
                                      f"sync only)", flush=True)
                                continue
                            for rep in range(1, args.reps + 1):
                                matrix.append((topo, scale, tname, (targs, tenv),
                                               kv, exch, rep))

    # per-topology single-GPU baseline: one naive run per topo (generated once,
    # reused across reps/runs).  Failure is non-fatal -- 1.3b runs just lose the
    # PSNR gate / speed reference.
    if not args.no_baseline and "1.3b" in args.scales:
        print("[sweep] baselines ...", flush=True)
        for topo in args.topos:
            run_baseline(topo, force=args.rebuild_baseline)
        print("", flush=True)

    total = len(matrix)
    port = args.base_port
    print(f"[sweep] {total} runs planned (topos={args.topos} scales={args.scales} "
          f"tiers={args.tiers} exch={args.exch} reps={args.reps})", flush=True)
    for i, (topo, scale, tname, (targs, tenv), kv, exch, rep) in enumerate(matrix, 1):
        tag = f"{topo}_{scale}_{tname}_{kv}_{exch}_r{rep}"
        port += 1
        cur_ref = (baseline_e2e_s(topo) if (scale == "1.3b" and baseline_ready(topo))
                   else None)
        # skip only if a record exists AND its baseline fingerprint still matches;
        # done.get(tag) == None for tags never run, and cur_ref is None for 14b
        # (no baseline) -- plain None == None would falsely skip every 14b run.
        if tag in done and done.get(tag) == cur_ref:
            print(f"  [{i}/{total}] {tag} SKIP (already OK)", flush=True)
            continue
        print(f"[{i}/{total}]", end="")
        r = run_one(topo, scale, tname, targs, tenv, exch, rep, port,
                    args.timeout, kv)
        # replace any prior record for this tag
        results = [x for x in results if x["tag"] != tag] + [r]
        json.dump(results, open(res_path, "w"), indent=2)
        render(results, sum_path, args.topos, args.tiers, args.exch,
               args.kv_contexts)

    print(f"\n[sweep] DONE. results -> {res_path}\n           summary -> {sum_path}")
    print(open(sum_path).read())


if __name__ == "__main__":
    main()
