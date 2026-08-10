"""Parse WAVE_TIMELINE diff_r*.json -> per-tick and steady-state phase timing.

Usage: python scripts/tl_parse.py --dir /tmp/tl_TAG [--rank 0] [--ticks 6,7]
Prints per-tick fwd/bc/mig/tick(ms) for the rank, then the mean over --ticks.
Forward here is the REAL (overlap-preserving) wall time incl. exposed comm + send drain.
"""
import argparse, glob, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--ticks", type=str, default="6,7")
    args = ap.parse_args()

    f = os.path.join(args.dir, f"diff_r{args.rank}.json")
    if not os.path.isfile(f):
        raise SystemExit(f"missing {f}")
    d = json.load(open(f))
    steady = {int(x) for x in args.ticks.split(",") if x.strip()}
    print(f"# dir={args.dir} rank={args.rank} mode={d.get('mode')} kv={d.get('kv')}")
    print(f"{'tick':>4} {'role':>8} {'chunk':>5} {'fwd':>7} {'bc':>6} {'mig':>6} "
          f"{'tick_ms':>7} {'klen':>6}")
    acc = {"fwd": [], "bc": [], "mig": [], "tick": []}
    for e in d["events"]:
        fwd = (e["fwd"] - e["t0"]) * 1000
        bc = (e["bc"] - e["fwd"]) * 1000
        mig = (e["mig"] - e["bc"]) * 1000
        tick = (e["mig"] - e["t0"]) * 1000
        star = "*" if e["tick"] in steady else " "
        print(f"{e['tick']:>4}{star}{e['role']:>7} {e['chunk']:>5} {fwd:>7.1f} "
              f"{bc:>6.1f} {mig:>6.1f} {tick:>7.1f} {e.get('klen',0):>6}")
        if e["tick"] in steady:
            acc["fwd"].append(fwd); acc["bc"].append(bc)
            acc["mig"].append(mig); acc["tick"].append(tick)
    if acc["fwd"]:
        n = len(acc["fwd"])
        mean = lambda k: sum(acc[k]) / n
        print(f"\n# STEADY mean over ticks {sorted(steady)} (n={n}):")
        print(f"#   fwd={mean('fwd'):.1f}ms bc={mean('bc'):.1f}ms "
              f"mig={mean('mig'):.1f}ms  TICK={mean('tick'):.1f}ms")


if __name__ == "__main__":
    main()
