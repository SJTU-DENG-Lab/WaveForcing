"""Render a per-GPU wavefront timeline (Gantt) as a dependency-free SVG.

Reads /tmp/wrt_timeline/diff_r*.json (per diffusion rank, from WAVE_TIMELINE) and
draws x=time(ms), y=rank(chunk/GPU), bars per tick phase:
  forward (compute + inline KV comm) | broadcast (clean KV) | migration (latent).
Collective phases include the barrier wait, so a fast rank shows a long comm bar
stretching to the barrier -> visualizes the skew payback.

Usage:
  python scripts/wp_timeline.py --dir /tmp/wrt_timeline --out timeline.svg [--ticks 3 9]
"""
import argparse
import glob
import json
import os

C_FWD = "#3b6db3"; C_BC = "#e08a1e"; C_MIG = "#b33b3b"; C_IDLE = "#dddddd"; C_VAE = "#3b9b6d"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/wrt_timeline")
    ap.add_argument("--out", default="/tmp/wrt_timeline/timeline.svg")
    ap.add_argument("--ticks", type=int, nargs=2, default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--pxperms", type=float, default=1.2)
    args = ap.parse_args()

    ranks = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.dir, "diff_r*.json")))]
    if not ranks:
        raise SystemExit(f"no diff_r*.json in {args.dir}")
    ranks.sort(key=lambda d: d["rank"])
    mode, kv = ranks[0].get("mode"), ranks[0].get("kv")
    vaes = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.dir, "vae_s*.json")))]
    vaes.sort(key=lambda d: d["stage"])

    def keep(e):
        return not args.ticks or (args.ticks[0] <= e["tick"] <= args.ticks[1])

    # common time origin across BOTH diffusion and VAE (same host clock)
    t_all = [e["t0"] for d in ranks for e in d["events"] if keep(e)]
    t_all += [ev[1] for v in vaes for ev in v["events"]]
    t0o = min(t_all)
    tmax = max([e["mig"] for d in ranks for e in d["events"] if keep(e)]
               + [ev[2] for v in vaes for ev in v["events"]])
    span_ms = (tmax - t0o) * 1000
    n_lanes = len(ranks) + len(vaes)
    L, R, T = 100, 30, 60          # margins
    lane_h, gap = 42, 10
    W = int(L + R + span_ms * args.pxperms)
    H = T + n_lanes * (lane_h + gap) + 30
    ms = lambda t: L + (t - t0o) * 1000 * args.pxperms

    def rect(x, w, y, h, c, txt=""):
        w = max(0.4, w)
        s = f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" fill="{c}" stroke="#fff" stroke-width="0.3"/>'
        if txt and w > 12:
            s += f'<text x="{x + w/2:.1f}" y="{y + h/2 + 3}" font-size="9" fill="#fff" text-anchor="middle">{esc(txt)}</text>'
        return s

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">']
    out.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    title = args.title or f"wavefront timeline — kv={kv} exchange={mode}  (x=ms)"
    out.append(f'<text x="{L}" y="24" font-size="14" font-weight="bold">{esc(title)}</text>')
    # legend
    lx = L
    for c, lab in [(C_FWD, "forward (compute+KV exch)"), (C_BC, "broadcast"),
                   (C_MIG, "migration + barrier wait"), (C_VAE, "VAE decode (per frame)")]:
        out.append(f'<rect x="{lx}" y="34" width="12" height="12" fill="{c}"/>')
        out.append(f'<text x="{lx+16}" y="44" font-size="10">{esc(lab)}</text>')
        lx += 12 + 10 + len(lab) * 6.2

    # diffusion lanes (rank 0 at top)
    for i, d in enumerate(ranks):
        y = T + i * (lane_h + gap)
        out.append(f'<text x="6" y="{y + lane_h/2 + 4}" font-size="10">diff r{d["rank"]}</text>')
        for e in d["events"]:
            if not keep(e):
                continue
            x0, xf, xb, xm = ms(e["t0"]), ms(e["fwd"]), ms(e["bc"]), ms(e["mig"])
            if e["role"] != "idle":
                out.append(rect(x0, xf - x0, y, lane_h, C_FWD, f"C{e['chunk']}"))
            else:
                out.append(rect(x0, xf - x0, y, lane_h, C_IDLE))
            out.append(rect(xf, xb - xf, y, lane_h, C_BC))
            out.append(rect(xb, xm - xb, y, lane_h, C_MIG))

    # VAE lanes (below diffusion): per-frame run_seg bars
    for j, v in enumerate(vaes):
        y = T + (len(ranks) + j) * (lane_h + gap)
        tag = f'VAE s{v["stage"]}' + ("(first)" if v.get("is_first") else "") + ("(last)" if v.get("is_last") else "")
        out.append(f'<text x="6" y="{y + lane_h/2 + 4}" font-size="9">{esc(tag)}</text>')
        for (_fi, ta, tb) in v["events"]:
            out.append(rect(ms(ta), ms(tb) - ms(ta), y, lane_h, C_VAE))

    out.append("</svg>")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {args.out}  span={span_ms:.0f}ms  diff_ranks={len(ranks)}  "
          f"vae_stages={len(vaes)}  mode={mode}/{kv}")


if __name__ == "__main__":
    main()
