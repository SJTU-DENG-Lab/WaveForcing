"""Optional denoiser profiling and diagnostic reports."""

from __future__ import annotations

import json
import os

import torch

from wave_rt.denoiser import settings


class DiagnosticsMixin:
    def _dump_ag_ts(self) -> None:
        """Resolve and report all-gather timing events."""
        torch.cuda.synchronize()
        from collections import defaultdict

        per_tick = defaultdict(list)
        for tick, e0, e1 in self._ag_records:
            per_tick[tick].append(e0.elapsed_time(e1))
        if not per_tick:
            print("[ag_ts] no all_gather records (sync path only)", flush=True)
            return
        ticks = sorted(per_tick)
        all_ms = [ms for t in ticks for ms in per_tick[t]]
        n_calls = len(all_ms)
        per_layer = sum(all_ms) / n_calls
        tick_totals = [sum(per_tick[t]) for t in ticks]
        per_tick_mean = sum(tick_totals) / len(tick_totals)
        print(
            f"\n[ag_ts] rank0 sync all_gather CUDA-event timing over steady ticks "
            f"{ticks}",
            flush=True,
        )
        for t in ticks:
            v = per_tick[t]
            print(
                f"[ag_ts]   t={t}: {len(v)} calls, "
                f"sum={sum(v):.1f}ms, mean/call={sum(v) / len(v):.4f}ms",
                flush=True,
            )
        print(
            f"[ag_ts] SUMMARY: {n_calls} calls over {len(ticks)} steady ticks; "
            f"per-layer all_gather mean={per_layer:.4f} ms/call; "
            f"per-tick total mean={per_tick_mean:.1f} ms/tick "
            f"(Ln={self.Ln})",
            flush=True,
        )

    def _dump_machprof(self) -> None:
        """Resolve and report per-tick machine profiling events."""
        torch.cuda.synchronize()
        rows = []
        for r in self._mp_records:
            fwd = r["e_start"].elapsed_time(r["e_fwd"])
            promo = r["e_fwd"].elapsed_time(r["e_mig0"])
            mig = r["e_mig0"].elapsed_time(r["e_mig1"])
            total = r["e_start"].elapsed_time(r["e_mig1"])
            rows.append(
                dict(
                    tick=r["tick"],
                    rank=r["rank"],
                    is_dn=r["is_dn"],
                    is_st=r["is_st"],
                    fwd=fwd,
                    promo=promo,
                    mig=mig,
                    total=total,
                    submit=r["submit_cpu"],
                    alloc=r["alloc_cpu"],
                    drain=r["drain_cpu"],
                    mig_submit=r["mig_submit_cpu"],
                )
            )
        os.makedirs(settings.MACHINE_PROFILE_DIR, exist_ok=True)
        with open(
            os.path.join(settings.MACHINE_PROFILE_DIR, f"mp_r{self.rank}.json"), "w"
        ) as f:
            json.dump(
                dict(
                    rank=self.rank,
                    world=self.world,
                    store_rank=self.store_rank,
                    num_layers=self.Ln,
                    num_ticks=self.num_ticks,
                    mode=self.cfg.exchange_mode,
                    kv=self.cfg.kv_context,
                    rows=rows,
                ),
                f,
            )
        lo, hi = self.world - 1, self.num_blocks - 1
        steady = [
            x for x in rows if lo <= x["tick"] <= hi and (x["is_dn"] or x["is_st"])
        ]
        tag = f"r{self.rank}"
        print(
            f"\n[machprof {tag}] mode={self.cfg.exchange_mode} "
            f"kv={self.cfg.kv_context} steady ticks [{lo},{hi}] "
            f"({len(steady)} active rows); regions=GPU-event ms, "
            f"submit/alloc/drain/mig_sub=CPU ms",
            flush=True,
        )
        hdr = (
            f"{'tick':>4} {'role':>4} | {'total':>7} {'fwd':>7} {'promo':>6} "
            f"{'mig':>6} | {'submit':>6} {'alloc':>6} {'drain':>6} {'mig_sub':>7}"
        )
        print(f"[machprof {tag}] " + hdr, flush=True)
        for x in steady:
            role = "dn" if x["is_dn"] else ("st" if x["is_st"] else "--")
            print(
                f"[machprof {tag}] {x['tick']:>4} {role:>4} | {x['total']:>7.1f} "
                f"{x['fwd']:>7.1f} {x['promo']:>6.1f} {x['mig']:>6.1f} | "
                f"{x['submit']:>6.2f} {x['alloc']:>6.2f} {x['drain']:>6.2f} "
                f"{x['mig_submit']:>7.2f}",
                flush=True,
            )
        if steady:
            n = len(steady)

            def _m(k):
                return sum(x[k] for x in steady) / n

            print(
                f"[machprof {tag}] MEAN over {n} steady ticks: "
                f"total={_m('total'):.1f} fwd={_m('fwd'):.1f} "
                f"promo={_m('promo'):.1f} mig={_m('mig'):.1f} || "
                f"submit={_m('submit'):.2f} alloc={_m('alloc'):.2f} "
                f"drain={_m('drain'):.2f} mig_submit={_m('mig_submit'):.2f}",
                flush=True,
            )

    def _dump_layer_prof(self) -> None:
        """WAVE_LAYER_PROF: resolve the per-layer CUDA events captured by the exch
        closures into pub/wait/cat/total ms, write raw JSON per rank, and (rank0) a
        steady-state per-layer mean.  Buckets:
          pub  = enter->pub  (publish latency)
          wait = pub->wait   (wait_peer latency; store has none -> 0)
          cat  = wait->cat   (context assembly latency)
          total= enter->cat  (per-layer total)"""
        out_dir = settings.LAYER_PROFILE_DIR
        os.makedirs(out_dir, exist_ok=True)
        # 1) raw JSON per rank
        torch.cuda.synchronize()
        raw = []
        for rec in self._lp_records:
            layers_out = []
            for ly in rec["layers"]:
                ly["enter"].synchronize()
                d = dict(
                    layer=ly["layer"],
                    pub_ms=ly["enter"].elapsed_time(ly["pub"]),
                    wait_ms=ly["pub"].elapsed_time(ly["wait"]),
                    cat_ms=ly["wait"].elapsed_time(ly["cat"]),
                    total_ms=ly["enter"].elapsed_time(ly["cat"]),
                )
                # staggered: pure residual gate latency (w0->w1, excludes L+lead kick)
                if ly.get("w0") is not None and ly.get("w1") is not None:
                    d["residual_wait_ms"] = ly["w0"].elapsed_time(ly["w1"])
                layers_out.append(d)
            raw.append(
                dict(
                    tick=rec["tick"],
                    rank=rec["rank"],
                    is_dn=rec["is_dn"],
                    is_st=rec["is_st"],
                    layers=layers_out,
                )
            )
        path = os.path.join(out_dir, f"layer_prof_r{self.rank}.json")
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)
        # 2) steady-state mean per layer (ticks >= world, exclude fill ramp).
        # All ranks emit their own summary so critical-rank migration is visible.
        from collections import defaultdict

        layer_sums = defaultdict(
            lambda: {
                "pub_ms": 0,
                "wait_ms": 0,
                "cat_ms": 0,
                "total_ms": 0,
                "n": 0,
                "residual_wait_ms": 0.0,
                "res_n": 0,
            }
        )
        ss_recs = [r for r in raw if r["tick"] >= self.world]
        for r in ss_recs:
            for ly in r["layers"]:
                k = layer_sums[ly["layer"]]
                k["pub_ms"] += ly["pub_ms"]
                k["wait_ms"] += ly["wait_ms"]
                k["cat_ms"] += ly["cat_ms"]
                k["total_ms"] += ly["total_ms"]
                k["n"] += 1
                if "residual_wait_ms" in ly:
                    k["residual_wait_ms"] += ly["residual_wait_ms"]
                    k["res_n"] += 1
        summary = []
        sum_total = 0.0
        for L in sorted(layer_sums):
            k = layer_sums[L]
            n = max(k["n"], 1)
            mean_total = k["total_ms"] / n
            sum_total += mean_total
            row = dict(
                layer=L,
                pub_ms=round(k["pub_ms"] / n, 4),
                wait_ms=round(k["wait_ms"] / n, 4),
                cat_ms=round(k["cat_ms"] / n, 4),
                total_ms=round(mean_total, 4),
                n=k["n"],
            )
            if k["res_n"] > 0:
                row["residual_wait_ms"] = round(k["residual_wait_ms"] / k["res_n"], 4)
            summary.append(row)
        # ss ticks / mean per-tick sum_total so cross-rank comparison is one number
        n_ss_ticks = len(set(r["tick"] for r in ss_recs))
        # staggered residual: mean over layers>=1 (layer0 warmup excluded) + layer0
        res_rows = [s for s in summary if "residual_wait_ms" in s]
        res_body = [s["residual_wait_ms"] for s in res_rows if s["layer"] >= 1]
        res_l0 = next(
            (s["residual_wait_ms"] for s in res_rows if s["layer"] == 0), None
        )
        mean_residual = round(sum(res_body) / len(res_body), 4) if res_body else None
        out_obj = dict(
            rank=self.rank,
            n_ticks=len(raw),
            n_ss_ticks=n_ss_ticks,
            sum_total_ms=round(sum_total, 4),
            mean_residual_wait_ms=mean_residual,  # layers 1..N-1
            residual_wait_ms_layer0=res_l0,
            layers=summary,
        )
        sum_path = os.path.join(out_dir, f"layer_summary_r{self.rank}.json")
        with open(sum_path, "w") as f:
            json.dump(out_obj, f, indent=2)
        _res_str = (
            f", mean_residual(L>=1)={mean_residual:.4f}ms (L0={res_l0})"
            if mean_residual is not None
            else ""
        )
        print(
            f"[WAVE_LAYER_PROF] rank{self.rank}: {len(raw)} ticks, "
            f"{len(ss_recs)} steady-state recs, sum_total={sum_total:.2f}ms"
            f"{_res_str}, dumped to {out_dir}/",
            flush=True,
        )

    def _dump_edge_ts(self) -> None:
        """WAVE_EDGE_TS (rank0): synchronize once, read every recorded CUDA event as
        ms-since-reference, write raw JSON + print a per-layer table.  No analysis --
        just the numbers (per the experiment spec)."""
        torch.cuda.synchronize()
        ref = self._edge_ref
        recs = []
        for rec in self._edge_records:
            enter = ref.elapsed_time(rec["enter"])
            attn = ref.elapsed_time(rec["attn"])
            recv = [(int(src), ref.elapsed_time(ev)) for src, ev in rec["recv"]]
            recv_ms = [m for _, m in recv]
            earliest = min(recv_ms) if recv_ms else enter
            latest = max(recv_ms) if recv_ms else enter
            recs.append(
                dict(
                    tick=rec["tick"],
                    layer=rec["layer"],
                    srcs=rec["srcs"],
                    enter=enter,
                    attn=attn,
                    recv=recv,
                    earliest=earliest,
                    latest=latest,
                    skew=latest - earliest,
                    wait=attn - earliest,
                    exposed=attn - enter,
                )
            )
        os.makedirs(settings.EDGE_TIMING_DIR, exist_ok=True)
        path = os.path.join(settings.EDGE_TIMING_DIR, "edge_r0.json")
        with open(path, "w") as f:
            json.dump(
                dict(
                    rank=0,
                    world=self.world,
                    store_rank=self.store_rank,
                    num_layers=self.Ln,
                    num_ticks=self.num_ticks,
                    mode=self.cfg.exchange_mode,
                    kv=self.cfg.kv_context,
                    records=recs,
                ),
                f,
            )

        # steady-state = past the pipeline-fill ramp (tick >= world-1); rank0 is
        # active only for ticks in [0, num_blocks).
        steady = [x for x in recs if x["tick"] >= self.world - 1]
        if not steady:
            steady = recs
        steady_ticks = sorted({x["tick"] for x in steady})
        Ln = self.Ln

        def _agg(rows, key):
            vals = [r[key] for r in rows]
            return sum(vals) / len(vals) if vals else 0.0

        print(
            f"\n[edge_ts] rank0 prefetch-overlap per-edge timestamps "
            f"(mode={self.cfg.exchange_mode} kv={self.cfg.kv_context})",
            flush=True,
        )
        print(
            f"[edge_ts] {len(recs)} records; steady ticks {steady_ticks} "
            f"({len(steady)} records); srcs order = recv_srcs "
            f"(ranks>0 then store={self.store_rank}); raw -> {path}",
            flush=True,
        )
        print(
            "[edge_ts] ms averaged over steady ticks per layer; "
            "skew=latest-earliest recv, wait=attn_start-earliest, "
            "exposed=attn_start-enter",
            flush=True,
        )
        hdr = (
            f"{'layer':>5} | {'n':>3} | {'mean_skew':>9} | {'mean_wait':>9} | "
            f"{'mean_exp':>8} | {'max_skew':>8} | {'max_exp':>8}"
        )
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)
        per_layer = []
        for L in range(Ln):
            rows = [x for x in steady if x["layer"] == L]
            if not rows:
                continue
            ms_skew = _agg(rows, "skew")
            ms_wait = _agg(rows, "wait")
            ms_exp = _agg(rows, "exposed")
            mx_skew = max(r["skew"] for r in rows)
            mx_exp = max(r["exposed"] for r in rows)
            per_layer.append(
                dict(
                    layer=L,
                    n=len(rows),
                    skew=ms_skew,
                    wait=ms_wait,
                    exp=ms_exp,
                    max_skew=mx_skew,
                    max_exp=mx_exp,
                )
            )
            print(
                f"{L:>5} | {len(rows):>3} | {ms_skew:>9.3f} | {ms_wait:>9.3f} | "
                f"{ms_exp:>8.3f} | {mx_skew:>8.3f} | {mx_exp:>8.3f}",
                flush=True,
            )

        # per-source incremental wait (delta from prior wait in recv_srcs order; the
        # first delta is measured from exch entry).  Attributes stall to each source.
        src_delta: dict = {}
        src_cnt: dict = {}
        for x in steady:
            prev = x["enter"]
            for src, m in x["recv"]:
                src_delta[src] = src_delta.get(src, 0.0) + (m - prev)
                src_cnt[src] = src_cnt.get(src, 0) + 1
                prev = m
        if src_delta:
            print(
                "[edge_ts] mean incremental wait per source (ms, recv order; "
                "delta beyond the previous source's wait):",
                flush=True,
            )
            for src in sorted(src_delta):
                print(
                    f"[edge_ts]   src {src}: "
                    f"{src_delta[src] / src_cnt[src]:.3f} ms  "
                    f"(n={src_cnt[src]})",
                    flush=True,
                )

        if per_layer:
            mean_skew = sum(p["skew"] for p in per_layer) / len(per_layer)
            mean_wait = sum(p["wait"] for p in per_layer) / len(per_layer)
            mean_exp = sum(p["exp"] for p in per_layer) / len(per_layer)
            top_skew = sorted(per_layer, key=lambda p: p["skew"], reverse=True)[:5]
            half = Ln // 2
            fh_rows = [p for p in per_layer if p["layer"] < half]
            sh_rows = [p for p in per_layer if p["layer"] >= half]
            fh = sum(p["skew"] for p in fh_rows) / max(1, len(fh_rows))
            sh = sum(p["skew"] for p in sh_rows) / max(1, len(sh_rows))
            print(
                f"[edge_ts] SUMMARY: mean_skew/layer={mean_skew:.3f}ms  "
                f"mean_wait/layer={mean_wait:.3f}ms  "
                f"mean_exposed/layer={mean_exp:.3f}ms",
                flush=True,
            )
            print(
                f"[edge_ts] top-5 skew layers: "
                f"{[(p['layer'], round(p['skew'], 3)) for p in top_skew]}",
                flush=True,
            )
            print(
                f"[edge_ts] skew first-half(L<{half})={fh:.3f}ms  "
                f"second-half={sh:.3f}ms  (slack-accumulation trend)",
                flush=True,
            )
