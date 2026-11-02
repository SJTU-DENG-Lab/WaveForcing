/* Wave Forcing project page: the two runtime explainers in #runtime (K/V delivery, VAE pipeline).
   Each mode is a fixed schedule; a playhead that moves in whole steps reveals it, as in the promo. */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var STEP = 1000 / 12, HOLD = 24; // 12 steps a second; the finished schedule holds for 2 s
  var FONT = '12px "WF Pixel", "IBM Plex Mono", monospace';
  var COL = {
    ink: "#07111f", text3: "#647083", line: "#d9e1de", line2: "#c9d3d1", paper: "#f4f7f5",
    tealInk: "#0f6f6d", wash: "#d8efea", p: "#b9e3dc", cyan: "#20d7d0", hi: "#70fff1", tan: "#e2b977",
    v1: "#c4bcff", v2: "#8b7cff", v3: "#5b4bd6", vq: "#b2a7ff", vqBg: "#f1efff", coral: "#c8553d"
  };

  var ctx = document.createElement("canvas").getContext("2d");
  var widths = {};
  var measure = function (s) {
    if (!(s in widths)) {
      ctx.font = FONT;
      widths[s] = Math.ceil(ctx.measureText(s).width);
    }
    return widths[s];
  };

  var el = function (parent, name, attrs, txt) {
    var n = document.createElementNS(NS, name);
    for (var a in attrs) n.setAttribute(a, attrs[a]);
    if (txt != null) n.textContent = txt;
    if (parent) parent.appendChild(n);
    return n;
  };
  var rect = function (g, x, y, w, h, fill) {
    return w > 0 && h > 0 ? el(g, "rect", { x: x, y: y, width: w, height: h, fill: fill }) : null;
  };
  // the pixel face stays crisp only on whole pixels: start anchor, integer x and baseline
  var text = function (g, x, y, s, fill) {
    return el(g, "text", { x: x, y: y, fill: fill || COL.ink, "font-size": 12 }, s);
  };
  var centered = function (g, x0, w, y, s, fill) {
    return text(g, x0 + Math.floor((w - measure(s)) / 2), y, s, fill);
  };
  var label2 = function (g, y, name, role) {
    text(g, 0, y + 13, name, COL.ink);
    text(g, 0, y + 28, role, COL.text3);
  };
  var dashes = function (g, x0, x1, y, on, off, fill) {
    for (var x = x0; x < x1; x += on + off) rect(g, x, y, Math.min(on, x1 - x), 2, fill);
  };
  var vdashes = function (g, x, y0, y1, fill) {
    for (var y = y0; y < y1; y += 8) rect(g, x, y, 2, Math.min(4, y1 - y), fill);
  };
  // 2 px checker, the same dither as the legend swatches
  var pattern = function (defs, id, bg, fg) {
    var p = el(defs, "pattern", { id: id, width: 4, height: 4, patternUnits: "userSpaceOnUse" });
    rect(p, 0, 0, 4, 4, bg);
    rect(p, 2, 0, 2, 2, fg);
    rect(p, 0, 2, 2, 2, fg);
    return "url(#" + id + ")";
  };
  // a tile from x0 to x1 with a 2 px gap on the right, labelled with the longest tier that fits
  var block = function (g, x0, x1, y, h, fill, o) {
    var w = x1 - x0 - 2;
    if (w <= 0) return;
    rect(g, x0, y, w, h, fill);
    if (o.hi) rect(g, x0, y, w, 4, o.hi);
    var tiers = o.tiers || [], pad = o.plate ? 16 : 8;
    for (var i = 0; i < tiers.length; i++) {
      var tw = measure(tiers[i]);
      if (tw + pad > w) continue;
      var tx = x0 + Math.floor((w - tw) / 2), ty = y + (o.base || Math.floor((h + 9) / 2));
      if (o.plate) rect(g, tx - 4, ty - 12, tw + 8, 16, o.plate);
      text(g, tx, ty, tiers[i], o.ink || COL.ink);
      return;
    }
  };

  /* ---------- K/V delivery inside one round, two GPUs, three layers ---------- */
  // schematic costs: QKV + RoPE 3, producer attention 4, consumer attention 6 (it reads more keys), delivery 3
  var kvSchedule = function (mode) {
    var P = 3, CP = 4, CC = 6, TAU = 3;
    var sc = { prod: [], cons: [], links: [], barriers: [], notes: [] };
    var pt = 0, t = 0;
    for (var l = 1; l <= 3; l++) {
      if (mode === "sync") {
        var b = t + P;
        // the faster producer idles until the barrier, then both wait out the transfer
        sc.prod.push(["p", pt, pt + P], ["idle", pt + P, b], ["wait", b, b + TAU], ["c", b + TAU, b + TAU + CP]);
        sc.cons.push(["p", t, b], ["wait", b, b + TAU], ["c", b + TAU, b + TAU + CC]);
        sc.links.push({ l: l, t0: b, t1: b + TAU, use: b + TAU });
        sc.barriers.push(b);
        sc.notes.push({ t: t, s: "L" + l + " · barrier, w = τ", bad: true });
        pt = b + TAU + CP;
        t = b + TAU + CC;
      } else {
        // paged delivery leaves 5.91 of the 19.02 ms one-sided assembly per tick
        var copy = mode === "paged" ? 5.91 / 19.02 : 1;
        var ready = pt + P, land = ready + TAU, need = t + P, use = Math.max(need, land);
        sc.prod.push(["p", pt, ready], ["c", ready, ready + CP]);
        sc.cons.push(["p", t, need]);
        if (land > need) sc.cons.push(["wait", need, land]);
        sc.cons.push(["copy", use, use + copy], ["c", use + copy, use + copy + CC]);
        sc.links.push({ l: l, t0: ready, t1: land, use: use });
        sc.notes.push(land > need ? { t: t, s: "L" + l + " · w = τ, exposed", bad: true } : { t: t, s: "L" + l + " · Δ > τ → w = 0" });
        pt = ready + CP;
        t = use + copy + CC;
      }
    }
    sc.end = t;
    return sc;
  };

  var KV = {
    T: 36,
    steps: 58,
    modes: ["sync", "early", "paged"],
    ours: "paged",
    text: {
      sync: '<span class="px">Sync</span> Each layer ends in a barrier: both GPUs stop until the K/V gather finishes, so every transfer adds its full cost τ to the round.',
      early: '<span class="px">Early publish</span> GPU 2 publishes K/V right after QKV + RoPE and keeps computing. From layer 2 on, K/V lands before GPU 1 needs it (Δ > τ, so w = 0); only the first wait stays exposed, and GPU 1 still copies K/V into its attention buffer.',
      paged: '<span class="px">Paged</span> K/V is written straight into GPU 1\'s ordered attention slots and flagged ready, so the copy nearly disappears: context assembly falls from 19.02 to 5.91 ms per tick (one-sided → paged). Sync → Paged cuts tick p50 by 14.2%.'
    },
    aria: {
      sync: "Schedule diagram, synchronized: at every layer both GPUs stop at a barrier while K/V is gathered, so each transfer is exposed.",
      early: "Schedule diagram, early publication: GPU 2 publishes K/V after QKV and RoPE and keeps computing; only the first layer's transfer is exposed, and GPU 1 copies K/V into its attention buffer before each attention.",
      paged: "Schedule diagram, paged delivery: K/V is written straight into GPU 1's attention slots, so only a short assembly step remains before each attention."
    },
    bars: {
      head: "Tick p50 · 1.3B BF16 · ms",
      max: 91.956,
      rows: [["sync", "Sync", 91.956], ["overlap", "Overlap", 85.73], ["onesided", "One-sided", 81.669], ["staggered", "Staggered", 81.953], ["paged", "Paged", 78.882]],
      on: { sync: ["sync"], early: ["overlap"], paged: ["paged"] }
    },
    draw: function (d, W, mode) {
      var LW = 136, u = Math.max(16, Math.floor((W - LW - 16) / 36));
      var X = function (t) { return LW + Math.round(t * u); };
      var yP = 20, yL = 72, yBuf = 106, yC = 124, yF = 176, end = X(36);
      var g = d.reset(W, 188, 160), sc = kvSchedule(mode);
      var KIND = {
        p: { fill: COL.p, tiers: ["QKV + RoPE", "QKV"] },
        c: { fill: COL.tealInk, tiers: ["attn + MLP", "attn"], ink: "#fff" },
        copy: { fill: COL.tan, tiers: ["copy"] },
        wait: { fill: d.fill.wait, tiers: ["wait"], ink: COL.text3, plate: "#fff" },
        idle: { fill: d.fill.wait }
      };

      rect(g.bg, LW, yP, end - LW - 2, 32, COL.paper);
      rect(g.bg, LW, yC, end - LW - 2, 32, COL.paper);
      dashes(g.bg, LW, end - 2, yL + 9, 4, 4, COL.line2);
      label2(g.labels, yP, "GPU 2 · t=500", "producer");
      text(g.labels, 0, yL + 14, "K/V link", COL.text3);
      label2(g.labels, yC, "GPU 1 · t=750", "consumer");
      text(g.labels, 0, yF, "exposed wait", COL.text3);

      [[sc.prod, yP], [sc.cons, yC]].forEach(function (lane) {
        lane[0].forEach(function (b) {
          var k = KIND[b[0]];
          block(g.tl, X(b[1]), X(b[2]), lane[1], 32, k.fill, k);
        });
      });
      sc.links.forEach(function (lk) {
        var x0 = X(lk.t0), x1 = X(lk.t1), xu = X(lk.use);
        rect(g.tl, x0, yP + 32, 2, yL - yP - 32, COL.ink);
        block(g.tl, x0, x1, yL, 20, COL.cyan, { hi: COL.hi, tiers: ["K/V L" + lk.l, "L" + lk.l], base: 16 });
        rect(g.tl, x1 - 4, yL + 20, 2, yBuf - yL - 20, COL.ink);
        if (xu > x1) {
          // landed early: K/V sits ready in GPU 1 until attention needs it
          dashes(g.tl, x1 - 4, xu, yBuf, 4, 2, COL.ink);
          var tw = measure("ready");
          if (xu - x1 >= tw + 8) text(g.tl, x1 + Math.floor((xu - x1 - tw) / 2), yBuf - 3, "ready", COL.tealInk);
        } else {
          rect(g.tl, x1 - 4, yBuf, xu - x1 + 4, 2, COL.ink);
        }
        rect(g.tl, xu, yBuf, 2, yC - yBuf, COL.ink);
      });
      sc.barriers.forEach(function (b) { vdashes(g.tl, X(b) - 2, yP - 4, yC + 36, COL.ink); });
      sc.notes.forEach(function (n) { text(g.tl, X(n.t), yF, n.s, n.bad ? COL.coral : COL.tealInk); });
      if (sc.end < 36) block(g.fin, X(sc.end), end, yC, 32, COL.wash, { tiers: ["saved"], ink: COL.tealInk });
      return { W: W, T: 36, X: X };
    }
  };

  /* ---------- VAE decoding over eight DiT rounds (measured times, ms) ---------- */
  var R = 78.88;
  // decoder blocks: the profiled cut sums to 73.15 / 60.91 / 80.89, the FLOP cut's middle stage to 115.76
  var UNITS = [23.575, 23.575, 26.0, 30.455, 30.455, 28.85, 26.02, 26.02];
  var CUTS = { flop: [2, 6], profiled: [3, 5] };
  var stageOf = function (cut, i) { return !cut ? 1 : i < cut[0] ? 0 : i < cut[1] ? 1 : 2; };

  var vaeSchedule = function (mode) {
    var T = 8 * R, k, t;
    var sc = { dit: [], waits: [], lanes: [[], [], []], queue: [], outs: [], qs: -1 };
    if (mode === "blocking") {
      var dec = 0;
      UNITS.forEach(function (x) { dec += x; });
      for (t = 0, k = 0; ; k++) {
        sc.dit.push({ t0: t, t1: t + R, k: k });
        if (t + R >= T) break;
        sc.waits.push({ t0: t + R, t1: t + R + dec });
        sc.lanes[0].push({ t0: t + R, t1: t + R + dec, k: k });
        sc.outs.push({ t: t + R + dec, k: k, lane: 0 });
        t += R + dec;
      }
    } else {
      var st = [0, 0, 0], free = [0, 0, 0], rows = [0, 0];
      UNITS.forEach(function (x, i) { st[stageOf(CUTS[mode], i)] += x; });
      for (k = 0; k < 8; k++) {
        sc.dit.push({ t0: k * R, t1: (k + 1) * R, k: k });
        var ready = (k + 1) * R;
        for (var si = 0; si < 3; si++) {
          var start = Math.max(ready, free[si]);
          if (start > ready + 1e-9) {
            var row = rows[0] <= ready ? 0 : 1;
            sc.queue.push({ t0: ready, t1: start, row: row });
            rows[row] = start;
            sc.qs = si;
          }
          sc.lanes[si].push({ t0: start, t1: start + st[si], k: k });
          free[si] = ready = start + st[si];
        }
        sc.outs.push({ t: ready, k: k, lane: 2 });
      }
    }
    sc.outs = sc.outs.filter(function (o) { return o.t <= T + 1e-9; });
    sc.qLabel = sc.qs < 0 ? "queue" : "queue → VAE " + (sc.qs + 1);
    return sc;
  };

  var VAE = {
    T: 8 * R,
    steps: 72,
    modes: ["blocking", "flop", "profiled"],
    ours: "profiled",
    text: {
      blocking: '<span class="px">Blocking</span> The DiT hands a chunk to the decoder and waits for it, so the four denoising GPUs sit idle through every decode.',
      flop: '<span class="px">FLOP split</span> Decoding now overlaps the DiT, but splitting by FLOPs leaves a 115.76 ms middle stage, longer than the 78.88 ms round. Chunks queue in front of it and come out at its pace.',
      profiled: '<span class="px">Profiled split</span> Cutting by measured time gives 73.15 / 60.91 / 80.89 ms, each about one round, so after the fill about one chunk comes out per round. The last stage runs 2 ms over the round.'
    },
    aria: {
      blocking: "Schedule diagram, blocking decode: the DiT waits while each chunk is decoded, so only two chunks come out in eight rounds.",
      flop: "Schedule diagram, FLOP split: the middle VAE stage takes 115.76 ms, longer than a 78.88 ms DiT round, so chunks queue in front of it and come out every 115.76 ms.",
      profiled: "Schedule diagram, profiled split: stages of 73.15, 60.91 and 80.89 ms are each about one round, so after the pipeline fills about one chunk comes out per round."
    },
    bars: {
      head: "Slowest stage vs one round · ms",
      max: 115.76,
      rows: [["round", "DiT round", 78.88], ["flop", "FLOP split", 115.76], ["prof", "Profiled", 80.89]],
      on: { blocking: ["round"], flop: ["round", "flop"], profiled: ["round", "prof"] }
    },
    draw: function (d, W, mode) {
      var LW = 136, T = 8 * R, s = (W - LW - 16) / T;
      var X = function (t) { return LW + Math.round(Math.min(t, T) * s); };
      var yD = 20, yQ = 56, yV = [80, 108, 136], yO = 172, yA = 200, yS = 264, end = X(T);
      var g = d.reset(W, 324, 196), sc = vaeSchedule(mode), blocking = mode === "blocking";
      var LANE = [{ fill: COL.v1, ink: COL.ink }, { fill: COL.v2, ink: COL.ink }, { fill: COL.v3, ink: "#fff" }];
      var cut = CUTS[mode], k;

      rect(g.bg, LW, yD, end - LW - 2, 28, COL.paper);
      yV.forEach(function (y) { rect(g.bg, LW, y, end - LW - 2, 24, COL.paper); });
      rect(g.bg, LW - 2, yA, end - LW + 2, 2, COL.line2);
      for (k = 0; k <= 8; k++) rect(g.bg, X(k * R) - 2, yA + 2, 2, 6, COL.line2);
      for (k = 0; k < 8; k++) centered(g.bg, X(k * R), X((k + 1) * R) - X(k * R) - 2, yA + 20, String(k + 1), COL.text3);
      text(g.labels, 0, yD + 18, "DiT · 4+1 GPUs", COL.ink);
      text(g.labels, 0, yQ + 12, sc.qLabel, blocking ? COL.line2 : COL.text3);
      yV.forEach(function (y, si) { text(g.labels, 0, y + 16, "VAE " + (si + 1), blocking && si ? COL.line2 : COL.ink); });
      var outLabel = text(g.labels, 0, yO + 14, "out · 0", COL.ink);
      text(g.labels, 0, yA + 20, "round · 78.88 ms", COL.text3);

      sc.dit.forEach(function (b) {
        block(g.tl, X(b.t0), X(b.t1), yD, 28, COL.tealInk, { tiers: ["c" + b.k], ink: "#fff" });
      });
      sc.waits.forEach(function (b) {
        block(g.tl, X(b.t0), X(b.t1), yD, 28, d.fill.wait, { tiers: ["waiting for decode", "waiting"], ink: COL.text3, plate: "#fff" });
      });
      sc.queue.forEach(function (q) {
        if (q.t0 >= T) return;
        rect(g.tl, X(q.t0), q.row ? yQ : yQ + 8, Math.max(2, X(q.t1) - X(q.t0)), 8, d.fill.queue);
      });
      sc.lanes.forEach(function (lane, si) {
        lane.forEach(function (b) {
          if (blocking) block(g.tl, X(b.t0), X(b.t1), yV[si], 24, COL.v2, { tiers: ["decode c" + b.k, "c" + b.k] });
          else block(g.tl, X(b.t0), X(b.t1), yV[si], 24, LANE[si].fill, { tiers: ["c" + b.k], ink: LANE[si].ink });
        });
      });
      // a decoded chunk pops out whole once the playhead reaches its finish time
      var pops = sc.outs.map(function (o) {
        var x = X(o.t), y = yV[o.lane] + 24, p = el(g.labels, "g");
        rect(p, x - 4, y, 2, yO - y, COL.ink);
        rect(p, x - 22, yO, 20, 20, COL.cyan);
        rect(p, x - 22, yO, 20, 4, COL.hi);
        centered(p, x - 22, 20, yO + 16, "c" + o.k, COL.ink);
        return p;
      });

      // where the decoder is cut, on the same ms scale, next to one round
      dashes(g.bg, 0, W, 236, 8, 4, COL.line);
      text(g.labels, 0, yS + 14, "decoder split", COL.text3);
      var acc = 0, xs = [X(0)];
      UNITS.forEach(function (x) { acc += x; xs.push(X(acc)); });
      UNITS.forEach(function (x, i) { rect(g.bg, xs[i], yS, xs[i + 1] - xs[i] - 2, 20, LANE[stageOf(cut, i)].fill); });
      var stages = !cut ? [[0, 8, "whole decoder", "on one GPU"]] :
        [[0, cut[0], "VAE 1"], [cut[0], cut[1], "VAE 2"], [cut[1], 8, "VAE 3"]];
      stages.forEach(function (sg, i) {
        var x0 = xs[sg[0]], w = xs[sg[1]] - x0 - 2, ms = 0;
        for (var j = sg[0]; j < sg[1]; j++) ms += UNITS[j];
        var time = sg[3] || (mode === "profiled" || i === 1 ? ms.toFixed(2) + " ms" : "");
        if (sg[0]) {
          var tw = measure("▼");
          text(g.bg, x0 - 1 - Math.floor(tw / 2), yS - 4, "▼", COL.ink);
        }
        centered(g.bg, x0, w, yS + 38, sg[2], COL.ink);
        if (time) centered(g.bg, x0, w, yS + 52, time, cut && ms > R + 20 ? COL.coral : COL.text3);
      });
      var rx = xs[8] + 40, rw = Math.round(R * s);
      rect(g.bg, rx, yS + 9, rw, 2, COL.tealInk);
      rect(g.bg, rx, yS + 4, 2, 12, COL.tealInk);
      rect(g.bg, rx + rw - 2, yS + 4, 2, 12, COL.tealInk);
      centered(g.bg, rx, rw, yS + 38, "one round", COL.tealInk);
      centered(g.bg, rx, rw, yS + 52, "78.88 ms", COL.tealInk);

      return {
        W: W,
        T: T,
        X: X,
        live: function (t) {
          var n = 0;
          sc.outs.forEach(function (o, i) {
            var out = o.t <= t + 1e-9;
            if (out) n++;
            pops[i].setAttribute("visibility", out ? "visible" : "hidden");
          });
          outLabel.textContent = "out · " + n;
        }
      };
    }
  };

  /* ---------- player ---------- */
  function Demo(root, spec) {
    var self = this;
    var auto = !reduceMotion && "IntersectionObserver" in window;
    this.root = root;
    this.spec = spec;
    this.svg = root.querySelector(".demo-svg");
    this.box = this.svg.parentNode;
    this.modeText = root.querySelector(".demo-mode");
    this.barsBox = root.querySelector(".bars");
    this.buttons = Array.prototype.slice.call(root.querySelectorAll(".seg button[data-mode]"));
    this.playBtn = root.querySelector('[data-act="play"]');
    this.timer = null;
    this.userPaused = false;
    this.hold = 0;
    // walk through the modes until the reader picks one; with reduced motion, show our final schedule
    this.tour = auto;
    this.step = auto ? 0 : spec.steps;
    this.setMode(auto ? spec.modes[0] : spec.ours, false);

    this.buttons.forEach(function (b) {
      b.addEventListener("click", function () {
        self.tour = false;
        self.setMode(b.getAttribute("data-mode"), true);
        self.restart(!reduceMotion);
      });
    });
    root.querySelector('[data-act="replay"]').addEventListener("click", function () { self.restart(true); });
    this.playBtn.addEventListener("click", function () {
      if (self.timer) {
        self.userPaused = true;
        self.stop();
      } else {
        self.userPaused = false;
        self.play();
      }
    });
    if (auto) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) self.stop();
          else if (e.intersectionRatio >= 0.4 && !self.userPaused) self.play();
        });
      }, { threshold: [0, 0.4] }).observe(root);
    }
  }

  Demo.prototype.width = function () {
    return Math.max(720, Math.floor(this.box.getBoundingClientRect().width));
  };

  // empty the SVG at a 1:1 scale and lay out its layers; the clip reveals the timeline up to the playhead
  Demo.prototype.reset = function (W, H, headBottom) {
    var svg = this.svg, id = this.root.id;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    svg.style.width = W + "px";
    var defs = el(svg, "defs");
    this.fill = {
      wait: pattern(defs, id + "-wait", COL.paper, COL.line2),
      queue: pattern(defs, id + "-queue", COL.vqBg, COL.vq)
    };
    this.clip = el(el(defs, "clipPath", { id: id + "-clip" }), "rect", { x: 0, y: 0, width: 0, height: H });
    var g = this.g = {
      bg: el(svg, "g"),
      tl: el(svg, "g", { "clip-path": "url(#" + id + "-clip)" }),
      fin: el(svg, "g"),
      labels: el(svg, "g"),
      head: el(svg, "g")
    };
    rect(g.head, -8, 0, 16, 8, COL.ink);
    rect(g.head, -4, 8, 8, 4, COL.ink);
    rect(g.head, -1, 12, 2, headBottom - 12, COL.ink);
    return g;
  };

  Demo.prototype.build = function () {
    this.L = this.spec.draw(this, this.width(), this.mode);
    this.frame();
  };

  Demo.prototype.frame = function () {
    var L = this.L, n = this.spec.steps, done = this.step >= n;
    var t = Math.min(1, this.step / n) * L.T, x = L.X(t);
    this.clip.setAttribute("width", done ? L.W : x);
    this.g.head.setAttribute("transform", "translate(" + x + ",0)");
    this.g.head.setAttribute("visibility", done ? "hidden" : "visible");
    this.g.fin.setAttribute("visibility", done ? "visible" : "hidden");
    if (L.live) L.live(t);
  };

  Demo.prototype.setMode = function (m, user) {
    var sp = this.spec;
    this.mode = m;
    this.buttons.forEach(function (b) { b.setAttribute("aria-pressed", String(b.getAttribute("data-mode") === m)); });
    // announce a mode the reader picked, not every stop of the tour
    this.modeText.setAttribute("aria-live", user ? "polite" : "off");
    this.modeText.innerHTML = sp.text[m];
    this.svg.setAttribute("aria-label", sp.aria[m]);
    this.renderBars();
    this.build();
  };

  Demo.prototype.renderBars = function () {
    var b = this.spec.bars, on = b.on[this.mode];
    var html = '<p class="bh">' + b.head + "</p>";
    b.rows.forEach(function (r) {
      html += '<div class="bar-row' + (on.indexOf(r[0]) >= 0 ? " is-on" : "") + '"><span>' + r[1] + '</span><span class="bt"><i style="--w:' +
        (100 * r[2] / b.max).toFixed(2) + '%"></i></span><span class="bv">' + r[2].toFixed(2) + "</span></div>";
    });
    this.barsBox.innerHTML = html;
  };

  Demo.prototype.play = function () {
    if (this.timer) return;
    var self = this;
    if (this.step >= this.spec.steps) {
      this.step = 0;
      this.hold = 0;
      this.frame();
    }
    this.playBtn.textContent = "Pause";
    this.playBtn.setAttribute("aria-pressed", "true");
    this.timer = setInterval(function () { self.tick(); }, STEP);
  };

  Demo.prototype.stop = function () {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.playBtn.textContent = "Play";
    this.playBtn.setAttribute("aria-pressed", "false");
  };

  Demo.prototype.restart = function (animate) {
    this.stop();
    this.hold = 0;
    this.step = animate ? 0 : this.spec.steps;
    this.frame();
    if (animate) {
      this.userPaused = false;
      this.play();
    }
  };

  Demo.prototype.tick = function () {
    if (this.step < this.spec.steps) {
      this.step++;
      this.frame();
      return;
    }
    if (++this.hold < HOLD) return;
    this.hold = 0;
    // with reduced motion a requested pass plays once
    if (reduceMotion) {
      this.stop();
      return;
    }
    if (this.tour) {
      var ms = this.spec.modes;
      this.setMode(ms[(ms.indexOf(this.mode) + 1) % ms.length], false);
    }
    this.step = 0;
    this.frame();
  };

  var demos = [];
  Array.prototype.forEach.call(document.querySelectorAll("figure.px-demo[data-demo]"), function (root) {
    var spec = { kv: KV, vae: VAE }[root.getAttribute("data-demo")];
    if (spec && root.querySelector(".demo-svg")) demos.push(root.__demo = new Demo(root, spec));
  });

  var raf = 0;
  window.addEventListener("resize", function () {
    if (raf) return;
    raf = requestAnimationFrame(function () {
      raf = 0;
      demos.forEach(function (d) { if (d.width() !== d.L.W) d.build(); });
    });
  });
  // labels are fitted with measured widths, so measure again once the pixel face is in
  if (document.fonts && document.fonts.load) {
    document.fonts.load('12px "WF Pixel"').then(function () {
      widths = {};
      demos.forEach(function (d) { d.build(); });
    }, function () {});
  }
})();
