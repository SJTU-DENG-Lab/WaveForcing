(function () {
  "use strict";

  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  /* ---------- nav: highlight the section in view ---------- */
  var navLinks = $$(".nav-links a[href^='#']");
  var sections = navLinks.map(function (a) { return $(a.getAttribute("href")); }).filter(Boolean);
  function updateNav() {
    var y = window.scrollY + 120;
    var current = null;
    sections.forEach(function (sec) { if (sec.offsetTop <= y) current = sec.id; });
    navLinks.forEach(function (a) { a.classList.toggle("active", a.getAttribute("href") === "#" + current); });
  }
  window.addEventListener("scroll", updateNav, { passive: true });
  updateNav();

  /* ---------- reveal on scroll ---------- */
  var reveals = $$(".reveal");
  if (reduceMotion || !("IntersectionObserver" in window)) {
    reveals.forEach(function (el) { el.classList.add("in"); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px" });
    reveals.forEach(function (el) { io.observe(el); });
  }

  /* ---------- guide blink ---------- */
  $$("[data-blink]").forEach(function (fig) {
    if (reduceMotion) return;
    (function schedule() {
      setTimeout(function () {
        fig.classList.add("is-blinking");
        setTimeout(function () { fig.classList.remove("is-blinking"); schedule(); }, 150);
      }, 2600 + Math.random() * 3400);
    })();
  });

  /* ---------- promo video: show a placeholder until the file exists ---------- */
  var promoFrame = $("#promoFrame");
  var promo = $("#promoVideo");
  if (promoFrame && promo) {
    var markMissing = function () { promoFrame.classList.add("is-missing"); };
    promo.addEventListener("error", markMissing);
    var source = promo.querySelector("source");
    if (source) source.addEventListener("error", markMissing);
    // networkState is briefly NO_SOURCE while the element starts loading, so only trust it later
    promo.addEventListener("loadedmetadata", function () { promoFrame.classList.remove("is-missing"); });
    if (promo.error) markMissing();
    setTimeout(function () { if (promo.error || (promo.networkState === 3 && promo.readyState === 0)) markMissing(); }, 1500);
  }

  /* ---------- wavefront explorer (S = 4, 8 chunks) ---------- */
  var S = 4, CHUNKS = 8, K_MAX = CHUNKS - 1 + S;
  var grid = $("#wfGrid");
  if (grid) {
    var out = $("#wfOut"), mask = $("#wfMask");
    var kLabel = $("#wfK"), activeLabel = $("#wfActive"), doneLabel = $("#wfDone");
    var playBtn = $("#wfPlay");
    var cells = [], chips = [], maskRows = [];
    var k = 0, timer = null;

    grid.style.setProperty("--cols", CHUNKS);
    out.style.setProperty("--cols", CHUNKS);
    var html = '<span class="lab"></span>';
    for (var i = 0; i < CHUNKS; i++) html += '<span class="lab colh">c' + i + "</span>";
    for (var s = 0; s <= S; s++) {
      html += '<span class="lab">' + (s < S ? "step " + (s + 1) : "store") + "</span>";
      for (i = 0; i < CHUNKS; i++) {
        html += '<span class="wf-cell' + (s === S ? " store" : "") + '" data-i="' + i + '" data-s="' + s + '" data-k="' + i + "+" + s + '"></span>';
      }
    }
    grid.innerHTML = html;
    cells = $$(".wf-cell", grid);

    var outHtml = '<span class="lab">out</span>';
    for (i = 0; i < CHUNKS; i++) outHtml += '<span class="out-chip"></span>';
    out.innerHTML = outHtml;
    chips = $$(".out-chip", out);

    var maskHtml = '<span class="h"></span>';
    for (var j = 0; j < S; j++) maskHtml += '<span class="h">j' + j + "</span>";
    for (s = 0; s < S; s++) {
      maskHtml += '<span class="h" data-row="' + s + '">s' + s + "</span>";
      for (j = 0; j < S; j++) {
        maskHtml += '<span class="m' + (j === s ? " self" : j > s ? " on" : "") + '" title="slot ' + s + (j >= s ? " reads " : " does not read ") + "slot " + j + '"></span>';
      }
    }
    mask.innerHTML = maskHtml;
    maskRows = $$("[data-row]", mask);

    var render = function () {
      var active = 0;
      cells.forEach(function (c) {
        var ci = +c.dataset.i, cs = +c.dataset.s, ck = ci + cs;
        c.classList.toggle("active", ck === k);
        c.classList.toggle("done", ck < k);
        c.classList.toggle("next", ck === k + 1);
        if (ck === k) active++;
      });
      var emitted = 0;
      chips.forEach(function (chip, ci) {
        var on = k > ci + S - 1;
        chip.classList.toggle("on", on);
        if (on) emitted++;
      });
      maskRows.forEach(function (row, rs) {
        var ci = k - rs;
        row.title = "query slot s" + rs + (ci >= 0 && ci < CHUNKS ? ", chunk c" + ci : ", idle");
      });
      kLabel.textContent = "k = " + k;
      activeLabel.textContent = active + " / " + (S + 1);
      doneLabel.textContent = emitted + " / " + CHUNKS;
    };

    var stop = function () {
      if (timer) clearInterval(timer);
      timer = null;
      playBtn.textContent = "Play";
      playBtn.setAttribute("aria-pressed", "false");
    };
    var play = function () {
      if (timer) return;
      if (k >= K_MAX) { k = 0; render(); }
      playBtn.textContent = "Pause";
      playBtn.setAttribute("aria-pressed", "true");
      timer = setInterval(function () {
        if (k >= K_MAX) { stop(); return; }
        k++;
        render();
      }, 900);
    };
    playBtn.addEventListener("click", function () { timer ? stop() : play(); });
    $("#wfNext").addEventListener("click", function () { stop(); k = Math.min(K_MAX, k + 1); render(); });
    $("#wfPrev").addEventListener("click", function () { stop(); k = Math.max(0, k - 1); render(); });
    $("#wfReset").addEventListener("click", function () { stop(); k = 0; render(); });

    if (reduceMotion || !("IntersectionObserver" in window)) {
      k = 6;
      render();
    } else {
      render();
      var started = false;
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting && !started) { started = true; play(); }
        });
      }, { threshold: 0.45 }).observe($("#explorer"));
    }
  }

  /* ---------- WP vs CP chart (paper Table: DiT-only throughput) ---------- */
  // [scale, backend, ulysses, ring, wp, ratio, dagger flags {u, w, r}]
  var CP = [
    ["1.3B", "BF16", 80.60, 41.19, 150.09, 1.862, "wr"],
    ["1.3B", "Sage", 84.01, 56.99, 151.15, 1.799, ""],
    ["1.3B", "Sage+W8A8", 61.17, 45.96, 154.75, 2.530, ""],
    ["3B", "BF16", 75.38, 38.20, 88.86, 1.179, ""],
    ["3B", "Sage", 75.93, 37.00, 89.26, 1.176, ""],
    ["3B", "Sage+W8A8", 56.15, 36.04, 95.31, 1.697, ""],
    ["5B", "BF16", 58.40, 41.42, 68.33, 1.170, ""],
    ["5B", "Sage", 63.13, 30.37, 67.00, 1.061, ""],
    ["5B", "Sage+W8A8", 60.71, 31.17, 75.44, 1.243, ""],
    ["7B", "BF16", 49.89, 38.96, 53.28, 1.068, ""],
    ["7B", "Sage", 52.70, 25.75, 53.50, 1.015, "uwr"],
    ["7B", "Sage+W8A8", 56.12, 26.56, 61.55, 1.097, ""],
    ["9B", "BF16", 46.41, 38.94, 45.70, 0.985, "uwr"],
    ["9B", "Sage", 45.26, 22.73, 45.59, 1.007, ""],
    ["9B", "Sage+W8A8", 52.51, 24.16, 53.84, 1.025, ""],
    ["14B", "BF16", 26.72, 23.90, 28.71, 1.075, ""],
    ["14B", "Sage", 27.90, 16.32, 28.72, 1.029, ""],
    ["14B", "Sage+W8A8", 30.98, 17.16, 33.60, 1.085, ""]
  ];
  var chart = $("#cpChart");
  if (chart) {
    var viz = $("#cpViz"), tip = $("#cpTip"), sub = $("#cpSub");
    var backend = "Sage+W8A8";
    var SERIES = [
      { key: "wp", name: "WP (ours)", idx: 4, flag: "w", color: "var(--series-wp)" },
      { key: "uly", name: "Ulysses CP", idx: 2, flag: "u", color: "var(--series-uly)" },
      { key: "ring", name: "Ring CP", idx: 3, flag: "", color: "var(--series-ring)" }
    ];
    var fmt = function (v) { return v.toFixed(2); };
    var dag = function (row, flag) { return flag && row[6].indexOf(flag) >= 0 ? "†" : ""; };
    var NS = "http://www.w3.org/2000/svg";
    var el = function (name, attrs, text) {
      var n = document.createElementNS(NS, name);
      for (var a in attrs) n.setAttribute(a, attrs[a]);
      if (text != null) n.textContent = text;
      return n;
    };
    // square pixel bars on whole-pixel edges
    var barPath = function (x0, y, w, h) { return "M" + x0 + "," + y + "H" + (x0 + w) + "V" + (y + h) + "H" + x0 + "Z"; };
    var ctx = document.createElement("canvas").getContext("2d");
    var tw = function (s) {
      ctx.font = '12px "WF Pixel", "IBM Plex Mono", monospace';
      return ctx.measureText(s).width;
    };

    var draw = function () {
      var rows = CP.filter(function (r) { return r[1] === backend; });
      var W = Math.max(280, chart.clientWidth);
      var narrow = W < 520;
      var labelW = narrow ? 40 : 56, rightW = narrow ? 92 : 120, top = 8, axisH = 26;
      var barH = narrow ? 12 : 14, barGap = 2, groupGap = narrow ? 16 : 20;
      var groupH = SERIES.length * barH + (SERIES.length - 1) * barGap;
      var H = top + rows.length * (groupH + groupGap) - groupGap + axisH;
      var plotW = W - labelW - rightW, xMax = 160;
      var x = function (v) { return labelW + Math.round((v / xMax) * plotW); };

      chart.innerHTML = "";
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H });
      [0, 40, 80, 120, 160].forEach(function (t) {
        svg.appendChild(el("rect", { x: x(t) - (t === 0 ? 2 : 0), y: top - 4, width: t === 0 ? 2 : 1, height: H - axisH - top + 8, fill: t === 0 ? "var(--axis)" : "var(--grid)" }));
        // the pixel face is crisp only on whole pixels, so centre by measured width instead of text-anchor
        svg.appendChild(el("text", { x: Math.round(x(t) - tw(String(t)) / 2), y: H - 6, "font-size": 12, fill: "var(--text-3)" }, String(t)));
      });

      rows.forEach(function (row, gi) {
        var gy = top + gi * (groupH + groupGap);
        var g = el("g", { "data-gi": gi });
        g.appendChild(el("text", { x: 0, y: gy + Math.floor((groupH + 9) / 2), "font-size": 12, fill: "var(--ink)" }, row[0]));
        SERIES.forEach(function (sr, si) {
          var y = gy + si * (barH + barGap);
          var w = x(row[sr.idx]) - labelW;
          g.appendChild(el("path", { d: barPath(labelW, y, w, barH), fill: sr.color, "data-si": si }));
          if (sr.key === "wp") {
            var label = fmt(row[4]) + dag(row, "w") + " · " + row[5].toFixed(narrow ? 2 : 3) + "×" + dag(row, "r");
            g.appendChild(el("text", { x: x(row[4]) + 6, y: y + Math.floor((barH + 9) / 2), "font-size": 12, fill: "var(--ink)" }, label));
          }
        });
        // hit target: the whole band, larger than the marks
        var hit = el("rect", { x: 0, y: gy - groupGap / 2, width: W, height: groupH + groupGap, fill: "transparent", "data-gi": gi });
        g.appendChild(hit);
        svg.appendChild(g);
      });
      chart.appendChild(svg);

      var show = function (gi, evt) {
        var row = rows[gi];
        var html = '<span class="tt">' + row[0] + " · " + row[1] + " · " + (row[0] === "14B" ? "4 steps / 5 GPUs" : "5 steps / 6 GPUs") + "</span>";
        SERIES.forEach(function (sr) {
          html += '<div class="row' + (sr.key === "wp" ? " hl" : "") + '"><span><i style="background:' + sr.color + '"></i>' + sr.name + "</span><span>" + fmt(row[sr.idx]) + dag(row, sr.flag) + "</span></div>";
        });
        html += '<div class="row"><span>WP / faster CP</span><span>' + row[5].toFixed(3) + "×" + dag(row, "r") + "</span></div>";
        tip.innerHTML = html;
        var box = viz.getBoundingClientRect();
        var px = evt.clientX - box.left + 14, py = evt.clientY - box.top + 14;
        var tw = tip.offsetWidth || 190;
        if (px + tw > box.width - 8) px = Math.max(8, evt.clientX - box.left - tw - 14);
        tip.style.left = px + "px";
        tip.style.top = py + "px";
        tip.classList.add("show");
      };
      $$("rect[data-gi]", svg).forEach(function (r) {
        r.addEventListener("mousemove", function (e) { show(+r.dataset.gi, e); });
        r.addEventListener("mouseleave", function () { tip.classList.remove("show"); });
      });
    };

    var tbody = $("#cpTbody");
    tbody.innerHTML = CP.map(function (r, n) {
      var wpBest = r[4] >= r[2];
      var cell = function (v, best, d) { return best ? '<td class="best"><span>' + fmt(v) + d + "</span></td>" : "<td>" + fmt(v) + d + "</td>"; };
      var steps = r[0] === "14B" ? "4" : "5", gpus = r[0] === "14B" ? "5" : "6";
      return '<tr' + (n % 3 === 0 && n ? ' class="sep"' : "") + "><td>" + r[0] + '</td><td class="l">' + r[1] + "</td><td>" + steps + "</td><td>" + gpus + "</td>" +
        cell(r[2], !wpBest, dag(r, "u")) + "<td>" + fmt(r[3]) + "</td>" + cell(r[4], wpBest, dag(r, "w")) +
        (r[5] < 1 ? '<td class="issue"><span>' : "<td>") + r[5].toFixed(3) + "×" + dag(r, "r") + (r[5] < 1 ? "</span></td>" : "</td>") + "</tr>";
    }).join("");

    $$(".seg button[data-backend]", viz).forEach(function (b) {
      b.addEventListener("click", function () {
        backend = b.dataset.backend;
        $$(".seg button[data-backend]", viz).forEach(function (o) { o.setAttribute("aria-pressed", String(o === b)); });
        sub.textContent = backend + " · higher is better";
        draw();
      });
    });
    var toggle = $("#cpToggle"), table = $("#cpTable");
    toggle.addEventListener("click", function () {
      var open = table.hasAttribute("hidden");
      if (open) table.removeAttribute("hidden"); else table.setAttribute("hidden", "");
      toggle.setAttribute("aria-expanded", String(open));
      toggle.textContent = open ? "Hide table" : "Show table";
    });
    var lastW = 0;
    window.addEventListener("resize", function () {
      if (chart.clientWidth !== lastW) { lastW = chart.clientWidth; draw(); }
    });
    draw();
    lastW = chart.clientWidth;
    if (document.fonts && document.fonts.load) document.fonts.load('12px "WF Pixel"').then(draw, function () {});
  }

  /* ---------- sample gallery ---------- */
  $$(".gallery-head .seg button[data-set]").forEach(function (b, _, all) {
    b.addEventListener("click", function () {
      all.forEach(function (o) { o.setAttribute("aria-pressed", String(o === b)); });
      $$("[data-gallery]").forEach(function (g) {
        var on = g.dataset.gallery === b.dataset.set;
        if (on) g.removeAttribute("hidden"); else g.setAttribute("hidden", "");
        if (!on) $$("video", g).forEach(function (v) { v.pause(); v.parentNode.classList.remove("playing"); });
      });
    });
  });
  $$(".clip").forEach(function (clip) {
    var v = $("video", clip);
    clip.addEventListener("click", function () {
      if (v.paused) {
        v.play().then(function () { clip.classList.add("playing"); }).catch(function () {});
      } else {
        v.pause();
        clip.classList.remove("playing");
      }
    });
  });

  /* ---------- BibTeX copy ---------- */
  var copyBtn = $("#copyBib");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      var text = $("#bibText").textContent;
      var done = function () {
        copyBtn.textContent = "Copied";
        setTimeout(function () { copyBtn.textContent = "Copy"; }, 1600);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done);
      } else {
        var ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); done(); } catch (e) {}
        document.body.removeChild(ta);
      }
    });
  }
})();
