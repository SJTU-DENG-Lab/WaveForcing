(() => {
  'use strict';

  const colors = ['#20d7d0', '#8b7cff', '#c8ff58', '#ff9a5b', '#64a8ff', '#ff6f91', '#f7d154', '#70e3a2'];
  const maxTick = 14;
  let tick = 0;
  let playing = true;
  let intervalMs = 800;
  let timer;

  const $ = (selector) => document.querySelector(selector);
  const timeline = $('#timeline');
  const playButton = $('#playButton');
  const stepButton = $('#stepButton');
  const wavefrontMatrix = $('#wavefrontMatrix');
  const pipelineStages = [
    { id: 'D₁', role: 'denoise 1', group: 'diffusion' },
    { id: 'D₂', role: 'denoise 2', group: 'diffusion' },
    { id: 'D₃', role: 'denoise 3', group: 'diffusion' },
    { id: 'D₄', role: 'denoise 4', group: 'diffusion' },
    { id: 'KV', role: 'clean store', group: 'store' },
    { id: 'V₁', role: 'VAE early', group: 'vae' },
    { id: 'V₂', role: 'VAE middle', group: 'vae' },
    { id: 'V₃', role: 'VAE output', group: 'vae' }
  ];

  function buildWavefrontMatrix() {
    const cells = [
      '<div class="matrix-corner"><b>Stage ID →</b><span>GPU ID ↓</span></div>',
      '<div class="matrix-group diffusion-group">Wave diffusion · full-model replicas</div>',
      '<div class="matrix-group vae-group">Streaming VAE · time-balanced</div>'
    ];

    pipelineStages.forEach((stage, stageIndex) => {
      const boundary = stageIndex === 5 ? ' vae-boundary' : '';
      cells.push(`<div class="matrix-stage-label ${stage.group}${boundary}" style="grid-column:${stageIndex + 2};grid-row:2"><b>${stage.id}</b><span>${stage.role}</span></div>`);
    });

    pipelineStages.forEach((ownedStage, gpuIndex) => {
      const row = gpuIndex + 3;
      const rowBoundary = gpuIndex === 5 ? ' vae-row-boundary' : '';
      cells.push(`<div class="matrix-gpu-label ${ownedStage.group}${rowBoundary}" style="grid-column:1;grid-row:${row}"><b>GPU ${gpuIndex}</b><span>replica ${gpuIndex}</span></div>`);
      pipelineStages.forEach((stage, stageIndex) => {
        const mapped = gpuIndex === stageIndex;
        const kvReturn = gpuIndex < stageIndex && stageIndex <= 4;
        const divider = stageIndex === 5 ? ' vae-boundary' : '';
        const cellType = mapped ? `mapped ${stage.group}` : (kvReturn ? 'kv-return-cell' : 'unmapped');
        const classes = `matrix-cell ${cellType}${divider}${rowBoundary}`;
        const label = mapped
          ? `<span class="matrix-coordinate">GPU ${gpuIndex} · ${stage.id}</span><span class="matrix-idle">idle</span>`
          : (kvReturn ? '<span class="kv-pending">KV</span>' : '');
        const accessibility = mapped
          ? `aria-label="GPU ${gpuIndex}, stage ${stage.role}"`
          : (kvReturn ? `aria-label="GPU ${gpuIndex} receives KV from ${stage.role}"` : 'aria-hidden="true"');
        cells.push(`<div class="${classes}" data-stage="${stageIndex}" data-source-stage="${kvReturn ? stageIndex : ''}" style="grid-column:${stageIndex + 2};grid-row:${row}" ${accessibility}>${label}</div>`);
      });
    });

    wavefrontMatrix.innerHTML = cells.join('');
  }

  buildWavefrontMatrix();

  function chunkHTML(chunkIndex, labelPrefix = 'C') {
    const color = colors[chunkIndex % colors.length];
    return `<span class="matrix-chunk" style="--chunk:${color}"><b>${labelPrefix}${chunkIndex + 1}</b></span>`;
  }

  function phaseFor(currentTick) {
    if (currentTick < 4) return {
      title: 'Filling the wavefront',
      description: `Chunk C${currentTick + 1} enters GPU 0 / D₁ while earlier colors advance diagonally. Pale cells fan up-left as each cleaner stage publishes its KV.`
    };
    if (currentTick < 7) return {
      title: 'Diffusion at full occupancy',
      description: 'GPU 0–4 are active across D₁–D₄ and the clean-KV stage. A completed latent now crosses the vertical boundary toward V₁.'
    };
    if (currentTick < 9) return {
      title: 'Overlapping generation and decode',
      description: 'New colors continue across diffusion while earlier colors advance through the independent V₁–V₃ decode stages.'
    };
    return {
      title: 'Steady state: one chunk per tick',
      description: 'The active diagonal spans all eight GPU–stage pairs; the light upper triangle confirms that every required cleaner-stage KV has arrived.'
    };
  }

  function render() {
    let active = 0;
    wavefrontMatrix.querySelectorAll('.matrix-cell.mapped').forEach((cell, stageIndex) => {
      const chunkIndex = tick - stageIndex;
      const stage = pipelineStages[stageIndex];
      const coordinate = `<span class="matrix-coordinate">GPU ${stageIndex} · ${stage.id}</span>`;
      if (chunkIndex >= 0) {
        cell.classList.add('active');
        cell.innerHTML = coordinate + chunkHTML(chunkIndex);
        active += 1;
      } else {
        cell.classList.remove('active');
        cell.innerHTML = `${coordinate}<span class="matrix-idle">${stage.group === 'vae' ? 'waiting' : 'idle'}</span>`;
      }
    });

    wavefrontMatrix.querySelectorAll('.matrix-cell.kv-return-cell').forEach(cell => {
      const sourceStage = Number(cell.dataset.sourceStage);
      const chunkIndex = tick - sourceStage;
      if (chunkIndex >= 0) {
        const color = colors[chunkIndex % colors.length];
        cell.style.setProperty('--kv', color);
        cell.classList.add('kv-ready');
        cell.innerHTML = `<span class="kv-arrived" style="--kv:${color}"><i></i><b>KV</b><span>C${chunkIndex + 1}</span></span>`;
      } else {
        cell.style.removeProperty('--kv');
        cell.classList.remove('kv-ready');
        cell.innerHTML = '<span class="kv-pending">KV</span>';
      }
    });

    const phase = phaseFor(tick);
    $('#phaseTitle').textContent = phase.title;
    $('#phaseDescription').textContent = phase.description;
    $('#activeCount').textContent = active;
    $('#outputCount').textContent = Math.max(0, tick - 7);
    $('#tickOutput').textContent = tick;
    timeline.value = tick;
    timeline.style.setProperty('--progress', `${(tick / maxTick) * 100}%`);
  }

  function setPlaying(next) {
    playing = next;
    clearInterval(timer);
    const icon = playButton.querySelector('span');
    const label = playButton.querySelector('b');
    icon.className = playing ? 'pause-icon' : 'play-icon';
    label.textContent = playing ? 'Pause' : 'Play';
    playButton.setAttribute('aria-label', playing ? 'Pause animation' : 'Play animation');
    if (playing) timer = setInterval(advance, intervalMs);
  }

  function advance() {
    tick = tick >= maxTick ? 0 : tick + 1;
    render();
  }

  playButton.addEventListener('click', () => setPlaying(!playing));
  stepButton.addEventListener('click', () => {
    setPlaying(false);
    advance();
  });
  timeline.addEventListener('input', (event) => {
    tick = Number(event.target.value);
    setPlaying(false);
    render();
  });
  document.querySelectorAll('[data-speed]').forEach(button => {
    button.addEventListener('click', () => {
      document.querySelectorAll('[data-speed]').forEach(item => item.classList.remove('active'));
      button.classList.add('active');
      intervalMs = Number(button.dataset.speed);
      if (playing) setPlaying(true);
    });
  });

  // Layer-level A/B schedule: synchronized exchange vs. communication hidden by
  // the causal compute imbalance. Values are conceptual proportions, not a trace.
  const scheduleSpecs = {};
  const rankSchedules = [
    { key: 'r0', syncCompute: 12.0, overlapCompute: 15.7, type: 'compute' },
    { key: 'r1', syncCompute: 10.7, overlapCompute: 13.5, type: 'compute secondary shade-1' },
    { key: 'r2', syncCompute: 9.4, overlapCompute: 11.8, type: 'compute secondary shade-2' },
    { key: 'r3', syncCompute: 8.1, overlapCompute: 10.0, type: 'compute secondary shade-3' },
    { key: 'store', syncCompute: 6.8, overlapCompute: 8.4, type: 'compute secondary shade-4' }
  ];
  rankSchedules.forEach(rank => {
    scheduleSpecs[`.sync-${rank.key}`] = [];
    scheduleSpecs[`.overlap-${rank.key}`] = [];
  });
  scheduleSpecs['.sync-link'] = [];
  scheduleSpecs['.overlap-link'] = [];

  const layerCount = 6;
  const cycle = 100 / layerCount;
  const syncCommStart = 12.0;
  for (let layer = 0; layer < layerCount; layer += 1) {
    const base = layer * cycle;
    const layerName = `L${layer ? `+${layer}` : ''}`;
    rankSchedules.forEach(rank => {
      scheduleSpecs[`.sync-${rank.key}`].push(
        { start: base, width: rank.syncCompute, type: rank.type, text: `${rank.key} · ${layerName}` },
        { start: base + rank.syncCompute, width: cycle - rank.syncCompute, type: 'wait', text: rank.key === 'r0' ? 'barrier' : 'wait' }
      );
      scheduleSpecs[`.overlap-${rank.key}`].push(
        { start: base, width: rank.overlapCompute, type: rank.type, text: `${rank.key} · ${layerName}` },
        { start: base + rank.overlapCompute, width: cycle - rank.overlapCompute, type: rank.key === 'r0' ? 'wait' : 'slack', text: rank.key === 'r0' ? 'gate' : 'slack' }
      );
      if (rank.key !== 'r0') {
        scheduleSpecs[`.overlap-${rank.key}`].push(
          { start: base + rank.overlapCompute - 1.8, width: 1.8, type: 'comm publish', text: 'KV↑' }
        );
      }
    });
    scheduleSpecs['.sync-link'].push(
      { start: base + syncCommStart, width: cycle - syncCommStart, type: 'comm', text: `all_gather ${layerName}` }
    );
    scheduleSpecs['.overlap-link'].push(
      { start: base + 8.4, width: 5.8, type: 'comm', text: `prefetch ${layer < layerCount - 1 ? `L+${layer + 1}` : 'next'}` }
    );
  }

  Object.entries(scheduleSpecs).forEach(([selector, segments]) => {
    const track = $(selector);
    segments.forEach(spec => {
      const segment = document.createElement('span');
      segment.className = `timeline-segment ${spec.type}`;
      segment.style.left = `${spec.start}%`;
      segment.style.width = `${spec.width}%`;
      segment.dataset.start = spec.start;
      segment.dataset.end = spec.start + spec.width;
      segment.textContent = spec.text;
      track.appendChild(segment);
    });
  });

  const scheduleButton = $('#schedulePlay');
  const scheduleProgress = $('#scheduleProgress');
  const scheduleClock = $('#scheduleClock');
  let schedulePlaying = true;
  let schedulePosition = 0;
  let scheduleLast = performance.now();
  const scheduleDuration = 6200;

  function renderSchedule() {
    const percent = schedulePosition * 100;
    scheduleProgress.style.width = `${percent}%`;
    scheduleClock.textContent = `${Math.round(schedulePosition * 600)} μs`;
    document.querySelectorAll('.schedule-canvas').forEach(canvas => {
      const track = canvas.querySelector('.lane-track');
      const cursor = canvas.querySelector('.schedule-cursor');
      cursor.style.left = `${track.offsetLeft + track.clientWidth * schedulePosition}px`;
    });
    document.querySelectorAll('.timeline-segment').forEach(segment => {
      const start = Number(segment.dataset.start);
      const end = Number(segment.dataset.end);
      segment.classList.toggle('current', percent >= start && percent < end);
    });
  }

  function runSchedule(now) {
    if (schedulePlaying) {
      schedulePosition = (schedulePosition + (now - scheduleLast) / scheduleDuration) % 1;
      renderSchedule();
    }
    scheduleLast = now;
    requestAnimationFrame(runSchedule);
  }

  scheduleButton.addEventListener('click', () => {
    schedulePlaying = !schedulePlaying;
    const icon = scheduleButton.querySelector('span');
    icon.className = schedulePlaying ? 'pause-icon' : 'play-icon';
    scheduleButton.querySelector('b').textContent = schedulePlaying ? 'Pause comparison' : 'Play comparison';
  });
  window.addEventListener('resize', renderSchedule);
  requestAnimationFrame(runSchedule);

  const transportModes = {
    sync: {
      step: 'Baseline · synchronized BF16',
      title: 'Every rank joins; every byte is full width.',
      description: 'Blocking all_gather moves BF16 KV through a matching collective. The consumer then concatenates received segments into an attention buffer at every layer.',
      engine: 'NCCL collective', bandwidth: 'BF16 · 2 bytes / value', action: 'Receive, then assemble',
      metricLabel: 'Critical-path communication', metricValue: '≈79 ms', metricNote: '14B diagnostic'
    },
    fp8: {
      step: 'Optimization 1 · shrink the payload',
      title: 'FP8 cuts the KV payload in half.',
      description: 'Quantizing exchanged KV from BF16 to FP8 halves DMA bytes and raw communication time. The collective barrier and destination assembly remain, and quant/dequant adds a new toll.',
      engine: 'NCCL collective', bandwidth: 'FP8 · 1 byte / value', action: 'Dequantize, then assemble',
      metricLabel: 'Raw communication', metricValue: '≈40 ms', metricNote: '≈79 → 40 ms; standalone win'
    },
    ce: {
      step: 'Optimization 2 · change the transport',
      title: 'Copy Engine writes remote memory directly.',
      description: 'cuMemcpyPeerAsync pushes KV into an IPC-mapped consumer slab over NVLink. A generation flag releases the consumer, replacing matching NCCL send/recv synchronization without stealing SM compute.',
      engine: 'CE · one-sided NVLink', bandwidth: 'BF16 · remote IPC write', action: 'Wait on flag, then assemble',
      metricLabel: '14B steady tick', metricValue: '583 ms', metricNote: '645 → 583 ms · +10.4%'
    },
    paged: {
      step: 'Optimization 3 · write the final layout',
      title: 'Producers land KV in attention-ready slots.',
      description: 'The copy engine writes each segment into a fixed per-(generation, layer) slot. The consumer reads one contiguous prefix in place, eliminating the per-layer torch.cat assembly wall.',
      engine: 'CE · paged direct-write', bandwidth: 'BF16 · fixed destination slots', action: 'Read contiguous prefix in place',
      metricLabel: 'r0 per-layer assembly', metricValue: '4.08 ms', metricNote: '18.49 → 4.08 ms'
    }
  };
  const transportVisual = $('.transport-visual');
  const transportButtons = [...document.querySelectorAll('[data-transport]')];
  let transportIndex = 0;
  let transportAuto = true;

  function setTransportMode(mode) {
    const data = transportModes[mode];
    transportIndex = transportButtons.findIndex(button => button.dataset.transport === mode);
    transportVisual.dataset.mode = mode;
    $('#transportStep').textContent = data.step;
    $('#transportTitle').textContent = data.title;
    $('#transportDescription').textContent = data.description;
    $('#routeEngine').textContent = data.engine;
    $('#routeBandwidth').textContent = data.bandwidth;
    $('#consumerAction').textContent = data.action;
    $('#metricLabel').textContent = data.metricLabel;
    $('#metricValue').textContent = data.metricValue;
    $('#metricNote').textContent = data.metricNote;
    transportButtons.forEach(button => {
      const active = button.dataset.transport === mode;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', active ? 'true' : 'false');
    });
  }

  transportButtons.forEach(button => button.addEventListener('click', () => {
    transportAuto = false;
    setTransportMode(button.dataset.transport);
  }));
  $('.transport-explorer').addEventListener('mouseenter', () => { transportAuto = false; });
  setInterval(() => {
    if (!transportAuto) return;
    const next = (transportIndex + 1) % transportButtons.length;
    setTransportMode(transportButtons[next].dataset.transport);
  }, 4200);

  let vaePhase = 0;
  const vaePipelineGrid = $('#vaeWaveGrid');
  const vaeCursor = $('#vaeGridCursor');
  const vaeTicks = [...vaePipelineGrid.querySelectorAll('.vwg-tick')];
  function renderVaePhase() {
    vaePipelineGrid.querySelectorAll('[data-vae-phase]').forEach(cell => {
      cell.classList.toggle('vae-current', Number(cell.dataset.vaePhase) === vaePhase);
    });
    if (vaeTicks[vaePhase]) vaeCursor.style.left = `${vaeTicks[vaePhase].offsetLeft}px`;
  }
  renderVaePhase();
  setInterval(() => {
    vaePhase = (vaePhase + 1) % 4;
    renderVaePhase();
  }, 900);
  window.addEventListener('resize', renderVaePhase);

  const benchmarkTierLabels = {
    bf16: { name: 'BF16', detail: 'Base precision' },
    sage: { name: 'Sage', detail: 'SageAttention' },
    sagefp8: { name: 'Sage + W8A8', detail: 'Compute quantization' }
  };
  const benchmarkModes = ['sync', 'overlap', 'onesided', 'staggered', 'paged'];
  const benchmarkButtons = [...document.querySelectorAll('[data-benchmark]')];
  const benchmarkBody = $('#benchmarkBody');
  let benchmarkData;

  function benchmarkCell(cell, isE2eBest, isSteadyBest) {
    if (!cell) return '<span class="benchmark-empty" aria-label="Not measured">—</span>';
    const classes = ['benchmark-cell'];
    if (isE2eBest) classes.push('best');
    if (isSteadyBest) classes.push('steady-best');
    if (cell.anomaly) classes.push('anomaly');
    const speedup = Number.isFinite(cell.speedup) ? ` · ${cell.speedup.toFixed(1)}×` : '';
    const warning = cell.anomaly ? `<b class="benchmark-anomaly">${cell.anomaly}</b>` : '';
    return `<div class="${classes.join(' ')}"><strong class="e2e-hero">${cell.e2eFps.toFixed(1)}<em> E2E FPS</em></strong><small>${cell.p50.toFixed(1)} ms${speedup}</small><small class="wall-fps">DiT ${cell.ditFps.toFixed(1)} · FPS(p50) ${cell.fpsP50.toFixed(1)}</small>${warning}</div>`;
  }

  function renderBenchmark(key) {
    if (!benchmarkData || !benchmarkData.benchmarks[key]) return;
    const data = benchmarkData.benchmarks[key];
    const available = Object.values(data.rows).flatMap(row => benchmarkModes.map(mode => row[mode]).filter(Boolean));
    const bestP50 = Math.min(...available.map(cell => cell.p50));
    const bestE2e = Math.max(...available.map(cell => cell.e2eFps));

    $('#benchmarkTitle').textContent = `${data.scale} · ${data.topology}`;
    $('#benchmarkBaseline').textContent = Number.isFinite(data.baseline) ? `${benchmarkData.statistics.outputFrames} frames @ ${data.baselineFps.toFixed(1)} FPS` : 'Not recorded · dummy';
    const scope = Number.isFinite(data.baseline)
      ? `Pooled ${benchmarkData.statistics.steadyWindow} · ${benchmarkData.statistics.framesPerTick} frames/tick. DiT and E2E use ${benchmarkData.statistics.outputFrames} output frames; × uses the ${data.baseline.toFixed(3)} s baseline.`
      : `Pooled ${benchmarkData.statistics.steadyWindow} · shape-accurate ${data.weights}; DiT/E2E use ${benchmarkData.statistics.outputFrames} output frames. No 14B single-GPU baseline.`;
    $('#benchmarkFoot').textContent = `${scope} ${data.note || ''}`.trim();

    benchmarkBody.innerHTML = Object.entries(benchmarkTierLabels).map(([tier, label]) => {
      const row = data.rows[tier] || {};
      const cells = benchmarkModes.map(mode => {
        const cell = row[mode];
        return `<td>${benchmarkCell(cell, cell && cell.e2eFps === bestE2e, cell && cell.p50 === bestP50)}</td>`;
      }).join('');
      return `<tr><td><div class="benchmark-tier ${tier}"><i></i><div><span>${label.name}</span><b>${label.detail}</b></div></div></td>${cells}</tr>`;
    }).join('');

    benchmarkButtons.forEach(button => {
      const active = button.dataset.benchmark === key;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', active ? 'true' : 'false');
      button.tabIndex = active ? 0 : -1;
    });
  }

  benchmarkButtons.forEach(button => button.addEventListener('click', () => renderBenchmark(button.dataset.benchmark)));
  fetch('static/data/results_summary.json')
    .then(response => {
      if (!response.ok) throw new Error(`Benchmark data returned ${response.status}`);
      return response.json();
    })
    .then(data => {
      benchmarkData = data;
      renderBenchmark('1.3b-4+3');
    })
    .catch(error => {
      benchmarkBody.innerHTML = '<tr><td colspan="6" class="benchmark-load-error">Benchmark summary could not be loaded.</td></tr>';
      console.error(error);
    });

  const citationModal = $('#citationModal');
  const citationText = $('#blogCitation');
  const citationCopy = $('#copyCitation');
  const citationStatus = $('#citationCopyStatus');
  let citationReturnFocus = null;

  function openCitation() {
    citationReturnFocus = document.activeElement;
    citationModal.hidden = false;
    document.body.classList.add('cite-open');
    citationModal.querySelector('.citation-close').focus();
  }

  function closeCitation() {
    citationModal.hidden = true;
    document.body.classList.remove('cite-open');
    citationStatus.textContent = '';
    if (citationReturnFocus) citationReturnFocus.focus();
  }

  document.querySelectorAll('[data-cite-open]').forEach(button => button.addEventListener('click', openCitation));
  citationModal.querySelectorAll('[data-cite-close]').forEach(element => element.addEventListener('click', closeCitation));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !citationModal.hidden) closeCitation();
  });

  citationCopy.addEventListener('click', async () => {
    const bibtex = citationText.textContent.trim();
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(bibtex);
      } else {
        const helper = document.createElement('textarea');
        helper.value = bibtex;
        helper.setAttribute('readonly', '');
        helper.style.position = 'fixed';
        helper.style.opacity = '0';
        document.body.appendChild(helper);
        helper.select();
        document.execCommand('copy');
        helper.remove();
      }
      citationCopy.textContent = 'Copied';
      citationStatus.textContent = 'BibTeX copied to clipboard.';
      window.setTimeout(() => { citationCopy.textContent = 'Copy BibTeX'; }, 1600);
    } catch (error) {
      citationStatus.textContent = 'Copy failed — select the BibTeX above.';
      console.error(error);
    }
  });

  const heroWave = $('#heroWave');
  for (let index = 0; index < 15; index += 1) {
    const particle = document.createElement('i');
    particle.style.left = `${(index % 5) * 9}%`;
    particle.style.animationDelay = `${index * -0.31}s`;
    particle.style.animationDuration = `${4.2 + (index % 4) * 0.35}s`;
    heroWave.appendChild(particle);
  }

  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        if (entry.target.classList.contains('result-layout')) {
          $('.bar-chart').classList.add('animate');
          animateCount($('.count-up'));
        }
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.14 });
  document.querySelectorAll('.reveal').forEach(element => revealObserver.observe(element));

  function animateCount(element) {
    if (!element || element.dataset.done) return;
    element.dataset.done = 'true';
    const target = Number(element.dataset.target);
    const start = performance.now();
    const duration = 1200;
    const update = now => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      element.textContent = (target * eased).toFixed(1);
      if (progress < 1) requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
  }

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    setPlaying(false);
    schedulePlaying = false;
    schedulePosition = 0.72;
    tick = 10;
  } else {
    setPlaying(true);
  }
  render();
})();
