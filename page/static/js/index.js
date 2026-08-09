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
  const diffusionGrid = $('#diffusionGrid');
  const vaeGrid = $('#vaeGrid');

  function makeRank(label, role, index, type) {
    const rank = document.createElement('div');
    rank.className = 'rank';
    rank.dataset.index = index;
    rank.dataset.type = type;
    rank.innerHTML = `<div class="rank-head"><b>${label}</b><span>${role}</span></div><div class="rank-slot"><span class="idle">idle</span></div>`;
    return rank;
  }

  ['GPU 0|Denoise D₁', 'GPU 1|Denoise D₂', 'GPU 2|Denoise D₃', 'GPU 3|Denoise D₄', 'GPU 4|Clean KV store']
    .forEach((item, index) => {
      const [label, role] = item.split('|');
      diffusionGrid.appendChild(makeRank(label, role, index, 'diffusion'));
    });

  ['GPU 5|VAE early', 'GPU 6|VAE middle', 'GPU 7|VAE output']
    .forEach((item, index) => {
      const [label, role] = item.split('|');
      vaeGrid.appendChild(makeRank(label, role, index, 'vae'));
    });

  function chunkHTML(chunkIndex, labelPrefix = 'C') {
    const color = colors[chunkIndex % colors.length];
    return `<span class="chunk" style="--chunk:${color}">${labelPrefix}${chunkIndex + 1}</span>`;
  }

  function phaseFor(currentTick) {
    if (currentTick < 4) return {
      title: 'Filling the wavefront',
      description: `Chunk C${currentTick + 1} enters D₁ while earlier chunks move to later denoising ranks. Idle GPUs disappear one tick at a time.`
    };
    if (currentTick < 7) return {
      title: 'Diffusion at full occupancy',
      description: 'All diffusion ranks are active. The store rank receives a completed latent chunk and immediately streams it toward the VAE.'
    };
    if (currentTick < 9) return {
      title: 'Overlapping generation and decode',
      description: 'New latent chunks continue through diffusion while earlier chunks are decoded by the independent three-stage VAE pipeline.'
    };
    return {
      title: 'Steady state: one chunk per tick',
      description: 'The complete eight-GPU pipeline is occupied. Every tick advances all in-flight chunks and retires one decoded video chunk.'
    };
  }

  function render() {
    let active = 0;
    diffusionGrid.querySelectorAll('.rank').forEach((rank, rankIndex) => {
      const chunkIndex = tick - rankIndex;
      const slot = rank.querySelector('.rank-slot');
      if (chunkIndex >= 0) {
        rank.classList.add('active');
        slot.innerHTML = chunkHTML(chunkIndex);
        active += 1;
      } else {
        rank.classList.remove('active');
        slot.innerHTML = '<span class="idle">idle</span>';
      }
    });

    vaeGrid.querySelectorAll('.rank').forEach((rank, rankIndex) => {
      const chunkIndex = tick - 5 - rankIndex;
      const slot = rank.querySelector('.rank-slot');
      if (chunkIndex >= 0) {
        rank.classList.add('active');
        slot.innerHTML = chunkHTML(chunkIndex);
        active += 1;
      } else {
        rank.classList.remove('active');
        slot.innerHTML = '<span class="idle">waiting</span>';
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
