(() => {
  'use strict';

  // Each .demo-slot with a stage is an independent demo instance: its own
  // lanes, timeline, speed toggle, autostart observer, and auto-loop.
  document.querySelectorAll('.demo-slot').forEach(slot => {
    if (slot.querySelector('.generation-demo-stage')) initDemo(slot);
  });

  function initDemo(root) {
    const stage = root.querySelector('.generation-demo-stage');
    const lanes = [...stage.querySelectorAll('[data-demo-lane]')].map(element => ({
      arm: element.dataset.demoLane,
      cards: [...element.querySelectorAll('[data-demo-card]')],
      clock: element.querySelector('[data-lane-clock]'),
      fill: element.querySelector('[data-lane-progress]'),
      badge: element.querySelector('[data-lane-badge]'),
    }));
    const cards = lanes.flatMap(lane => lane.cards);
    const speedButtons = [...root.querySelectorAll('[data-demo-speed]')];
    let speed = 8;
    let startTime = null;
    let raf = null;
    let holdTimer = null;
    let started = false;
    const holdMs = 6000;

    // Every panel in a lane starts denoising at t=0 and resolves at its own
    // completion time, so finished samples land one after another in waves.
    lanes.forEach(lane => {
      if (lane.arm === 'wave') {
        let cursor = 0;
        lane.cards.forEach(card => {
          cursor += Number(card.dataset.generation);
          card.demoEnd = cursor;
        });
        lane.total = cursor;
        lane.badgeAt = cursor;
      } else {
        const sample = Number(lane.cards[0].dataset.generation);
        lane.cards.forEach((card, index) => {
          card.demoEnd = sample * (index + 1);
        });
        lane.total = sample;
        lane.badgeAt = sample;
      }
    });

    const waveLane = lanes.find(lane => lane.arm === 'wave');
    const rollingLane = lanes.find(lane => lane.arm === 'rolling');
    // Tick marks on the Wave Forcing progress bar: one per completed sample.
    waveLane.cards.forEach(card => {
      if (card.demoEnd >= waveLane.total) return;
      const tick = document.createElement('i');
      tick.className = 'demo-tick';
      tick.style.left = `${(card.demoEnd / waveLane.total) * 100}%`;
      waveLane.fill.parentNode.appendChild(tick);
    });
    // Finish flag at the end of each lane's bar.
    [waveLane, rollingLane].forEach(lane => {
      const finish = document.createElement('i');
      finish.className = 'demo-tick demo-tick-finish';
      finish.style.left = '100%';
      lane.fill.parentNode.appendChild(finish);
    });
    waveLane.badge.textContent = `${waveLane.cards.length} videos generated in ${waveLane.total.toFixed(1)} s`;
    rollingLane.badge.textContent = `1 video in ${rollingLane.badgeAt.toFixed(1)} s`;
    const endAt = Math.max(waveLane.total, rollingLane.total);

    function seededRandom(seed) {
      let state = seed >>> 0;
      return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 4294967296;
      };
    }

    function sizeCanvas(card) {
      const canvas = card.querySelector('canvas');
      const width = Math.max(1, Math.round(card.clientWidth / 2));
      const height = Math.max(1, Math.round(card.clientHeight / 2));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        card.noiseCanvas = null;
      }
      return canvas;
    }

    function makeNoise(card) {
      const canvas = sizeCanvas(card);
      if (card.noiseCanvas) return card.noiseCanvas;
      const noise = document.createElement('canvas');
      noise.width = canvas.width;
      noise.height = canvas.height;
      const context = noise.getContext('2d', { alpha: false });
      const image = context.createImageData(noise.width, noise.height);
      const random = seededRandom(Number(card.dataset.seed) * 2654435761);
      for (let index = 0; index < image.data.length; index += 4) {
        const value = 62 + Math.floor(random() * 156);
        image.data[index] = value * .86;
        image.data[index + 1] = value * .94;
        image.data[index + 2] = value;
        image.data[index + 3] = 255;
      }
      context.putImageData(image, 0, 0);
      card.noiseCanvas = noise;
      return noise;
    }

    // The video plays underneath from t=0; each frame we draw its current
    // frame over the noise field with alpha/blur ramping with progress.
    function paintFrame(card, progress) {
      const canvas = sizeCanvas(card);
      const context = canvas.getContext('2d', { alpha: false });
      context.globalAlpha = 1;
      context.filter = 'none';
      context.drawImage(makeNoise(card), 0, 0, canvas.width, canvas.height);
      const video = card.querySelector('video');
      if (video.paused && video.readyState >= 2) {
        // A stalled or rejected autoplay (big files over a slow link) gets
        // retried while denoising instead of freezing on the first frame.
        const now = Date.now();
        if (!card.playRetryAt || now - card.playRetryAt > 800) {
          card.playRetryAt = now;
          video.play().catch(() => {});
        }
      }
      if (video.readyState < 2) return;
      const eased = progress * progress * (3 - 2 * progress);
      context.globalAlpha = eased;
      context.filter = `blur(${Math.max(0, 11 * (1 - eased))}px)`;
      context.drawImage(video, -8, -8, canvas.width + 16, canvas.height + 16);
      context.filter = 'none';
      context.globalAlpha = 1;
    }

    function setState(card, state, progress = 0) {
      const previous = card.demoState;
      if (previous !== state) {
        card.classList.remove('is-generating', 'is-ready', 'is-paused', 'is-ended');
        card.classList.add(`is-${state}`);
        card.demoState = state;
      }
      if (state === 'generating') paintFrame(card, progress);
      // 'ready' only hides the canvas (CSS); the same video element keeps
      // playing underneath, so the noise-to-video handoff is seamless.
      if (state === 'ready') {
        const video = card.querySelector('video');
        if (video.paused) video.play().catch(() => {});
      }
    }

    function render(now) {
      if (startTime === null) startTime = now;
      const timeline = ((now - startTime) / 1000) * speed;
      cards.forEach(card => {
        if (timeline < card.demoEnd) {
          setState(card, 'generating', Math.max(0, Math.min(1, timeline / card.demoEnd)));
        } else {
          setState(card, 'ready');
        }
      });
      lanes.forEach(lane => {
        lane.clock.textContent = `${Math.min(timeline, lane.total).toFixed(1)} s`;
        lane.fill.style.width = `${Math.min(1, timeline / lane.total) * 100}%`;
        lane.badge.classList.toggle('is-shown', timeline >= lane.badgeAt);
      });
      if (timeline < endAt) {
        raf = requestAnimationFrame(render);
      } else {
        // End state reached: hold it for a fixed wall-clock beat, then loop.
        raf = null;
        if (holdTimer === null) holdTimer = setTimeout(reset, holdMs);
      }
    }

    function reset() {
      if (raf) cancelAnimationFrame(raf);
      if (holdTimer) {
        clearTimeout(holdTimer);
        holdTimer = null;
      }
      startTime = null;
      cards.forEach(card => {
        const video = card.querySelector('video');
        video.currentTime = 0;
        video.play().catch(() => card.classList.add('is-paused'));
        card.demoState = null;
        card.classList.remove('is-paused', 'is-ended');
      });
      lanes.forEach(lane => {
        lane.clock.textContent = '0.0 s';
        lane.fill.style.width = '0%';
        lane.badge.classList.remove('is-shown');
      });
      raf = requestAnimationFrame(render);
      started = true;
    }

    cards.forEach(card => {
      const video = card.querySelector('video');
      video.loop = true;
      video.addEventListener('ended', () => card.classList.add('is-ended'));
      video.addEventListener('play', () => card.classList.remove('is-paused', 'is-ended'));
      video.addEventListener('pause', () => {
        if (!video.ended && card.demoState === 'ready') card.classList.add('is-paused');
      });
      card.addEventListener('click', () => {
        if (card.demoState !== 'ready') return;
        if (video.ended) {
          video.currentTime = 0;
          video.play();
        } else if (video.paused) {
          video.play();
        } else {
          video.pause();
        }
      });
    });

    speedButtons.forEach(button => button.addEventListener('click', () => {
      const now = performance.now();
      if (startTime !== null) {
        const timeline = ((now - startTime) / 1000) * speed;
        speed = Number(button.dataset.demoSpeed);
        startTime = now - (timeline / speed) * 1000;
      } else {
        speed = Number(button.dataset.demoSpeed);
      }
      speedButtons.forEach(candidate => candidate.classList.toggle('active', candidate === button));
    }));

    const observer = new IntersectionObserver(entries => {
      if (!started && entries.some(entry => entry.isIntersecting)) reset();
    }, { threshold: .28 });
    observer.observe(stage);
  }
})();
