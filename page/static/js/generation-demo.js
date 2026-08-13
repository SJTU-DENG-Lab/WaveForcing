(() => {
  'use strict';

  const root = document.querySelector('[data-generation-demo]');
  if (!root) return;

  const cards = [...root.querySelectorAll('[data-demo-card]')];
  const waveCards = cards.filter(card => card.dataset.arm === 'wave');
  const baseline = cards.find(card => card.dataset.arm === 'rolling');
  const speedButtons = [...root.querySelectorAll('[data-demo-speed]')];
  const replayButton = root.querySelector('[data-demo-replay]');
  let speed = 4;
  let startTime = null;
  let raf = null;
  let started = false;

  let cursor = 0;
  waveCards.forEach(card => {
    card.demoStart = cursor;
    cursor += Number(card.dataset.generation);
    card.demoEnd = cursor;
  });
  baseline.demoStart = 0;
  baseline.demoEnd = Number(baseline.dataset.generation);

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
      card.noiseData = null;
    }
    return canvas;
  }

  function makeNoise(card) {
    const canvas = sizeCanvas(card);
    if (card.noiseData) return card.noiseData;
    const context = canvas.getContext('2d', { alpha: false });
    const image = context.createImageData(canvas.width, canvas.height);
    const random = seededRandom(Number(card.dataset.seed) * 2654435761);
    for (let index = 0; index < image.data.length; index += 4) {
      const value = 62 + Math.floor(random() * 156);
      image.data[index] = value * .86;
      image.data[index + 1] = value * .94;
      image.data[index + 2] = value;
      image.data[index + 3] = 255;
    }
    card.noiseData = image;
    return image;
  }

  function paintNoise(card, progress) {
    const canvas = sizeCanvas(card);
    const context = canvas.getContext('2d', { alpha: false });
    context.putImageData(makeNoise(card), 0, 0);
    const poster = card.posterImage;
    if (!poster || !poster.complete) return;
    const eased = progress * progress * (3 - 2 * progress);
    context.globalAlpha = eased;
    context.filter = `blur(${Math.max(0, 11 * (1 - eased))}px)`;
    context.drawImage(poster, -8, -8, canvas.width + 16, canvas.height + 16);
    context.filter = 'none';
    context.globalAlpha = 1;
  }

  function setState(card, state, progress = 0) {
    const previous = card.demoState;
    if (previous !== state) {
      card.classList.remove('is-queued', 'is-generating', 'is-ready', 'is-paused', 'is-ended');
      card.classList.add(`is-${state}`);
      card.demoState = state;
    }
    if (state === 'queued') paintNoise(card, 0);
    if (state === 'generating') paintNoise(card, progress);
    if (state === 'ready' && previous !== 'ready') {
      const video = card.querySelector('video');
      video.currentTime = 0;
      video.play().catch(() => card.classList.add('is-paused'));
    }
  }

  function render(now) {
    if (startTime === null) startTime = now;
    const timeline = ((now - startTime) / 1000) * speed;
    cards.forEach(card => {
      if (timeline < card.demoStart) {
        setState(card, 'queued');
      } else if (timeline < card.demoEnd) {
        const progress = (timeline - card.demoStart) / (card.demoEnd - card.demoStart);
        setState(card, 'generating', Math.max(0, Math.min(1, progress)));
      } else {
        setState(card, 'ready');
      }
    });
    if (timeline < 153) raf = requestAnimationFrame(render);
    else raf = null;
  }

  function reset() {
    if (raf) cancelAnimationFrame(raf);
    startTime = null;
    cards.forEach(card => {
      const video = card.querySelector('video');
      video.pause();
      video.currentTime = 0;
      card.demoState = null;
      card.classList.remove('is-paused', 'is-ended');
    });
    raf = requestAnimationFrame(render);
    started = true;
  }

  cards.forEach(card => {
    const video = card.querySelector('video');
    const poster = new Image();
    poster.src = video.poster;
    poster.onload = () => {
      card.posterImage = poster;
      if (card.demoState === 'generating') paintNoise(card, 0);
    };
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

  replayButton.addEventListener('click', reset);
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
  observer.observe(root.querySelector('.generation-demo-stage'));
})();
