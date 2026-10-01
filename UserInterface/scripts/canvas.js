/* ============================================================
   canvas.js — Background particle mesh + channel waveform
   Light-theme variant: soft pastel orbs + signal animation
   ============================================================ */

// ---- Background Orb Canvas ----------------------------------------
(function initBgCanvas() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W, H, orbs = [], raf;

  // Soft pastel colors for light theme
  const PALETTE = [
    'rgba(79,110,247,0.06)',
    'rgba(139,92,246,0.06)',
    'rgba(16,185,129,0.05)',
    'rgba(245,158,11,0.05)',
    'rgba(6,182,212,0.05)',
  ];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function makeOrb() {
    return {
      x:    Math.random() * W,
      y:    Math.random() * H,
      r:    160 + Math.random() * 200,
      vx:   (Math.random() - 0.5) * 0.3,
      vy:   (Math.random() - 0.5) * 0.3,
      color: PALETTE[Math.floor(Math.random() * PALETTE.length)],
      phase: Math.random() * Math.PI * 2,
    };
  }

  function init() {
    resize();
    orbs = Array.from({ length: 6 }, makeOrb);
  }

  function draw(ts) {
    ctx.clearRect(0, 0, W, H);

    orbs.forEach(o => {
      o.phase += 0.004;
      o.x += o.vx + Math.sin(o.phase) * 0.2;
      o.y += o.vy + Math.cos(o.phase * 0.7) * 0.2;

      // Bounce
      if (o.x < -o.r) o.x = W + o.r;
      if (o.x > W + o.r) o.x = -o.r;
      if (o.y < -o.r) o.y = H + o.r;
      if (o.y > H + o.r) o.y = -o.r;

      const g = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, o.r);
      g.addColorStop(0, o.color);
      g.addColorStop(1, 'transparent');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2);
      ctx.fill();
    });

    raf = requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  init();
  raf = requestAnimationFrame(draw);
})();


// ---- Channel Signal Canvas ----------------------------------------
(function initChannelCanvas() {
  let snrDb = 10;
  let animPhase = 0;
  let rafId;

  const canvas = document.getElementById('channel-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function getNoiseAmplitude(snr) {
    // Higher SNR → less noise amplitude visually
    const linear = Math.pow(10, snr / 10);
    return Math.max(2, 28 - Math.sqrt(linear) * 1.2);
  }

  function draw() {
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    canvas.width  = W * devicePixelRatio;
    canvas.height = H * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    ctx.clearRect(0, 0, W, H);

    const noiseAmp = getNoiseAmplitude(snrDb);
    const cy       = H / 2;
    const pts      = 80;

    // Grid lines
    ctx.strokeStyle = 'rgba(245,158,11,0.08)';
    ctx.lineWidth   = 1;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (H / 4) * i);
      ctx.lineTo(W, (H / 4) * i);
      ctx.stroke();
    }

    // Clean signal (thin, faded blue)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(79,110,247,0.25)';
    ctx.lineWidth   = 1.2;
    for (let i = 0; i <= pts; i++) {
      const x = (i / pts) * W;
      const y = cy + Math.sin((i / pts) * Math.PI * 6 - animPhase) * 10;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Noisy signal (main, amber gradient)
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0,   'rgba(245,158,11,0.7)');
    grad.addColorStop(0.5, 'rgba(249,115,22,0.9)');
    grad.addColorStop(1,   'rgba(245,158,11,0.7)');

    ctx.beginPath();
    ctx.strokeStyle = grad;
    ctx.lineWidth   = 1.8;
    for (let i = 0; i <= pts; i++) {
      const x    = (i / pts) * W;
      const base = Math.sin((i / pts) * Math.PI * 6 - animPhase) * 10;
      const noise = (Math.random() - 0.5) * noiseAmp;
      const y    = cy + base + noise;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // SNR label
    ctx.fillStyle = 'rgba(180,130,0,0.55)';
    ctx.font = `600 9px 'JetBrains Mono', monospace`;
    ctx.fillText(`SNR: ${snrDb} dB`, 8, H - 6);

    animPhase += 0.06;
    rafId = requestAnimationFrame(draw);
  }

  function start() { if (!rafId) rafId = requestAnimationFrame(draw); }
  function stop()  { if (rafId)  { cancelAnimationFrame(rafId); rafId = null; } }

  // Watch intersection (pause when off-screen)
  if ('IntersectionObserver' in window) {
    const obs = new IntersectionObserver(entries => {
      entries[0].isIntersecting ? start() : stop();
    }, { threshold: 0.1 });
    obs.observe(canvas);
  } else {
    start();
  }

  // Expose SNR update hook
  window.updateChannelCanvas = function(snr) { snrDb = snr; };
})();
