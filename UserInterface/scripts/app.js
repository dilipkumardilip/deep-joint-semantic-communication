/* ============================================================
   app.js — Main Application Logic
   Deep JSCC Web Interface
   ============================================================ */

// ---- State ----------------------------------------------------------
const State = {
  inputB64:     null,   // current loaded image base64
  vectorData:   null,   // last encoded flat vector
  activeModel:  null,   // active model metadata object
  encoding:     false,
  decoding:     false,
};

// ---- Toast ----------------------------------------------------------
let _toastTimer;
function toast(msg, type = 'info') {
  const el   = document.getElementById('toast');
  const icon = { info: 'ℹ️', success: '✅', error: '❌', warn: '⚠️' }[type] || 'ℹ️';
  el.innerHTML = `<div class="toast-inner"><span class="toast-icon">${icon}</span><span>${msg}</span></div>`;
  el.style.display = 'block';
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.style.display = 'none'; }, 3000);
}

// ---- Model Registry -------------------------------------------------
async function loadModels() {
  try {
    const res  = await fetch('/api/models');
    const data = await res.json();
    renderModelCards(data.models);
    const active = data.models.find(m => m.active);
    if (active) applyActiveModel(active, false);
  } catch (e) {
    toast('Could not load model registry', 'error');
  }
}

function renderModelCards(models) {
  const grid = document.getElementById('models-grid');
  grid.innerHTML = '';

  models.forEach(m => {
    const card = document.createElement('div');
    card.className = 'model-card' +
      (m.active      ? ' active'      : '') +
      (!m.available  ? ' unavailable' : '');
    card.id = `mc-${m.key}`;
    card.setAttribute('role', 'button');
    card.setAttribute('aria-pressed', m.active ? 'true' : 'false');
    card.onclick = () => selectModel(m.key, m.available);

    const datasetBadge = m.dataset === 'CIFAR-10'
      ? '<span class="badge badge-blue">CIFAR-10</span>'
      : m.dataset === 'DIV2K'
      ? '<span class="badge badge-purple">DIV2K HD</span>'
      : `<span class="badge badge-dim">${m.dataset}</span>`;

    const availBadge = m.available
      ? '<span class="badge badge-green">✓ Trained</span>'
      : '<span class="badge badge-dim">⚠ No Checkpoint</span>';

    const k = m.symbols_k.toLocaleString();

    card.innerHTML = `
      <div class="model-check">✓</div>
      <div class="model-card-top">
        <div class="model-card-name">${m.label}</div>
        <div class="model-card-badges">${datasetBadge}${availBadge}</div>
      </div>
      <div class="model-card-desc">${m.description}</div>
      <div class="model-card-meta">
        <span class="meta-chip">📐 ${m.patch_size}×${m.patch_size}px</span>
        <span class="meta-chip">🔢 c = ${m.channel_c}</span>
        <span class="meta-chip">📡 k = ${k}</span>
        <span class="meta-chip">📶 ${m.snr_db} dB</span>
      </div>`;
    grid.appendChild(card);
  });
}

async function selectModel(key, available) {
  if (!available) { toast('Model checkpoint not found — train it first', 'warn'); return; }
  if (key === State.activeModel?.key) return;

  try {
    const res  = await fetch('/api/select_model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key }),
    });
    const data = await res.json();
    if (data.error) { toast(data.error, 'error'); return; }
    renderModelCards(data.models);
    const active = data.models.find(m => m.active);
    if (active) applyActiveModel(active, true);
  } catch (e) {
    toast('Failed to switch model: ' + e.message, 'error');
  }
}

function applyActiveModel(meta, notify) {
  State.activeModel = meta;

  // Header pill
  document.getElementById('header-model-label').textContent = meta.label;

  // Encoder subtitle
  document.getElementById('enc-sub').textContent =
    `Encode image → semantic channel symbols (${meta.badge} · c=${meta.channel_c})`;
  document.getElementById('dropzone-hint').textContent =
    `PNG, JPEG, WebP — auto-resized to ${meta.patch_size}×${meta.patch_size} px`;
  document.getElementById('enc-tensor-info').textContent =
    `Tensor: (1, 3, ${meta.patch_size}, ${meta.patch_size}) · float32 · [0, 1]`;
  document.getElementById('fmap-label').textContent =
    `${meta.channel_c} Feature Channels (latent z)`;

  // Reset pipeline on model change
  if (notify) {
    resetPipeline();
    toast(`Switched to: ${meta.label}`, 'success');
  }
}

function resetPipeline() {
  State.inputB64   = null;
  State.vectorData = null;
  hide('input-preview-box');
  hide('vector-results');
  hide('decoder-results');
  document.getElementById('btn-encode').disabled = true;
  document.getElementById('btn-transmit').disabled = true;
  document.getElementById('btn-transmit-clean').disabled = true;
  document.getElementById('decoder-input-box').value = '';
}

// ---- Image Input ----------------------------------------------------
function setupDropzone() {
  const zone = document.getElementById('dropzone');
  if (!zone) return;

  ['dragenter', 'dragover'].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('drag-over'); })
  );
  ['dragleave', 'drop'].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('drag-over'); })
  );
  zone.addEventListener('drop', e => {
    const f = e.dataTransfer.files?.[0];
    if (f) processFile(f);
  });
}

function handleFileSelect(e) {
  const f = e.target.files?.[0];
  if (f) processFile(f);
}

function processFile(file) {
  const reader = new FileReader();
  reader.onload = ev => setImageSource(ev.target.result, file.name);
  reader.readAsDataURL(file);
}

function setImageSource(dataUrl, title = 'Uploaded Image') {
  State.inputB64 = dataUrl;

  const patch = State.activeModel?.patch_size || 32;
  show('input-preview-box');
  document.getElementById('input-img-preview').src = dataUrl;
  document.getElementById('input-img-title').textContent = title;
  document.getElementById('enc-tensor-info').textContent =
    `Tensor: (1, 3, ${patch}, ${patch}) · float32 · [0, 1]`;
  document.getElementById('decoded-original-img').src = dataUrl;
  document.getElementById('btn-encode').disabled = false;

  toast('Image loaded — ready to encode', 'success');
}

async function loadSampleImage() {
  try {
    const btn = document.getElementById('btn-sample');
    btn.textContent = 'Loading...'; btn.disabled = true;
    const res  = await fetch('/api/sample');
    const data = await res.json();
    if (data.image) setImageSource(data.image, 'Sample Image');
  } catch (e) {
    toast('Error loading sample: ' + e.message, 'error');
  } finally {
    const btn = document.getElementById('btn-sample');
    btn.textContent = '🎲 Load Sample Image'; btn.disabled = false;
  }
}

// ---- Encode ---------------------------------------------------------
async function encodeImage() {
  if (!State.inputB64 || State.encoding) return;
  State.encoding = true;

  const btn = document.getElementById('btn-encode');
  btn.textContent = 'Encoding…'; btn.disabled = true;

  try {
    const res  = await fetch('/api/encode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: State.inputB64 }),
    });
    const data = await res.json();
    if (data.error) { toast('Encoding failed: ' + data.error, 'error'); return; }

    State.vectorData = data.vector_flat;

    // Stats
    document.getElementById('stat-shape').textContent  = `[${data.shape.join(', ')}]`;
    document.getElementById('stat-k').textContent      = data.num_symbols.toLocaleString();
    document.getElementById('stat-power').textContent  = data.avg_power.toFixed(4);
    document.getElementById('stat-range').textContent  = `[${data.min_val}, ${data.max_val}]`;

    // Feature maps
    const fmapGrid = document.getElementById('fmap-grid');
    fmapGrid.innerHTML = '';
    data.feature_maps.forEach((b64, i) => {
      const cell = document.createElement('div');
      cell.className = 'fmap-cell';
      cell.title     = `Channel ${i + 1} / ${data.feature_maps.length}`;
      cell.onclick   = () => openFmapLightbox(b64, i + 1, data.feature_maps.length);
      const img = document.createElement('img');
      img.src = b64;
      cell.appendChild(img);
      fmapGrid.appendChild(cell);
    });

    // Vector box
    document.getElementById('vector-box').value      = JSON.stringify(data.vector_flat);
    document.getElementById('vector-len').textContent = `${data.vector_flat.length.toLocaleString()} floats`;

    show('vector-results');
    document.getElementById('btn-transmit').disabled      = false;
    document.getElementById('btn-transmit-clean').disabled = false;

    toast('Encoding complete — vector generated', 'success');
  } catch (e) {
    toast('Encode error: ' + e.message, 'error');
  } finally {
    btn.textContent = '⚡ Encode to Channel Symbols';
    btn.disabled = false;
    State.encoding = false;
  }
}

// ---- SNR Slider -----------------------------------------------------
function updateSnr(val) {
  document.getElementById('snr-display').textContent = parseFloat(val).toFixed(1) + ' dB';
  // Update CSS gradient for range fill
  const slider = document.getElementById('snr-slider');
  const min  = parseFloat(slider.min), max = parseFloat(slider.max);
  const pct  = ((val - min) / (max - min)) * 100;
  slider.style.setProperty('--pct', pct + '%');
  // Update channel canvas
  if (window.updateChannelCanvas) window.updateChannelCanvas(parseFloat(val));
}

function setSnr(val) {
  document.getElementById('snr-slider').value = val;
  updateSnr(val);
}

// ---- Transmit -------------------------------------------------------
async function transmit(noiseless = false) {
  if (!State.vectorData) { toast('Please encode an image first', 'warn'); return; }
  const snr = parseFloat(document.getElementById('snr-slider').value);

  if (noiseless) {
    document.getElementById('decoder-input-box').value = JSON.stringify(State.vectorData);
    toast('Vector passed noiseless → decoding…', 'info');
    await decodeVector();
    return;
  }

  try {
    toast(`Simulating AWGN @ SNR = ${snr} dB…`, 'info');
    const res  = await fetch('/api/channel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vector: State.vectorData, snr_db: snr, noiseless: false }),
    });
    const data = await res.json();
    if (data.error) { toast(data.error, 'error'); return; }
    document.getElementById('decoder-input-box').value = JSON.stringify(data.noisy_vector_flat);
    toast('Noisy vector received → decoding…', 'info');
    await decodeVector();
  } catch (e) {
    toast('Channel error: ' + e.message, 'error');
  }
}

// ---- Decode ---------------------------------------------------------
async function decodeVector() {
  if (State.decoding) return;
  const raw = document.getElementById('decoder-input-box').value.trim();
  if (!raw) { toast('Paste or transfer a vector first', 'warn'); return; }

  let parsed;
  try { parsed = JSON.parse(raw); }
  catch { toast('Invalid JSON vector', 'error'); return; }

  State.decoding = true;
  const btn = document.getElementById('btn-decode');
  btn.textContent = 'Decoding…'; btn.disabled = true;

  try {
    const res  = await fetch('/api/decode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vector: parsed, original_image: State.inputB64 }),
    });
    const data = await res.json();
    if (data.error) { toast('Decode error: ' + data.error, 'error'); return; }

    document.getElementById('decoded-recon-img').src = data.reconstructed_image;
    document.getElementById('btn-download-img').href = data.reconstructed_image;

    const m = data.metrics;
    if (m?.psnr !== undefined) {
      document.getElementById('stat-psnr').textContent = m.psnr.toFixed(2) + ' dB';
      document.getElementById('stat-mse').textContent  = m.mse.toFixed(5);
      // Color PSNR by quality
      const psnrEl = document.getElementById('stat-psnr');
      psnrEl.className = 'stat-val ' + (m.psnr >= 25 ? 'c-green' : m.psnr >= 15 ? 'c-amber' : 'c-rose');
    } else {
      document.getElementById('stat-psnr').textContent = 'N/A';
      document.getElementById('stat-mse').textContent  = 'N/A';
    }

    show('decoder-results');
    toast('Image reconstructed successfully!', 'success');
  } catch (e) {
    toast('Decode error: ' + e.message, 'error');
  } finally {
    btn.textContent = '✨ Decode to Image'; btn.disabled = false;
    State.decoding = false;
  }
}

// ---- Clipboard / Download -------------------------------------------
function copyVector() {
  const val = document.getElementById('vector-box').value;
  if (!val) return;
  navigator.clipboard.writeText(val).then(() => toast('Vector copied to clipboard', 'success'));
}

function downloadVector() {
  const val  = document.getElementById('vector-box').value;
  if (!val) return;
  const blob = new Blob([val], { type: 'application/json' });
  const a    = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(blob), download: 'semantic_vector.json'
  });
  a.click(); URL.revokeObjectURL(a.href);
}

function handleVectorFile(e) {
  const f = e.target.files?.[0];
  if (!f) return;
  const r = new FileReader();
  r.onload = ev => {
    document.getElementById('decoder-input-box').value = ev.target.result;
    toast('Vector loaded from file', 'success');
  };
  r.readAsText(f);
}

// ---- Feature Map Lightbox -------------------------------------------
function openFmapLightbox(imgSrc, chIdx, total) {
  const lb = document.getElementById('fmap-lightbox');
  document.getElementById('fmap-lb-title').textContent = `Feature Map — Channel ${chIdx} / ${total}`;
  document.getElementById('fmap-lb-img').src = imgSrc;
  lb.classList.add('open');
}
function closeFmapLightbox() {
  document.getElementById('fmap-lightbox').classList.remove('open');
}

// ---- Compare Lightbox (iframe) --------------------------------------
function openCompareLightbox() {
  if (!State.inputB64) { toast('No image loaded yet', 'warn'); return; }
  const lb = document.getElementById('compare-lightbox');
  const iframe = document.getElementById('compare-iframe');

  // Build comparison page URL with params
  const snr = document.getElementById('snr-slider').value;
  iframe.src = `/compare?snr=${snr}`;
  lb.classList.add('open');
}
function closeCompareLightbox() {
  const lb     = document.getElementById('compare-lightbox');
  const iframe = document.getElementById('compare-iframe');
  lb.classList.remove('open');
  iframe.src = 'about:blank';
}

// Close lightboxes on backdrop click
document.addEventListener('click', e => {
  if (e.target.id === 'fmap-lightbox')    closeFmapLightbox();
  if (e.target.id === 'compare-lightbox') closeCompareLightbox();
});

// Keyboard escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closeFmapLightbox(); closeCompareLightbox(); }
});

// ---- Utility --------------------------------------------------------
function show(id) {
  const el = document.getElementById(id);
  if (el) { el.style.display = 'flex'; el.style.flexDirection = 'column'; }
}
function hide(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = 'none';
}

// ---- Init -----------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  setupDropzone();
  loadModels();
  updateSnr(10);

  // Animate flow steps
  document.querySelectorAll('.flow-step').forEach((s, i) => {
    setTimeout(() => s.classList.add('lit'), i * 120);
  });
});
