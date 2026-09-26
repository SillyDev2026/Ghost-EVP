'use strict';

const $ = id => document.getElementById(id);
const ui = {
  startBtn: $('startBtn'), stopBtn: $('stopBtn'), markTalkBtn: $('markTalkBtn'), markEventBtn: $('markEventBtn'),
  downloadAudioBtn: $('downloadAudioBtn'), exportBtn: $('exportBtn'), clearBtn: $('clearBtn'), statusPill: $('statusPill'),
  dbValue: $('dbValue'), levelBar: $('levelBar'), scoreValue: $('scoreValue'), scoreBar: $('scoreBar'),
  adaptiveThresholdValue: $('adaptiveThresholdValue'), thresholdDetail: $('thresholdDetail'), baselineValue: $('baselineValue'),
  calibrationText: $('calibrationText'), classification: $('classification'), classDetail: $('classDetail'),
  stabilityValue: $('stabilityValue'), stabilityDetail: $('stabilityDetail'), sampleRateLabel: $('sampleRateLabel'),
  transcript: $('transcript'), speechSupport: $('speechSupport'), threshold: $('threshold'), thresholdValue: $('thresholdValue'),
  sensitivity: $('sensitivity'), sensitivityValue: $('sensitivityValue'), adaptSpeed: $('adaptSpeed'), adaptSpeedValue: $('adaptSpeedValue'),
  speechGuard: $('speechGuard'), autoRecord: $('autoRecord'), autoAdjust: $('autoAdjust'), engineMode: $('engineMode'),
  events: $('events'), waveCanvas: $('waveCanvas'), spectrumCanvas: $('spectrumCanvas'), spectrogramCanvas: $('spectrogramCanvas'),
  speechRatioValue: $('speechRatioValue'), centroidValue: $('centroidValue'), flatnessValue: $('flatnessValue'),
  peakFactorValue: $('peakFactorValue'), baselineUpdatesValue: $('baselineUpdatesValue'), driftValue: $('driftValue')
};

const state = {
  stream: null, audioContext: null, source: null, analyser: null, timeData: null, freqData: null, raf: 0,
  sessionStart: 0, startupStart: 0, startupMs: 5000, startupSamples: [], calibrated: false,
  baselineRms: 0.004, baselineDb: -48, baselineCentroid: 1000, baselineFlatness: 0.3, initialBaselineDb: -48,
  noiseMeanDb: -48, noiseVariance: 2, noiseStdDb: 1.4, environmentDriftDb: 0, baselineUpdates: 0,
  currentAdaptiveThreshold: 68, lastEventAt: 0, eventCooldownMs: 1800, events: [], userTalkingUntil: 0,
  likelySpeech: false, mediaRecorder: null, chunks: [], audioBlob: null, recognition: null, recognitionWanted: false,
  lastMetrics: null, lastScore: 0, scoreHistory: [], dbHistory: [], previousRms: 0, previousCentroid: 0,
  spectrogramX: 0, alertTimer: null, lastBaselineUiAt: 0, startupFinishedEvent: false
};

const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const lerp = (a, b, t) => a + (b - a) * t;
const mean = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
const median = arr => {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const percentile = (arr, p) => {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.floor((s.length - 1) * p)))];
};
const nowSessionSeconds = () => state.sessionStart ? (performance.now() - state.sessionStart) / 1000 : 0;

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 1000);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

function setStatus(label, mode = 'idle') {
  ui.statusPill.className = `status-pill ${mode}`;
  ui.statusPill.querySelector('strong').textContent = label;
}

function dbFromRms(rms) { return rms > 0 ? 20 * Math.log10(rms) : -120; }
function rmsFromDb(db) { return Math.pow(10, db / 20); }

function getRmsAndPeak(data) {
  let sum = 0, peak = 0;
  for (let i = 0; i < data.length; i++) {
    const x = data[i];
    sum += x * x;
    peak = Math.max(peak, Math.abs(x));
  }
  const rms = Math.sqrt(sum / data.length);
  return { rms, peak };
}

function getBandEnergy(freq, lowHz, highHz, sampleRate, fftSize) {
  const binHz = sampleRate / fftSize;
  const start = clamp(Math.floor(lowHz / binHz), 0, freq.length - 1);
  const end = clamp(Math.ceil(highHz / binHz), start + 1, freq.length);
  let sum = 0;
  for (let i = start; i < end; i++) sum += freq[i] * freq[i];
  return sum / Math.max(1, end - start);
}

function spectralCentroid(freq, sampleRate, fftSize) {
  const binHz = sampleRate / fftSize;
  let weighted = 0, total = 0;
  for (let i = 1; i < freq.length; i++) {
    const mag = freq[i];
    weighted += i * binHz * mag;
    total += mag;
  }
  return total > 0 ? weighted / total : 0;
}

function spectralFlatness(freq) {
  let logSum = 0, linearSum = 0, n = 0;
  for (let i = 1; i < freq.length; i++) {
    const v = Math.max(freq[i], 1e-10);
    logSum += Math.log(v);
    linearSum += v;
    n++;
  }
  return n && linearSum ? Math.exp(logSum / n) / (linearSum / n) : 0;
}

function computeMetrics() {
  state.analyser.getFloatTimeDomainData(state.timeData);
  state.analyser.getFloatFrequencyData(state.freqData);
  const { rms, peak } = getRmsAndPeak(state.timeData);
  const db = dbFromRms(rms);
  const sr = state.audioContext.sampleRate;
  const fft = state.analyser.fftSize;
  const magnitudes = new Float32Array(state.freqData.length);
  for (let i = 0; i < state.freqData.length; i++) magnitudes[i] = Math.pow(10, state.freqData[i] / 20);

  const sub = getBandEnergy(magnitudes, 20, 80, sr, fft);
  const low = getBandEnergy(magnitudes, 80, 280, sr, fft);
  const speech = getBandEnergy(magnitudes, 280, 3400, sr, fft);
  const presence = getBandEnergy(magnitudes, 3400, 6500, sr, fft);
  const high = getBandEnergy(magnitudes, 6500, 12000, sr, fft);
  const total = sub + low + speech + presence + high + 1e-12;
  const speechRatio = speech / total;
  const lowRatio = (sub + low) / total;
  const highRatio = (presence + high) / total;
  const centroid = spectralCentroid(magnitudes, sr, fft);
  const flatness = spectralFlatness(magnitudes);
  const peakFactor = peak / Math.max(rms, 1e-8);
  const rmsDelta = Math.abs(rms - state.previousRms) / Math.max(state.baselineRms, 1e-6);
  const centroidDeltaFrame = Math.abs(centroid - state.previousCentroid) / Math.max(state.baselineCentroid, 250);
  state.previousRms = rms;
  state.previousCentroid = centroid;

  return { rms, db, peak, peakFactor, sub, low, speech, presence, high, speechRatio, lowRatio, highRatio, centroid, flatness, rmsDelta, centroidDeltaFrame, magnitudes };
}

function startupCalibration(m) {
  if (state.calibrated) return;
  const elapsed = performance.now() - state.startupStart;
  state.startupSamples.push({ rms: m.rms, db: m.db, centroid: m.centroid, flatness: m.flatness });
  const progress = clamp(elapsed / state.startupMs, 0, 1);
  ui.baselineValue.textContent = `Learning ${Math.round(progress * 100)}%`;
  ui.calibrationText.textContent = `${Math.max(0, Math.ceil((state.startupMs - elapsed) / 1000))}s startup learning • normal room sound is okay.`;

  if (elapsed < state.startupMs) return;
  const dbs = state.startupSamples.map(x => x.db).filter(Number.isFinite);
  const rms = state.startupSamples.map(x => x.rms);
  const centroids = state.startupSamples.map(x => x.centroid).filter(v => v > 0 && v < 20000);
  const flats = state.startupSamples.map(x => x.flatness);
  const quietDb = percentile(dbs, 0.55);
  state.baselineDb = clamp(quietDb, -100, -10);
  state.baselineRms = Math.max(0.00001, rmsFromDb(state.baselineDb));
  state.baselineCentroid = Math.max(100, median(centroids));
  state.baselineFlatness = clamp(median(flats), 0.01, 1);
  state.initialBaselineDb = state.baselineDb;
  state.noiseMeanDb = state.baselineDb;
  state.noiseVariance = Math.max(0.25, mean(dbs.map(v => Math.pow(v - state.baselineDb, 2))));
  state.noiseStdDb = Math.sqrt(state.noiseVariance);
  state.calibrated = true;
  updateBaselineUi(true);
  if (!state.startupFinishedEvent) {
    state.startupFinishedEvent = true;
    addEvent('Environment learned', 0, `Baseline ${state.baselineDb.toFixed(1)} dB • variability ${state.noiseStdDb.toFixed(1)} dB`, false);
  }
}

function detectSpeech(m) {
  const manual = performance.now() < state.userTalkingUntil;
  const above = m.db - state.baselineDb;
  const nearSpeechShape = m.speechRatio > 0.43 && m.peakFactor < 12 && above > 4;
  const strongSpeechShape = m.speechRatio > 0.55 && above > 1.5;
  state.likelySpeech = manual || nearSpeechShape || strongSpeechShape;
  return state.likelySpeech;
}

function calculateRawScore(m) {
  const baseRms = Math.max(state.baselineRms, 0.00001);
  const dbRise = m.db - state.baselineDb;
  const levelScore = clamp((dbRise - 2) / 18, 0, 1);
  const z = (m.db - state.noiseMeanDb) / Math.max(state.noiseStdDb, 0.75);
  const statisticalScore = clamp((z - 1.2) / 4.5, 0, 1);
  const centroidDelta = Math.abs(m.centroid - state.baselineCentroid) / Math.max(state.baselineCentroid, 350);
  const spectrumScore = clamp(centroidDelta / 1.2, 0, 1);
  const textureDelta = Math.abs(m.flatness - state.baselineFlatness);
  const textureScore = clamp(textureDelta / 0.42, 0, 1);
  const transientScore = clamp(Math.max(m.rmsDelta / 5, m.centroidDeltaFrame / 1.6), 0, 1);
  const peakScore = clamp((m.peakFactor - 3.3) / 8, 0, 1);
  const isolatedVoiceShape = clamp((m.speechRatio - 0.28) / 0.46, 0, 1) * clamp((dbRise - 1) / 14, 0, 1);
  const broadbandPenalty = clamp((m.flatness - 0.55) / 0.35, 0, 0.35);

  let score = (
    levelScore * 0.20 + statisticalScore * 0.22 + spectrumScore * 0.15 + textureScore * 0.10 +
    transientScore * 0.16 + peakScore * 0.07 + isolatedVoiceShape * 0.10
  ) * 100;
  score *= Number(ui.sensitivity.value) / 100;
  score *= 1 - broadbandPenalty;
  if (ui.speechGuard.checked && state.likelySpeech) score *= 0.24;
  return clamp(score, 0, 100);
}

function computeAdaptiveThreshold() {
  const base = Number(ui.threshold.value);
  if (!ui.autoAdjust.checked || !state.calibrated) return base;
  const volatilityPenalty = clamp((state.noiseStdDb - 1.2) * 2.1, 0, 14);
  const driftPenalty = clamp(Math.abs(state.environmentDriftDb) * 0.7, 0, 8);
  const threshold = clamp(base + volatilityPenalty + driftPenalty, 45, 96);
  state.currentAdaptiveThreshold = lerp(state.currentAdaptiveThreshold, threshold, 0.08);
  return state.currentAdaptiveThreshold;
}

function isSafeAmbientFrame(m, score) {
  if (!state.calibrated || !ui.autoAdjust.checked) return false;
  const threshold = computeAdaptiveThreshold();
  const manualTalking = performance.now() < state.userTalkingUntil;
  if (manualTalking || state.likelySpeech) return false;
  if (score > Math.min(threshold - 18, 48)) return false;
  if (Math.abs(m.db - state.baselineDb) > Math.max(8, state.noiseStdDb * 3.2)) return false;
  if (m.peakFactor > 10) return false;
  return true;
}

function adaptEnvironment(m, score) {
  if (!isSafeAmbientFrame(m, score)) return;
  const speed = Number(ui.adaptSpeed.value) / 100;
  const alpha = 0.0007 + speed * 0.0045;
  const statAlpha = 0.002 + speed * 0.008;

  state.baselineDb = lerp(state.baselineDb, m.db, alpha);
  state.baselineRms = rmsFromDb(state.baselineDb);
  state.baselineCentroid = lerp(state.baselineCentroid, m.centroid, alpha * 0.65);
  state.baselineFlatness = lerp(state.baselineFlatness, m.flatness, alpha * 0.6);

  const delta = m.db - state.noiseMeanDb;
  state.noiseMeanDb += statAlpha * delta;
  state.noiseVariance = (1 - statAlpha) * (state.noiseVariance + statAlpha * delta * delta);
  state.noiseStdDb = clamp(Math.sqrt(Math.max(0.16, state.noiseVariance)), 0.4, 12);
  state.environmentDriftDb = state.baselineDb - state.initialBaselineDb;
  state.baselineUpdates++;
  updateBaselineUi(false);
}

function updateBaselineUi(force) {
  const now = performance.now();
  if (!force && now - state.lastBaselineUiAt < 250) return;
  state.lastBaselineUiAt = now;
  ui.baselineValue.textContent = `${state.baselineDb.toFixed(1)} dB`;
  ui.calibrationText.textContent = ui.autoAdjust.checked
    ? `Adaptive • centroid ${Math.round(state.baselineCentroid)} Hz • learning ambient frames`
    : `Locked • centroid ${Math.round(state.baselineCentroid)} Hz`;
  ui.baselineUpdatesValue.textContent = state.baselineUpdates.toLocaleString();
  ui.driftValue.textContent = `${state.environmentDriftDb >= 0 ? '+' : ''}${state.environmentDriftDb.toFixed(1)} dB`;
}

function updateStabilityUi() {
  const s = state.noiseStdDb;
  let label = 'Stable', detail = 'Background is consistent.';
  if (s > 5) { label = 'Changing'; detail = 'Room noise varies significantly; auto threshold is raised.'; }
  else if (s > 2.7) { label = 'Moderate'; detail = 'Some environmental variation detected.'; }
  ui.stabilityValue.textContent = label;
  ui.stabilityDetail.textContent = `${detail} σ ${s.toFixed(1)} dB`;
}

function classify(m, score, threshold) {
  if (!state.calibrated) return ['Learning room', 'Fast startup calibration is building an initial baseline.'];
  if (performance.now() < state.userTalkingUntil) return ['User speech', 'Manual speech marker is active.'];
  if (state.likelySpeech) return ['Likely speech', 'Speech-band shape suggests nearby human speech.'];
  if (score >= threshold) return ['Audio anomaly', 'Signal differs strongly from the adaptive room model.'];
  if (m.db > state.baselineDb + Math.max(5, state.noiseStdDb * 2.2)) return ['Sound activity', 'Audio is elevated but below the anomaly trigger.'];
  return ['Ambient', ui.autoAdjust.checked ? 'Normal frames are being used to refine the room model.' : 'Signal is within the locked baseline range.'];
}

function addEvent(type, score, details, alert = false) {
  const event = {
    at: new Date().toISOString(), sessionTime: nowSessionSeconds(), type, score: Math.round(score), details,
    adaptiveThreshold: Number(state.currentAdaptiveThreshold.toFixed(2)), baselineDb: Number(state.baselineDb.toFixed(2)),
    metrics: state.lastMetrics ? {
      db: Number(state.lastMetrics.db.toFixed(2)), rms: Number(state.lastMetrics.rms.toFixed(6)), centroid: Math.round(state.lastMetrics.centroid),
      speechRatio: Number(state.lastMetrics.speechRatio.toFixed(3)), flatness: Number(state.lastMetrics.flatness.toFixed(3)),
      peakFactor: Number(state.lastMetrics.peakFactor.toFixed(2)), noiseStdDb: Number(state.noiseStdDb.toFixed(2))
    } : null
  };
  state.events.unshift(event);
  renderEvents();
  ui.exportBtn.disabled = false;
  if (alert && navigator.vibrate) navigator.vibrate([35, 30, 35]);
}

function renderEvents() {
  if (!state.events.length) {
    ui.events.innerHTML = '<div class="empty-state">No events yet.</div>';
    return;
  }
  ui.events.innerHTML = state.events.map(e => `
    <div class="event-row ${e.type === 'Audio anomaly' ? 'alert' : ''}">
      <span>${formatTime(e.sessionTime)}</span>
      <span class="tag">${escapeHtml(e.type)}</span>
      <span class="score">${e.score ? `${e.score}%` : '—'}</span>
      <span>${escapeHtml(e.details)}</span>
    </div>`).join('');
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
}

function maybeTrigger(m, score, threshold) {
  if (!state.calibrated) return;
  const now = performance.now();
  if (score < threshold || now - state.lastEventAt < state.eventCooldownMs) return;
  state.lastEventAt = now;
  const detail = `${m.db.toFixed(1)} dB • threshold ${threshold.toFixed(0)}% • centroid ${Math.round(m.centroid)} Hz • speech ${Math.round(m.speechRatio * 100)}% • σ ${state.noiseStdDb.toFixed(1)} dB`;
  addEvent('Audio anomaly', score, detail, true);
  setStatus('ANOMALY', 'alert');
  clearTimeout(state.alertTimer);
  state.alertTimer = setTimeout(() => { if (state.stream) setStatus('LIVE', 'live'); }, 850);
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    if (canvas === ui.spectrogramCanvas) state.spectrogramX = 0;
  }
  return { ratio, width, height };
}

function drawWaveform() {
  const { width, height } = resizeCanvas(ui.waveCanvas);
  const ctx = ui.waveCanvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = 'rgba(99,230,190,.95)';
  ctx.lineWidth = Math.max(1, window.devicePixelRatio || 1);
  ctx.beginPath();
  const step = width / Math.max(1, state.timeData.length - 1);
  for (let i = 0; i < state.timeData.length; i++) {
    const x = i * step;
    const y = (0.5 - state.timeData[i] * 0.42) * height;
    if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.strokeStyle = 'rgba(255,255,255,.08)';
  ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke();
}

function drawSpectrum(m) {
  const { width, height } = resizeCanvas(ui.spectrumCanvas);
  const ctx = ui.spectrumCanvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  const bins = 180;
  const nyquist = state.audioContext.sampleRate / 2;
  const maxHz = Math.min(20000, nyquist);
  const maxBin = Math.floor((maxHz / nyquist) * m.magnitudes.length);
  const stride = Math.max(1, Math.floor(maxBin / bins));
  const barW = width / bins;
  for (let i = 0; i < bins; i++) {
    let peak = 0;
    const start = i * stride;
    for (let j = start; j < Math.min(start + stride, m.magnitudes.length); j++) peak = Math.max(peak, m.magnitudes[j]);
    const normalized = clamp((20 * Math.log10(Math.max(peak, 1e-9)) + 100) / 100, 0, 1);
    const h = normalized * height;
    const grad = ctx.createLinearGradient(0, height, 0, 0);
    grad.addColorStop(0, 'rgba(110,168,254,.22)');
    grad.addColorStop(1, 'rgba(99,230,190,.95)');
    ctx.fillStyle = grad;
    ctx.fillRect(i * barW, height - h, Math.max(1, barW - 1), h);
  }
}

function drawSpectrogram() {
  const { width, height } = resizeCanvas(ui.spectrogramCanvas);
  const ctx = ui.spectrogramCanvas.getContext('2d');
  if (state.spectrogramX === 0) {
    ctx.fillStyle = '#070b12';
    ctx.fillRect(0, 0, width, height);
  }
  const x = state.spectrogramX % width;
  const bins = state.freqData.length;
  const maxHz = Math.min(12000, state.audioContext.sampleRate / 2);
  const maxBin = Math.max(1, Math.floor(maxHz / (state.audioContext.sampleRate / 2) * bins));
  for (let y = 0; y < height; y += 2) {
    const norm = 1 - y / height;
    const bin = Math.min(maxBin - 1, Math.floor(norm * maxBin));
    const db = state.freqData[bin];
    const intensity = clamp((db + 100) / 85, 0, 1);
    const r = Math.round(20 + intensity * 85);
    const g = Math.round(30 + intensity * 205);
    const b = Math.round(55 + intensity * 155);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, y, 2, 2);
  }
  state.spectrogramX = (x + 2) % width;
}

function updateLiveUi(m, score, threshold) {
  const dbShown = Number.isFinite(m.db) ? `${m.db.toFixed(1)} dB` : '-∞ dB';
  ui.dbValue.textContent = dbShown;
  ui.levelBar.style.width = `${clamp((m.db + 80) / 70 * 100, 0, 100)}%`;
  ui.scoreValue.textContent = `${Math.round(score)}%`;
  ui.scoreBar.style.width = `${score}%`;
  ui.adaptiveThresholdValue.textContent = `${Math.round(threshold)}%`;
  ui.thresholdDetail.textContent = ui.autoAdjust.checked ? `Base ${ui.threshold.value}% + environment compensation` : 'Using fixed manual threshold.';
  ui.engineMode.textContent = ui.autoAdjust.checked ? 'AUTO' : 'MANUAL';
  ui.speechRatioValue.textContent = `${Math.round(m.speechRatio * 100)}%`;
  ui.centroidValue.textContent = `${Math.round(m.centroid)} Hz`;
  ui.flatnessValue.textContent = m.flatness.toFixed(2);
  ui.peakFactorValue.textContent = `${m.peakFactor.toFixed(2)}×`;
  updateStabilityUi();
}

function renderLoop() {
  if (!state.analyser) return;
  const m = computeMetrics();
  state.lastMetrics = m;
  startupCalibration(m);
  detectSpeech(m);
  const score = state.calibrated ? calculateRawScore(m) : 0;
  const threshold = computeAdaptiveThreshold();
  state.lastScore = score;
  state.scoreHistory.push(score); if (state.scoreHistory.length > 600) state.scoreHistory.shift();
  state.dbHistory.push(m.db); if (state.dbHistory.length > 600) state.dbHistory.shift();
  if (state.calibrated) adaptEnvironment(m, score);
  const [label, detail] = classify(m, score, threshold);
  ui.classification.textContent = label;
  ui.classDetail.textContent = detail;
  updateLiveUi(m, score, threshold);
  drawWaveform();
  drawSpectrum(m);
  drawSpectrogram();
  maybeTrigger(m, score, threshold);
  state.raf = requestAnimationFrame(renderLoop);
}

function setupRecorder(stream) {
  state.chunks = [];
  state.audioBlob = null;
  ui.downloadAudioBtn.disabled = true;
  if (!ui.autoRecord.checked || !window.MediaRecorder) return;
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
  const supported = candidates.find(type => MediaRecorder.isTypeSupported?.(type));
  const options = supported ? { mimeType: supported } : {};
  try {
    state.mediaRecorder = new MediaRecorder(stream, options);
    state.mediaRecorder.ondataavailable = e => { if (e.data?.size) state.chunks.push(e.data); };
    state.mediaRecorder.onstop = () => {
      if (!state.chunks.length) return;
      state.audioBlob = new Blob(state.chunks, { type: state.mediaRecorder.mimeType || 'audio/webm' });
      ui.downloadAudioBtn.disabled = false;
    };
    state.mediaRecorder.start(1000);
  } catch (err) {
    console.warn('MediaRecorder unavailable:', err);
    state.mediaRecorder = null;
  }
}

function setupSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { ui.speechSupport.textContent = 'not supported'; return; }
  ui.speechSupport.textContent = 'available';
  state.recognitionWanted = true;
  const rec = new SR();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = navigator.language || 'en-US';
  rec.onresult = event => {
    let interim = '';
    const finals = [];
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const text = event.results[i][0].transcript.trim();
      if (event.results[i].isFinal) finals.push(text); else interim += `${text} `;
    }
    if (finals.length) {
      state.userTalkingUntil = Math.max(state.userTalkingUntil, performance.now() + 900);
      const stamp = formatTime(nowSessionSeconds());
      ui.transcript.textContent = `[${stamp}] ${finals.join(' ')}\n${ui.transcript.textContent}`.slice(0, 12000);
      addEvent('Speech', 0, finals.join(' ').slice(0, 200), false);
    } else if (interim) {
      ui.speechSupport.textContent = `hearing: ${interim.trim().slice(0, 35)}`;
    }
  };
  rec.onend = () => {
    ui.speechSupport.textContent = 'available';
    if (state.recognitionWanted && state.stream) { try { rec.start(); } catch (_) {} }
  };
  rec.onerror = () => { ui.speechSupport.textContent = 'speech unavailable'; };
  state.recognition = rec;
  try { rec.start(); } catch (_) {}
}

async function startSession() {
  if (state.stream) return;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 }, video: false
    });
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
    await state.audioContext.resume();
    state.source = state.audioContext.createMediaStreamSource(state.stream);
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 4096;
    state.analyser.smoothingTimeConstant = 0.35;
    state.source.connect(state.analyser);
    state.timeData = new Float32Array(state.analyser.fftSize);
    state.freqData = new Float32Array(state.analyser.frequencyBinCount);

    Object.assign(state, {
      sessionStart: performance.now(), startupStart: performance.now(), startupSamples: [], calibrated: false,
      baselineUpdates: 0, lastEventAt: 0, events: [], scoreHistory: [], dbHistory: [], previousRms: 0,
      previousCentroid: 0, spectrogramX: 0, startupFinishedEvent: false, currentAdaptiveThreshold: Number(ui.threshold.value)
    });
    renderEvents();
    ui.sampleRateLabel.textContent = `${Math.round(state.audioContext.sampleRate / 1000)} kHz • ${state.analyser.fftSize}-point FFT`;
    ui.startBtn.disabled = true; ui.stopBtn.disabled = false; ui.markTalkBtn.disabled = false; ui.markEventBtn.disabled = false;
    ui.exportBtn.disabled = true;
    ui.transcript.textContent = 'Listening… you can talk normally. “I\'m Talking” forces a 5-second speech exclusion marker.';
    setStatus('LIVE', 'live');
    setupRecorder(state.stream);
    setupSpeechRecognition();
    renderLoop();
  } catch (err) {
    console.error(err);
    setStatus('MIC BLOCKED', 'alert');
    ui.classification.textContent = 'Microphone unavailable';
    ui.classDetail.textContent = 'Allow microphone access and serve the page over HTTPS (GitHub Pages works).';
  }
}

async function stopSession() {
  cancelAnimationFrame(state.raf); state.raf = 0;
  state.recognitionWanted = false;
  if (state.recognition) { try { state.recognition.stop(); } catch (_) {} }
  state.recognition = null;
  if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') state.mediaRecorder.stop();
  state.mediaRecorder = null;
  state.stream?.getTracks().forEach(t => t.stop()); state.stream = null;
  if (state.audioContext) { try { await state.audioContext.close(); } catch (_) {} }
  state.audioContext = null; state.analyser = null;
  ui.startBtn.disabled = false; ui.stopBtn.disabled = true; ui.markTalkBtn.disabled = true; ui.markEventBtn.disabled = true;
  ui.exportBtn.disabled = state.events.length === 0;
  setStatus('STOPPED', 'idle');
  ui.classification.textContent = 'Session stopped';
  ui.classDetail.textContent = 'Review the timeline, download audio, or export the evidence log.';
}

function markTalking() {
  state.userTalkingUntil = performance.now() + 5000;
  addEvent('User speech', 0, 'Manual 5-second speech exclusion marker.', false);
  ui.markTalkBtn.textContent = 'Talking marked ✓';
  setTimeout(() => { ui.markTalkBtn.textContent = "I'm Talking"; }, 900);
}

function markManualEvent() {
  addEvent('Manual marker', state.lastScore, 'Investigator manually marked this moment.', false);
}

function downloadAudio() {
  if (!state.audioBlob) return;
  const ext = state.audioBlob.type.includes('mp4') ? 'm4a' : state.audioBlob.type.includes('ogg') ? 'ogg' : 'webm';
  downloadBlob(state.audioBlob, `evp-session-${Date.now()}.${ext}`);
}

function exportLog() {
  const payload = {
    app: 'EVP Field Lab', version: '2.0.0', exportedAt: new Date().toISOString(),
    note: 'Scores indicate deviation from the learned audio environment, not evidence that a paranormal entity caused the signal.',
    baseline: {
      db: state.baselineDb, rms: state.baselineRms, centroidHz: state.baselineCentroid, flatness: state.baselineFlatness,
      noiseStdDb: state.noiseStdDb, environmentDriftDb: state.environmentDriftDb, updateCount: state.baselineUpdates, calibrated: state.calibrated
    },
    settings: {
      autoAdjust: ui.autoAdjust.checked, baseTrigger: Number(ui.threshold.value), currentAdaptiveThreshold: state.currentAdaptiveThreshold,
      sensitivityPercent: Number(ui.sensitivity.value), adaptationSpeedPercent: Number(ui.adaptSpeed.value), speechGuard: ui.speechGuard.checked
    },
    events: state.events.slice().reverse()
  };
  downloadBlob(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }), `evp-evidence-${Date.now()}.json`);
}

function downloadBlob(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = filename; document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

ui.startBtn.addEventListener('click', startSession);
ui.stopBtn.addEventListener('click', stopSession);
ui.markTalkBtn.addEventListener('click', markTalking);
ui.markEventBtn.addEventListener('click', markManualEvent);
ui.downloadAudioBtn.addEventListener('click', downloadAudio);
ui.exportBtn.addEventListener('click', exportLog);
ui.clearBtn.addEventListener('click', () => { state.events = []; renderEvents(); ui.exportBtn.disabled = true; });
ui.threshold.addEventListener('input', () => { ui.thresholdValue.textContent = `${ui.threshold.value}%`; });
ui.sensitivity.addEventListener('input', () => { ui.sensitivityValue.textContent = `${ui.sensitivity.value}%`; });
ui.adaptSpeed.addEventListener('input', () => { ui.adaptSpeedValue.textContent = `${ui.adaptSpeed.value}%`; });
ui.autoAdjust.addEventListener('change', () => { ui.engineMode.textContent = ui.autoAdjust.checked ? 'AUTO' : 'MANUAL'; updateBaselineUi(true); });
window.addEventListener('beforeunload', () => state.stream?.getTracks().forEach(t => t.stop()));

if (!navigator.mediaDevices?.getUserMedia) {
  ui.startBtn.disabled = true;
  ui.classification.textContent = 'Unsupported browser';
  ui.classDetail.textContent = 'This browser does not expose microphone capture to web pages.';
}
ui.speechSupport.textContent = (window.SpeechRecognition || window.webkitSpeechRecognition) ? 'available' : 'not supported';
