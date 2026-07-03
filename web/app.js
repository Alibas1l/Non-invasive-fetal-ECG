import * as ort from "./public/ort/ort.wasm.min.mjs";
import { WINDOW_LEN, FS, parseCsv, findPeaks, estimateFhr, runInferenceWindows } from "./lib.js";

// onnxruntime-web resolves its .wasm binaries relative to this imported .mjs
// file's own URL, which is already ./public/ort/ -- no wasmPaths override needed.

const MODEL_PATH = "./public/model_int8.onnx";

const COLOR_SIGNAL = "#6b7280";
const COLOR_WAVE = "#dc2626";
const COLOR_QRS = "#2563eb";

const els = {
  input: document.getElementById("csv-input"),
  status: document.getElementById("status"),
  results: document.getElementById("results"),
  fhr: document.getElementById("fhr-value"),
  beats: document.getElementById("beats-value"),
  time: document.getElementById("time-value"),
  chartInput: document.getElementById("chart-input"),
  chartWave: document.getElementById("chart-wave"),
  chartQrs: document.getElementById("chart-qrs"),
};

let session = null;

function setStatus(msg, isError = false) {
  els.status.textContent = msg;
  els.status.classList.toggle("error", isError);
}

async function getSession() {
  if (!session) {
    setStatus("Loading model (first time only)…");
    session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ["wasm"] });
  }
  return session;
}

async function runWindow(inputData) {
  const sess = await getSession();
  const inputTensor = new ort.Tensor("float32", inputData, [1, 3, WINDOW_LEN]);
  const outputs = await sess.run({ input: inputTensor });
  return { wave: outputs.wave.data, qrs: outputs.qrs.data };
}

function drawLine(canvas, series, color, { fixedRange } = {}) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  for (let gy = 0; gy <= 4; gy++) {
    const y = (h / 4) * gy;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  const n = series.length;
  let min = fixedRange ? fixedRange[0] : Math.min(...series);
  let max = fixedRange ? fixedRange[1] : Math.max(...series);
  if (max - min < 1e-6) max = min + 1;

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * w;
    const y = h - ((series[i] - min) / (max - min)) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

async function handleFile(file) {
  try {
    setStatus("Reading CSV…");
    const text = await file.text();
    const channels = parseCsv(text);
    const originalLen = channels[0].length;
    if (originalLen < WINDOW_LEN) {
      setStatus(`CSV too short: need at least ${WINDOW_LEN} samples (2s at ${FS}Hz), got ${originalLen}.`, true);
      return;
    }

    setStatus(`Running inference on ${originalLen.toLocaleString()} samples…`);
    const t0 = performance.now();
    const { wave, qrs } = await runInferenceWindows(channels, originalLen, runWindow);
    const dt = performance.now() - t0;

    const peaks = findPeaks(qrs, 0.5, 0.25 * FS);
    const fhr = estimateFhr(peaks);

    els.fhr.textContent = fhr ? `${fhr.toFixed(0)} bpm` : "n/a (too few beats)";
    els.beats.textContent = peaks.length.toString();
    els.time.textContent = `${dt.toFixed(0)} ms`;

    const previewLen = Math.min(originalLen, 10 * FS);
    drawLine(els.chartInput, channels[0].subarray(0, previewLen), COLOR_SIGNAL);
    drawLine(els.chartWave, wave.subarray(0, previewLen), COLOR_WAVE);
    drawLine(els.chartQrs, qrs.subarray(0, previewLen), COLOR_QRS, { fixedRange: [0, 1] });

    els.results.hidden = false;
    setStatus(`Done. ${peaks.length} fetal beats detected over ${(originalLen / FS).toFixed(1)}s.`);
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`, true);
  }
}

els.input.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) handleFile(file);
});
