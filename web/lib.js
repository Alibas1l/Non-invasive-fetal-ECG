// Pure signal-processing logic shared between the browser app (app.js, using
// onnxruntime-web) and the Node CLI test (test_inference.mjs, using
// onnxruntime-node) -- kept free of DOM/ORT-backend specifics so both can
// exercise the exact same code path.

export const WINDOW_LEN = 2000; // must match model.py training window_len / export_onnx.py EXPORT_WINDOW_LEN
export const STRIDE = 1000;
export const FS = 1000; // Hz, per the NIFECGDB recordings this model was trained on

/** Parse a CSV of 3 columns (abdominal channels) into a Float32Array[channel][time]. */
export function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const first = lines[0].split(",");
  const hasHeader = first.some((v) => Number.isNaN(Number.parseFloat(v)));
  const rows = hasHeader ? lines.slice(1) : lines;

  const channels = [[], [], []];
  for (const line of rows) {
    if (!line.trim()) continue;
    const parts = line.split(",");
    for (let c = 0; c < 3; c++) {
      channels[c].push(Number.parseFloat(parts[c]));
    }
  }
  return channels.map((ch) => Float32Array.from(ch));
}

export function zscoreWindow(channels, start, len) {
  const out = new Float32Array(3 * len);
  for (let c = 0; c < 3; c++) {
    let mean = 0;
    for (let i = 0; i < len; i++) mean += channels[c][start + i];
    mean /= len;
    let variance = 0;
    for (let i = 0; i < len; i++) {
      const d = channels[c][start + i] - mean;
      variance += d * d;
    }
    const std = Math.sqrt(variance / len) + 1e-8;
    for (let i = 0; i < len; i++) {
      out[c * len + i] = (channels[c][start + i] - mean) / std;
    }
  }
  return out;
}

/** Pad the tail of each channel by repeating the last sample, so the signal
 * length is covered by whole WINDOW_LEN-length windows at stride STRIDE. */
export function padChannels(channels, originalLen) {
  const nWindows = Math.max(1, Math.ceil((originalLen - WINDOW_LEN) / STRIDE) + 1);
  const paddedLen = (nWindows - 1) * STRIDE + WINDOW_LEN;
  const padded = channels.map((ch) => {
    const out = new Float32Array(paddedLen);
    out.set(ch);
    const last = ch[ch.length - 1] ?? 0;
    for (let i = ch.length; i < paddedLen; i++) out[i] = last;
    return out;
  });
  return { padded, paddedLen, nWindows };
}

/** Mirrors losses.peaks_from_qrs_signal: local-maxima peak picking with a
 * height threshold and a minimum-distance refractory period. */
export function findPeaks(signal, height, minDistance) {
  const peaks = [];
  for (let i = 1; i < signal.length - 1; i++) {
    if (signal[i] >= height && signal[i] > signal[i - 1] && signal[i] >= signal[i + 1]) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDistance) {
        peaks.push(i);
      } else if (signal[i] > signal[peaks[peaks.length - 1]]) {
        peaks[peaks.length - 1] = i; // keep the taller of two close peaks
      }
    }
  }
  return peaks;
}

export function estimateFhr(qrsPeaks) {
  if (qrsPeaks.length < 2) return null;
  const rrSamples = [];
  for (let i = 1; i < qrsPeaks.length; i++) rrSamples.push(qrsPeaks[i] - qrsPeaks[i - 1]);
  const meanRr = rrSamples.reduce((a, b) => a + b, 0) / rrSamples.length;
  return 60 / (meanRr / FS);
}

/** Run inference over sliding windows via the given async runner
 * (start, inputTensorData) -> {wave: Float32Array, qrs: Float32Array}
 * and overlap-add the outputs back into full-length arrays (averaged where
 * windows overlap). Backend-agnostic: caller supplies how a window is run. */
export async function runInferenceWindows(channels, originalLen, runWindow) {
  const { padded, paddedLen, nWindows } = padChannels(channels, originalLen);

  const waveSum = new Float32Array(paddedLen);
  const qrsSum = new Float32Array(paddedLen);
  const counts = new Float32Array(paddedLen);

  for (let w = 0; w < nWindows; w++) {
    const start = w * STRIDE;
    const inputData = zscoreWindow(padded, start, WINDOW_LEN);
    const { wave, qrs } = await runWindow(inputData);
    for (let i = 0; i < WINDOW_LEN; i++) {
      waveSum[start + i] += wave[i];
      qrsSum[start + i] += qrs[i];
      counts[start + i] += 1;
    }
  }

  const wave = new Float32Array(originalLen);
  const qrs = new Float32Array(originalLen);
  for (let i = 0; i < originalLen; i++) {
    wave[i] = waveSum[i] / counts[i];
    qrs[i] = qrsSum[i] / counts[i];
  }
  return { wave, qrs };
}
