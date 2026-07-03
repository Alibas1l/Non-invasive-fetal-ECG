// CLI end-to-end test of the exact browser inference logic (lib.js), using
// onnxruntime-node instead of onnxruntime-web so it can run without a
// browser. Validates against a real held-out test-record CSV
// (src/export_test_csv.py) whose true fetal beat count is known.
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-node";
import { WINDOW_LEN, FS, parseCsv, findPeaks, estimateFhr, runInferenceWindows } from "./lib.js";

const MODEL_PATH = "./public/model_int8.onnx";
const CSV_PATH = process.argv[2] ?? "./test_sample.csv";

async function main() {
  const session = await ort.InferenceSession.create(MODEL_PATH);

  async function runWindow(inputData) {
    const inputTensor = new ort.Tensor("float32", inputData, [1, 3, WINDOW_LEN]);
    const outputs = await session.run({ input: inputTensor });
    return { wave: outputs.wave.data, qrs: outputs.qrs.data };
  }

  const text = readFileSync(CSV_PATH, "utf-8");
  const channels = parseCsv(text);
  const originalLen = channels[0].length;
  console.log(`Loaded ${CSV_PATH}: ${originalLen} samples (${(originalLen / FS).toFixed(1)}s), 3 channels`);

  const t0 = performance.now();
  const { wave, qrs } = await runInferenceWindows(channels, originalLen, runWindow);
  const dt = performance.now() - t0;

  const peaks = findPeaks(qrs, 0.5, 0.25 * FS);
  const fhr = estimateFhr(peaks);

  console.log(`Inference time: ${dt.toFixed(0)}ms`);
  console.log(`wave range: [${Math.min(...wave).toFixed(3)}, ${Math.max(...wave).toFixed(3)}]`);
  console.log(`qrs range: [${Math.min(...qrs).toFixed(3)}, ${Math.max(...qrs).toFixed(3)}]`);
  console.log(`Detected beats: ${peaks.length}`);
  console.log(`Estimated FHR: ${fhr ? fhr.toFixed(1) + " bpm" : "n/a"}`);
  console.log(`Peak sample indices (first 10): ${peaks.slice(0, 10).join(", ")}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
