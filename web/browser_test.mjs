// Drives the actual browser build (onnxruntime-web + Wasm) against the local
// static server, to catch anything the Node/onnxruntime-node test can't:
// ES module loading, Wasm fetch/instantiate, file input handling, canvas
// rendering, console errors.
import { chromium } from "playwright";
import path from "node:path";

const URL = "http://localhost:8080/index.html";
const CSV_PATH = path.resolve("./test_sample.csv");

const browser = await chromium.launch();
const page = await browser.newPage();

const consoleErrors = [];
page.on("console", (msg) => {
  if (msg.type() === "error") consoleErrors.push(msg.text());
});
page.on("pageerror", (err) => consoleErrors.push(String(err)));

await page.goto(URL);
await page.setInputFiles("#csv-input", CSV_PATH);

// Wait for the "Done." status message (inference completes) or an error status.
await page.waitForFunction(
  () => {
    const el = document.getElementById("status");
    return el && (el.textContent.startsWith("Done.") || el.classList.contains("error"));
  },
  { timeout: 30000 }
);

const status = await page.textContent("#status");
const statusIsError = await page.evaluate(() => document.getElementById("status").classList.contains("error"));
const resultsHidden = await page.evaluate(() => document.getElementById("results").hidden);
const fhr = await page.textContent("#fhr-value");
const beats = await page.textContent("#beats-value");
const time = await page.textContent("#time-value");

await page.screenshot({ path: "browser_test_screenshot.png", fullPage: true });

console.log(`status: "${status}" (error=${statusIsError})`);
console.log(`results hidden: ${resultsHidden}`);
console.log(`FHR: ${fhr}, beats: ${beats}, inference time: ${time}`);
console.log(`console errors: ${consoleErrors.length}`);
for (const e of consoleErrors) console.log(`  - ${e}`);

await browser.close();

if (statusIsError || resultsHidden || consoleErrors.length > 0) {
  console.error("BROWSER TEST FAILED");
  process.exit(1);
}
console.log("BROWSER TEST PASSED");
