// Copies the onnxruntime-web runtime (JS glue + .wasm binaries) out of
// node_modules into public/ort/ so the page loads everything from same-origin
// files -- no CDN dependency at runtime, matching the "offline after page
// load" requirement.
import { cpSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "node_modules", "onnxruntime-web", "dist");
const dest = join(here, "public", "ort");

mkdirSync(dest, { recursive: true });
cpSync(src, dest, { recursive: true });
console.log(`Copied onnxruntime-web assets: ${src} -> ${dest}`);
