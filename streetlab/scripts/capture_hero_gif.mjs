// Regenerates docs/screenshots/hero.gif for the root README. Not part of the
// test suite -- run manually against a live dev server, exactly as
// capture_screenshots.mjs does:
//
//   cd streetlab-backend && tail -f /dev/null | uv run streetlab serve --source osm &
//   cd streetlab && npm run dev &
//   node scripts/capture_hero_gif.mjs
//
// (`tail -f /dev/null |` keeps the backend's stdin open; see
// capture_screenshots.mjs for why.)
//
// Frames come from Chrome's CDP screencast (compositor output, already scaled
// down by Chrome) rather than page.screenshot(), which on this WebGPU page
// manages only ~4 fps. Each frame's delay is the gap between Chrome's own
// frame timestamps, so playback speed matches wall-clock. JPEG frames are
// decoded with jpeg-js and encoded with gifenc, both pure JS, so no ffmpeg is
// needed.
//
// Tunables (env): HERO_SCALE (output is 1440/SCALE wide, default 2),
// HERO_SECONDS, HERO_COLORS, HERO_MAX_FPS. If the GIF is over ~6 MB, raise
// HERO_SCALE or lower HERO_COLORS / HERO_MAX_FPS and re-run.
import { chromium } from '@playwright/test';
import { PNG } from 'pngjs';
import gifenc from 'gifenc'; // CommonJS: named imports fail under Node ESM
import jpeg from 'jpeg-js'; // CommonJS as well
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const { GIFEncoder, quantize, applyPalette } = gifenc;

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(HERE, '../../docs/screenshots');
mkdirSync(OUT_DIR, { recursive: true });

// Lay the page out at the app's design size (the Tauri window opens at 1440x900)
// so the full layout shows; below the 1320 px breakpoint the right panel is
// cropped. Chrome scales the screencast down to WIDTH/SCALE.
const WIDTH = 1440;
const HEIGHT = 900;
const SCALE = Number(process.env.HERO_SCALE ?? 2);
const SECONDS = Number(process.env.HERO_SECONDS ?? 10);
const COLORS = Number(process.env.HERO_COLORS ?? 128);
const MIN_GAP_MS = 1000 / Number(process.env.HERO_MAX_FPS ?? 15);
const LAST_FRAME_MS = 100;

const LAUNCH_ARGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=default'];

const browser = await chromium.launch({ args: LAUNCH_ARGS });
const context = await browser.newContext({ viewport: { width: WIDTH, height: HEIGHT } });
const page = await context.newPage();

console.log('goto app (real backend)...');
await page.goto('http://localhost:1420/');
await page.getByRole('heading', { name: /./ }).first().waitFor({ timeout: 15000 });
await page.waitForTimeout(6000); // let the car pick up speed and reach an open stretch

// Record: keep each frame's JPEG bytes and Chrome's timestamp (seconds).
const cdp = await context.newCDPSession(page);
const shots = [];
cdp.on('Page.screencastFrame', (e) => {
  shots.push({ jpeg: Buffer.from(e.data, 'base64'), tMs: e.metadata.timestamp * 1000 });
  cdp.send('Page.screencastFrameAck', { sessionId: e.sessionId }).catch(() => {});
});
await cdp.send('Page.startScreencast', {
  format: 'jpeg',
  quality: 85,
  maxWidth: Math.round(WIDTH / SCALE),
  maxHeight: Math.round(HEIGHT / SCALE),
  everyNthFrame: 1,
});
await page.waitForTimeout(SECONDS * 1000);
await cdp.send('Page.stopScreencast');
await browser.close();

// Thin to the target fps, then encode.
const kept = [];
for (const s of shots) {
  if (kept.length === 0 || s.tMs - kept[kept.length - 1].tMs >= MIN_GAP_MS) kept.push(s);
}
if (kept.length < 2) throw new Error(`only ${kept.length} screencast frames captured`);

const gif = GIFEncoder();
let midPng = null;
kept.forEach((s, i) => {
  const { data, width, height } = jpeg.decode(s.jpeg, { useTArray: true, formatAsRGBA: true });
  const palette = quantize(data, COLORS);
  const index = applyPalette(data, palette);
  const delay = i + 1 < kept.length ? Math.round(kept[i + 1].tMs - s.tMs) : LAST_FRAME_MS;
  gif.writeFrame(index, width, height, { palette, delay });
  if (i === Math.floor(kept.length / 2)) {
    const preview = new PNG({ width, height });
    preview.data = Buffer.from(data);
    midPng = PNG.sync.write(preview);
  }
});
gif.finish();
const bytes = gif.bytes();
writeFileSync(resolve(OUT_DIR, 'hero.gif'), bytes);
if (midPng) writeFileSync(resolve(OUT_DIR, '.hero-midframe-preview.png'), midPng);
const spanS = (kept[kept.length - 1].tMs - kept[0].tMs) / 1000;
console.log(
  `hero.gif: ${kept.length} frames over ${spanS.toFixed(1)} s ` +
    `(${(kept.length / spanS).toFixed(1)} fps), ${(bytes.length / 1048576).toFixed(2)} MB`,
);
