// Regenerates docs/screenshots/*.png for the root README. Not part of the
// test suite -- run manually against a live dev server:
//
//   cd streetlab-backend && tail -f /dev/null | uv run streetlab serve --source osm &
//   cd streetlab && npm run dev &
//   node scripts/capture_screenshots.mjs
//
// (the stdin-EOF watchdog in server/cli.py exits the backend the instant its
// stdin closes, which happens immediately under most non-interactive
// shells -- `tail -f /dev/null |` keeps a pipe open indefinitely so the
// watchdog never fires.)
import { chromium } from '@playwright/test';
import { PNG } from 'pngjs';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(HERE, '../../docs/screenshots');
mkdirSync(OUT_DIR, { recursive: true });

const LAUNCH_ARGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=default'];

/** Count roughly-orange pixels in the 3D viewport -- see e2e/app.spec.ts's
 * identical helper. The hazard box and warning billboard are the only
 * orange things in the scene, so a jump here means the overlay rendered. */
async function orangePixels(page) {
  const buf = await page.locator('canvas.viewport-canvas').screenshot({ type: 'png' });
  const png = PNG.sync.read(buf);
  let n = 0;
  for (let i = 0; i < png.data.length; i += 4) {
    const r = png.data[i];
    const g = png.data[i + 1];
    const b = png.data[i + 2];
    if (r > 185 && g > 75 && g < 180 && b < 110 && r - b > 100) n++;
  }
  return n;
}

async function newPage(browser) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  return context.newPage();
}

const browser = await chromium.launch({ args: LAUNCH_ARGS });

// hero.png + address-search.png + performance-overlay.png: real backend,
// real OSM data, driven live.
{
  const page = await newPage(browser);
  console.log('goto app (real backend)...');
  await page.goto('http://localhost:1420/');
  await page.getByRole('heading', { name: /./ }).first().waitFor({ timeout: 15000 });
  await page.waitForTimeout(6000); // let the car pick up speed and reach an open stretch

  console.log('hero.png');
  await page.screenshot({ path: resolve(OUT_DIR, 'hero.png') });

  console.log('searching address...');
  await page.getByLabel('Load a location').fill('Golden Gate Park, San Francisco');
  await page.keyboard.press('Enter');
  const pending = page.locator('.location-pending');
  try {
    await pending.waitFor({ state: 'visible', timeout: 5000 });
    await pending.waitFor({ state: 'hidden', timeout: 30000 });
    await page.waitForTimeout(1500);
  } catch (err) {
    console.warn('address load did not complete in time, screenshotting current state:', err.message);
  }
  console.log('address-search.png');
  await page.screenshot({ path: resolve(OUT_DIR, 'address-search.png') });

  console.log('toggling performance overlay...');
  await page.getByRole('button', { name: 'Toggle performance overlay' }).click();
  await page.waitForTimeout(2000);
  console.log('performance-overlay.png');
  await page.screenshot({ path: resolve(OUT_DIR, 'performance-overlay.png') });

  await page.close();
}

// hazard-injection.png: offline mock mode, same as e2e/app.spec.ts's hazard
// test -- Nob Hill's real OSM traffic is single-lane along ~88% of the
// driven route, so a scripted cut-in rarely has an adjacent lane to merge
// from. Mock mode scripts one deterministically.
{
  const page = await newPage(browser);
  console.log('goto app (mock traffic)...');
  await page.goto('http://localhost:1420/?mock=1');
  await page.locator('.viewport-stats').waitFor({ timeout: 15000 });
  await page.waitForTimeout(1500);

  const baseline = await orangePixels(page);
  await page.getByText('Inject cut-in hazard').click();

  let last = baseline;
  for (let i = 0; i < 24 && last <= baseline + 200; i++) {
    await page.waitForTimeout(1000);
    last = await orangePixels(page);
  }
  console.log('hazard-injection.png (orange px delta:', last - baseline, ')');
  await page.screenshot({ path: resolve(OUT_DIR, 'hazard-injection.png') });
  await page.close();
}

await browser.close();
console.log('done');
