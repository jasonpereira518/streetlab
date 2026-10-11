// Regenerates docs/screenshots/social-preview.png (1280x640, GitHub's
// recommended social-preview size). Run against a live dev server exactly as
// capture_screenshots.mjs does:
//
//   cd streetlab-backend && tail -f /dev/null | uv run streetlab serve --source osm &
//   cd streetlab && npm run dev &
//   node scripts/capture_social_preview.mjs
import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(HERE, '../../docs/screenshots');
mkdirSync(OUT_DIR, { recursive: true });

const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=default'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 640 } });
await page.goto('http://localhost:1420/');
await page.getByRole('heading', { name: /./ }).first().waitFor({ timeout: 15000 });
await page.waitForTimeout(6000);
await page.screenshot({ path: resolve(OUT_DIR, 'social-preview.png') });
console.log('social-preview.png written');
await browser.close();
