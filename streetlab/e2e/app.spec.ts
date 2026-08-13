import { expect, test } from '@playwright/test';
import { PNG } from 'pngjs';

/** The stats chip is the app's own report of backend, frame rate and draw calls. */
async function readStats(page: import('@playwright/test').Page) {
  const chip = page.locator('.viewport-stats');
  await expect(chip).toBeVisible();
  const raw = (await chip.textContent()) ?? '';
  const [, backend] = raw.match(/(WebGPU|WebGL2)/) ?? [];
  const [, fps] = raw.match(/(\d+)\s*fps/) ?? [];
  const [, draws] = raw.match(/(\d+)\s*draws/) ?? [];
  return { backend, fps: Number(fps), draws: Number(draws) };
}

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (err) => {
    throw new Error(`uncaught page error: ${err.message}`);
  });
  await page.goto('/?mock=1');
});

test('boots on the mock with no backend present', async ({ page }) => {
  await expect(page).toHaveTitle('StreetLab');
  await expect(page.locator('.brand-name')).toHaveText('StreetLab');
  // "mock" chip proves the default source, not a live socket.
  await expect(page.locator('.link-chip')).toHaveText('mock');
  await expect(page.getByRole('heading', { name: 'Nob Hill', exact: true })).toBeVisible();
});

test('renders the 3D scene within the draw-call budget', async ({ page }) => {
  await expect(page.locator('canvas.viewport-canvas')).toBeVisible();
  // Let the scene build and the frame-rate window fill.
  await page.waitForTimeout(2500);

  const stats = await readStats(page);
  expect(['WebGPU', 'WebGL2']).toContain(stats.backend);
  expect(stats.draws).toBeGreaterThan(10);
  expect(stats.draws).toBeLessThan(150);
  expect(stats.fps).toBeGreaterThan(20);
});

test('the toolbar shows live telemetry and pause halts the clock', async ({ page }) => {
  const speed = page.locator('.readout').first().locator('.readout-value');
  await expect(speed).not.toHaveText('—', { timeout: 10_000 });

  await expect(page.getByText('Assist active')).toBeVisible();

  await page.getByLabel('Pause simulation').click();
  await expect(page.getByLabel('Resume simulation')).toBeVisible();

  const paused = await speed.textContent();
  await page.waitForTimeout(1200);
  expect(await speed.textContent()).toBe(paused);

  await page.getByLabel('Resume simulation').click();
  await expect(page.getByLabel('Pause simulation')).toBeVisible();
});

test('all six telemetry widgets are present and animating', async ({ page }) => {
  for (const title of ['Speed', 'Lane position', 'Radar', 'Vehicle', 'Trajectory', 'Steering']) {
    await expect(page.locator('.tcard-head h4', { hasText: title })).toBeVisible();
  }
  const canvases = page.locator('.tcard-body canvas');
  await expect(canvases).toHaveCount(6);

  // Compare a widget's pixels across time: a static canvas means it is dead.
  const speedCanvas = canvases.first();
  const a = await speedCanvas.screenshot();
  await page.waitForTimeout(1500);
  const b = await speedCanvas.screenshot();
  expect(Buffer.compare(a, b)).not.toBe(0);
});

test('loading a scenario from the sidebar swaps the scene', async ({ page }) => {
  await expect(page.getByRole('heading', { name: 'Nob Hill', exact: true })).toBeVisible();
  await page.getByLabel('Load Hyde St Descent').click();
  await expect(page.getByRole('heading', { name: 'Russian Hill', exact: true })).toBeVisible();
  await expect(page.locator('.scenario.is-active')).toContainText('Hyde St Descent');
});

test('toggling the detections layer keeps the renderer alive', async ({ page }) => {
  await page.getByRole('tab', { name: 'Layers' }).click();
  const toggle = page.getByRole('switch', { name: 'Detections' });
  await expect(toggle).toHaveAttribute('aria-checked', 'true');

  const before = await page.locator('canvas.viewport-canvas').screenshot();
  await toggle.click();
  await expect(toggle).toHaveAttribute('aria-checked', 'false');
  await page.waitForTimeout(900);
  const after = await page.locator('canvas.viewport-canvas').screenshot();
  expect(Buffer.compare(before, after)).not.toBe(0);

  await toggle.click();
  await expect(toggle).toHaveAttribute('aria-checked', 'true');
});

test('a parameter slider round-trips through an ack', async ({ page }) => {
  const slider = page.getByLabel(/Max speed/);
  await slider.fill('62');
  await expect(page.locator('.ctl--slider .ctl-value').first()).toContainText('62');
  await expect(page.locator('.ack')).toContainText('ego_speed_cap_mph=62');
});

test('an unreachable ?backend= degrades without crashing the app', async ({ page }) => {
  // Port 9 (discard) refuses connections, so this exercises the retry path.
  await page.goto('/?backend=ws://127.0.0.1:9');
  await expect(page.locator('.link-chip')).toHaveText('ws://127.0.0.1:9');
  // No frames arrive, so the widgets sit in their placeholder state — but the
  // shell, the 3D canvas and the controls are all still up.
  await expect(page.locator('canvas.viewport-canvas')).toBeVisible();
  await expect(page.locator('.tcard-body canvas')).toHaveCount(6);
  await expect(page.getByLabel('Pause simulation')).toBeVisible();
});

/**
 * Count roughly-orange pixels in a screenshot of the 3D viewport.
 *
 * The pixels are read from a real Playwright screenshot rather than
 * `drawImage(webgpuCanvas)` — a WebGPU swap-chain texture is not retained as a
 * canvas image source, so a 2D readback comes back fully transparent.
 * The hazard box and the warning billboard are the only orange things in the
 * scene, so a jump in this count is direct evidence the overlay rendered.
 */
async function orangePixels(page: import('@playwright/test').Page): Promise<number> {
  const png = PNG.sync.read(
    await page.locator('canvas.viewport-canvas').screenshot({ type: 'png' }),
  );
  let n = 0;
  for (let i = 0; i < png.data.length; i += 4) {
    const r = png.data[i];
    const g = png.data[i + 1];
    const b = png.data[i + 2];
    if (r > 185 && g > 75 && g < 180 && b < 110 && r - b > 100) n++;
  }
  return n;
}

test('an injected hazard draws an orange overlay in the 3D view', async ({ page }) => {
  await expect(page.locator('.viewport-stats')).toBeVisible();
  await page.waitForTimeout(1500);

  const baseline = await orangePixels(page);
  await page.getByText('Inject cut-in hazard').click();

  // The scripted cut-in needs a few seconds to pull alongside and merge before
  // the detection is flagged as a hazard.
  await expect
    .poll(() => orangePixels(page), { timeout: 30_000, intervals: [750] })
    .toBeGreaterThan(baseline + 200);
});
