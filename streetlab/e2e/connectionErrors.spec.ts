import { expect, test, type Page } from '@playwright/test';

/**
 * The hosted shell used to load with a blank viewport and a small
 * "reconnecting" chip when the simulator was unreachable. These drive the real
 * app against a dead port and against scripted servers (`routeWebSocket`), and
 * assert the visible error and its distinct wording per failure.
 */

const overlay = (page: Page) => page.getByTestId('connection-error');

test('a dead port shows "can\'t reach the simulator" with a Retry button', async ({ page }) => {
  await page.goto('/?backend=ws://127.0.0.1:1');
  await expect(overlay(page)).toHaveAttribute('data-kind', 'backend_down');
  await expect(overlay(page)).toContainText("Can't reach the simulator");
  await expect(overlay(page).getByRole('button', { name: 'Retry' })).toBeVisible();
});

test('Retry makes a fresh attempt against the dead port', async ({ page }) => {
  let attempts = 0;
  await page.routeWebSocket('ws://dead.test/', (ws) => {
    attempts += 1;
    ws.close({ code: 1006 });
  });
  await page.goto('/?backend=ws://dead.test/');
  await expect(overlay(page)).toHaveAttribute('data-kind', 'backend_down');
  const before = attempts;
  await overlay(page).getByRole('button', { name: 'Retry' }).click();
  await expect.poll(() => attempts).toBeGreaterThan(before);
});

test('close 1008 is reported as a refused origin, distinct from "down"', async ({ page }) => {
  await page.routeWebSocket('ws://refuse.test/', (ws) => ws.close({ code: 1008 }));
  await page.goto('/?backend=ws://refuse.test/');
  await expect(overlay(page)).toHaveAttribute('data-kind', 'origin_rejected');
  await expect(overlay(page)).toContainText('refused this page');
});

test('close 4429 is reported as busy, and Retry asks again', async ({ page }) => {
  let attempts = 0;
  await page.routeWebSocket('ws://busy.test/', (ws) => {
    attempts += 1;
    ws.close({ code: 4429, reason: 'server busy' });
  });
  await page.goto('/?backend=ws://busy.test/');
  await expect(overlay(page)).toHaveAttribute('data-kind', 'server_busy');
  await expect(overlay(page)).toContainText('busy');
  await page.waitForTimeout(1500); // a halted transport must not retry on a timer
  expect(attempts).toBe(1);
  await overlay(page).getByRole('button', { name: 'Retry' }).click();
  await expect.poll(() => attempts).toBe(2);
});

test('a server on another protocol is reported as a version mismatch', async ({ page }) => {
  await page.routeWebSocket('ws://old.test/', (ws) => {
    ws.send(JSON.stringify({ type: 'scene_description', protocol: 999 }));
  });
  await page.goto('/?backend=ws://old.test/');
  await expect(overlay(page)).toHaveAttribute('data-kind', 'protocol_mismatch');
  await expect(overlay(page)).toContainText('999');
});

test('"Use offline demo" leaves the error state and runs the mock', async ({ page }) => {
  await page.routeWebSocket('ws://refuse.test/', (ws) => ws.close({ code: 1008 }));
  await page.goto('/?backend=ws://refuse.test/');
  await overlay(page).getByRole('button', { name: /offline demo/i }).click();
  await expect(overlay(page)).toHaveCount(0);
  await expect(page.locator('.viewport')).toBeVisible();
});
