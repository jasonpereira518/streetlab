import { expect, test } from '@playwright/test';

/** Driver warnings, end to end on the in-process mock. */

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (err) => {
    throw new Error(`uncaught page error: ${err.message}`);
  });
  await page.goto('/?mock=1');
  await expect(page.locator('.link-chip')).toHaveText('mock');
});

test('the settings have an Active safety group', async ({ page }) => {
  await expect(page.getByText('Active safety', { exact: true })).toBeVisible();
  await expect(page.getByText('Forward collision warning')).toBeVisible();
});

test('driving into a cut-in raises a forward collision warning over the road', async ({ page }) => {
  // FSD brakes for it, which silences FCW (as Tesla's): drive it yourself.
  await page.getByRole('button', { name: 'Cut-in', exact: true }).click();
  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.keyboard.down('s'); // brake: take over in Manual
  await page.waitForTimeout(300);
  await page.keyboard.up('s');
  await page.keyboard.down('w');
  await expect(page.locator('.ack-toast')).toContainText('Forward collision', { timeout: 30_000 });
  await page.keyboard.up('w');
});

test('drifting over the line in Manual warns of a lane departure', async ({ page }) => {
  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.keyboard.down('s'); // brake: take over in Manual
  await page.waitForTimeout(300);
  await page.keyboard.up('s');
  await page.keyboard.down('w');
  await page.waitForTimeout(3000); // past the 10 mph floor
  await page.keyboard.down('a');
  await expect(page.locator('.ack-toast')).toContainText('Lane departure', { timeout: 10_000 });
  await page.keyboard.up('a');
  await page.keyboard.up('w');
});
