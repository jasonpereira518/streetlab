import { expect, test } from '@playwright/test';

/** Every visible control either works or is not there. */

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (err) => {
    throw new Error(`uncaught page error: ${err.message}`);
  });
  await page.goto('/?mock=1');
  await expect(page.locator('.link-chip')).toHaveText('mock');
});

test('the stub buttons and the cut-in slider are gone', async ({ page }) => {
  for (const name of ['New session', 'Save scenario', 'Undo']) {
    await expect(page.getByRole('button', { name, exact: true })).toHaveCount(0);
  }
  await expect(page.locator('.sidebar-foot')).toHaveCount(0);
  await expect(page.getByText('Cut-in interval')).toHaveCount(0);
});

test('a bookmark survives a reload', async ({ page }) => {
  const mark = page.getByRole('button', { name: /bookmark for Nob Hill Loop/ });
  await expect(mark).toBeVisible();
  const before = await mark.getAttribute('aria-pressed');
  await mark.click();
  await page.reload();
  const after = page.getByRole('button', { name: /bookmark for Nob Hill Loop/ });
  await expect(after).toHaveAttribute('aria-pressed', before === 'true' ? 'false' : 'true');
});

test('ml_limitation is shown on the hazard menu and acks reach the viewport', async ({ page }) => {
  await expect(page.locator('.hazard-limit').first()).toBeVisible();
  // Leave the Params tab: the toast must still show the answer.
  await page.getByRole('tab', { name: 'Layers' }).click();
  await page.getByRole('tab', { name: 'Parameters' }).click();
  await page.getByRole('button', { name: 'Jaywalker' }).click();
  await page.getByRole('tab', { name: 'Layers' }).click();
  await expect(page.locator('.ack-toast')).toContainText('only stages cut_in');
});

test('shortcuts pause, reset and open help; the list is visible', async ({ page }) => {
  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.keyboard.press('Space');
  await expect(page.getByRole('button', { name: 'Resume simulation' })).toBeVisible();
  await page.keyboard.press('Space');
  await expect(page.getByRole('button', { name: 'Pause simulation' })).toBeVisible();

  await page.keyboard.press('?');
  const dialog = page.getByRole('dialog', { name: /help/i });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('Pause / resume');
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
});

test('the perception menu says why it is disabled', async ({ page }) => {
  const wrapper = page.locator('.menu[title*="--perception"]');
  await expect(wrapper).toHaveCount(1);
  await expect(wrapper.locator('button')).toBeDisabled();
});
