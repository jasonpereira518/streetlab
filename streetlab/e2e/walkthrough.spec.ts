/**
 * Walkthrough presets against the in-process mock (`?mock=1`): Run the 20 s
 * pinned preset, land on the Run tab, get a scorecard, and Replay onto the
 * same seed. Assertions read store-driven text only (run-seed, scorecard),
 * never the throttled Elapsed readout.
 */
import { expect, test } from '@playwright/test';

const TITLE = 'Mock cut-in, pinned';

test('Run lands a scorecard and Replay keeps the seed', async ({ page }) => {
  test.setTimeout(90_000);
  page.on('pageerror', (err) => {
    throw new Error(`uncaught page error: ${err.message}`);
  });

  await page.goto('/?mock=1');
  const presets = page.getByRole('list', { name: 'Walkthrough presets' });
  await expect(presets.getByText(TITLE)).toBeVisible();

  const replay = page.getByRole('button', { name: `Replay ${TITLE}` });
  await expect(replay).toBeDisabled();

  await page.getByRole('button', { name: `Run ${TITLE}` }).click();
  await expect(page.getByRole('tab', { name: 'Run' })).toHaveAttribute('aria-selected', 'true');

  const scorecard = page.getByRole('table', { name: 'Scorecard' });
  await expect(scorecard).toBeVisible({ timeout: 40_000 });
  // Read the seed off the run_summary's column: run-seed shows the previous
  // scene's seed until the load_preset ack lands.
  const header = (await scorecard.getByRole('columnheader').nth(1).textContent())!;
  const first = header.replace('seed ', '');
  const seed = page.getByTestId('run-seed');
  await expect(seed).toHaveText(first);

  await expect(replay).toBeEnabled();
  await replay.click();
  await expect(seed).toHaveText(first);
  // The replay's own summary joins the first: two columns, same seed.
  await expect(scorecard.getByRole('columnheader', { name: `seed ${first}` })).toHaveCount(2, {
    timeout: 40_000,
  });
});
