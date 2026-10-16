import { expect, test } from '@playwright/test';

/**
 * Needs a real hosted-mode backend (`streetlab serve --max-sessions N`, or the
 * Docker image). Skipped unless STREETLAB_E2E_BACKEND names it, e.g.
 *
 *   STREETLAB_E2E_BACKEND=ws://127.0.0.1:8080 npx playwright test hostedSessions
 *
 * The point: with one shared world, pausing in one tab paused every visitor.
 */
const BACKEND = process.env.STREETLAB_E2E_BACKEND;
test.skip(!BACKEND, 'set STREETLAB_E2E_BACKEND to a hosted-mode backend');

test('two tabs get independent simulations', async ({ browser }) => {
  const ctx = await browser.newContext();
  const a = await ctx.newPage();
  const b = await ctx.newPage();
  await a.goto(`/?backend=${BACKEND}`);
  await b.goto(`/?backend=${BACKEND}`);

  for (const page of [a, b]) {
    await expect(page.getByRole('button', { name: 'Pause simulation' })).toBeVisible({
      timeout: 20_000,
    });
  }

  await a.getByRole('button', { name: 'Pause simulation' }).click();
  await expect(a.getByRole('button', { name: 'Resume simulation' })).toBeVisible();

  // On a shared world B would flip to "Resume" within a frame or two.
  await b.waitForTimeout(1500);
  await expect(b.getByRole('button', { name: 'Pause simulation' })).toBeVisible();
  await expect(a.getByRole('button', { name: 'Resume simulation' })).toBeVisible();
  await ctx.close();
});

test('a third tab beyond the cap sees the busy state', async ({ browser }) => {
  const cap = Number(process.env.STREETLAB_E2E_CAP ?? '2');
  const ctx = await browser.newContext();
  const open = [];
  for (let i = 0; i < cap; i++) {
    const p = await ctx.newPage();
    await p.goto(`/?backend=${BACKEND}`);
    await expect(p.getByRole('button', { name: 'Pause simulation' })).toBeVisible({
      timeout: 20_000,
    });
    open.push(p);
  }
  const extra = await ctx.newPage();
  await extra.goto(`/?backend=${BACKEND}`);
  await expect(extra.getByTestId('connection-error')).toHaveAttribute('data-kind', 'server_busy');

  // Free a slot; Retry on the refused tab now gets in.
  await open[0].close();
  await extra.waitForTimeout(500);
  await extra.getByRole('button', { name: 'Retry' }).click();
  await expect(extra.getByRole('button', { name: 'Pause simulation' })).toBeVisible({
    timeout: 20_000,
  });
  await ctx.close();
});
