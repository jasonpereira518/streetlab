/**
 * The restart button, end to end against a real `streetlab serve`.
 *
 * Its contract is an ordering — reset the simulator, and only then reload the
 * page — and neither half proves the feature on its own: a reload alone comes
 * back to a simulation still running from wherever it had got to, and a reset
 * alone leaves the app exactly where it was. So this asserts both, and reads
 * the backend's half from the backend's own `/health` rather than from
 * anything the UI claims about it.
 */
import { spawn } from 'node:child_process';
import type { ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';

const HERE = dirname(fileURLToPath(import.meta.url));
const BACKEND_DIR = resolve(HERE, '../../streetlab-backend');

interface Handshake {
  ws: string;
  http: string;
  pid: number;
  protocol: number;
}

/** Same shape as `faultInjection.spec.ts`'s own spawner. */
function spawnBackend(): Promise<{
  proc: ChildProcessWithoutNullStreams;
  handshake: Handshake;
}> {
  return new Promise((resolvePromise, reject) => {
    const proc = spawn('uv', ['run', 'streetlab', 'serve', '--port', '0'], {
      cwd: BACKEND_DIR,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    const rl = createInterface({ input: proc.stdout });
    const timeout = setTimeout(() => {
      rl.close();
      proc.kill('SIGKILL');
      reject(new Error('backend did not print STREETLAB_READY within 20s'));
    }, 20_000);

    rl.on('line', (line) => {
      if (!line.startsWith('STREETLAB_READY ')) return;
      clearTimeout(timeout);
      rl.close();
      resolvePromise({
        proc,
        handshake: JSON.parse(line.slice('STREETLAB_READY '.length)) as Handshake,
      });
    });

    proc.on('error', reject);
    proc.on('exit', (code) => {
      if (code !== null && code !== 0) {
        clearTimeout(timeout);
        reject(new Error(`backend exited early with code ${code}`));
      }
    });
  });
}

async function simClock(http: string): Promise<number> {
  const res = await fetch(`${http}/health`);
  return ((await res.json()) as { t: number }).t;
}

test('the restart button resets the simulator and reloads the app', async ({ page }) => {
  const { proc, handshake } = await spawnBackend();

  try {
    page.on('pageerror', (err) => {
      throw new Error(`uncaught page error: ${err.message}`);
    });

    await page.goto(`/?backend=${encodeURIComponent(handshake.ws)}`);
    await expect(page.locator('.link-chip')).toHaveClass(/link-chip--open/, {
      timeout: 10_000,
    });
    const speed = page.locator('.readout').first().locator('.readout-value');
    await expect(speed).not.toHaveText('—', { timeout: 10_000 });

    // Let the simulation build up a clock worth resetting.
    await page.waitForTimeout(3000);
    const before = await simClock(handshake.http);
    expect(before).toBeGreaterThan(2);

    // A marker on the current document: if it survives, the page never
    // reloaded, whatever else the app did.
    await page.evaluate(() => {
      (window as unknown as { __preRestart?: true }).__preRestart = true;
    });

    // Registered before the click: this waits for the NEXT load event, where
    // `waitForLoadState('load')` would happily resolve against the document
    // that is still on screen and let a never-reloading button pass.
    const reloaded = page.waitForEvent('load');
    await page.getByLabel('Restart session').click();
    await reloaded;

    expect(
      await page.evaluate(
        () => (window as unknown as { __preRestart?: true }).__preRestart,
      ),
    ).toBeUndefined();

    // The backend's own clock went backwards: it really did restart, and it
    // did so before the reload tore the socket down.
    expect(await simClock(handshake.http)).toBeLessThan(before);

    // And the app came back alive on the same backend.
    await expect(page.locator('.link-chip')).toHaveClass(/link-chip--open/, {
      timeout: 10_000,
    });
    await expect(page.locator('canvas.viewport-canvas')).toBeVisible();
    await expect(speed).not.toHaveText('—', { timeout: 10_000 });
  } finally {
    if (proc.exitCode === null && proc.signalCode === null) proc.kill('SIGKILL');
  }
});
