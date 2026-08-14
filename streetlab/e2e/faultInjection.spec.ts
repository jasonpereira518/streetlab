/**
 * The first real integration test between the two halves: a genuine
 * `streetlab serve` subprocess, killed mid-stream, proving the frontend's
 * reconnect/backoff (already unit-tested against a fake socket in
 * wsClient.test.ts) holds against the real transport end to end.
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

function spawnBackend(): Promise<{ proc: ChildProcessWithoutNullStreams; handshake: Handshake }> {
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
      const handshake = JSON.parse(line.slice('STREETLAB_READY '.length)) as Handshake;
      resolvePromise({ proc, handshake });
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

test('a real backend killed mid-stream leaves the app reconnecting, not crashed', async ({
  page,
}) => {
  const { proc, handshake } = await spawnBackend();

  try {
    page.on('pageerror', (err) => {
      throw new Error(`uncaught page error: ${err.message}`);
    });

    await page.goto(`/?backend=${encodeURIComponent(handshake.ws)}`);
    await expect(page.locator('.link-chip')).toHaveText(handshake.ws);
    await expect(page.locator('.link-chip')).toHaveClass(/link-chip--open/, {
      timeout: 10_000,
    });

    // Prove the connection is really live — real telemetry, not a placeholder.
    const speed = page.locator('.readout').first().locator('.readout-value');
    await expect(speed).not.toHaveText('—', { timeout: 10_000 });

    proc.kill('SIGKILL');

    await expect(page.locator('.link-chip')).toHaveClass(/link-chip--reconnecting/, {
      timeout: 10_000,
    });

    // The shell survives a dead backend: still rendering, controls still work.
    await expect(page.locator('canvas.viewport-canvas')).toBeVisible();
    await expect(page.locator('.tcard-body canvas')).toHaveCount(6);
    await expect(page.getByLabel('Pause simulation')).toBeVisible();
  } finally {
    if (proc.exitCode === null && proc.signalCode === null) proc.kill('SIGKILL');
  }
});
