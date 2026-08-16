/**
 * The other half of the Phase 2 integration story: `faultInjection.spec.ts`
 * proves a real backend subprocess survives being killed mid-stream; this
 * file proves `load_location` — the address search box — works end to end
 * against a real backend subprocess, in both the happy and the failure path.
 *
 * Both tests start `streetlab serve --source osm`, the one configuration
 * where the backend's `SceneSource` actually implements `build_location`
 * (`map/osm_source.py`'s `OsmSceneSource`; the default `--source synthetic`
 * has no such method, so `load_location` against it fails at the ack itself
 * — see `sim/loop.py`'s `_cmd_load_location`). The initial scene it serves
 * (`osm-nob-hill`, the one bundled location) never touches the network: its
 * geocode result is baked into `BUNDLED` and its Overpass extract ships in
 * `streetlab-backend/bundled/`, so both specs boot offline regardless of
 * what the *searched* address needs.
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

/** Same shape as `faultInjection.spec.ts`'s own `spawnBackend`, generalised
 * with an `extraArgs` parameter so callers here can pass `--source osm`. */
function spawnBackend(
  extraArgs: string[] = [],
): Promise<{ proc: ChildProcessWithoutNullStreams; handshake: Handshake }> {
  return new Promise((resolvePromise, reject) => {
    const proc = spawn('uv', ['run', 'streetlab', 'serve', '--port', '0', ...extraArgs], {
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

function killBackend(proc: ChildProcessWithoutNullStreams): void {
  if (proc.exitCode === null && proc.signalCode === null) proc.kill('SIGKILL');
}

/**
 * Live Overpass/Nominatim by design — "Alamo Square, San Francisco" is not
 * one of the bundled locations, so this exercises a real geocode plus a real
 * Overpass fetch. Measured locally: a genuinely uncached address end to end
 * through this exact test (browser, real backend subprocess, real network)
 * lands around 3-5s; a bare-CLI build with no server/browser involved
 * (`uv run streetlab build "<address>"`, first hit) measured ~12s for a
 * denser neighbourhood. Either way, the 60s expect timeout and 90s test
 * timeout below leave generous room on top of the common case, and for
 * Overpass's own occasional slowness — it returned real 504s twice during
 * Phase 1 verification, and a stalled fetch can retry for up to ~90s before
 * giving up (`OverpassClient`'s 3 retries × 30s timeout, `map/overpass.py`).
 * Set STREETLAB_OFFLINE=1 to skip this spec in an offline or flaky-network
 * run.
 */
test('searching an address loads and drives it', async ({ page }) => {
  test.skip(
    Boolean(process.env.STREETLAB_OFFLINE),
    'requires live Nominatim/Overpass network access',
  );
  test.setTimeout(90_000);

  const { proc, handshake } = await spawnBackend(['--source', 'osm']);
  try {
    page.on('pageerror', (err) => {
      throw new Error(`uncaught page error: ${err.message}`);
    });

    await page.goto(`/?backend=${encodeURIComponent(handshake.ws)}`);
    await expect(page.getByText(/OpenStreetMap contributors/)).toBeVisible();

    const sceneName = page.getByTestId('scene-name');
    const before = await sceneName.textContent();
    // Scoped to `.location-pending` specifically, not a bare `getByText`
    // regex: `_cmd_load_location`'s ack message is ALSO literally "building
    // {query}" (`sim/loop.py`), and the Parameters tab (the default active
    // tab) renders `lastAck.message` in its own `<span>` — which, unlike
    // this transient pending indicator, never disappears once the scene
    // lands. A regex-based `getByText` matches both elements at once and
    // can never observe `toBeHidden()` as a result.
    const pending = page.locator('.location-pending');

    await page.getByLabel('Load a location').fill('Alamo Square, San Francisco');
    await page.keyboard.press('Enter');

    // Immediate ack -> pending state, before any scene arrives.
    await expect(pending).toBeVisible();
    await expect(pending).toContainText(/alamo square/i);

    // The scene arrives unsolicited, via the epoch push -- not as a direct
    // reply to the command above.
    await expect(sceneName).not.toHaveText(before ?? '', { timeout: 60_000 });
    await expect(pending).toBeHidden();

    // The box re-enables once the build lands, same as the failure path.
    await expect(page.getByLabel('Load a location')).toBeEnabled();

    // ...and the ego actually DRIVES the new location. Everything above is
    // satisfied by a scene that merely arrived: a location that builds into
    // a route with no drivable road, or one where the planner never gets a
    // target, changes the heading and clears the pending flag exactly like a
    // good one. Without this the spec's own name would be half untrue.
    const speed = page.locator('.readout').first().locator('.readout-value');
    await expect(speed).not.toHaveText('—', { timeout: 15_000 });
    await expect
      .poll(async () => Number((await speed.textContent())?.trim() ?? '0'), {
        timeout: 15_000,
      })
      .toBeGreaterThan(0);
  } finally {
    killBackend(proc);
  }
});

/**
 * No network required: `NominatimGeocoder.raw()` (`map/geocode.py`) wraps
 * every httpx failure — a genuine "no results" response from a reachable
 * Nominatim, or a connection failure because there is no network at all —
 * in the same `GeocodeError`, which `submit_scene` (`sim/loop.py`) turns
 * into the same `location_failed` event either way. So this spec produces
 * its target event regardless of whether the sandbox running it has
 * outbound network access.
 */
test('a nonsense address surfaces an event and clears the pending state', async ({ page }) => {
  const { proc, handshake } = await spawnBackend(['--source', 'osm']);
  try {
    page.on('pageerror', (err) => {
      throw new Error(`uncaught page error: ${err.message}`);
    });

    await page.goto(`/?backend=${encodeURIComponent(handshake.ws)}`);
    // Wait for the initial (bundled, offline) scene so the catalog — and
    // therefore a scenario "load" button — actually exists before the
    // gating assertions below.
    await expect(page.getByText(/OpenStreetMap contributors/)).toBeVisible();
    const loadScenarioButton = page.locator('.scenario .play-btn').first();
    await expect(loadScenarioButton).toBeEnabled();

    await page.getByLabel('Load a location').fill('zzzqqq not a real place 99999');
    await page.keyboard.press('Enter');

    // Both the search box and the scenario list gate on the same pending
    // flag (Task 6, and Task 6's own follow-up fix for the scenario list).
    await expect(page.getByLabel('Load a location')).toBeDisabled();
    await expect(loadScenarioButton).toBeDisabled();

    await page.getByRole('tab', { name: /events/i }).click();
    await expect(page.getByText(/location_failed/)).toBeVisible({ timeout: 30_000 });

    // The box — and the scenario list — must not stay stuck spinning on a
    // failure.
    await expect(page.getByLabel('Load a location')).toBeEnabled();
    await expect(loadScenarioButton).toBeEnabled();
  } finally {
    killBackend(proc);
  }
});
