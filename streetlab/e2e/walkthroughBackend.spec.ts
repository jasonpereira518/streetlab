/**
 * Walkthrough presets against a real `streetlab serve` (synthetic grid):
 * Run the shortest preset, wait for its run_summary to land in the
 * scorecard, then Replay and check the run is back on the same seed.
 * The sim runs in real time, so this takes at least the preset's duration.
 */
import { spawn } from "node:child_process";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const HERE = dirname(fileURLToPath(import.meta.url));
const BACKEND_DIR = resolve(HERE, "../../streetlab-backend");

// ponytail: copied from faultInjection.spec.ts; extract a shared helper if a third spec needs it.
function spawnBackend(): Promise<{
  proc: ChildProcessWithoutNullStreams;
  ws: string;
}> {
  return new Promise((resolvePromise, reject) => {
    const proc = spawn("uv", ["run", "streetlab", "serve", "--port", "0"], {
      cwd: BACKEND_DIR,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const rl = createInterface({ input: proc.stdout });
    const timeout = setTimeout(() => {
      rl.close();
      proc.kill("SIGKILL");
      reject(new Error("backend did not print STREETLAB_READY within 60s"));
    }, 60_000);
    rl.on("line", (line) => {
      if (!line.startsWith("STREETLAB_READY ")) return;
      clearTimeout(timeout);
      rl.close();
      resolvePromise({
        proc,
        ws: JSON.parse(line.slice("STREETLAB_READY ".length)).ws,
      });
    });
    proc.on("error", reject);
    proc.on("exit", (code) => {
      if (code !== null && code !== 0) {
        clearTimeout(timeout);
        reject(new Error(`backend exited early with code ${code}`));
      }
    });
  });
}

test("a real preset run lands a scorecard and Replay reuses its seed", async ({
  page,
}) => {
  // The shortest real preset is 90 s of real-time sim; leave room for a loaded machine.
  test.setTimeout(300_000);
  const { proc, ws } = await spawnBackend();

  try {
    page.on("pageerror", (err) => {
      throw new Error(`uncaught page error: ${err.message}`);
    });
    // Keep only frames that carry a preset_loaded event (backend message
    // "<id>: seed N"); the full telemetry stream is too big to buffer.
    const frames: string[] = [];
    page.on("websocket", (sock) =>
      sock.on("framereceived", ({ payload }) => {
        const text = String(payload);
        if (text.includes("preset_loaded")) frames.push(text);
      }),
    );
    await page.goto(`/?backend=${encodeURIComponent(ws)}`);

    const cards = page
      .getByRole("list", { name: "Walkthrough presets" })
      .getByRole("listitem");
    await expect(cards.first()).toBeVisible({ timeout: 30_000 });

    // Pick the card with the smallest duration rather than hard-coding an id.
    const durations = await cards
      .getByTestId("preset-duration")
      .allTextContents();
    const secs = durations.map((d) => parseFloat(d));
    const shortest = cards.nth(secs.indexOf(Math.min(...secs)));
    const title = (await shortest.locator("h3.scenario-name").textContent())!;

    await page.getByRole("button", { name: `Run ${title}` }).click();
    await expect(page.getByRole("tab", { name: "Run" })).toHaveAttribute(
      "aria-selected",
      "true",
    );

    // The seed comes from the scorecard column (the run_summary itself), not
    // from run-seed, which shows the previous scene's seed until the ack lands.
    const scorecard = page.getByRole("table", { name: "Scorecard" });
    await expect(scorecard).toBeVisible({
      timeout: (Math.min(...secs) + 120) * 1000,
    });
    const header = (await scorecard
      .getByRole("columnheader")
      .nth(1)
      .textContent())!;
    const seed = header.replace("seed ", "");
    await expect(page.getByTestId("run-seed")).toHaveText(seed);

    // Watch the wire for the backend confirming the replay on that seed:
    // run-seed alone would already read the right number before the click.
    frames.length = 0;
    const replay = page.getByRole("button", { name: `Replay ${title}` });
    await expect(replay).toBeEnabled();
    await replay.click();
    await expect
      .poll(() => frames.some((f) => f.includes(`: seed ${seed}`)), {
        timeout: 30_000,
      })
      .toBe(true);
    await expect(page.getByTestId("run-seed")).toHaveText(seed);
  } finally {
    if (proc.exitCode === null && proc.signalCode === null)
      proc.kill("SIGKILL");
  }
});
