// Companion to run_capture.sh: drives the StreetLab frontend against a
// running backend so a capture accumulates frames. Must be Playwright, not
// the Browser pane -- the pane's tab is background-throttled to ~1
// frame/minute, which would make a 150-frame capture take hours.
//
// Deliberately does nothing scenario-specific: the backend is already
// configured with --scenario/--seed/--traffic on the command line, and the
// frontend connects to it with no query params at all (ws://127.0.0.1:8765
// is the CLI's own default, matching `npm run dev` + `streetlab serve` with
// no arguments on either side -- see src/net/wsClient.ts's
// createTransportFromLocation). Once connected, the ego drives itself
// autonomously and streams camera frames back over the same socket, which
// is what the backend's --capture sink records.
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on('pageerror', (err) => console.error('pageerror:', err.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') console.error('console error:', msg.text());
  });
  await page.goto(process.env.CAPTURE_URL || 'http://localhost:1420/');
  console.log('page loaded, letting the sim run...');

  // Optional hazard staging (M2 benchmark-hazards): CAPTURE_HAZARDS is a comma list of
  // hazard button labels ("Jaywalker,Cyclist drift"), injected in turn every
  // HAZARD_EVERY_S seconds. A declined injection is harmless: the backend acks false.
  const hazards = (process.env.CAPTURE_HAZARDS || '').split(',').map((h) => h.trim()).filter(Boolean);
  if (hazards.length) {
    const everyMs = Number(process.env.HAZARD_EVERY_S || 25) * 1000;
    for (let i = 0; ; i++) {
      await page.waitForTimeout(everyMs);
      const label = hazards[i % hazards.length];
      try {
        await page.getByRole('button', { name: label, exact: true }).click({ timeout: 5000 });
        console.log('injected', label);
      } catch (err) {
        console.error('could not inject', label, err.message);
      }
    }
  }

  // Keep the process (and page) alive; run_capture.sh polls frames on disk
  // and kills this process once the target is reached or it times out.
  await new Promise(() => {});
})();
