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
  await page.goto('http://localhost:1420/');
  console.log('page loaded, letting the sim run...');

  // Keep the process (and page) alive; run_capture.sh polls frames on disk
  // and kills this process once the target is reached or it times out.
  await new Promise(() => {});
})();
