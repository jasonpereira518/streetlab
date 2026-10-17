import { defineConfig, devices } from '@playwright/test';

// Locally Chromium gets WebGPU so the primary render path is exercised.
const LOCAL_CHROMIUM_ARGS = [
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan,WebGPU',
  '--use-angle=default',
];

// A CI runner has no GPU. Force-enabling WebGPU there hands the app a software
// Vulkan adapter that dies mid-test ("Instance dropped in popErrorScope"), so
// CI leaves WebGPU off and renders through SwiftShader-backed WebGL2 -- the
// fallback the app takes on its own, which the tests already accept.
const CI_CHROMIUM_ARGS = [
  '--use-gl=angle',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
];

/**
 * End-to-end smoke tests against the real Vite build.
 *
 * Chromium is launched with WebGPU enabled so the primary path gets exercised;
 * if the CI machine has no GPU the app falls back to WebGL2 on its own, and the
 * tests assert whichever backend actually came up.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 45_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'line' : [['list']],
  use: {
    baseURL: 'http://localhost:1420',
    viewport: { width: 1440, height: 900 },
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        // `devices['Desktop Chrome']` carries its own `viewport` (1280x720),
        // and a project's `use` overrides the config's, so spreading it here
        // silently discarded the 1440x900 above -- the size the Tauri window
        // actually opens at (`tauri.conf.json`). Every e2e test had therefore
        // only ever run at 1280, which is BELOW the 1320 px breakpoint in
        // `styles.css`: the whole suite, including `shell.spec.ts`'s layout
        // contract, was exercising the narrow layout and calling it the
        // default. Restate it after the spread so the intent survives.
        viewport: { width: 1440, height: 900 },
        launchOptions: { args: process.env.CI ? CI_CHROMIUM_ARGS : LOCAL_CHROMIUM_ARGS },
      },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:1420',
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
