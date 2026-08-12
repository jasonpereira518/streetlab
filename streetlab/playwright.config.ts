import { defineConfig, devices } from '@playwright/test';

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
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan,WebGPU',
            '--use-angle=default',
          ],
        },
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
