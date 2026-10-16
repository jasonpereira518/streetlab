import { describe, expect, it } from 'vitest';
import { backendUrlProblem } from '../buildGuard';

describe('backendUrlProblem', () => {
  it('fails a Vercel production build with no backend URL', () => {
    expect(backendUrlProblem({ VERCEL_ENV: 'production' })).toMatch(/VITE_BACKEND_WS_URL is not set/);
    expect(backendUrlProblem({ VERCEL_ENV: 'production', VITE_BACKEND_WS_URL: '  ' })).toMatch(
      /not set/,
    );
  });

  it('requires wss:// in production and a ws(s) scheme everywhere', () => {
    expect(
      backendUrlProblem({ VERCEL_ENV: 'production', VITE_BACKEND_WS_URL: 'ws://x.fly.dev' }),
    ).toMatch(/wss:\/\//);
    expect(backendUrlProblem({ VITE_BACKEND_WS_URL: 'https://x.fly.dev' })).toMatch(/ws:\/\/ or wss/);
  });

  it('lets production with wss://, previews, local and Tauri builds through', () => {
    expect(
      backendUrlProblem({ VERCEL_ENV: 'production', VITE_BACKEND_WS_URL: 'wss://x.fly.dev' }),
    ).toBeNull();
    expect(backendUrlProblem({ VERCEL_ENV: 'preview' })).toBeNull();
    expect(backendUrlProblem({})).toBeNull();
    expect(backendUrlProblem({ TAURI_ENV_PLATFORM: 'darwin' })).toBeNull();
  });
});
