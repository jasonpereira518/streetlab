/**
 * Fail a hosted production build that would ship a page pointing at nothing.
 *
 * `VITE_BACKEND_WS_URL` is baked in at build time. Unset, the page falls back
 * to ws://127.0.0.1:8765 -- which on a stranger's machine is their own
 * localhost, so the deployment "works" in every check except the one that
 * matters. Only Vercel production builds are gated: the Tauri bundle, local
 * `npm run build` and preview deployments legitimately run without it.
 */
export function backendUrlProblem(env: Record<string, string | undefined>): string | null {
  const url = (env.VITE_BACKEND_WS_URL ?? '').trim();
  const production = env.VERCEL_ENV === 'production';

  if (url && !/^wss?:\/\//i.test(url)) {
    return `VITE_BACKEND_WS_URL must start with ws:// or wss:// (got "${url}").`;
  }
  if (production && !url) {
    return (
      'VITE_BACKEND_WS_URL is not set for this Vercel production build. ' +
      'Without it the deployed page tries ws://127.0.0.1:8765 on every visitor\'s own machine. ' +
      'Set it in Vercel -> Project Settings -> Environment Variables (Production) to the ' +
      'backend\'s wss:// URL, e.g. wss://streetlab-sim.fly.dev, then redeploy.'
    );
  }
  if (production && !/^wss:\/\//i.test(url)) {
    return `VITE_BACKEND_WS_URL must be wss:// for a production (https) page; browsers block ws:// from https (got "${url}").`;
  }
  return null;
}
