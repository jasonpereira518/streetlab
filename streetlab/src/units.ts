/** Unit conversion + small scalar helpers shared by the UI and the renderer. */

export const MPS_TO_MPH = 2.236936292054402;
export const MPH_TO_MPS = 1 / MPS_TO_MPH;
export const RAD_TO_DEG = 180 / Math.PI;
export const DEG_TO_RAD = Math.PI / 180;

export const toMph = (mps: number): number => mps * MPS_TO_MPH;
export const toMps = (mph: number): number => mph * MPH_TO_MPS;

/** "32" — integer mph, the form the toolbar and speedometer render. */
export const formatMph = (mps: number): string =>
  String(Math.max(0, Math.round(toMph(mps))));

/** "2.4s", or "—" when nothing is closing. */
export function formatTtc(ttc: number | null | undefined): string {
  if (ttc == null || !Number.isFinite(ttc) || ttc > 99) return '—';
  return `${ttc.toFixed(1)}s`;
}

export const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

export const invLerp = (a: number, b: number, v: number): number =>
  a === b ? 0 : (v - a) / (b - a);

/**
 * Frame-rate independent exponential smoothing.
 * `smoothing` is the fraction of the remaining gap left after one second.
 */
export function damp(
  current: number,
  target: number,
  smoothing: number,
  dt: number,
): number {
  return lerp(target, current, Math.pow(smoothing, dt));
}

/** Shortest signed angular difference `a - b`, wrapped to (-pi, pi]. */
export function angleDelta(a: number, b: number): number {
  let d = (a - b) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d <= -Math.PI) d += Math.PI * 2;
  return d;
}

/** Angle-aware damping — interpolates the short way around the circle. */
export function dampAngle(
  current: number,
  target: number,
  smoothing: number,
  dt: number,
): number {
  return current + angleDelta(target, current) * (1 - Math.pow(smoothing, dt));
}

/** Deterministic 32-bit PRNG so the mock world is identical every run. */
export function makeRng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 0x100000000;
  };
}
