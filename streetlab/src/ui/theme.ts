/**
 * Design tokens. This module is the single source of truth for colour and
 * spacing: the DOM gets them as CSS custom properties (installed once at
 * startup), while the canvas widgets and the three.js scene read the same
 * literals directly. Nothing hard-codes a hex value twice.
 */

export const color = {
  /* surfaces */
  bg: '#F1F4F8',
  surface: '#FFFFFF',
  surfaceAlt: '#F7F9FC',
  surfaceSunken: '#EDF1F6',
  border: '#E4E9F0',
  borderStrong: '#D2DAE5',

  /* type */
  text: '#141D28',
  textMuted: '#5F6E80',
  textFaint: '#9AA7B6',

  /* brand */
  accent: '#0FB5C9',
  accentDark: '#0A8DA0',
  accentSoft: '#E1F6F9',
  accentRing: 'rgba(15, 181, 201, 0.28)',

  /* semantic */
  ok: '#22C55E',
  okSoft: '#E6F9EE',
  warn: '#FF7A1A',
  warnSoft: '#FFF0E4',
  danger: '#E5484D',
  dangerSoft: '#FDECEC',

  /* domain */
  plan: '#2F80ED',
  planSoft: '#E8F1FE',
  ego: '#FFFFFF',
  laneMark: '#C6D0DC',
} as const;

export const radius = {
  sm: '6px',
  md: '10px',
  lg: '14px',
  xl: '18px',
  pill: '999px',
} as const;

export const shadow = {
  card: '0 1px 2px rgba(20, 29, 40, 0.04), 0 4px 14px rgba(20, 29, 40, 0.06)',
  raised: '0 2px 6px rgba(20, 29, 40, 0.07), 0 14px 32px rgba(20, 29, 40, 0.10)',
  inset: 'inset 0 1px 2px rgba(20, 29, 40, 0.06)',
} as const;

export const font = {
  sans: `'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif`,
  mono: `'SF Mono', ui-monospace, 'JetBrains Mono', Menlo, Consolas, monospace`,
} as const;

/** Detection-class colours, shared by the 3D overlay and the lane widget. */
export const classColor: Record<string, string> = {
  car: '#5B7A99',
  truck: '#7C6BA8',
  bus: '#7C6BA8',
  motorcycle: '#4E8FA8',
  cyclist: '#4E8FA8',
  pedestrian: '#D4A03C',
  unknown: '#94A3B8',
};

/** Traffic-signal lamp colours. */
export const signalColor = {
  red: '#E5484D',
  yellow: '#F5A524',
  green: '#22C55E',
  flashing_yellow: '#F5A524',
  off: '#3A4450',
} as const;

/**
 * Sun/sky presets driven by the `time_of_day` render parameter. `ambient` is deliberately high relative to `intensity`: in a
 * light theme a shadow must read as a soft grey wash, not a black hole, so the
 * hemisphere fill carries most of the exposure and the sun supplies contrast.
 */
export const lighting = {
  morning: { sky: '#DCEAF6', horizon: '#F3EEE4', sun: '#FFE9C9', ground: '#D8D2C6', elev: 0.42, azim: 1.9, intensity: 1.35, ambient: 2.5 },
  midday: { sky: '#CFE3F5', horizon: '#EFF4F8', sun: '#FFFAF0', ground: '#D6D3CB', elev: 0.95, azim: 0.9, intensity: 1.55, ambient: 2.7 },
  golden: { sky: '#E8D9E8', horizon: '#FFE2C2', sun: '#FFD39B', ground: '#DCCBB8', elev: 0.24, azim: 2.6, intensity: 1.5, ambient: 2.35 },
  overcast: { sky: '#DEE4EA', horizon: '#E9EDF1', sun: '#F2F5F8', ground: '#D5D8DB', elev: 0.7, azim: 1.2, intensity: 0.55, ambient: 3.1 },
} as const;

export type LightingPreset = keyof typeof lighting;

/** Flatten the token set into CSS custom properties on `:root`. */
export function installThemeVars(root: HTMLElement = document.documentElement): void {
  const set = (k: string, v: string) => root.style.setProperty(k, v);
  for (const [k, v] of Object.entries(color)) set(`--c-${kebab(k)}`, v);
  for (const [k, v] of Object.entries(radius)) set(`--r-${k}`, v);
  for (const [k, v] of Object.entries(shadow)) set(`--sh-${k}`, v);
  set('--font-sans', font.sans);
  set('--font-mono', font.mono);
}

const kebab = (s: string): string =>
  s.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`);

/** `#rrggbb` -> `rgba(r, g, b, a)`, for canvas fills that need transparency. */
export function alpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/** `#rrggbb` -> `0xrrggbb`, for three.js `Color` construction. */
export const hexInt = (hex: string): number => parseInt(hex.slice(1), 16);
