/**
 * Warning chimes, synthesised with Web Audio oscillators: no assets, no
 * dependency.
 *
 * A browser (and the Tauri webview) keeps an `AudioContext` suspended until the
 * page has seen a user gesture. `unlockAudioOnGesture` resumes it on the first
 * key or pointer press, which every driver makes before anything can warn
 * them. Where there is no Web Audio at all (jsdom), every call is a no-op.
 */

export type ChimeKind = 'info' | 'warn' | 'critical';

const PATTERNS: Record<ChimeKind, { hz: number; beeps: number }> = {
  info: { hz: 660, beeps: 1 },
  warn: { hz: 880, beeps: 2 },
  critical: { hz: 1200, beeps: 4 },
};
const BEEP_S = 0.09;
const GAP_S = 0.06;
const GAIN = 0.12;

/** How loud each kind is relative to the others, for picking one per batch. */
export const CHIME_RANK: Record<ChimeKind, number> = { info: 0, warn: 1, critical: 2 };

let ctx: AudioContext | null = null;

function context(): AudioContext | null {
  if (ctx) return ctx;
  const Ctor = globalThis.AudioContext;
  if (!Ctor) return null;
  ctx = new Ctor();
  return ctx;
}

function resume(ac: AudioContext): void {
  if (ac.state === 'suspended') ac.resume().catch(() => {});
}

export function chime(kind: ChimeKind): void {
  const ac = context();
  if (!ac) return;
  resume(ac);
  const { hz, beeps } = PATTERNS[kind];
  const t0 = ac.currentTime;
  for (let i = 0; i < beeps; i++) {
    const start = t0 + i * (BEEP_S + GAP_S);
    const osc = ac.createOscillator();
    const gain = ac.createGain();
    osc.type = 'sine';
    osc.frequency.value = hz;
    gain.gain.setValueAtTime(0, start);
    gain.gain.linearRampToValueAtTime(GAIN, start + 0.01);
    gain.gain.linearRampToValueAtTime(0, start + BEEP_S);
    osc.connect(gain).connect(ac.destination);
    osc.start(start);
    osc.stop(start + BEEP_S + 0.02);
  }
}

/** Resume audio on the first key or pointer press. Returns the remover. */
export function unlockAudioOnGesture(): () => void {
  const unlock = () => {
    const ac = context();
    if (ac) resume(ac);
  };
  window.addEventListener('keydown', unlock);
  window.addEventListener('pointerdown', unlock);
  return () => {
    window.removeEventListener('keydown', unlock);
    window.removeEventListener('pointerdown', unlock);
  };
}

/** Tests install a fake context (jsdom has none) and clear it after. */
export function setAudioContextForTest(fake: AudioContext | null): void {
  ctx = fake;
}
