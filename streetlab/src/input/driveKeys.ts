/**
 * Keyboard driving: held keys -> smoothed analog axes -> `drive_input` at ~30 Hz.
 *
 * A key is binary, a pedal is not. `shapeAxes` ramps each axis toward its
 * key's target (steer reaches full lock over 0.3 s and recentres over 0.2 s;
 * pedals over 0.25 s), so a tap is a nudge and a hold is a sweep. The backend
 * then rate- and jerk-limits on top (`plan/manual.py`): this is feel, that is
 * the guarantee.
 */
import { useEffect } from 'react';
import { useSimStore } from '../store/simStore';
import { ownsTheKey } from '../ui/HelpDialog';

export interface Axes {
  /** -1 full right .. +1 full left, matching `steering_angle`. */
  steer: number;
  throttle: number;
  brake: number;
}

export const ZERO_AXES: Axes = { steer: 0, throttle: 0, brake: 0 };

type Control = 'throttle' | 'brake' | 'left' | 'right';

const KEYS: Record<string, Control> = {
  w: 'throttle',
  arrowup: 'throttle',
  s: 'brake',
  arrowdown: 'brake',
  a: 'left',
  arrowleft: 'left',
  d: 'right',
  arrowright: 'right',
};

export const SEND_HZ = 30;
const STEER_ENGAGE_S = 0.3;
const STEER_RETURN_S = 0.2;
const PEDAL_S = 0.25;

export function controlFor(key: string): Control | undefined {
  return KEYS[key.toLowerCase()];
}

function approach(value: number, target: number, rate: number, dt: number): number {
  const step = rate * dt;
  // Snap the float residue a sum of steps leaves, so a released key reads as
  // exactly zero and the hook stops sending.
  if (Math.abs(target - value) <= step + 1e-9) return target;
  return value < target ? value + step : value - step;
}

/** One step of the held-key -> axes ramp. Pure, for tests. */
export function shapeAxes(prev: Axes, held: ReadonlySet<Control>, dt: number): Axes {
  const steerTarget = (held.has('left') ? 1 : 0) - (held.has('right') ? 1 : 0);
  // Moving away from centre engages at one rate; heading back to (or through)
  // centre recentres faster, like a wheel springing back.
  const recentring = steerTarget === 0 || Math.sign(steerTarget) !== Math.sign(prev.steer);
  const steerRate = 1 / (recentring && prev.steer !== 0 ? STEER_RETURN_S : STEER_ENGAGE_S);
  return {
    steer: approach(prev.steer, steerTarget, steerRate, dt),
    throttle: approach(prev.throttle, held.has('throttle') ? 1 : 0, 1 / PEDAL_S, dt),
    brake: approach(prev.brake, held.has('brake') ? 1 : 0, 1 / PEDAL_S, dt),
  };
}

export function isZero(a: Axes): boolean {
  return a.steer === 0 && a.throttle === 0 && a.brake === 0;
}

/** Mount once (App). Sends while any axis is live, plus one final zero frame. */
export function useDriveKeys(): void {
  useEffect(() => {
    const held = new Set<Control>();
    let axes: Axes = ZERO_AXES;
    let last = performance.now();

    const onDown = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
      const control = controlFor(e.key);
      if (!control) return;
      const s = useSimStore.getState();
      if (s.failure || s.queue || ownsTheKey(e.target, e.key)) return;
      e.preventDefault(); // arrows would scroll the page
      held.add(control);
    };
    const onUp = (e: KeyboardEvent) => {
      const control = controlFor(e.key);
      if (control) held.delete(control);
    };
    // Alt-tab with a key down never delivers its keyup.
    const onBlur = () => held.clear();

    const timer = setInterval(() => {
      const now = performance.now();
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const next = shapeAxes(axes, held, dt);
      if (!isZero(next) || !isZero(axes)) {
        useSimStore.getState().send({ cmd: 'drive_input', ...next });
      }
      axes = next;
    }, 1000 / SEND_HZ);

    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup', onUp);
    window.addEventListener('blur', onBlur);
    return () => {
      clearInterval(timer);
      window.removeEventListener('keydown', onDown);
      window.removeEventListener('keyup', onUp);
      window.removeEventListener('blur', onBlur);
    };
  }, []);
}
