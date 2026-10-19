/**
 * Gamepad driving through the browser's native Gamepad API, "standard"
 * mapping (Xbox / PlayStation layout): left stick steers, right trigger is
 * throttle, left trigger is brake. A stick is already analog, so no ramp is
 * applied here; the backend rate- and jerk-limits it like any other input.
 */
import type { Axes } from './driveKeys';

/** Radial deadzone on the stick: a resting thumb is not a steering input. */
export const STICK_DEADZONE = 0.12;
/** Triggers rest a hair above zero on some pads. */
export const TRIGGER_DEADZONE = 0.05;

/** One-shot pad buttons, by standard-mapping index. */
export type PadAction =
  | 'fsd'
  | 'assist'
  | 'cruise'
  | 'faster'
  | 'slower'
  | 'signalLeft'
  | 'signalRight';

export const PAD_BUTTONS: Record<number, PadAction> = {
  3: 'fsd', // Y / triangle
  2: 'assist', // X / square
  1: 'cruise', // B / circle
  12: 'faster', // d-pad up
  13: 'slower', // d-pad down
  4: 'signalLeft', // LB / L1
  5: 'signalRight', // RB / R1
};

/** What `navigator.getGamepads()` hands back, narrowed to what is read here. */
export interface PadLike {
  axes: readonly number[];
  buttons: readonly { pressed: boolean; value: number }[];
}

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

/** A pad's driving axes. Stick X is negated: pushing left steers left (+). */
export function padAxes(pad: PadLike): Axes {
  const x = pad.axes[0] ?? 0;
  const y = pad.axes[1] ?? 0;
  // Radial deadzone, rescaled so full deflection still reaches full lock.
  const r = Math.hypot(x, y);
  const steer =
    r < STICK_DEADZONE ? 0 : -(x / r) * Math.min(1, (r - STICK_DEADZONE) / (1 - STICK_DEADZONE));
  const trigger = (i: number) => {
    const v = pad.buttons[i]?.value ?? 0;
    return v < TRIGGER_DEADZONE ? 0 : clamp(v, 0, 1);
  };
  return { steer: clamp(steer, -1, 1), throttle: trigger(7), brake: trigger(6) };
}

/** Actions for buttons that went down since `prev`, and the new pressed state. */
export function pressedSince(
  pad: PadLike,
  prev: readonly boolean[],
): { actions: PadAction[]; now: boolean[] } {
  const now = pad.buttons.map((b) => b.pressed);
  const actions: PadAction[] = [];
  now.forEach((down, i) => {
    const action = PAD_BUTTONS[i];
    if (down && !prev[i] && action) actions.push(action);
  });
  return { actions, now };
}

/** Keyboard and pad together: per axis, whichever is pushed harder wins. */
export function mergeAxes(a: Axes, b: Axes): Axes {
  return {
    steer: Math.abs(b.steer) > Math.abs(a.steer) ? b.steer : a.steer,
    throttle: Math.max(a.throttle, b.throttle),
    brake: Math.max(a.brake, b.brake),
  };
}

/** The first connected pad, if any. */
export function firstPad(): PadLike | null {
  if (typeof navigator === 'undefined' || !navigator.getGamepads) return null;
  for (const pad of navigator.getGamepads()) if (pad && pad.connected) return pad;
  return null;
}
