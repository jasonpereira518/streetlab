/**
 * Driver attention monitoring, Tesla-style, while Assist or FSD drives.
 *
 * Tesla watches the driver with a cabin camera and, failing that, wheel
 * torque. The sim's stand-in is the person's own input: any key, click, wheel
 * or gamepad movement counts as attention. Ignored long enough, the prompts
 * escalate; then FSD drops to Assist (Autosteer), which keeps the lane and
 * stops for lights while the set speed is walked down to a stop with the
 * hazards on, and Assist/FSD stay locked out until the scene is reset (one
 * strike per drive; reset forgives it). Tesla's stage timings are unpublished,
 * so these are named sim guesses.
 *
 * It runs on the client and drives the car with commands that already exist
 * (`set_drive_mode` with a set speed), so headless runs, tests and contract
 * fixtures never see it. Off by default: the self-serve demo is someone
 * watching FSD drive, and a nag would stop the car on them. `ponytail:` the
 * backend sees only a driver lowering the set speed; upgrade is backend
 * `attention` events if the scorecard should count strikes.
 */

import { useEffect } from 'react';
import type { CruiseMode, ParamValue, StateUpdate } from '../schema';
import { frameBus, useSimStore } from '../store/simStore';
import { chime, type ChimeKind } from './chime';
import type { Warning } from './warnings';

export type AttentionStage = 'ok' | 'prompt' | 'escalated' | 'stopping';

/** Seconds of no input before the prompt, the escalation and the strike. */
export const ATTENTION_TIMERS: Record<string, readonly [number, number, number]> = {
  normal: [30, 40, 50],
  demo: [8, 12, 16],
};
/** Repeat chimes this often once escalated. */
const REPEAT_S = 2;
/** The stop: every this many seconds the set speed drops by this much. */
const SLOW_EVERY_S = 2;
const SLOW_BY_MPS = 2.2;

const ENGAGED: ReadonlySet<CruiseMode> = new Set(['autosteer', 'fsd']);

export interface AttentionStep {
  stage: AttentionStage;
  locked: boolean;
  events: Warning[];
  sound: ChimeKind | null;
  /** `set_drive_mode` commands to send, in order. */
  commands: Array<{ mode: CruiseMode; set_speed_mps: number }>;
}

export class AttentionMonitor {
  stage: AttentionStage = 'ok';
  locked = false;
  private lastActiveT = -Infinity;
  private lastChimeT = -Infinity;
  private lastStepT = -Infinity;
  private lastT = -Infinity;

  update(frame: StateUpdate, params: Record<string, ParamValue>, active: boolean): AttentionStep {
    const out: AttentionStep = { stage: this.stage, locked: this.locked, events: [], sound: null, commands: [] };
    const t = frame.t;
    if (t < this.lastT) this.reset(); // the scene was reset: the strike is forgiven
    this.lastT = t;
    if (this.lastActiveT === -Infinity) this.lastActiveT = t; // the clock starts now

    const timers = ATTENTION_TIMERS[String(params.attention_monitor ?? 'off')];
    const { cruise, speed_mps } = frame.ego;
    const engaged = ENGAGED.has(cruise.mode);
    if (!timers) {
      this.stage = 'ok';
      this.lastActiveT = t;
      return this.result(out);
    }
    const [promptS, escalateS, strikeS] = timers;

    if (this.stage === 'stopping') {
      // Only a real takeover (the ladder: steer or brake) ends the stop.
      if (!engaged) {
        this.stage = 'ok';
        this.lastActiveT = t;
      }
    } else if (active || !engaged) {
      this.stage = 'ok';
      this.lastActiveT = t;
    }

    const idle = t - this.lastActiveT;
    if (this.stage !== 'stopping' && engaged) {
      if (idle >= strikeS) {
        this.stage = 'stopping';
        this.locked = true;
        out.events.push({
          code: 'attention_strike',
          level: 'critical',
          message: 'Driver attention warnings ignored: Assist and FSD are off for this drive',
          sound: 'critical',
        });
        out.sound = 'critical';
        this.lastChimeT = t;
        this.lastStepT = t;
        out.commands.push({ mode: 'autosteer', set_speed_mps: Math.max(0, speed_mps - SLOW_BY_MPS) });
      } else if (idle >= escalateS) {
        if (this.stage !== 'escalated') {
          this.stage = 'escalated';
          this.lastChimeT = -Infinity;
        }
        if (t - this.lastChimeT >= REPEAT_S) {
          out.sound = 'warn';
          this.lastChimeT = t;
        }
      } else if (idle >= promptS && this.stage === 'ok') {
        this.stage = 'prompt';
        out.events.push({
          code: 'attention',
          level: 'warn',
          message: 'Please pay attention to the road',
          sound: 'warn',
        });
        out.sound = 'warn';
      }
    } else if (this.stage === 'stopping') {
      // Continuous chime, and the set speed walked down to a stop.
      if (t - this.lastChimeT >= REPEAT_S) {
        out.sound = 'critical';
        this.lastChimeT = t;
      }
      if (t - this.lastStepT >= SLOW_EVERY_S && cruise.set_speed_mps > 0) {
        this.lastStepT = t;
        out.commands.push({ mode: 'autosteer', set_speed_mps: Math.max(0, cruise.set_speed_mps - SLOW_BY_MPS) });
      }
    }
    return this.result(out);
  }

  private result(out: AttentionStep): AttentionStep {
    out.stage = this.stage;
    out.locked = this.locked;
    return out;
  }

  reset(): void {
    this.stage = 'ok';
    this.locked = false;
    this.lastActiveT = -Infinity;
    this.lastChimeT = -Infinity;
    this.lastStepT = -Infinity;
  }
}

/** Driving input the key/pad sender saw (not every key reaches `window`). */
let activeSinceFrame = false;
export function markDriverActive(): void {
  activeSinceFrame = true;
}

/** Mount once (App): the attention monitor, its banner state and its commands. */
export function useAttentionEffects(): void {
  useEffect(() => {
    const monitor = new AttentionMonitor();
    const onInput = () => markDriverActive();
    const inputs = ['keydown', 'pointerdown', 'wheel'] as const;
    for (const e of inputs) window.addEventListener(e, onInput);
    const off = frameBus.subscribe((frame) => {
      const store = useSimStore.getState();
      const active = activeSinceFrame;
      activeSinceFrame = false;
      const step = monitor.update(frame, store.params, active);
      if (step.stage !== store.attention.stage || step.locked !== store.attention.locked) {
        store.setAttention({ stage: step.stage, locked: step.locked });
      }
      if (step.events.length) {
        store.pushClientEvents(
          step.events.map((w) => ({ t: frame.t, level: w.level, code: w.code, message: w.message, progress: null })),
        );
      }
      // Straight to `send`: `setDriveMode` refuses Assist/FSD once locked.
      for (const c of step.commands) store.send({ cmd: 'set_drive_mode', ...c });
      if (step.sound && store.params.chimes !== 'off') chime(step.sound);
    });
    return () => {
      off();
      for (const e of inputs) window.removeEventListener(e, onInput);
    };
  }, []);
}
