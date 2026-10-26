/**
 * Tesla-style driver warnings, derived on the client from each frame.
 *
 * Every warning here is a function of fields the wire already carries (TTC,
 * the plan label, speed against the posted limit, the drive mode), so none of
 * them needs a backend change or a protocol bump. `WarningMonitor` is pure
 * edge detection; `useWarningEffects` is the one subscription that chimes and
 * logs each new warning as a client event, which `AckToast` and the Events tab
 * already show. The scorecard never sees them. `ponytail:` client events, not
 * sim events; upgrade is a backend `_emit` per warning (event codes are free
 * strings, still no bump) when the scorecard should count them.
 *
 * Thresholds Tesla does not publish are sim guesses, named here and listed in
 * docs/measurements.
 */

import { useEffect } from 'react';
import type { ParamValue, StateUpdate } from '../schema';
import { frameBus, useSimStore } from '../store/simStore';
import { toMph } from '../units';
import { CHIME_RANK, type ChimeKind, chime, unlockAudioOnGesture } from './chime';

export interface Warning {
  code: string;
  level: 'info' | 'warn' | 'critical';
  message: string;
  sound: ChimeKind | null;
}

type Params = Record<string, ParamValue>;

/** Forward Collision Warning: TTC per sensitivity. Tesla publishes none;
 * Medium matches the toolbar's critical TTC (2.5 s). */
export const FCW_TTC_S: Record<string, number> = { late: 2.0, medium: 2.5, early: 3.0 };
/** Tesla's FCW and AEB window starts at 5 km/h. */
const FCW_MIN_MPS = 1.4;
/** The driver (or planner) is already braking: FCW keeps quiet. */
const BRAKING = 0.3;
/** Stop warning: this far over the speed the planner would be holding for
 * the stop line counts as "likely to run it". Unpublished; a sim guess. */
const STOP_WARN_OVER_MPS = 3;
const STOPPED_MPS = 0.3;
/** Over the limit this long before Speed Assist reacts, so the one-frame
 * limit dip at junction service roads cannot trip it. */
const SPEED_HOLD_S = 1;

const DRIVER_MODES = new Set(['off', 'cruise']);
const ENGAGED_MODES = new Set(['autosteer', 'fsd']);

/** Whether the car is over the Speed Assist threshold (relative or absolute). */
export function overSpeedLimit(frame: StateUpdate, params: Params): boolean {
  if ((params.speed_limit_warning ?? 'display') === 'off') return false;
  const mph = toMph(Math.abs(frame.ego.speed_mps));
  if (params.speed_limit_mode === 'absolute') {
    return mph > Number(params.speed_limit_absolute_mph ?? 70);
  }
  if (frame.ego.speed_limit_mps <= 0) return false; // no limit known: no sign, no warning
  return mph > toMph(frame.ego.speed_limit_mps) + Number(params.speed_limit_offset_mph ?? 5);
}

export class WarningMonitor {
  private active = new Set<string>();
  private prevManeuver: string | null = null;
  private overSince: number | null = null;
  private lastT = -Infinity;

  /** The warnings that started on this frame. */
  update(frame: StateUpdate, params: Params): Warning[] {
    if (frame.t < this.lastT) this.reset();
    this.lastT = frame.t;

    const now = this.conditions(frame, params);
    const fresh = now.filter((w) => !this.active.has(w.code));
    this.active = new Set(now.map((w) => w.code));

    // One-shot: the stop the car was waiting at has released.
    const { ego, plan } = frame;
    if (
      params.green_light_chime === 'on' &&
      DRIVER_MODES.has(ego.cruise.mode) &&
      this.prevManeuver === 'stop' &&
      plan.maneuver !== 'stop' &&
      Math.abs(ego.speed_mps) <= STOPPED_MPS
    ) {
      fresh.push({ code: 'green_light', level: 'info', message: 'Clear to go', sound: 'info' });
    }
    this.prevManeuver = plan.maneuver;
    return fresh;
  }

  /** Every warning whose condition holds on this frame. */
  private conditions(frame: StateUpdate, params: Params): Warning[] {
    const { ego, plan, telemetry } = frame;
    const mode = ego.cruise.mode;
    const speed = Math.abs(ego.speed_mps);
    const out: Warning[] = [];

    const ttcLimit = FCW_TTC_S[String(params.fcw_sensitivity ?? 'medium')];
    const ttc = telemetry.ttc_s;
    if (ttcLimit !== undefined && ttc !== null && ttc <= ttcLimit && speed > FCW_MIN_MPS && ego.brake < BRAKING) {
      const hazard = frame.detections.find((d) => d.hazard);
      out.push({
        code: 'fcw',
        level: 'critical',
        message: hazard?.hazard_label ?? 'Forward collision warning',
        sound: 'critical',
      });
    }

    if (plan.maneuver === 'emergency_brake') {
      out.push({ code: 'aeb', level: 'critical', message: 'Automatic emergency braking', sound: 'critical' });
    }

    if (
      DRIVER_MODES.has(mode) &&
      plan.maneuver === 'stop' &&
      speed > plan.target_speed_mps + STOP_WARN_OVER_MPS &&
      ego.brake < BRAKING
    ) {
      out.push({ code: 'stop_warning', level: 'critical', message: 'Stop ahead: take corrective action', sound: 'critical' });
    }

    if (overSpeedLimit(frame, params)) {
      this.overSince ??= frame.t;
      if (params.speed_limit_warning === 'chime' && frame.t - this.overSince >= SPEED_HOLD_S) {
        out.push({ code: 'speed_limit', level: 'info', message: 'Over the speed limit', sound: 'info' });
      }
    } else {
      this.overSince = null;
    }

    if (ENGAGED_MODES.has(mode) && !ego.cruise.fsd_available) {
      out.push({ code: 'take_over', level: 'critical', message: 'Take over immediately', sound: 'critical' });
    }

    return out;
  }

  reset(): void {
    this.active.clear();
    this.prevManeuver = null;
    this.overSince = null;
  }
}

/** The loudest chime among new warnings: one sound per frame, not a chord. */
export function loudest(warnings: Warning[]): ChimeKind | null {
  let best: ChimeKind | null = null;
  for (const w of warnings) {
    if (w.sound && (best === null || CHIME_RANK[w.sound] > CHIME_RANK[best])) best = w.sound;
  }
  return best;
}

/** Mount once (App): watches the frame stream, chimes and logs new warnings. */
export function useWarningEffects(): void {
  useEffect(() => {
    const monitor = new WarningMonitor();
    const offUnlock = unlockAudioOnGesture();
    const off = frameBus.subscribe((frame) => {
      const { params, pushClientEvents } = useSimStore.getState();
      const fresh = monitor.update(frame, params);
      if (!fresh.length) return;
      pushClientEvents(
        fresh.map((w) => ({ t: frame.t, level: w.level, code: w.code, message: w.message, progress: null })),
      );
      const sound = loudest(fresh);
      if (sound && params.chimes !== 'off') chime(sound);
    });
    return () => {
      off();
      offUnlock();
    };
  }, []);
}
