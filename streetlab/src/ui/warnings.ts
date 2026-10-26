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

/** Lane Departure Warning arms above this. Tesla's 64 km/h floor would never
 * arm on a 25 mph Nob Hill street; `ponytail:` a sim floor (10 mph), not
 * Tesla's figure. */
export const LDW_MIN_MPS = 4.5;
/** Back this far inside the line before a departure warning can re-arm. */
const LDW_HYSTERESIS_M = 0.25;
/** Blind-spot zone in the ego's frame, metres: from 8 m behind the centre to
 * 1 m ahead of it, one lane either side. Tesla's is unpublished. */
const BSW_BEHIND_M = 8;
const BSW_AHEAD_M = 1;
const BSW_INNER_M = 1.2;
const BSW_OUTER_M = 5.4;
/** Tesla's blind-spot monitor works above about 7 mph. */
const BSW_MIN_MPS = 3;
/** Slower than this a car beside you is parked, not passing. */
const BSW_MOVING_MPS = 0.5;

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

/** Which blind spots hold a moving vehicle. Ground truth (`world_agents`):
 * the driving feed deliberately cannot see these sectors, so this stands in
 * for Tesla's side sensors. `ponytail:` no occlusion, and an ego-frame box
 * ignores road curvature (a few false frames on a tight corner). */
export function blindSpots(frame: StateUpdate): { left: boolean; right: boolean } {
  const out = { left: false, right: false };
  const { ego } = frame;
  if (Math.abs(ego.speed_mps) < BSW_MIN_MPS) return out;
  const c = Math.cos(ego.pose.heading);
  const sn = Math.sin(ego.pose.heading);
  for (const a of frame.world_agents) {
    if (a.cls === 'pedestrian' || Math.abs(a.speed_mps) < BSW_MOVING_MPS) continue;
    const rx = a.pose.x - ego.pose.x;
    const ry = a.pose.y - ego.pose.y;
    if (rx * rx + ry * ry > 100) continue; // cheap reject: nothing past 10 m
    const along = c * rx + sn * ry;
    const side = -sn * rx + c * ry; // + = left
    if (along < -BSW_BEHIND_M || along > BSW_AHEAD_M) continue;
    if (Math.abs(side) < BSW_INNER_M || Math.abs(side) > BSW_OUTER_M) continue;
    if (side > 0) out.left = true;
    else out.right = true;
  }
  return out;
}

export class WarningMonitor {
  private active = new Set<string>();
  private prevManeuver: string | null = null;
  private overSince: number | null = null;
  private ldwSide: 'left' | 'right' | null = null;
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

    // Where the planner's braking is the car's. In Manual the backend's own
    // `aeb` event announces it, and only when it actually acts.
    const plannerBrakes = ENGAGED_MODES.has(mode) || (mode === 'cruise' && ego.cruise.fsd_available);
    if (plannerBrakes && plan.maneuver === 'emergency_brake') {
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

    const ldw = this.laneDeparture(frame, params);
    if (ldw) {
      out.push({ code: 'lane_departure', level: 'warn', message: `Drifting ${ldw}`, sound: 'warn' });
    }

    // Blind spot: only a warning once the indicator points into an occupied
    // side (Tesla's blinking light); the steady light is the HUD pill.
    const spots = blindSpots(frame);
    for (const side of ['left', 'right'] as const) {
      if (spots[side] && ego.turn_signal === side) {
        out.push({
          code: 'blind_spot',
          level: 'warn',
          message: `Vehicle in the ${side} blind spot`,
          sound: params.blind_spot_chime === 'on' ? 'warn' : null,
        });
      }
    }

    // An emergency vehicle: the planner pulls over for one behind (FSD,
    // Assist, on-route Cruise); anyone else is told it is there.
    if (plannerBrakes && plan.maneuver === 'pull_over') {
      out.push({ code: 'pull_over', level: 'warn', message: 'Pulling over for an emergency vehicle', sound: 'warn' });
    } else if (frame.detections.some((d) => d.emergency)) {
      out.push({ code: 'emergency_vehicle', level: 'info', message: 'Emergency vehicle nearby', sound: 'info' });
    }

    if (ENGAGED_MODES.has(mode) && !ego.cruise.fsd_available) {
      out.push({ code: 'take_over', level: 'critical', message: 'Take over immediately', sound: 'critical' });
    }

    return out;
  }

  /** The side a front wheel has crossed an unsignalled marking on, if any. */
  private laneDeparture(frame: StateUpdate, params: Params): 'left' | 'right' | null {
    const { ego, telemetry } = frame;
    const lane = telemetry.lane;
    if (
      (params.lane_departure_avoidance ?? 'warning') === 'off' ||
      !DRIVER_MODES.has(ego.cruise.mode) ||
      Math.abs(ego.speed_mps) < LDW_MIN_MPS
    ) {
      this.ldwSide = null;
      return null;
    }
    // A front wheel on the line: half a lane less half the car.
    const edge = lane.lane_width_m / 2 - ego.size.width / 2;
    const side = lane.offset_m > 0 ? 'left' : 'right';
    const marking = side === 'left' ? lane.left_marking : lane.right_marking;
    const held = this.ldwSide !== null && Math.abs(lane.offset_m) > edge - LDW_HYSTERESIS_M;
    const crossing = Math.abs(lane.offset_m) > edge && marking !== 'none';
    this.ldwSide = (held || crossing) && ego.turn_signal !== side ? (held ? this.ldwSide : side) : null;
    return this.ldwSide;
  }

  reset(): void {
    this.active.clear();
    this.prevManeuver = null;
    this.overSince = null;
    this.ldwSide = null;
  }
}

/** Backend events that sound like the client's warnings: interventions the
 * sim made while a person drives, and the incidents themselves. */
export const EVENT_SOUNDS: Record<string, ChimeKind> = {
  aeb: 'critical',
  obstacle_aware: 'warn',
  lane_assist: 'warn',
  collision_evasion: 'critical',
  collision: 'critical',
  off_road: 'warn',
};

/** The loudest chime among new warnings: one sound per frame, not a chord. */
export function loudest(sounds: Array<ChimeKind | null>): ChimeKind | null {
  let best: ChimeKind | null = null;
  for (const s of sounds) {
    if (s && (best === null || CHIME_RANK[s] > CHIME_RANK[best])) best = s;
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
      if (fresh.length) {
        pushClientEvents(
          fresh.map((w) => ({ t: frame.t, level: w.level, code: w.code, message: w.message, progress: null })),
        );
      }
      const sound = loudest([
        ...fresh.map((w) => w.sound),
        ...frame.events.map((e) => EVENT_SOUNDS[e.code] ?? null),
      ]);
      if (sound && params.chimes !== 'off') chime(sound);
    });
    return () => {
      off();
      offUnlock();
    };
  }, []);
}
