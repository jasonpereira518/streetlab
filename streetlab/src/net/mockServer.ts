/**
 * In-process mock simulator.
 *
 * Emits one `SceneDescription` on connect, then streams `StateUpdate` at a
 * fixed 60 Hz using an accumulator so the rate is independent of the display
 * refresh. Everything it produces is validated against the same zod schema the
 * real backend will be held to — a shape mismatch fails loudly here rather than
 * silently in the renderer.
 */
import {
  PROTOCOL_VERSION,
  StateUpdateSchema,
  SceneDescriptionSchema,
  formatIssues,
} from '../schema';
import type {
  Ack,
  Command,
  Detection,
  LaneNeighbor,
  RadarPoint,
  SceneDescription,
  SignalPhase,
  SignalState,
  SimEvent,
  StateUpdate,
  TrajectorySample,
  Vec2,
} from '../schema';
import { clamp, makeRng } from '../units';
import type { Transport, TransportHandlers } from './transport';
import { makeRectRoute, Route } from './route';
import {
  EGO_LANE_INSET,
  KERB_LANE_INSET,
  LANE_W,
  LOOP_BLOCK,
  SCENARIOS,
  STREETS,
  buildScene,
  halfWidth,
  signalGroup,
} from './mockCity';

const SIM_HZ = 60;
const DT = 1 / SIM_HZ;
const MAX_CATCHUP_STEPS = 4;

const EGO_SIZE = { length: 4.9, width: 1.96, height: 1.44 };
const CAR_SIZE = { length: 4.6, width: 1.9, height: 1.46 };
const TRUCK_SIZE = { length: 8.2, width: 2.5, height: 3.1 };

const WHEELBASE = 2.96;
const MAX_LAT_ACCEL = 2.6;
const MAX_ACCEL = 2.1;
const MAX_DECEL = -4.4;

/** Signal cycle, seconds. */
const CYCLE = 32;
const NS_GREEN_END = 13;
const NS_YELLOW_END = 16;
const EW_GREEN_END = 29;

/** How often the scripted cut-in fires, seconds. */
const DEFAULT_CUTIN_PERIOD = 22;

/* ------------------------------------------------------------------ */

interface Agent {
  id: string;
  cls: Detection['cls'];
  size: { length: number; width: number; height: number };
  route: Route;
  s: number;
  speed: number;
  /** Current lateral offset from its route centreline, + = left. */
  lateral: number;
  targetLateral: number;
  role: 'lead' | 'cutin' | 'oncoming';
}

interface Params {
  cutin_period_s: number;
  traffic_speed_scale: number;
  assist_enabled: boolean;
  ego_speed_cap_mph: number;
}

const DEFAULT_PARAMS: Params = {
  cutin_period_s: DEFAULT_CUTIN_PERIOD,
  traffic_speed_scale: 1,
  assist_enabled: true,
  ego_speed_cap_mph: 45,
};

interface ControlPoint {
  s: number;
  kind: 'signal' | 'stop';
  /** Signal id prefix for phase lookup, e.g. "tl_0_0". */
  key: string;
  /** Which phase group governs ego's approach. */
  group: 'ns' | 'ew';
  /** Phase offset for this intersection. */
  offset: number;
}

/**
 * Normalize a `load_location` query. schema.ts's `query: z.string().min(1)`
 * lets a whitespace-only string like `"   "` through; trimming is what
 * decides whether there's actually anything to build, so both `apply()`
 * (the ack) and `createMockTransport`'s `send()` (the relabelled scene) use
 * this one function rather than each trimming `command.query` themselves.
 */
function normalizeLocationQuery(raw: string): string {
  return raw.trim();
}

/* ------------------------------------------------------------------ */
/* Simulator                                                           */
/* ------------------------------------------------------------------ */

export class MockSim {
  scene: SceneDescription;
  private route: Route;
  private agents: Agent[] = [];
  private controls: ControlPoint[] = [];
  private clearedStop = new Set<number>();
  private stopHold = 0;

  t = 0;
  seq = 0;
  paused = false;
  private params: Params = { ...DEFAULT_PARAMS };

  private egoS = 0;
  private egoSpeed = 0;
  private egoAccel = 0;
  private egoLatOffset = 0;
  private planLatTarget = 0;
  private planLat = 0;
  private steering = 0;
  private battery = 78;

  /** Cut-in scripting. */
  private nextCutinAt = 8;
  private cutinPhase: 'idle' | 'approach' | 'merging' | 'settled' = 'idle';
  private cutinTimer = 0;

  private pendingEvents: SimEvent[] = [];
  private latHistory: Array<{ t: number; ego: number; cutin: number | null }> = [];
  private rng = makeRng(0x2f81b3);

  constructor(scenarioId = SCENARIOS[0].id) {
    this.scene = buildScene(scenarioId);
    this.route = makeRectRoute(
      LOOP_BLOCK.x0 + EGO_LANE_INSET,
      LOOP_BLOCK.y0 + EGO_LANE_INSET,
      LOOP_BLOCK.x1 - EGO_LANE_INSET,
      LOOP_BLOCK.y1 - EGO_LANE_INSET,
      10,
      true,
    );
    this.controls = this.findControlPoints();
    this.resetDynamics();
  }

  /* ---------------- setup ---------------- */

  private resetDynamics(): void {
    this.t = 0;
    this.seq = 0;
    this.egoS = 20;
    this.egoSpeed = 8;
    this.egoAccel = 0;
    this.egoLatOffset = 0;
    this.planLat = 0;
    this.planLatTarget = 0;
    this.steering = 0;
    this.battery = 78;
    this.nextCutinAt = 8;
    this.cutinPhase = 'idle';
    this.cutinTimer = 0;
    this.clearedStop.clear();
    this.stopHold = 0;
    this.latHistory = [];
    this.pendingEvents = [];

    const oncoming = makeRectRoute(
      LOOP_BLOCK.x0 - EGO_LANE_INSET,
      LOOP_BLOCK.y0 - EGO_LANE_INSET,
      LOOP_BLOCK.x1 + EGO_LANE_INSET,
      LOOP_BLOCK.y1 + EGO_LANE_INSET,
      12,
      false,
    );

    this.agents = [
      {
        id: 'veh_lead',
        cls: 'car',
        size: CAR_SIZE,
        route: this.route,
        s: this.egoS + 44,
        speed: 10,
        lateral: 0,
        targetLateral: 0,
        role: 'lead',
      },
      {
        id: 'veh_cutin',
        cls: 'car',
        size: CAR_SIZE,
        route: this.route,
        s: this.egoS - 16,
        speed: 12,
        lateral: -(EGO_LANE_INSET - KERB_LANE_INSET),
        targetLateral: -(EGO_LANE_INSET - KERB_LANE_INSET),
        role: 'cutin',
      },
      {
        id: 'veh_oncoming',
        cls: 'truck',
        size: TRUCK_SIZE,
        route: oncoming,
        s: 120,
        speed: 9,
        lateral: 0,
        targetLateral: 0,
        role: 'oncoming',
      },
    ];
  }

  /**
   * Locate where the ego route passes each controlled intersection by scanning
   * for the closest approach. Cheap (one pass at construction) and immune to
   * how the fillets shift the geometry.
   */
  private findControlPoints(): ControlPoint[] {
    const targets: Array<{ p: Vec2; kind: 'signal' | 'stop'; offset: number }> = [
      { p: [0, 0], kind: 'signal', offset: 0 },
      { p: [80, 80], kind: 'signal', offset: 14 },
      { p: [0, 80], kind: 'stop', offset: 0 },
      { p: [80, 0], kind: 'stop', offset: 0 },
    ];
    const STEP = 0.5;
    return targets.map((tgt) => {
      let bestS = 0;
      let bestD = Infinity;
      for (let s = 0; s < this.route.length; s += STEP) {
        const p = this.route.sample(s);
        const d = Math.hypot(p.x - tgt.p[0], p.y - tgt.p[1]);
        if (d < bestD) {
          bestD = d;
          bestS = s;
        }
      }
      const at = this.route.sample(bestS);
      // Ego's heading through the intersection decides which phase group
      // governs it: mostly-east/west travel is controlled by the EW group.
      const group: 'ns' | 'ew' =
        Math.abs(Math.cos(at.heading)) > Math.abs(Math.sin(at.heading))
          ? 'ew'
          : 'ns';
      return {
        s: bestS,
        kind: tgt.kind,
        key: `tl_${tgt.p[0]}_${tgt.p[1]}`,
        group,
        offset: tgt.offset,
      };
    });
  }

  /* ---------------- signals ---------------- */

  private phaseFor(group: 'ns' | 'ew', offset: number): SignalPhase {
    const c = (this.t + offset) % CYCLE;
    if (group === 'ns') {
      if (c < NS_GREEN_END) return 'green';
      if (c < NS_YELLOW_END) return 'yellow';
      return 'red';
    }
    if (c < NS_YELLOW_END) return 'red';
    if (c < EW_GREEN_END) return 'green';
    return 'yellow';
  }

  private signalStates(): SignalState[] {
    return this.scene.traffic_lights.map((tl) => {
      const control = this.controls.find((c) => tl.id.startsWith(c.key));
      const offset = control?.offset ?? 0;
      const group = signalGroup(tl.id);
      const phase = this.phaseFor(group, offset);
      const c = (this.t + offset) % CYCLE;
      const bounds =
        group === 'ns'
          ? [NS_GREEN_END, NS_YELLOW_END, CYCLE]
          : [NS_YELLOW_END, EW_GREEN_END, CYCLE];
      const next = bounds.find((b) => b > c) ?? CYCLE;
      return {
        id: tl.id,
        phase,
        time_to_change_s: Math.round((next - c) * 10) / 10,
      };
    });
  }

  /* ---------------- stepping ---------------- */

  step(): void {
    if (this.paused) {
      this.seq++;
      return;
    }
    const dt = DT;
    this.t += dt;
    this.seq++;

    this.stepCutinScript(dt);
    this.stepAgents(dt);
    this.stepEgo(dt);

    this.battery = Math.max(4, this.battery - dt * 0.0055);

    // 3 s of lateral history feeds the trajectory graph's past half.
    const cutin = this.agents.find((a) => a.role === 'cutin')!;
    this.latHistory.push({
      t: this.t,
      ego: this.egoLatOffset,
      cutin: this.cutinPhase === 'idle' ? null : cutin.lateral,
    });
    while (this.latHistory.length && this.latHistory[0].t < this.t - 3.2) {
      this.latHistory.shift();
    }
  }

  private stepCutinScript(dt: number): void {
    const cutin = this.agents.find((a) => a.role === 'cutin')!;
    const kerb = -(EGO_LANE_INSET - KERB_LANE_INSET);

    switch (this.cutinPhase) {
      case 'idle':
        if (this.t >= this.nextCutinAt) {
          this.cutinPhase = 'approach';
          this.cutinTimer = 0;
          // Only reposition if it has drifted out of the scripted window —
          // teleporting a visible car reads as a rendering glitch.
          if (Math.abs(this.route.gap(this.egoS, cutin.s)) > 40) {
            cutin.s = this.route.wrap(this.egoS - 12);
          }
          cutin.lateral = kerb;
          cutin.targetLateral = kerb;
          cutin.speed = this.egoSpeed + 3.2;
        }
        break;
      case 'approach': {
        this.cutinTimer += dt;
        const gap = this.route.gap(this.egoS, cutin.s);
        if (gap > 8 || this.cutinTimer > 8) {
          this.cutinPhase = 'merging';
          this.cutinTimer = 0;
          cutin.targetLateral = 0;
          this.pushEvent(
            'warn',
            'CUTIN_DETECTED',
            'Vehicle cutting in from right lane',
          );
        }
        break;
      }
      case 'merging':
        this.cutinTimer += dt;
        if (Math.abs(cutin.lateral) < 0.15 && this.cutinTimer > 2.2) {
          this.cutinPhase = 'settled';
          this.cutinTimer = 0;
        }
        break;
      case 'settled':
        this.cutinTimer += dt;
        if (this.cutinTimer > 4.5) {
          this.cutinPhase = 'idle';
          this.nextCutinAt = this.t + this.params.cutin_period_s;
          cutin.targetLateral = kerb;
          this.pushEvent('info', 'CUTIN_CLEARED', 'Cut-in vehicle settled');
        }
        break;
    }

    // Lateral tracking: bounded merge rate, so the box slides rather than snaps.
    const rate = this.cutinPhase === 'merging' ? 1.7 : 0.9;
    const d = cutin.targetLateral - cutin.lateral;
    cutin.lateral += clamp(d, -rate * dt, rate * dt);
  }

  private stepAgents(dt: number): void {
    for (const a of this.agents) {
      const scale = this.params.traffic_speed_scale;
      switch (a.role) {
        case 'lead': {
          // Gentle speed oscillation so ego has to modulate its following gap.
          const target = (9.5 + Math.sin(this.t * 0.31) * 3.4) * scale;
          a.speed += clamp(target - a.speed, -2.4 * dt, 1.6 * dt);
          break;
        }
        case 'cutin': {
          const gap = this.route.gap(this.egoS, a.s);
          const target =
            this.cutinPhase === 'approach'
              ? (this.egoSpeed + 3.4) * scale
              : gap > 0
                ? (this.egoSpeed - 0.6) * scale
                : (this.egoSpeed + 2.0) * scale;
          a.speed += clamp(target - a.speed, -3.5 * dt, 2.6 * dt);
          break;
        }
        case 'oncoming': {
          const target = (8.5 + Math.sin(this.t * 0.22 + 1.1) * 2.0) * scale;
          a.speed += clamp(target - a.speed, -2.0 * dt, 1.4 * dt);
          break;
        }
      }
      a.speed = Math.max(0, a.speed);
      a.s = a.route.wrap(a.s + a.speed * dt);
    }
  }

  private stepEgo(dt: number): void {
    const limitCap = this.params.ego_speed_cap_mph * 0.44704;
    const roadLimit = Math.min(this.currentSpeedLimit(), limitCap);

    // Curve speed: hold lateral acceleration under a comfort ceiling.
    let maxCurv = 0;
    for (let d = 0; d <= 26; d += 2) {
      maxCurv = Math.max(maxCurv, Math.abs(this.route.sample(this.egoS + d).curvature));
    }
    const curveSpeed =
      maxCurv > 1e-4 ? Math.sqrt(MAX_LAT_ACCEL / maxCurv) : Infinity;

    // Car following.
    const lead = this.nearestLead();
    let followSpeed = Infinity;
    if (lead) {
      const desiredGap = 6 + this.egoSpeed * 1.5;
      followSpeed = lead.speed + (lead.gap - desiredGap) * 0.55;
    }

    // Signals and stop signs.
    const stopSpeed = this.controlSpeed(dt);

    const target = Math.max(
      0,
      Math.min(roadLimit, curveSpeed, followSpeed, stopSpeed),
    );
    const raw = (target - this.egoSpeed) * 1.35;
    this.egoAccel = clamp(raw, MAX_DECEL, MAX_ACCEL);
    this.egoSpeed = Math.max(0, this.egoSpeed + this.egoAccel * dt);
    this.egoS = this.route.wrap(this.egoS + this.egoSpeed * dt);

    // Lane keeping: a slow wander plus an evasive nudge away from a cut-in.
    const evade = this.cutinPhase === 'merging' ? 0.55 : 0;
    this.planLatTarget = evade + Math.sin(this.t * 0.42) * 0.12;
    this.planLat += clamp(this.planLatTarget - this.planLat, -0.9 * dt, 0.9 * dt);
    this.egoLatOffset += clamp(
      this.planLat - this.egoLatOffset,
      -0.8 * dt,
      0.8 * dt,
    );

    const curv = this.route.sample(this.egoS).curvature;
    const steerTarget = Math.atan(WHEELBASE * curv) + this.planLat * 0.06;
    this.steering += clamp(steerTarget - this.steering, -2.6 * dt, 2.6 * dt);
  }

  /** Speed limit of the street the ego is currently travelling along. */
  private currentSpeedLimit(): number {
    const p = this.route.sample(this.egoS);
    let best = 11.176;
    let bestD = Infinity;
    for (const s of STREETS) {
      const d = s.axis === 'ns' ? Math.abs(p.x - s.at) : Math.abs(p.y - s.at);
      if (d < bestD && d < halfWidth(s) + 4) {
        bestD = d;
        best = s.speed_mph * 0.44704;
      }
    }
    return best;
  }

  /** Deceleration target imposed by the next red light or stop sign. */
  private controlSpeed(dt: number): number {
    if (this.stopHold > 0) {
      this.stopHold -= dt;
      return 0;
    }
    let limit = Infinity;
    for (let i = 0; i < this.controls.length; i++) {
      const c = this.controls[i];
      const gap = this.route.gap(this.egoS, c.s);
      if (gap > 60 || gap < -8) {
        if (gap < -20 || gap > 80) this.clearedStop.delete(i);
        continue;
      }
      const stopLine = gap - 5.5;
      let mustStop: boolean;
      if (c.kind === 'signal') {
        const phase = this.phaseFor(c.group, c.offset);
        // Yellow only forces a stop when there is room to do it comfortably.
        mustStop = phase === 'red' || (phase === 'yellow' && stopLine > 12);
      } else {
        mustStop = !this.clearedStop.has(i);
      }
      if (!mustStop) continue;

      // Inside the final couple of metres, demand a full stop rather than
      // following the sqrt profile — that profile only reaches zero speed
      // asymptotically, which leaves the car creeping forever at a stop sign.
      if (stopLine < 2) {
        if (c.kind === 'stop' && this.egoSpeed < 1) {
          this.clearedStop.add(i);
          this.stopHold = 0.9;
          this.pushEvent('info', 'STOP_SIGN', 'Stopped at all-way stop');
        }
        return 0;
      }
      // v = sqrt(2 a d) with a comfortable 2.6 m/s^2.
      limit = Math.min(limit, Math.sqrt(2 * 2.6 * stopLine));
    }
    return limit;
  }

  private nearestLead(): { speed: number; gap: number } | null {
    let best: { speed: number; gap: number } | null = null;
    for (const a of this.agents) {
      if (a.route !== this.route) continue;
      const gap = this.route.gap(this.egoS, a.s);
      if (gap <= 0 || gap > 90) continue;
      // Only vehicles substantially inside the ego lane block it.
      if (Math.abs(a.lateral - this.egoLatOffset) > LANE_W * 0.6) continue;
      const bumperGap = gap - (a.size.length + EGO_SIZE.length) / 2;
      if (!best || bumperGap < best.gap) {
        best = { speed: a.speed, gap: bumperGap };
      }
    }
    return best;
  }

  private pushEvent(
    level: SimEvent['level'],
    code: string,
    message: string,
  ): void {
    this.pendingEvents.push({
      t: Math.round(this.t * 100) / 100,
      level,
      code,
      message,
    });
    if (this.pendingEvents.length > 8) this.pendingEvents.shift();
  }

  /* ---------------- frame assembly ---------------- */

  frame(): StateUpdate {
    const ego = this.route.sample(this.egoS, this.egoLatOffset);
    const cos = Math.cos(ego.heading);
    const sin = Math.sin(ego.heading);

    /** World point -> ego frame (forward, left). */
    const toEgo = (x: number, y: number): [number, number] => {
      const dx = x - ego.x;
      const dy = y - ego.y;
      return [dx * cos + dy * sin, -dx * sin + dy * cos];
    };

    const egoVx = this.egoSpeed * cos;
    const egoVy = this.egoSpeed * sin;

    const detections: Detection[] = [];
    const neighbors: LaneNeighbor[] = [];
    const radar: RadarPoint[] = [];
    let criticalTtc: number | null = null;

    for (const a of this.agents) {
      const p = a.route.sample(a.s, a.lateral);
      const [fwd, left] = toEgo(p.x, p.y);
      const av = { x: a.speed * Math.cos(p.heading), y: a.speed * Math.sin(p.heading) };

      // Closing speed along the line of sight; negative means approaching.
      const range = Math.hypot(p.x - ego.x, p.y - ego.y);
      const ux = range > 1e-3 ? (p.x - ego.x) / range : 1;
      const uy = range > 1e-3 ? (p.y - ego.y) / range : 0;
      const rangeRate = (av.x - egoVx) * ux + (av.y - egoVy) * uy;

      const sameLane = Math.abs(left) < LANE_W * 0.7 && fwd > 0;
      const bumperGap = Math.max(
        0.1,
        fwd - (a.size.length + EGO_SIZE.length) / 2,
      );
      const ttc =
        sameLane && rangeRate < -0.3 ? bumperGap / -rangeRate : null;

      const hazard =
        a.role === 'cutin' &&
        (this.cutinPhase === 'merging' || this.cutinPhase === 'settled') &&
        fwd > 0 &&
        fwd < 45;

      if (ttc != null && (criticalTtc == null || ttc < criticalTtc)) {
        criticalTtc = ttc;
      }

      detections.push({
        id: a.id,
        cls: a.cls,
        pose: { x: p.x, y: p.y, heading: p.heading },
        size: a.size,
        velocity: [av.x, av.y],
        speed_mps: a.speed,
        confidence: hazard ? 0.97 : 0.82 + this.rng() * 0.15,
        hazard,
        hazard_label: hazard ? 'Cut-in vehicle' : null,
        ttc_s: ttc == null ? null : Math.round(ttc * 100) / 100,
        lane_offset: clamp(Math.round(left / LANE_W), -2, 2),
      });

      if (Math.abs(fwd) < 90) {
        neighbors.push({
          id: a.id,
          cls: a.cls,
          lane_offset: clamp(Math.round(left / LANE_W), -2, 2),
          longitudinal_m: Math.round(fwd * 100) / 100,
          lateral_m: Math.round((left + this.egoLatOffset) * 100) / 100,
          speed_mps: a.speed,
          hazard,
        });
      }

      // Two returns per vehicle (near corner + centroid) reads like real radar.
      for (const jitter of [0, 1]) {
        const jx = p.x + (jitter ? -Math.cos(p.heading) * a.size.length * 0.4 : 0);
        const jy = p.y + (jitter ? -Math.sin(p.heading) * a.size.length * 0.4 : 0);
        const [f2, l2] = toEgo(jx, jy);
        const r2 = Math.hypot(f2, l2);
        if (r2 > 90) continue;
        radar.push({
          id: jitter === 0 ? a.id : null,
          azimuth: Math.atan2(l2, f2),
          range_m: Math.round(r2 * 100) / 100,
          range_rate_mps: Math.round(rangeRate * 100) / 100,
          rcs_db: a.cls === 'truck' ? 18 + this.rng() * 3 : 9 + this.rng() * 4,
          tracked: true,
        });
      }
    }

    // Kerb and building clutter across the forward sensor arc.
    for (let i = 0; i < 12; i++) {
      const az = (-1 + (2 * i) / 11) * 1.05 + (this.rng() - 0.5) * 0.06;
      const r = 9 + Math.abs(az) * 18 + this.rng() * 16;
      radar.push({
        id: null,
        azimuth: az,
        range_m: Math.round(r * 100) / 100,
        range_rate_mps: Math.round(-this.egoSpeed * Math.cos(az) * 100) / 100,
        rcs_db: -6 + this.rng() * 8,
        tracked: false,
      });
    }

    const maneuver = this.currentManeuver();
    const plan = this.buildPlan(ego.heading);
    const trajectory = this.buildTrajectory();

    const events = this.pendingEvents;
    this.pendingEvents = [];

    const frame: StateUpdate = {
      type: 'state_update',
      protocol: PROTOCOL_VERSION,
      seq: this.seq,
      t: Math.round(this.t * 1000) / 1000,
      sim_rate_hz: SIM_HZ,
      paused: this.paused,
      assist_active: this.params.assist_enabled,
      scenario_id: this.scene.scenario_id,
      ego: {
        pose: { x: ego.x, y: ego.y, heading: ego.heading },
        speed_mps: this.egoSpeed,
        accel_mps2: this.egoAccel,
        steering_angle: this.steering,
        yaw_rate: this.egoSpeed * ego.curvature,
        throttle: clamp(this.egoAccel / MAX_ACCEL, 0, 1),
        brake: clamp(-this.egoAccel / -MAX_DECEL, 0, 1),
        gear: 'D',
        speed_limit_mps: this.currentSpeedLimit(),
        cruise: {
          mode: this.params.assist_enabled ? 'fsd' : 'off',
          set_speed_mps: this.params.ego_speed_cap_mph * 0.44704,
        },
        size: EGO_SIZE,
      },
      detections,
      plan: {
        polyline: plan,
        target_speed_mps: Math.min(
          this.currentSpeedLimit(),
          this.params.ego_speed_cap_mph * 0.44704,
        ),
        maneuver,
        confidence: this.cutinPhase === 'merging' ? 0.71 : 0.94,
      },
      telemetry: {
        radar,
        lane: {
          lane_index: 0,
          lane_count: 2,
          lane_width_m: LANE_W,
          offset_m: this.egoLatOffset,
          heading_error: 0,
          left_marking: 'double_yellow',
          right_marking: 'dashed_white',
          neighbors,
        },
        ttc_s:
          criticalTtc == null ? null : Math.round(criticalTtc * 100) / 100,
        vehicle: {
          battery_pct: Math.round(this.battery * 10) / 10,
          range_km: Math.round(this.battery * 4.6 * 10) / 10,
          motor_temp_c: Math.round((42 + this.egoSpeed * 1.1) * 10) / 10,
          tire_pressure_kpa: [248, 249, 245, 246],
          subsystems: [
            { key: 'perception', label: 'Perception', status: 'ok', detail: '8 cameras nominal' },
            { key: 'planning', label: 'Planning', status: 'ok', detail: null },
            { key: 'steering', label: 'Steering', status: 'ok', detail: null },
            { key: 'brakes', label: 'Brakes', status: 'ok', detail: null },
            { key: 'battery', label: 'Battery', status: this.battery < 15 ? 'warn' : 'ok', detail: null },
          ],
          overall: this.battery < 15 ? 'warn' : 'ok',
        },
        trajectory,
      },
      signals: this.signalStates(),
      events,
      // The mock server has no ML perception path; ground-truth detections
      // above are all it ever produces.
      perception: null,
    };

    return frame;
  }

  private currentManeuver(): StateUpdate['plan']['maneuver'] {
    if (this.egoSpeed < 0.4 && this.egoAccel <= 0) return 'stop';
    const ahead = this.route.sample(this.egoS + 14).curvature;
    if (ahead < -0.02) return 'turn_right';
    if (ahead > 0.02) return 'turn_left';
    if (this.cutinPhase === 'merging') return 'yield';
    return 'keep_lane';
  }

  /**
   * Forward plan in world coordinates. The lateral offset eases from where the
   * car actually is to where the planner wants it, so the ribbon leaves the
   * bumper rather than jumping sideways.
   */
  private buildPlan(_heading: number): Vec2[] {
    const LEN = 58;
    const N = 28;
    const out: Vec2[] = [];
    for (let i = 0; i <= N; i++) {
      const f = i / N;
      const lat =
        this.egoLatOffset + (this.planLat - this.egoLatOffset) * Math.min(1, f * 3);
      const p = this.route.sample(this.egoS + LEN * f, lat);
      out.push([p.x, p.y]);
    }
    return out;
  }

  private buildTrajectory(): StateUpdate['telemetry']['trajectory'] {
    const HORIZON = 5;
    const STEP = 0.25;
    const planned: TrajectorySample[] = [];
    const cutinSeries: TrajectorySample[] = [];
    const cutin = this.agents.find((a) => a.role === 'cutin')!;
    const active = this.cutinPhase !== 'idle';

    // Past: downsample the history ring.
    for (let tt = -3; tt < 0; tt += STEP) {
      const want = this.t + tt;
      let best = this.latHistory[0];
      for (const h of this.latHistory) {
        if (Math.abs(h.t - want) < Math.abs((best?.t ?? -1e9) - want)) best = h;
      }
      if (!best) continue;
      planned.push({ t: Math.round(tt * 100) / 100, lateral_m: best.ego });
      if (active && best.cutin != null) {
        cutinSeries.push({ t: Math.round(tt * 100) / 100, lateral_m: best.cutin });
      }
    }

    // Future: the planner's lateral target, first-order approach.
    for (let tt = 0; tt <= HORIZON + 1e-6; tt += STEP) {
      const k = 1 - Math.exp(-tt / 0.9);
      planned.push({
        t: Math.round(tt * 100) / 100,
        lateral_m: this.egoLatOffset + (this.planLatTarget - this.egoLatOffset) * k,
      });
      if (active) {
        const k2 = 1 - Math.exp(-tt / 1.3);
        cutinSeries.push({
          t: Math.round(tt * 100) / 100,
          lateral_m: cutin.lateral + (cutin.targetLateral - cutin.lateral) * k2,
        });
      }
    }

    return {
      horizon_s: HORIZON,
      planned,
      cutin: active ? cutinSeries : null,
      cutin_label: active ? 'Cut-in vehicle' : null,
    };
  }

  /* ---------------- commands ---------------- */

  // `camera_frame` is deliberately excluded from this parameter type: the
  // real backend intercepts it at the socket, before the command queue
  // (`ws_server.py` `_handle` -> `_ingest_frame`, never through `submit()`/
  // `_apply()`), so `apply()` — the mock's equivalent of the sim-thread
  // command executor — should never see one either. `createMockTransport`'s
  // `send()` below is the mock's equivalent of `_handle` and does the
  // intercepting.
  apply(
    command: Exclude<Command, { cmd: 'camera_frame' }>,
  ): { ok: boolean; message: string | null; scene?: SceneDescription } {
    switch (command.cmd) {
      case 'set_paused':
        this.paused = command.paused;
        return { ok: true, message: command.paused ? 'paused' : 'running' };
      case 'step':
        for (let i = 0; i < command.frames; i++) {
          const was = this.paused;
          this.paused = false;
          this.step();
          this.paused = was;
        }
        return { ok: true, message: `stepped ${command.frames}` };
      case 'reset':
        this.resetDynamics();
        return { ok: true, message: 'reset' };
      case 'load_scenario': {
        const found = SCENARIOS.find((s) => s.id === command.scenario_id);
        if (!found) {
          return { ok: false, message: `unknown scenario ${command.scenario_id}` };
        }
        this.scene = buildScene(found.id);
        this.resetDynamics();
        return { ok: true, message: found.name, scene: this.scene };
      }
      case 'set_param': {
        const { key, value } = command;
        if (key in DEFAULT_PARAMS) {
          (this.params as unknown as Record<string, unknown>)[key] = value;
          return { ok: true, message: `${key}=${String(value)}` };
        }
        // Unknown params are accepted and ignored: the UI may expose knobs the
        // simulator does not model yet.
        return { ok: true, message: `${key} ignored` };
      }
      case 'toggle_layer':
        // Layer visibility is purely a client concern; acknowledged so the
        // command path is uniform.
        return { ok: true, message: `${command.layer}=${command.visible}` };
      case 'set_camera':
        return { ok: true, message: command.view };
      case 'set_perception':
        // The mock never builds a perception pipeline, so this always
        // refuses — mirroring `sim/loop.py`'s `_cmd_set_perception`, which
        // refuses the same way when `self.perception_pipeline is None`.
        return { ok: false, message: 'no perception pipeline: start with --perception' };
      case 'inject_hazard':
        this.nextCutinAt = this.t;
        this.cutinPhase = 'idle';
        return { ok: true, message: `hazard ${command.kind} queued` };
      case 'load_location': {
        // The mock has no geocoder or map pipeline, so it never builds
        // anything new — it fakes the *shape* of a real build instead. A
        // real backend acks immediately and delivers the finished scene
        // later through the epoch push (sim/loop.py `_cmd_load_location`);
        // this only decides the ack. The delayed scene swap is the
        // transport wrapper's job (createMockTransport.send(), below),
        // which is the thing that actually owns emitScene and timers.
        const query = normalizeLocationQuery(command.query);
        if (!query) {
          // A whitespace-only query normalizes to empty; don't hand a
          // blank name/location to the UI.
          return { ok: false, message: 'load_location requires a non-empty query' };
        }
        return { ok: true, message: `building ${query}` };
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* Transport wrapper                                                   */
/* ------------------------------------------------------------------ */

export interface MockTransportOptions {
  scenarioId?: string;
  /** Validate every outbound frame. On by default; the cost is ~0.1 ms. */
  validate?: boolean;
}

/**
 * Fake `load_location` build time. The real backend's build takes seconds
 * (geocode plus an Overpass fetch); this just needs to be long enough that
 * the scene visibly arrives *after* the ack rather than alongside it. A
 * value under ~100 ms risks disappearing entirely behind a UI's minimum-
 * visible-duration spinner throttle (a standard pattern), which would
 * defeat the point — exercising the pending state under `?mock=1` — rather
 * than merely making it brief. 600 ms sits comfortably clear of that and
 * matches what Task 6 found perceptible by hand in a running browser.
 */
const MOCK_LOCATION_BUILD_MS = 600;

export function createMockTransport(
  opts: MockTransportOptions = {},
): Transport {
  const sim = new MockSim(opts.scenarioId);
  const validate = opts.validate ?? true;
  let handlers: TransportHandlers | null = null;
  let running = false;
  let rafId = 0;
  let timerId: ReturnType<typeof setInterval> | null = null;
  let last = 0;
  let acc = 0;
  // At most one load_location build is ever in flight: a newer request
  // cancels an older one still pending. This matches the *effect* of the
  // real backend's single pending-scene slot (sim/loop.py `_pending_scene`,
  // set — not appended — around line 776-779) — only the latest request's
  // scene is ever delivered — but not its mechanism: the backend never
  // cancels an in-flight build; an earlier one keeps running and is
  // discarded only if a later one *completes* after it, a completion-order
  // race. Here submission order and completion order coincide because the
  // delay is fixed, so proactive cancellation and that race land on the
  // same outcome — they are not the same thing.
  let pendingLocationTimer: ReturnType<typeof setTimeout> | null = null;

  // The mock never touches a real socket, so `onRawFrame` is computed here
  // for parity — it's what the message would have cost in bytes on the wire,
  // letting the perf overlay show a real number under ?mock=1 too.
  const emitScene = (scene: SceneDescription) => {
    if (!handlers) return;
    if (validate) {
      const res = SceneDescriptionSchema.safeParse(scene);
      if (!res.success) {
        handlers.onInvalid(formatIssues(res.error), scene);
        return;
      }
      handlers.onRawFrame?.(JSON.stringify(res.data).length);
      handlers.onMessage(res.data);
      return;
    }
    handlers.onRawFrame?.(JSON.stringify(scene).length);
    handlers.onMessage(scene);
  };

  const emitFrame = () => {
    if (!handlers) return;
    const f = sim.frame();
    if (validate) {
      const res = StateUpdateSchema.safeParse(f);
      if (!res.success) {
        handlers.onInvalid(formatIssues(res.error), f);
        return;
      }
      handlers.onRawFrame?.(JSON.stringify(res.data).length);
      handlers.onMessage(res.data);
      return;
    }
    handlers.onRawFrame?.(JSON.stringify(f).length);
    handlers.onMessage(f);
  };

  const tick = (now: number) => {
    if (!running) return;
    const elapsed = Math.min(0.25, (now - last) / 1000);
    last = now;
    acc += elapsed;
    let steps = 0;
    while (acc >= DT && steps < MAX_CATCHUP_STEPS) {
      sim.step();
      acc -= DT;
      steps++;
      emitFrame();
    }
    if (steps === MAX_CATCHUP_STEPS) acc = 0;
  };

  const loop = (now: number) => {
    tick(now);
    if (running) rafId = requestAnimationFrame(loop);
  };

  return {
    kind: 'mock',
    label: 'mock',
    connect(h) {
      handlers = h;
      running = true;
      h.onStatus('connecting');
      // Deliver asynchronously so callers can finish wiring up first.
      queueMicrotask(() => {
        if (!running || !handlers) return;
        handlers.onStatus('open', 'in-process mock');
        emitScene(sim.scene);
        if (typeof requestAnimationFrame === 'function') {
          last = performance.now();
          rafId = requestAnimationFrame(loop);
        } else {
          last = Date.now();
          timerId = setInterval(() => tick(Date.now()), 1000 / SIM_HZ);
        }
      });
    },
    send(command) {
      if (!running) return;
      // Mirrors `ws_server.py` `_handle`'s early-out for `camera_frame`: real
      // frames bypass the command queue entirely and are never acked. The
      // mock has nowhere to route a frame either — `StubDetector`/the
      // perception pipeline don't exist client-side — so the faithful
      // behaviour is the same early return, not a fabricated ack.
      if (command.cmd === 'camera_frame') return;
      const res = sim.apply(command);
      const ack: Ack = {
        type: 'ack',
        protocol: PROTOCOL_VERSION,
        id: command.id,
        cmd: command.cmd,
        ok: res.ok,
        message: res.message,
        t: Math.round(sim.t * 1000) / 1000,
      };
      // Mimic network latency so the UI cannot depend on synchronous acks.
      queueMicrotask(() => {
        if (!handlers) return;
        if (res.scene) emitScene(res.scene);
        handlers.onMessage(ack);
      });

      if (command.cmd === 'load_location' && res.ok) {
        // A later request cancels an earlier one still building — see the
        // `pendingLocationTimer` declaration above for how this compares to
        // the real backend's behaviour.
        if (pendingLocationTimer != null) clearTimeout(pendingLocationTimer);
        const query = normalizeLocationQuery(command.query);
        pendingLocationTimer = setTimeout(() => {
          pendingLocationTimer = null;
          // Relabelling the existing mock city is the right level of
          // fidelity here — the mock has no map pipeline to actually run.
          sim.scene = { ...sim.scene, name: query, location: query };
          emitScene(sim.scene);
        }, MOCK_LOCATION_BUILD_MS);
      }
    },
    close() {
      running = false;
      if (rafId) cancelAnimationFrame(rafId);
      if (timerId) clearInterval(timerId);
      if (pendingLocationTimer != null) {
        clearTimeout(pendingLocationTimer);
        pendingLocationTimer = null;
      }
      handlers?.onStatus('closed');
      handlers = null;
    },
    pendingCount() {
      // The mock never buffers commands while disconnected — `send()` above
      // simply no-ops when `!running`.
      return 0;
    },
  };
}
