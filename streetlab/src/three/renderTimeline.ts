/**
 * Render-time playback clock for the ego.
 *
 * Three clocks run free in this system and none of them is phase-locked to
 * the others: the sim thread steps at 60 Hz, the websocket push loop samples
 * the sim's latest-wins slot at its own 60 Hz, and the display refreshes at
 * whatever the monitor does (60 Hz, 120 Hz, or nothing at all while the tab
 * is hidden). Measured against the real backend, the sim time carried by
 * consecutively *delivered* frames advances by 0, 1, 2 or 3 ticks — 87.5% of
 * frames are one tick, the other 12.5% are duplicates or double-jumps.
 *
 * Binding the ego's rendered pose to frame *arrival* therefore makes it stall
 * for a frame and then lurch, while the chase camera — damped against real
 * display `dt` — keeps gliding. That relative motion is what reads on screen
 * as the car jumping back and forth.
 *
 * The fix is to stop rendering arrivals and start rendering *time*. This
 * class keeps a short history of frames, runs a local clock that advances
 * with real display `dt` and is phase-locked to the stream a fraction of a
 * frame in the past, and interpolates between the two frames bracketing it.
 * The cost is that buffered fraction of latency (~25 ms at 60 Hz), which is
 * well under what anyone can see and is the standard price for smooth
 * playback of a discretely-sampled stream. Traffic agents already solve the
 * same problem their own way (a critically-damped follow in `agents.ts`); the
 * ego was the one moving object left snapping, which is exactly why it was
 * the one that looked wrong.
 */
import type { Pose, StateUpdate, Vec2 } from '../schema';
import { angleDelta, clamp, damp, lerp } from '../units';

/** Everything the renderer draws on the playback clock rather than on arrival. */
export interface RenderSample {
  pose: Pose;
  speed_mps: number;
  steering_angle: number;
  accel_mps2: number;
  /**
   * The plan ribbon. It rides this clock too, and not because it flickers on
   * its own — it is a fixed-count arc-length sampling of the route ahead of
   * the car, so its near end is pinned to the bumper. Left on arrival time
   * while the ego moved on playback time, it would shimmer against a car that
   * had just been made smooth.
   */
  plan: Vec2[];
}

interface Keyframe extends RenderSample {
  /** Sim time, seconds. */
  t: number;
}

/**
 * How far behind the newest frame the clock sits, as a multiple of the
 * measured delivery interval. Needs to exceed 1 so an ordinary skipped frame
 * still has a frame on both sides of the clock to interpolate between; much
 * beyond 1.5 is latency spent for nothing.
 */
const BUFFER_FRAMES = 1.5;
const MIN_BUFFER_S = 1 / 120;
const MAX_BUFFER_S = 0.12;

/**
 * Frames of history kept. Has to cover the buffer with room to spare — at the
 * 1.5-frame buffer above, four is already generous, and eight costs nothing.
 */
const HISTORY = 8;

/** Weight given to each new inter-frame gap when estimating the interval. */
const INTERVAL_EMA = 0.1;
/** Assumed stream rate until enough frames have arrived to measure one. */
const INITIAL_INTERVAL_S = 1 / 60;

/**
 * Fraction of the clock's phase error left after one second. The clock runs
 * at real time and this pulls it back onto the stream, so it is a phase-locked
 * loop rather than a follower: a one-tick correction is spread over ~250 ms,
 * far below the threshold where a speed change is visible.
 */
const CLOCK_SMOOTHING = 0.02;

/**
 * A sim-time discontinuity larger than this is a scenario load, a reset or a
 * long stall — not jitter. Sliding the car across it would be wrong, so the
 * clock jumps instead.
 */
const RESYNC_S = 0.5;

/**
 * Real seconds without a *new* frame after which the stream counts as
 * stalled and the clock is allowed to close the buffer gap. This is the
 * ordinary paused case: the server keeps re-sending the same frame (`paused`
 * is a field on it), those arrive as duplicates and are dropped, and without
 * this the car would sit frozen a buffer's worth behind the pose the
 * telemetry panels are reading off the same frame.
 */
const STALL_FRAMES = 3;

function keyframe(frame: StateUpdate): Keyframe {
  const { ego } = frame;
  return {
    t: frame.t,
    pose: { x: ego.pose.x, y: ego.pose.y, heading: ego.pose.heading },
    speed_mps: ego.speed_mps,
    steering_angle: ego.steering_angle,
    accel_mps2: ego.accel_mps2,
    plan: frame.plan.polyline,
  };
}

/**
 * Blend two plan polylines vertex by vertex. Both are the same fixed number
 * of points sampled at the same fixed arc-length step, differing only in
 * where along the route the sampling starts, so matching indices are the same
 * point on the plan and lerping them is exactly interpolating that start
 * offset. A length mismatch means the route itself changed underneath — the
 * one case where the vertices are not comparable, so the newer plan is taken
 * whole.
 */
function blendPlan(a: Vec2[], b: Vec2[], u: number): Vec2[] {
  if (a.length !== b.length) return b;
  const out: Vec2[] = new Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = [lerp(a[i][0], b[i][0], u), lerp(a[i][1], b[i][1], u)];
  }
  return out;
}

function snapshot(k: Keyframe): RenderSample {
  return {
    pose: { ...k.pose },
    speed_mps: k.speed_mps,
    steering_angle: k.steering_angle,
    accel_mps2: k.accel_mps2,
    plan: k.plan,
  };
}

function blend(a: Keyframe, b: Keyframe, u: number): RenderSample {
  return {
    pose: {
      x: lerp(a.pose.x, b.pose.x, u),
      y: lerp(a.pose.y, b.pose.y, u),
      // Short way around: a car crossing due west must not sweep back
      // through zero on the way from +pi to -pi.
      heading: a.pose.heading + angleDelta(b.pose.heading, a.pose.heading) * u,
    },
    speed_mps: lerp(a.speed_mps, b.speed_mps, u),
    steering_angle: lerp(a.steering_angle, b.steering_angle, u),
    accel_mps2: lerp(a.accel_mps2, b.accel_mps2, u),
    plan: blendPlan(a.plan, b.plan, u),
  };
}

export class RenderTimeline {
  /** Ascending by `t`, newest last, at most `HISTORY` long. */
  private frames: Keyframe[] = [];
  private clock = 0;
  private locked = false;
  private interval = INITIAL_INTERVAL_S;
  private sinceFrame = 0;

  /** Sim seconds the rendered pose trails the newest frame by. */
  get bufferSeconds(): number {
    return clamp(this.interval * BUFFER_FRAMES, MIN_BUFFER_S, MAX_BUFFER_S);
  }

  /**
   * Take a delivered frame. Duplicates and out-of-order frames — both of
   * which the latest-wins wire protocol legitimately produces — are dropped,
   * so a stalled stream holds its pose instead of jittering on re-sends. A
   * backwards jump big enough to be a reset, rather than a re-send, restarts
   * the history.
   */
  push(frame: StateUpdate): void {
    const next = keyframe(frame);
    const latest = this.frames[this.frames.length - 1];
    if (latest) {
      const dt = next.t - latest.t;
      if (dt <= 0) {
        if (dt > -RESYNC_S) return; // duplicate or stale re-send
        this.frames.length = 0; // sim time restarted: nothing to blend from
        this.locked = false;
      } else if (dt < RESYNC_S) {
        this.interval = lerp(this.interval, dt, INTERVAL_EMA);
      } else {
        this.frames.length = 0; // long gap: the two sides are unrelated
        this.locked = false;
      }
    }
    this.frames.push(next);
    if (this.frames.length > HISTORY) this.frames.shift();
    this.sinceFrame = 0;
  }

  /** Advance the playback clock by one display frame's worth of real time. */
  advance(dt: number): void {
    const latest = this.frames[this.frames.length - 1];
    if (!latest) return;
    this.sinceFrame += dt;

    // A stream that has gone quiet has no future to interpolate toward, so
    // stop holding the buffer open and settle on the newest pose.
    const stalled = this.sinceFrame > this.interval * STALL_FRAMES;
    const target = latest.t - (stalled ? 0 : this.bufferSeconds);

    if (!this.locked || Math.abs(this.clock - target) > RESYNC_S) {
      this.clock = target;
      this.locked = true;
      return;
    }
    this.clock = damp(this.clock + dt, target, CLOCK_SMOOTHING, dt);
  }

  /** The scene as it should be drawn right now, or `null` before any frame. */
  sample(): RenderSample | null {
    return this.sampleAt(this.clock);
  }

  /**
   * Sample at an explicit sim time, clamped to the history at both ends — so
   * the ego never extrapolates past the newest frame the simulator has
   * actually produced.
   */
  sampleAt(t: number): RenderSample | null {
    const { frames } = this;
    if (frames.length === 0) return null;
    if (frames.length === 1 || t <= frames[0].t) return snapshot(frames[0]);
    const last = frames[frames.length - 1];
    if (t >= last.t) return snapshot(last);

    for (let i = frames.length - 2; i >= 0; i--) {
      const a = frames[i];
      if (t < a.t) continue;
      const b = frames[i + 1];
      return blend(a, b, clamp((t - a.t) / (b.t - a.t), 0, 1));
    }
    return snapshot(frames[0]);
  }

  /** Drop all history — a new scene has nothing to interpolate from. */
  reset(): void {
    this.frames.length = 0;
    this.locked = false;
    this.interval = INITIAL_INTERVAL_S;
    this.sinceFrame = 0;
  }
}
