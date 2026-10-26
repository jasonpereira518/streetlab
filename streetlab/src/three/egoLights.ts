/**
 * Automatic hazard lights, derived from the frames the client already has.
 *
 * Tesla switches the hazards on when forceful braking or AEB brings the car to
 * a stop, after a crash, and when the car pulls over or stops for an
 * inattentive driver; pressing the accelerator switches them off. None of that
 * is on the wire, so this latch reads it off speed, the pedals, the plan label
 * and the event stream. `ponytail:` a client latch, not sim state: other agents
 * cannot see the flashers. Upgrade is a wire `Ego.hazard_lights` (a protocol
 * bump) if anything in the sim ever needs to.
 */

import type { StateUpdate } from '../schema';
import { DBL_BRAKE, DBL_MIN_MPS, type TurnSignal } from './ego';

/** At or below this the car counts as stopped. */
const STOPPED_MPS = 0.3;
/** A fresh accelerator press: the wire throttle crossing this upward. */
const THROTTLE_PRESS = 0.1;

export class EgoLights {
  private armed = false;
  private on = false;
  private forced = false;
  private lastT = -Infinity;
  private lastCollisionT = -Infinity;
  private prevThrottle = 0;

  /** Hazards held on by something outside the frame (the attention stop). */
  force(on: boolean): void {
    this.forced = on;
  }

  /** The indicator state to draw: the wire's, or `hazard`. */
  update(frame: StateUpdate): TurnSignal {
    const { ego, plan } = frame;
    if (frame.t < this.lastT) this.reset(); // the sim was reset
    this.lastT = frame.t;

    const speed = Math.abs(ego.speed_mps);
    const pressed = ego.throttle > THROTTLE_PRESS && this.prevThrottle <= THROTTLE_PRESS;
    this.prevThrottle = ego.throttle;

    if (pressed) {
      this.on = false;
      this.armed = false;
    }
    if (plan.maneuver === 'emergency_brake' || (ego.brake >= DBL_BRAKE && speed > DBL_MIN_MPS)) {
      this.armed = true;
    } else if (ego.throttle > THROTTLE_PRESS && speed > STOPPED_MPS) {
      this.armed = false; // it braked hard and drove on
    }
    if (this.armed && speed <= STOPPED_MPS) this.on = true;
    if (plan.maneuver === 'pull_over' && speed <= STOPPED_MPS) this.on = true;
    for (const e of frame.events) {
      if (e.code === 'collision' && e.t > this.lastCollisionT) {
        this.lastCollisionT = e.t;
        this.on = true;
      }
    }
    return this.on || this.forced ? 'hazard' : ego.turn_signal;
  }

  reset(): void {
    this.armed = false;
    this.on = false;
    this.lastCollisionT = -Infinity;
    this.prevThrottle = 0;
  }
}
