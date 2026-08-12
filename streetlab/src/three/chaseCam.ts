/**
 * Camera rig.
 *
 * The chase view follows a *virtual* ego — a damped copy of the real pose —
 * rather than the car itself. Trailing the smoothed pose is what makes a turn
 * feel like a camera swinging in behind the car instead of a rigid boom bolted
 * to the roof. Distance and height open up with speed.
 */
import * as THREE from 'three/webgpu';
import type { CameraView, Pose } from '../schema';
import { clamp, damp, dampAngle } from '../units';

const CHASE = {
  /** Trail distance at 0 m/s and at 30 m/s. */
  distNear: 8.4,
  distFar: 14.5,
  heightNear: 3.1,
  heightFar: 4.3,
  lookAhead: 9,
  /** Fraction of the gap left after one second — lower is snappier. */
  poseSmoothing: 0.0009,
  camSmoothing: 0.0016,
  targetSmoothing: 0.0012,
};

export class ChaseCamera {
  readonly camera: THREE.PerspectiveCamera;

  /** Damped ego pose the rig hangs off. */
  private vx = 0;
  private vz = 0;
  private vHeading = 0;
  private started = false;

  private readonly camPos = new THREE.Vector3();
  private readonly lookAt = new THREE.Vector3();
  private readonly desired = new THREE.Vector3();
  private readonly desiredLook = new THREE.Vector3();

  /** Free-orbit state, driven by pointer input. */
  private orbitAzimuth = Math.PI * 0.25;
  private orbitElevation = 0.62;
  private orbitDistance = 90;

  constructor(aspect = 16 / 9) {
    this.camera = new THREE.PerspectiveCamera(52, aspect, 0.3, 1400);
    this.camera.position.set(0, 12, 24);
  }

  setAspect(aspect: number): void {
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }

  /**
   * Snap the rig to a pose without interpolating — used on scenario load. The
   * camera object is moved too, so the first rendered frame is already in the
   * trail position instead of easing in from wherever it was left.
   */
  reset(pose: Pose): void {
    this.vx = pose.x;
    this.vz = -pose.y;
    this.vHeading = pose.heading;
    this.started = true;

    const fx = Math.cos(pose.heading);
    const fz = -Math.sin(pose.heading);
    this.camPos.set(
      pose.x - fx * CHASE.distNear,
      CHASE.heightNear,
      -pose.y - fz * CHASE.distNear,
    );
    this.lookAt.set(
      pose.x + fx * CHASE.lookAhead,
      1.15,
      -pose.y - fz * CHASE.lookAhead,
    );
    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.lookAt);
  }

  update(pose: Pose, speed: number, view: CameraView, dt: number): void {
    const ex = pose.x;
    const ez = -pose.y;

    if (!this.started) this.reset(pose);

    // Virtual ego.
    this.vx = damp(this.vx, ex, CHASE.poseSmoothing, dt);
    this.vz = damp(this.vz, ez, CHASE.poseSmoothing, dt);
    this.vHeading = dampAngle(this.vHeading, pose.heading, CHASE.poseSmoothing, dt);

    // three.js forward for a world heading `h` is (cos h, 0, -sin h).
    const fx = Math.cos(this.vHeading);
    const fz = -Math.sin(this.vHeading);
    const t = clamp(speed / 30, 0, 1);
    const dist = CHASE.distNear + (CHASE.distFar - CHASE.distNear) * t;
    const height = CHASE.heightNear + (CHASE.heightFar - CHASE.heightNear) * t;

    switch (view) {
      case 'chase':
        this.desired.set(this.vx - fx * dist, height, this.vz - fz * dist);
        this.desiredLook.set(
          ex + Math.cos(pose.heading) * CHASE.lookAhead,
          1.15,
          ez - Math.sin(pose.heading) * CHASE.lookAhead,
        );
        break;

      case 'overhead':
        this.desired.set(this.vx - fx * 6, 46 + t * 18, this.vz - fz * 6);
        this.desiredLook.set(ex + fx * 10, 0, ez + fz * 10);
        break;

      case 'cockpit':
        this.desired.set(
          ex + Math.cos(pose.heading) * 0.15,
          1.33,
          ez - Math.sin(pose.heading) * 0.15,
        );
        this.desiredLook.set(
          ex + Math.cos(pose.heading) * 40,
          1.15,
          ez - Math.sin(pose.heading) * 40,
        );
        break;

      case 'free': {
        const ce = Math.cos(this.orbitElevation);
        this.desired.set(
          ex + Math.cos(this.orbitAzimuth) * ce * this.orbitDistance,
          Math.sin(this.orbitElevation) * this.orbitDistance,
          ez + Math.sin(this.orbitAzimuth) * ce * this.orbitDistance,
        );
        this.desiredLook.set(ex, 0, ez);
        break;
      }
    }

    // The cockpit is rigidly mounted; every other view floats.
    if (view === 'cockpit') {
      this.camPos.copy(this.desired);
      this.lookAt.copy(this.desiredLook);
    } else {
      const s = view === 'free' ? 0.004 : CHASE.camSmoothing;
      this.camPos.x = damp(this.camPos.x, this.desired.x, s, dt);
      this.camPos.y = damp(this.camPos.y, this.desired.y, s, dt);
      this.camPos.z = damp(this.camPos.z, this.desired.z, s, dt);
      this.lookAt.x = damp(this.lookAt.x, this.desiredLook.x, CHASE.targetSmoothing, dt);
      this.lookAt.y = damp(this.lookAt.y, this.desiredLook.y, CHASE.targetSmoothing, dt);
      this.lookAt.z = damp(this.lookAt.z, this.desiredLook.z, CHASE.targetSmoothing, dt);
    }

    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.lookAt);
  }

  /* ---- free-view input ---- */

  orbit(dxPixels: number, dyPixels: number): void {
    this.orbitAzimuth -= dxPixels * 0.005;
    this.orbitElevation = clamp(this.orbitElevation + dyPixels * 0.004, 0.12, 1.45);
  }

  zoom(deltaY: number): void {
    this.orbitDistance = clamp(this.orbitDistance * (1 + deltaY * 0.001), 14, 420);
  }
}
