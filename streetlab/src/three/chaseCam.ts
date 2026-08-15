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
  /**
   * Occlusion recovery, in the same "fraction left after one second" units
   * as the smoothing constants above. Pulling in is urgent (a wall is about
   * to fill the screen), so it is far snappier than easing back out once
   * clear — the asymmetry is what keeps a flickering ray (e.g. rounding a
   * building corner) from reading as a jump: a spurious one-frame hit snaps
   * the trail in a little, and the slow recovery smooths that back out
   * instead of the whole thing oscillating at hard-clamp speed.
   */
  occlusionPullIn: 0.00005,
  occlusionEaseOut: 0.02,
  /** Clearance kept between the camera and whatever it was clamped against. */
  clipMargin: 0.6,
  /** Never trail in closer than this, even against a wall at the bumper. */
  minTrailDist: 1.2,
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

  /**
   * How far the chase view is currently pulled in from its natural,
   * speed-based trail distance to clear a blocker — eased, so it is 0 both
   * before anything is ever occluded and, asymptotically, again once clear.
   * Starting and staying at exactly 0 when nothing is ever hit (`damp(0, 0,
   * ...)` is exactly 0, not just close) is what keeps this a true no-op on
   * scenes with open space behind the car: the synthetic grid renders
   * bit-identically to before this feature existed. Reused (never
   * reallocated) across frames so a single ray per frame is the whole
   * per-frame cost of this feature.
   */
  private pullback = 0;
  private readonly raycaster = new THREE.Raycaster();
  private readonly rayOrigin = new THREE.Vector3();
  private readonly rayDir = new THREE.Vector3();

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
   *
   * `blockers` is optional so callers that have no scene geometry handy (or
   * tests that don't care) can omit it; a route start sitting hard against a
   * building is exactly the case a scene swap can produce, so when it is
   * given the rest distance is clamped immediately rather than only on the
   * next `update()` — otherwise the very first rendered frame would show the
   * camera already embedded in the wall, easing out over the following few
   * frames instead of never having been inside it.
   */
  reset(pose: Pose, blockers?: THREE.Object3D | null): void {
    this.vx = pose.x;
    this.vz = -pose.y;
    this.vHeading = pose.heading;
    this.started = true;

    const fx = Math.cos(pose.heading);
    const fz = -Math.sin(pose.heading);
    const dist = this.clampTrailDistance(
      this.vx,
      this.vz,
      fx,
      fz,
      CHASE.distNear,
      CHASE.heightNear,
      blockers,
    );
    this.pullback = CHASE.distNear - dist;
    this.camPos.set(pose.x - fx * dist, CHASE.heightNear, -pose.y - fz * dist);
    this.lookAt.set(
      pose.x + fx * CHASE.lookAhead,
      1.15,
      -pose.y - fz * CHASE.lookAhead,
    );
    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.lookAt);
  }

  /**
   * How far the trail can sit before a blocker gets in the way. Casts a
   * single ray, horizontal at the given height, from `(ex, height, ez)`
   * toward the trail point `desiredDist` behind it along `(-fx, -fz)` — the
   * exact segment the chase view would otherwise place the camera on.
   * Buildings are uniform vertical extrusions (`world.ts`'s
   * `ExtrudeGeometry`), so testing at the camera's own height is exact: if a
   * building's footprint covers this point it fills that height too, and if
   * the camera height already clears the roof there is nothing to catch.
   *
   * Returns `desiredDist` unchanged when there is nothing to hit, so this is
   * a no-op (not just a coincidentally-large clamp) on scenes with no
   * blocker geometry or with geometry farther away than the trail — i.e. the
   * synthetic grid, which insets buildings behind sidewalks and lot margins.
   */
  private clampTrailDistance(
    ex: number,
    ez: number,
    fx: number,
    fz: number,
    desiredDist: number,
    height: number,
    blockers: THREE.Object3D | null | undefined,
  ): number {
    if (!blockers || desiredDist <= 0) return desiredDist;
    this.rayOrigin.set(ex, height, ez);
    this.rayDir.set(-fx, 0, -fz);
    this.raycaster.set(this.rayOrigin, this.rayDir);
    this.raycaster.near = 0;
    this.raycaster.far = desiredDist;
    const hits = this.raycaster.intersectObject(blockers, true);
    if (!hits.length) return desiredDist;
    return clamp(hits[0].distance - CHASE.clipMargin, CHASE.minTrailDist, desiredDist);
  }

  /**
   * `blockers` is the merged building mesh the renderer already built for
   * drawing (`world.ts`), reused as-is — no separate collision structure.
   * Only the chase view needs it: it is the only rig whose rest pose can end
   * up behind the car close enough to land inside real, kerb-flush
   * buildings; overhead/cockpit/free either stay outside building volumes by
   * construction or are rigidly mounted to the car.
   */
  update(
    pose: Pose,
    speed: number,
    view: CameraView,
    dt: number,
    blockers?: THREE.Object3D | null,
  ): void {
    const ex = pose.x;
    const ez = -pose.y;

    if (!this.started) this.reset(pose, blockers);

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
      case 'chase': {
        const targetDist = this.clampTrailDistance(this.vx, this.vz, fx, fz, dist, height, blockers);
        // Ease the *pullback* (how far short of the natural distance we're
        // sitting), not the distance itself — so when nothing is occluded
        // (targetPullback stays 0 every frame) `pullback` never leaves 0 and
        // the trail distance is exactly `dist`, unchanged from before this
        // feature existed.
        const targetPullback = dist - targetDist;
        const smoothing =
          targetPullback > this.pullback ? CHASE.occlusionPullIn : CHASE.occlusionEaseOut;
        this.pullback = damp(this.pullback, targetPullback, smoothing, dt);
        const trailDist = dist - this.pullback;
        this.desired.set(this.vx - fx * trailDist, height, this.vz - fz * trailDist);
        this.desiredLook.set(
          ex + Math.cos(pose.heading) * CHASE.lookAhead,
          1.15,
          ez - Math.sin(pose.heading) * CHASE.lookAhead,
        );
        break;
      }

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
