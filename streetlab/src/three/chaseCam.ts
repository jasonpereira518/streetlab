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
import type { HeightFn } from './terrain';

/** Closest a floating camera may get to the ground beneath it. */
const MIN_CLEARANCE_M = 1.2;

/** Where on the car the camera's line of sight lands, above the ground under it. */
const EGO_EYE_M = 1.0;

/**
 * The occlusion ray is re-cast at most this often. It is the one expensive
 * thing the rig does -- a raycast over the merged building mesh, ~0.7 ms on
 * 450 buildings and several times that on Nob Hill's 2,224 -- and its answer
 * only feeds a damped pull-in, so at 100+ fps casting every frame bought
 * nothing but frame time. 30 Hz is under 0.4 m of travel at 24 mph.
 */
const RAY_INTERVAL_S = 1 / 30;

/**
 * What may hide the car: the merged building mesh, and (optionally) the tree
 * group. One object or several; none means an open scene.
 */
export type Blockers = THREE.Object3D | readonly THREE.Object3D[] | null | undefined;

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
  /** Where `pullback` is heading, as of the last cast, and the time since it. */
  private targetPullback = 0;
  private sinceCast = 0;
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
  /**
   * The scene's ground. Every view's heights are measured from the ground
   * under the car, and a floating camera is kept above the ground under
   * itself, so a car cresting a hill never drags the view into the slope.
   */
  private ground: HeightFn | null = null;

  setGround(ground: HeightFn | null): void {
    this.ground = ground;
  }

  private groundAt(x: number, z: number): number {
    return this.ground ? this.ground(x, -z) : 0;
  }

  reset(pose: Pose, blockers?: Blockers): void {
    const g = this.groundAt(pose.x, -pose.y);
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
      g + CHASE.heightNear,
      g + EGO_EYE_M,
      blockers,
    );
    this.pullback = CHASE.distNear - dist;
    this.targetPullback = this.pullback;
    this.sinceCast = 0;
    this.camPos.set(pose.x - fx * dist, g + CHASE.heightNear, -pose.y - fz * dist);
    this.camPos.y = Math.max(this.camPos.y, this.groundAt(this.camPos.x, this.camPos.z) + MIN_CLEARANCE_M);
    this.lookAt.set(
      pose.x + fx * CHASE.lookAhead,
      g + 1.15,
      -pose.y - fz * CHASE.lookAhead,
    );
    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.lookAt);
  }

  /**
   * How far the trail can sit before a blocker gets in the way. Casts one
   * ray along the actual line of sight: from the car's eye point
   * `(ex, eyeY, ez)` to the trail point `desiredDist` behind it along
   * `(-fx, -fz)` at camera `height`. The old ray ran level at camera height,
   * which is exact for buildings taller than the camera but passes over a
   * low wall or a tree canopy that still hides the car from a camera looking
   * down at it. The returned figure is still a HORIZONTAL trail distance.
   *
   * Returns `desiredDist` unchanged when there is nothing to hit, so this is
   * a no-op (not just a coincidentally-large clamp) on scenes with no
   * blocker geometry or with geometry farther away than the trail — i.e. the
   * synthetic grid, which insets buildings behind sidewalks and lot margins.
   *
   * Blockers are tested double-sided (`castBothSides`). The real building
   * material is single-sided, so a car already inside a footprint (an OSM
   * extract can overlap a lane) used to see only back faces on the way out
   * and got an unclamped trail on the far side of the wall. Now the exit
   * face is a hit and the camera stays in the volume with the car. A ray
   * that starts outside meets a front face first, so nothing changes there.
   */
  private clampTrailDistance(
    ex: number,
    ez: number,
    fx: number,
    fz: number,
    desiredDist: number,
    height: number,
    eyeY: number,
    blockers: Blockers,
  ): number {
    if (!blockers || desiredDist <= 0) return desiredDist;
    this.rayOrigin.set(ex, eyeY, ez);
    this.rayDir.set(-fx * desiredDist, height - eyeY, -fz * desiredDist);
    const length = this.rayDir.length();
    this.rayDir.divideScalar(length);
    this.raycaster.set(this.rayOrigin, this.rayDir);
    this.raycaster.near = 0;
    this.raycaster.far = length;
    const hits = this.castBothSides(blockers);
    if (!hits.length) return desiredDist;
    // 3D distance along the sight line back to horizontal trail distance.
    const horizontal = hits[0].distance * (desiredDist / length);
    return clamp(horizontal - CHASE.clipMargin, CHASE.minTrailDist, desiredDist);
  }

  /**
   * Raycast with every blocker material temporarily double-sided, restored
   * before returning so the renderer never sees the change. Three's raycast
   * reads `material.side` to cull back faces; there is no per-call override.
   */
  private castBothSides(blockers: Blockers): THREE.Intersection[] {
    const list: THREE.Object3D[] = Array.isArray(blockers)
      ? [...(blockers as readonly THREE.Object3D[])]
      : [blockers as THREE.Object3D];
    const saved: [THREE.Material, THREE.Side][] = [];
    for (const root of list) {
      // A layer the user has switched off cannot hide the car.
      if (!root.visible) continue;
      root.traverse((o: THREE.Object3D) => {
        const m = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
        if (!m) return;
        for (const mat of Array.isArray(m) ? m : [m]) {
          saved.push([mat, mat.side]);
          mat.side = THREE.DoubleSide;
        }
      });
    }
    try {
      return this.raycaster.intersectObjects(
        list.filter((r) => r.visible),
        true,
      );
    } finally {
      for (const [mat, side] of saved) mat.side = side;
    }
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
    blockers?: Blockers,
  ): void {
    const ex = pose.x;
    const ez = -pose.y;
    const g = this.groundAt(ex, ez);

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
        this.sinceCast += dt;
        if (!blockers) {
          this.targetPullback = 0; // nothing to cast against, nothing to wait for
        } else if (this.sinceCast >= RAY_INTERVAL_S - 1e-9) {
          this.sinceCast = 0;
          this.targetPullback =
            dist -
            this.clampTrailDistance(
              this.vx,
              this.vz,
              fx,
              fz,
              dist,
              g + height,
              g + EGO_EYE_M,
              blockers,
            );
        }
        // Ease the *pullback* (how far short of the natural distance we're
        // sitting), not the distance itself — so when nothing is occluded
        // (targetPullback stays 0 every frame) `pullback` never leaves 0 and
        // the trail distance is exactly `dist`, unchanged from before this
        // feature existed. A held target is clamped to the CURRENT natural
        // distance, which speed can shrink between casts.
        const targetPullback = Math.min(this.targetPullback, dist - CHASE.minTrailDist);
        const smoothing =
          targetPullback > this.pullback ? CHASE.occlusionPullIn : CHASE.occlusionEaseOut;
        this.pullback = damp(this.pullback, targetPullback, smoothing, dt);
        const trailDist = dist - this.pullback;
        this.desired.set(this.vx - fx * trailDist, g + height, this.vz - fz * trailDist);
        this.desiredLook.set(
          ex + Math.cos(pose.heading) * CHASE.lookAhead,
          g + 1.15,
          ez - Math.sin(pose.heading) * CHASE.lookAhead,
        );
        break;
      }

      case 'overhead':
        this.desired.set(this.vx - fx * 6, g + 46 + t * 18, this.vz - fz * 6);
        this.desiredLook.set(ex + fx * 10, g, ez + fz * 10);
        break;

      case 'cockpit':
        this.desired.set(
          ex + Math.cos(pose.heading) * 0.15,
          g + 1.33,
          ez - Math.sin(pose.heading) * 0.15,
        );
        this.desiredLook.set(
          ex + Math.cos(pose.heading) * 40,
          g + 1.15,
          ez - Math.sin(pose.heading) * 40,
        );
        break;

      case 'free': {
        const ce = Math.cos(this.orbitElevation);
        this.desired.set(
          ex + Math.cos(this.orbitAzimuth) * ce * this.orbitDistance,
          g + Math.sin(this.orbitElevation) * this.orbitDistance,
          ez + Math.sin(this.orbitAzimuth) * ce * this.orbitDistance,
        );
        this.desiredLook.set(ex, g, ez);
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
      this.camPos.y = Math.max(this.camPos.y, this.groundAt(this.camPos.x, this.camPos.z) + MIN_CLEARANCE_M);
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
