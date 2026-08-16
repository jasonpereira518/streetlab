// @vitest-environment jsdom
/**
 * Scene-graph tests. None of these need a GPU: geometry, transforms, pooling
 * and visibility are all CPU-side, so they run headless and deterministically.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import * as THREE from 'three/webgpu';
import { MockSim } from '../src/net/mockServer';
import { buildWorld } from '../src/three/world';
import { HazardOverlay } from '../src/three/hazardOverlay';
import { PathRibbon } from '../src/three/pathRibbon';
import { ChaseCamera } from '../src/three/chaseCam';
import { TrafficFleet } from '../src/three/agents';
import { EgoVehicle } from '../src/three/ego';
import { Polyline, subtractIntervals, worldToThree } from '../src/three/meshBuilder';
import type { Detection, StateUpdate } from '../src/schema';

/** Objects that would each cost at least one draw call. */
function drawables(root: THREE.Object3D): THREE.Object3D[] {
  const out: THREE.Object3D[] = [];
  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh || (o as THREE.Sprite).isSprite) out.push(o);
  });
  return out;
}

function frameWithHazard(): { frame: StateUpdate; hazard: Detection } {
  const sim = new MockSim();
  sim.apply({ id: 'h', cmd: 'inject_hazard', kind: 'cutin' });
  for (let i = 0; i < 60 * 30; i++) {
    sim.step();
    const f = sim.frame();
    const hazard = f.detections.find((d) => d.hazard);
    if (hazard) return { frame: f, hazard };
  }
  throw new Error('mock never produced a hazard');
}

/* ------------------------------------------------------------------ */

describe('coordinate mapping', () => {
  it('sends world +y (north) to three -z', () => {
    expect(worldToThree(3, 5, 1)).toEqual([3, 1, -5]);
  });
});

describe('meshBuilder helpers', () => {
  it('subtracts intersection holes from a road span', () => {
    expect(subtractIntervals(100, [{ from: 40, to: 60 }])).toEqual([
      { from: 0, to: 40 },
      { from: 60, to: 100 },
    ]);
    // Overlapping holes merge rather than producing negative spans.
    expect(subtractIntervals(100, [
      { from: 10, to: 50 },
      { from: 40, to: 70 },
    ])).toEqual([
      { from: 0, to: 10 },
      { from: 70, to: 100 },
    ]);
  });

  it('offsets a polyline to the left of travel', () => {
    const line = new Polyline([
      [0, 0],
      [10, 0],
    ]);
    expect(line.length).toBe(10);
    expect(line.offsetAt(5, 2)).toEqual([5, 2]);
    expect(line.offsetAt(5, -2)).toEqual([5, -2]);
  });
});

describe('buildWorld', () => {
  const scene = new MockSim().scene;
  const world = buildWorld(scene);

  it('stays inside the draw-call budget', () => {
    const objects = drawables(world.root);
    expect(objects.length).toBeGreaterThan(10);
    expect(objects.length).toBeLessThan(150);
  });

  it('merges the whole city into a handful of buffers', () => {
    const named = new Set<string>();
    world.root.traverse((o) => {
      if (o.name) named.add(o.name);
    });
    for (const name of ['roads', 'sidewalks', 'lane-markings', 'crosswalks', 'buildings', 'trees']) {
      expect(named.has(name), `missing ${name}`).toBe(true);
    }
  });

  it('instances the repeated furniture', () => {
    const instanced = drawables(world.root).filter(
      (o) => (o as THREE.InstancedMesh).isInstancedMesh,
    );
    expect(instanced.length).toBeGreaterThan(3);
    const trees = instanced.find((o) => (o as THREE.InstancedMesh).count === scene.trees.length);
    expect(trees).toBeTruthy();
  });

  it('toggles a layer without disturbing the rest', () => {
    const buildings = world.root.getObjectByName('buildings')!;
    const roads = world.root.getObjectByName('roads')!;
    world.setLayerVisible('buildings', false);
    expect(buildings.visible).toBe(false);
    expect(roads.visible).toBe(true);
    world.setLayerVisible('buildings', true);
    expect(buildings.visible).toBe(true);
  });

  it('lights the signal head that matches the reported phase', () => {
    const group = world.root.getObjectByName('traffic-lights')!;
    // Lamp meshes are the ones painted in the three signal colours; poles,
    // arms, housings and the unlit lenses all use other materials.
    const LAMP_HEXES = new Set(['e5484d', 'f5a524', '22c55e']);
    const lamps = group.children.filter((c) => {
      const mat = (c as THREE.InstancedMesh).material as THREE.MeshBasicNodeMaterial;
      return Boolean(mat?.color) && LAMP_HEXES.has(mat.color.getHexString());
    }) as THREE.InstancedMesh[];
    expect(lamps).toHaveLength(3);
    const first = scene.traffic_lights[0];

    /**
     * How many of the three lamps on head 0 are scaled up (i.e. lit). Read from
     * the matrix elements rather than `decompose`, which reports a scale of 1
     * for the zero-determinant matrix used to hide an instance.
     */
    const litCount = () => {
      let n = 0;
      const m = new THREE.Matrix4();
      for (const lamp of lamps) {
        lamp.getMatrixAt(0, m);
        const e = m.elements;
        const rowLen = Math.hypot(e[0], e[1], e[2]);
        if (rowLen > 0.5) n++;
      }
      return n;
    };

    world.updateSignals([{ id: first.id, phase: 'off', time_to_change_s: null }], 0);
    expect(litCount()).toBe(0);
    world.updateSignals([{ id: first.id, phase: 'green', time_to_change_s: 4 }], 0);
    expect(litCount()).toBe(1);
    world.updateSignals([{ id: first.id, phase: 'red', time_to_change_s: 4 }], 0);
    expect(litCount()).toBe(1);
  });
});

describe('HazardOverlay', () => {
  it('draws an orange box and a label for each hazard', () => {
    const overlay = new HazardOverlay();
    const camera = new THREE.PerspectiveCamera(52, 16 / 9, 0.3, 1000);
    camera.position.set(0, 3, 20);
    const { frame, hazard } = frameWithHazard();

    overlay.update(frame.detections, camera);

    const boxes = overlay.group.children.filter(
      (c) => (c as THREE.Mesh).isMesh && c.visible,
    ) as THREE.Mesh[];
    const sprites = overlay.group.children.filter(
      (c) => (c as THREE.Sprite).isSprite && c.visible,
    ) as THREE.Sprite[];

    expect(boxes).toHaveLength(1);
    expect(sprites).toHaveLength(1);

    // Box wraps the detection, in three.js coordinates, with a little padding.
    expect(boxes[0].position.x).toBeCloseTo(hazard.pose.x, 5);
    expect(boxes[0].position.z).toBeCloseTo(-hazard.pose.y, 5);
    expect(boxes[0].rotation.y).toBeCloseTo(hazard.pose.heading, 5);
    expect(boxes[0].scale.x).toBeGreaterThan(hazard.size.length);
    expect(boxes[0].scale.x).toBeLessThan(hazard.size.length + 1);

    // Label sits above the roof and carries a texture.
    expect(sprites[0].position.y).toBeGreaterThan(hazard.size.height);
    const mat = sprites[0].material as THREE.SpriteNodeMaterial;
    expect(mat.map).toBeTruthy();

    overlay.dispose();
  });

  it('hides everything when there is no hazard', () => {
    const overlay = new HazardOverlay();
    const camera = new THREE.PerspectiveCamera();
    const { frame } = frameWithHazard();

    overlay.update(frame.detections, camera);
    overlay.update(
      frame.detections.map((d) => ({ ...d, hazard: false })),
      camera,
    );
    expect(overlay.group.children.every((c) => !c.visible)).toBe(true);
    overlay.dispose();
  });

  it('keeps labels a constant size on screen as distance changes', () => {
    const overlay = new HazardOverlay();
    const camera = new THREE.PerspectiveCamera(52, 16 / 9, 0.3, 1000);
    const { frame, hazard } = frameWithHazard();

    camera.position.set(hazard.pose.x, 3, -hazard.pose.y + 10);
    overlay.update(frame.detections, camera);
    const near = (overlay.group.children.find((c) => (c as THREE.Sprite).isSprite) as THREE.Sprite).scale.y;

    camera.position.set(hazard.pose.x, 3, -hazard.pose.y + 60);
    overlay.update(frame.detections, camera);
    const far = (overlay.group.children.find((c) => (c as THREE.Sprite).isSprite) as THREE.Sprite).scale.y;

    // Six times the distance means roughly six times the world size.
    expect(far / near).toBeGreaterThan(3);
    overlay.dispose();
  });

  it('reuses label textures instead of rebuilding them every frame', () => {
    const overlay = new HazardOverlay();
    const camera = new THREE.PerspectiveCamera();
    const { frame, hazard } = frameWithHazard();

    const seen = new Set<unknown>();
    const record = () => {
      const sprite = overlay.group.children.find(
        (c) => (c as THREE.Sprite).isSprite,
      ) as THREE.Sprite;
      seen.add((sprite.material as THREE.SpriteNodeMaterial).map);
    };

    // TTC drifts continuously; the label must not be re-rasterised per frame.
    // Over 61 frames the 0.5 s quantiser can only cross a bucket boundary once,
    // so at most three distinct plates exist (the initial one plus two buckets).
    overlay.update(frame.detections, camera);
    record();
    for (let i = 1; i <= 60; i++) {
      const nudged = frame.detections.map((d) =>
        d.id === hazard.id ? { ...d, ttc_s: (hazard.ttc_s ?? 3) + i * 0.005 } : d,
      );
      overlay.update(nudged, camera);
      record();
    }
    expect(seen.size).toBeLessThanOrEqual(3);
    overlay.dispose();
  });
});

describe('PathRibbon', () => {
  it('lays a flat strip along the plan polyline', () => {
    const ribbon = new PathRibbon();
    const sim = new MockSim();
    for (let i = 0; i < 200; i++) sim.step();
    const frame = sim.frame();

    ribbon.update(frame.plan.polyline);
    expect(ribbon.mesh.visible).toBe(true);
    expect(ribbon.pointCount).toBe(frame.plan.polyline.length);

    const pos = ribbon.mesh.geometry.attributes.position as THREE.BufferAttribute;
    const n = frame.plan.polyline.length;
    for (let i = 0; i < n; i++) {
      // Both rails hug the ground at one constant height.
      expect(pos.getY(i * 2)).toBeCloseTo(pos.getY(0), 6);
      expect(pos.getY(i * 2 + 1)).toBeCloseTo(pos.getY(0), 6);
      // And straddle the centreline symmetrically.
      const [wx, wy] = frame.plan.polyline[i];
      const midX = (pos.getX(i * 2) + pos.getX(i * 2 + 1)) / 2;
      const midZ = (pos.getZ(i * 2) + pos.getZ(i * 2 + 1)) / 2;
      expect(midX).toBeCloseTo(wx, 4);
      expect(midZ).toBeCloseTo(-wy, 4);
    }
    expect(pos.getY(0)).toBeGreaterThan(0);
    expect(pos.getY(0)).toBeLessThan(0.2);

    ribbon.dispose();
  });

  it('goes dark for a degenerate plan and comes back', () => {
    const ribbon = new PathRibbon();
    ribbon.update([[0, 0]]);
    expect(ribbon.mesh.visible).toBe(false);
    ribbon.update([
      [0, 0],
      [10, 0],
      [20, 0],
    ]);
    expect(ribbon.mesh.visible).toBe(true);
    ribbon.setVisible(false);
    expect(ribbon.mesh.visible).toBe(false);
    ribbon.dispose();
  });
});

describe('ChaseCamera', () => {
  const pose = (x: number, y: number, heading: number) => ({ x, y, heading });

  it('settles behind the ego and looks ahead of it', () => {
    const cam = new ChaseCamera(16 / 9);
    // Ego heading north (+y world) at the origin.
    const p = pose(0, 0, Math.PI / 2);
    cam.reset(p);
    for (let i = 0; i < 240; i++) cam.update(p, 12, 'chase', 1 / 60);

    // North is -z in three.js, so "behind" the car is +z.
    expect(cam.camera.position.z).toBeGreaterThan(6);
    expect(Math.abs(cam.camera.position.x)).toBeLessThan(0.5);
    expect(cam.camera.position.y).toBeGreaterThan(2.5);

    // The camera faces roughly -z, i.e. the direction of travel.
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.camera.quaternion);
    expect(forward.z).toBeLessThan(-0.8);
  });

  it('trails smoothly through a turn instead of snapping', () => {
    const cam = new ChaseCamera(16 / 9);
    cam.reset(pose(0, 0, 0));
    let prev = cam.camera.position.clone();
    let maxStep = 0;
    for (let i = 0; i < 300; i++) {
      // Drive a quarter circle of radius 20 at 10 m/s.
      const t = i / 60;
      const a = (10 * t) / 20;
      cam.update(pose(Math.sin(a) * 20, 20 - Math.cos(a) * 20, a), 10, 'chase', 1 / 60);
      maxStep = Math.max(maxStep, cam.camera.position.distanceTo(prev));
      prev = cam.camera.position.clone();
    }
    // At 10 m/s a frame moves ~0.17 m; allow generous headroom but no jumps.
    expect(maxStep).toBeLessThan(1.2);
  });

  it('puts the cockpit view at the driver and the overhead view above', () => {
    const cam = new ChaseCamera(16 / 9);
    const p = pose(10, -4, 0);
    cam.reset(p);
    cam.update(p, 8, 'cockpit', 1 / 60);
    expect(cam.camera.position.y).toBeGreaterThan(1);
    expect(cam.camera.position.y).toBeLessThan(2);
    expect(cam.camera.position.distanceTo(new THREE.Vector3(10, 1.33, 4))).toBeLessThan(1);

    for (let i = 0; i < 400; i++) cam.update(p, 8, 'overhead', 1 / 60);
    expect(cam.camera.position.y).toBeGreaterThan(30);
  });

  describe('building occlusion', () => {
    // Heading pi/2 is "north" in this rig's convention (see the test above):
    // forward is three.js -z, so the trail sits behind the car at +z.
    const p = pose(0, 0, Math.PI / 2);

    /** A flush-to-kerb wall, the shape SyntheticGrid never produces but real OSM does. */
    function wallAt(z: number): THREE.Mesh {
      const wall = new THREE.Mesh(new THREE.BoxGeometry(40, 20, 1), new THREE.MeshBasicMaterial());
      wall.position.set(0, 10, z);
      // Standalone (not parented into a rendered scene), so nothing else
      // will ever sync its matrixWorld from `position` for the raycaster.
      wall.updateMatrixWorld(true);
      return wall;
    }

    it('pulls the camera in rather than sitting inside a building', () => {
      const cam = new ChaseCamera(16 / 9);
      cam.reset(p);
      // Near face at z=3.5 — well inside the ~8.4 m rest distance, so an
      // unclamped camera would end up on the far side of the wall.
      const wall = wallAt(4);
      for (let i = 0; i < 60; i++) cam.update(p, 0, 'chase', 1 / 60, wall);

      const d = Math.hypot(cam.camera.position.x, cam.camera.position.z);
      expect(d).toBeLessThan(4);
      // And it should not have been clamped down to nothing either.
      expect(d).toBeGreaterThan(1);
    });

    // NOTE ON THE NEXT TWO TESTS: neither can fail against the pre-fix
    // chaseCam.ts, which ignores `blockers` entirely — any geometry placed
    // beyond the ray's `far` bound produces identical output whether the
    // occlusion feature exists or not, by construction. They are regression
    // guards against a *different* bug (over-eager or unconditionally wrong
    // clamping), not proof that occlusion works — that proof is the two
    // tests above. Kept and labelled honestly rather than deleted, per
    // task-8 review round 1.

    it('regression: a blocker just past the farthest possible trail is not treated as a hit', () => {
      const cam = new ChaseCamera(16 / 9);
      cam.reset(p);
      // Top speed puts the rest distance at CHASE.distFar (14.5 m); the wall's
      // near face sits a full metre past that — tight enough to catch an
      // off-by-one on the `raycaster.far = desiredDist` bound (e.g. treating
      // it as inclusive when it shouldn't be, or a stray +margin on the far
      // side), without sitting exactly on the boundary where float rounding
      // could make the assertion flaky.
      const wall = wallAt(15.5);
      for (let i = 0; i < 60; i++) cam.update(p, 30, 'chase', 1 / 60, wall);

      const d = Math.hypot(cam.camera.position.x, cam.camera.position.z);
      expect(d).toBeGreaterThan(13);
    });

    it('opens back up smoothly and without overshoot once the building is gone', () => {
      const cam = new ChaseCamera(16 / 9);
      cam.reset(p);
      const wall = wallAt(4);
      // Sit behind the wall long enough to fully pull in.
      for (let i = 0; i < 60; i++) cam.update(p, 0, 'chase', 1 / 60, wall);
      const pulledIn = Math.hypot(cam.camera.position.x, cam.camera.position.z);
      expect(pulledIn).toBeLessThan(4);

      // The building is gone (car has turned a corner) — track the distance
      // opening back up frame by frame: it should climb steadily and never
      // overshoot past the natural rest distance.
      let prev = pulledIn;
      let sawGradualStep = false;
      for (let i = 0; i < 90; i++) {
        cam.update(p, 0, 'chase', 1 / 60, null);
        const d = Math.hypot(cam.camera.position.x, cam.camera.position.z);
        expect(d).toBeGreaterThanOrEqual(prev - 1e-6); // monotonically opening up
        expect(d).toBeLessThanOrEqual(8.4 + 1e-6); // never overshoots the rest distance
        const step = d - prev;
        if (step > 1e-6 && step < 1) sawGradualStep = true;
        prev = d;
      }
      expect(sawGradualStep).toBe(true);
      expect(prev).toBeGreaterThan(8);
    });

    it('damps out a flickering ray instead of chattering every frame', () => {
      // Rounding a building corner, the ray can toggle hit/no-hit from one
      // frame to the next as it grazes the edge. A hard clamp would recompute
      // `desired` all the way back and forth between the clamped and full
      // distance every single frame; proven below by running the identical
      // scenario against a hard-clamped variant, whose peak step is ~18x
      // larger (see task-8-report.md for the measured numbers).
      const cam = new ChaseCamera(16 / 9);
      cam.reset(p);
      const wall = wallAt(4);

      let prev = cam.camera.position.clone();
      let maxStep = 0;
      for (let i = 0; i < 240; i++) {
        cam.update(p, 0, 'chase', 1 / 60, i % 2 === 0 ? wall : null);
        if (i > 60) maxStep = Math.max(maxStep, cam.camera.position.distanceTo(prev));
        prev = cam.camera.position.clone();
      }
      // Comfortably above the eased implementation's actual peak (~0.016 m)
      // and comfortably below a hard clamp's (~0.30 m) for this scenario.
      expect(maxStep).toBeLessThan(0.1);
    });

    it('regression: does not disturb the synthetic grid, which always has open space behind the car', () => {
      // Reproduces the actual call shape used against SyntheticGrid: buildings
      // exist (blockers is non-null) but are inset well behind the trail. See
      // the NOTE above the previous test — this also cannot fail against the
      // pre-fix code (it never reads `blockers` either), so it isn't proof
      // the feature works, only proof it stays out of the way when it
      // shouldn't engage. The real guarantee this documents is structural,
      // not just empirical: `pullback` eases a *delta* from the natural
      // distance (`dist - clampTrailDistance(...)`), and that delta is
      // exactly 0 — not merely close to 0 — every frame nothing is hit,
      // because `desiredDist - desiredDist` is exact IEEE-754 zero and
      // `damp(0, 0, s, dt)` is exactly 0 for any `s`/`dt` (see `units.ts`'s
      // `damp`/`lerp`). This test is the black-box confirmation of that
      // white-box argument.
      const withBlocker = new ChaseCamera(16 / 9);
      withBlocker.reset(p);
      const farWall = wallAt(200);
      for (let i = 0; i < 120; i++) withBlocker.update(p, 12, 'chase', 1 / 60, farWall);

      const withoutBlocker = new ChaseCamera(16 / 9);
      withoutBlocker.reset(p);
      for (let i = 0; i < 120; i++) withoutBlocker.update(p, 12, 'chase', 1 / 60, null);

      expect(withBlocker.camera.position.distanceTo(withoutBlocker.camera.position)).toBeLessThan(
        1e-6,
      );
    });

    it('known limitation: does not clamp when the car itself starts inside a blocker', () => {
      // Pins current behaviour rather than asserting a fix — see task-8
      // review round 1, "Important 2". clampTrailDistance always casts
      // outward from the car; that is correct for entering a wall from
      // outside (the normal case, and the only one covered by the tests
      // above), but if the car's own position is already inside a blocker's
      // volume — e.g. an OSM building footprint overlapping the drivable
      // lane — the ray would need to register an *exiting* hit against an
      // interior-facing triangle. `world.ts`'s `buildingMaterial()` sets no
      // `side`, so it defaults to `THREE.FrontSide`, and a FrontSide
      // raycast from inside a solid mesh cannot see faces whose front
      // (outward) normal points the same way as the ray — they're
      // back-facing from the ray's perspective and get culled. Deliberately
      // not fixed with a second ray: see task-8-report.md's fix-report
      // section for the reachability assessment and cost tradeoff.
      const cam = new ChaseCamera(16 / 9);
      cam.reset(p);

      // A solid, FrontSide box that *encloses the car itself* — the
      // rest-pose ray origin (vx, height, vz) = (0, ~3.1, 0) sits well
      // inside it — with an exit face at z=3, comfortably inside the ~8.4 m
      // desired trail distance.
      const enclosing = new THREE.Mesh(new THREE.BoxGeometry(40, 20, 6), new THREE.MeshBasicMaterial());
      enclosing.position.set(0, 10, 0);
      enclosing.updateMatrixWorld(true);

      for (let i = 0; i < 30; i++) cam.update(p, 0, 'chase', 1 / 60, enclosing);

      // If this ever starts failing because the distance came back clamped,
      // that's good news — it means the limitation above got fixed, and
      // this test should be deleted (or flipped into a real regression
      // test) rather than "fixed" to keep passing.
      const d = Math.hypot(cam.camera.position.x, cam.camera.position.z);
      expect(d).toBeGreaterThan(8);
    });
  });
});

describe('TrafficFleet', () => {
  it('pools one mesh per detection and recycles on disappearance', () => {
    const fleet = new TrafficFleet();
    const sim = new MockSim();
    for (let i = 0; i < 120; i++) sim.step();
    const frame = sim.frame();

    fleet.update(frame.detections, 1 / 60);
    expect(fleet.group.children.filter((c) => c.visible)).toHaveLength(3);
    // One draw call per vehicle: the whole car is a single merged mesh.
    expect(drawables(fleet.group)).toHaveLength(3);

    fleet.update(frame.detections.slice(0, 1), 1 / 60);
    expect(fleet.group.children.filter((c) => c.visible)).toHaveLength(1);

    fleet.update(frame.detections, 1 / 60);
    expect(fleet.group.children.filter((c) => c.visible)).toHaveLength(3);
    fleet.dispose();
  });

  it('converges on the reported pose', () => {
    const fleet = new TrafficFleet();
    const sim = new MockSim();
    for (let i = 0; i < 120; i++) sim.step();
    const frame = sim.frame();
    for (let i = 0; i < 30; i++) fleet.update(frame.detections, 1 / 60);

    const holder = fleet.group.children.find((c) => c.visible)!;
    const d = frame.detections[0];
    expect(holder.position.x).toBeCloseTo(d.pose.x, 2);
    expect(holder.position.z).toBeCloseTo(-d.pose.y, 2);
    fleet.dispose();
  });
});

describe('EgoVehicle', () => {
  it('is a single white mesh placed by pose', () => {
    const ego = new EgoVehicle({ length: 4.9, width: 1.96, height: 1.44 });
    expect(drawables(ego.group)).toHaveLength(1);

    ego.setPose({ x: 12, y: -7, heading: Math.PI / 2 });
    expect(ego.group.position.toArray()).toEqual([12, 0, 7]);
    expect(ego.group.rotation.y).toBeCloseTo(Math.PI / 2, 6);

    // Body roll follows steering, bounded so it never looks like a capsize.
    ego.setAttitude(0.4, 2);
    expect(ego.mesh.rotation.z).toBeLessThan(0);
    expect(Math.abs(ego.mesh.rotation.z)).toBeLessThan(0.05);
    ego.dispose();
  });
});

describe('ChaseCamera on the real Nob Hill route', () => {
  /**
   * The test the original occlusion fix never had.
   *
   * `ChaseCamera`'s other tests exercise `clampTrailDistance` in isolation —
   * they hand it a single pose and check the clamp shortened the trail. But
   * the clamp only decides a *desired* point on a straight ray cast back from
   * the damped virtual ego; `camera.position` is then separately damped toward
   * that point, so the path the camera actually travels is never the segment
   * that was ray-tested. Nothing pinned the thing users care about: across a
   * real drive, does the camera ever end up inside a building, or lose sight
   * of the car?
   *
   * Fixture is a real recorded Nob Hill drive — poses straight off the
   * backend at 60 Hz, and every OSM building within 60 m of the route (the
   * camera never trails more than ~15 m, so nothing reachable is trimmed).
   * `buildWorld` builds the mesh, so this runs against the same
   * `ExtrudeGeometry` the renderer draws, not a reimplementation of it.
   *
   * `stride` matters and is not cosmetic: the render loop consumes
   * `frameBus.latest` once per DISPLAY frame, so at 30 fps it skips every
   * other 60 Hz sim frame. Replaying all 60 Hz poses at a 30 fps `dt` would
   * hand the damping twice the settling time per metre travelled and quietly
   * flatter the result.
   *
   * WHAT THESE TESTS DO NOT PROVE, measured rather than assumed: they still
   * pass with `clampTrailDistance` stubbed out to `return desiredDist`, so
   * they are NOT evidence that the occlusion clamp works. The reason is that
   * on this route the clamp never engages at all — instrumenting `pullback`
   * across a full lap gives 0 engaged frames out of 750, max pullback 0.00 m.
   * The ego simply never gets close enough to a kerb-flush facade for the
   * trail ray to hit one. These are forward-looking guards: they fail if a
   * future change to the trail geometry (`distNear`/`distFar`, the heights,
   * `lookAhead`) starts driving the camera into buildings on real streets.
   * Anyone wanting a test that pins the CLAMP itself needs a location whose
   * geometry actually triggers it; Nob Hill is not one, and the freeform
   * address box means users can load places that are.
   */
  const fixture = JSON.parse(
    readFileSync(resolve(process.cwd(), 'tests/fixtures/nobHillChaseRoute.json'), 'utf8'),
  ) as {
    buildings: { footprint: [number, number][]; height_m: number }[];
    poses: { pose: { x: number; y: number; heading: number }; speed: number }[];
  };

  /** World (x, y, z) maps to footprint (x, -z) at height y — see world.ts's `rotateX`. */
  function insideFootprint(px: number, pz: number, poly: [number, number][]): boolean {
    let hit = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i];
      const [xj, yj] = poly[j];
      if (yi > pz !== yj > pz && px < ((xj - xi) * (pz - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  }

  function drive(dt: number, stride: number) {
    const scene = {
      buildings: fixture.buildings.map((b, i) => ({
        id: `b${i}`,
        footprint: b.footprint,
        height_m: b.height_m,
        color: '#b09070',
        roof_color: '#8a7057',
      })),
      roads: [],
      crosswalks: [],
      trees: [],
      traffic_lights: [],
      stop_signs: [],
      street_signs: [],
    } as never;
    const world = buildWorld(scene);
    const blockers = world.root.getObjectByName('buildings') ?? null;
    expect(blockers).not.toBeNull();

    const cam = new ChaseCamera(16 / 9);
    cam.reset(fixture.poses[0].pose, blockers);
    const ray = new THREE.Raycaster();
    const dir = new THREE.Vector3();
    const ego = new THREE.Vector3();
    let inside = 0;
    let blocked = 0;
    for (let i = 0; i < fixture.poses.length; i += stride) {
      const p = fixture.poses[i];
      cam.update(p.pose, p.speed, 'chase', dt, blockers);
      const c = cam.camera.position;
      for (const b of fixture.buildings) {
        if (c.y > b.height_m) continue;
        if (insideFootprint(c.x, -c.z, b.footprint)) {
          inside++;
          break;
        }
      }
      ego.set(p.pose.x, 1.0, -p.pose.y);
      dir.copy(ego).sub(c);
      const dist = dir.length();
      dir.normalize();
      ray.set(c, dir);
      ray.near = 0;
      ray.far = dist;
      if (ray.intersectObject(blockers!, true).length) blocked++;
    }
    world.dispose();
    return { inside, blocked, frames: Math.ceil(fixture.poses.length / stride) };
  }

  it('never puts the camera inside a building, at any frame rate', () => {
    for (const [dt, stride] of [
      [1 / 60, 1],
      [1 / 30, 2],
      [1 / 20, 3],
    ] as [number, number][]) {
      const { inside, frames } = drive(dt, stride);
      expect({ dt, inside, frames }).toEqual({ dt, inside: 0, frames });
    }
  });

  it('never loses sight of the car behind a building', () => {
    // Line of sight from the camera to the ego, against the same merged mesh
    // the clamp uses. A camera that is technically outside every footprint but
    // parked on the wrong side of a wall is just as broken to look at.
    const { blocked, frames } = drive(1 / 30, 2);
    expect({ blocked, frames }).toEqual({ blocked: 0, frames });
  });
});
