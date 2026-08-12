// @vitest-environment jsdom
/**
 * Scene-graph tests. None of these need a GPU: geometry, transforms, pooling
 * and visibility are all CPU-side, so they run headless and deterministically.
 */
import { describe, expect, it } from 'vitest';
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
