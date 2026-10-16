// @vitest-environment jsdom
/**
 * Detector-view isolation. The detector camera must photograph world geometry
 * and nothing else -- no annotation we draw for humans, no ego mesh -- and no UI
 * toggle may ever take world geometry out of its view. Pinned against the real
 * scene objects, not a list of names, so a new overlay that forgets to opt in
 * fails here.
 */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { MockSim } from '../src/net/mockServer';
import { TrafficFleet } from '../src/three/agents';
import { EgoVehicle } from '../src/three/ego';
import { HazardOverlay } from '../src/three/hazardOverlay';
import { PathRibbon } from '../src/three/pathRibbon';
import { createShadowBoxes } from '../src/three/shadowBoxes';
import { buildWorld } from '../src/three/world';
import {
  CH,
  DETECTOR_LAYER_MASK,
  TOGGLE_CHANNEL,
  applyMainLayers,
  assignChannel,
  setDetectorLayers,
} from '../src/three/layers';
import { LAYER_KEYS } from '../src/schema';
import type { LayerKey } from '../src/schema';

const allLayers = (on: boolean) =>
  Object.fromEntries(LAYER_KEYS.map((k: LayerKey) => [k, on])) as Record<LayerKey, boolean>;

function meshes(root: THREE.Object3D): THREE.Object3D[] {
  const out: THREE.Object3D[] = [];
  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh || (o as THREE.Sprite).isSprite) out.push(o);
  });
  return out;
}

function build() {
  const sim = new MockSim();
  for (let i = 0; i < 120; i++) sim.step();
  const frame = sim.frame();
  const scene = new THREE.Scene();
  const world = buildWorld(sim.scene);
  const fleet = new TrafficFleet();
  const hazards = new HazardOverlay();
  const shadow = createShadowBoxes(scene);
  const ribbon = new PathRibbon();
  const ego = new EgoVehicle({ length: 4.9, width: 1.96, height: 1.44 });
  const radar = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
  ego.group.add(radar);
  assignChannel(ego.group, CH.EGO);
  assignChannel(radar, CH.OVERLAY);
  assignChannel(ribbon.mesh, CH.OVERLAY);

  fleet.update(frame.world_agents, 1 / 60);
  // Force pooled overlay objects into existence: a hazard and a shadow box.
  hazards.update(
    frame.detections.map((d) => ({ ...d, hazard: true, hazard_label: 'x' })),
    new THREE.PerspectiveCamera(),
  );
  shadow.update(frame.detections);
  scene.add(world.root, fleet.group, hazards.group, ribbon.mesh, ego.group);
  return { scene, world, fleet, hazards, ribbon, ego, radar, frame };
}

describe('detector view isolation', () => {
  const detector = new THREE.PerspectiveCamera();
  setDetectorLayers(detector);
  const sees = (o: THREE.Object3D) => o.layers.test(detector.layers);

  it('excludes every overlay object and the ego, includes the fleet and the world', () => {
    const s = build();
    expect(s.fleet.group.children.length).toBeGreaterThan(0);

    const overlays = [
      ...meshes(s.hazards.group),
      ...meshes(s.scene.getObjectByName('shadow-detections') ?? new THREE.Group()),
      s.ribbon.mesh,
      s.radar,
    ];
    expect(meshes(s.hazards.group).length).toBeGreaterThan(0);
    for (const o of overlays) expect(sees(o), o.name || o.type).toBe(false);
    for (const o of meshes(s.ego.group)) {
      if (o !== s.radar) expect(sees(o), 'ego mesh').toBe(false);
    }

    for (const o of meshes(s.fleet.group)) expect(sees(o), 'fleet').toBe(true);
    for (const o of meshes(s.world.root)) {
      // The reference path is an annotation of the route, not a road feature.
      if (o.name === 'reference-path') expect(sees(o)).toBe(false);
      else expect(sees(o), o.name || o.type).toBe(true);
    }
  });

  it('keeps world geometry in the detector mask whatever the layer toggles say', () => {
    const s = build();
    for (const on of [true, false]) {
      const main = new THREE.PerspectiveCamera();
      applyMainLayers(main, allLayers(on));
      // Toggling never touches the detector camera.
      expect(detector.layers.mask).toBe(DETECTOR_LAYER_MASK);
      for (const o of meshes(s.fleet.group)) expect(sees(o)).toBe(true);
    }
  });

  it('lets a toggle hide a category from the main camera only', () => {
    const s = build();
    const buildings = s.world.root.getObjectByName('buildings')!;
    const main = new THREE.PerspectiveCamera();

    applyMainLayers(main, allLayers(true));
    expect(buildings.layers.test(main.layers)).toBe(true);

    applyMainLayers(main, { ...allLayers(true), buildings: false });
    expect(buildings.layers.test(main.layers)).toBe(false);
    expect(buildings.layers.test(detector.layers)).toBe(true);
  });

  it('puts every toggle on a channel the detector mask contains and the overlay is not in', () => {
    for (const channel of Object.values(TOGGLE_CHANNEL)) {
      expect(DETECTOR_LAYER_MASK & (1 << channel!)).not.toBe(0);
    }
    expect(DETECTOR_LAYER_MASK & (1 << CH.OVERLAY)).toBe(0);
    expect(DETECTOR_LAYER_MASK & (1 << CH.EGO)).toBe(0);
  });

  it('shows the user overlay, ego and fleet regardless of toggles', () => {
    const s = build();
    const main = new THREE.PerspectiveCamera();
    applyMainLayers(main, allLayers(false));
    expect(s.ribbon.mesh.layers.test(main.layers)).toBe(true);
    expect(s.ego.group.layers.test(main.layers)).toBe(true);
    for (const o of meshes(s.fleet.group)) expect(o.layers.test(main.layers)).toBe(true);
  });
});
