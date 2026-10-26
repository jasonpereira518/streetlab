/**
 * Renders `world_agents` -- ground truth, independent of perception -- as vehicles.
 *
 * Meshes are pooled per detection class and recycled by id, so a scenario that
 * cycles agents in and out does not churn GPU buffers. Poses are damped toward
 * the incoming frame, which hides the 60 Hz -> display-rate mismatch when the
 * renderer runs faster than the simulator.
 */
import * as THREE from 'three/webgpu';
import type { WorldAgent } from '../schema';
import { dampAngle } from '../units';
import {
  TRAFFIC_STYLES,
  buildVehicleGeometry,
  vehicleMaterial,
} from './ego';
import { attitudeOn, type HeightFn } from './terrain';

/** Canonical body sizes per class; the frame's own size scales these. */
const REFERENCE_SIZE: Record<string, { length: number; width: number; height: number }> = {
  car: { length: 4.6, width: 1.9, height: 1.46 },
  truck: { length: 8.2, width: 2.5, height: 3.1 },
  bus: { length: 11, width: 2.55, height: 3.2 },
  motorcycle: { length: 2.1, width: 0.8, height: 1.3 },
  cyclist: { length: 1.8, width: 0.7, height: 1.7 },
  pedestrian: { length: 0.6, width: 0.6, height: 1.75 },
  unknown: { length: 4.4, width: 1.9, height: 1.5 },
};

/** An emergency vehicle's roof bar: red and blue, alternating at 2 Hz. */
const LIGHT_HZ = 2;
const LIGHT_SIZE = { length: 0.3, height: 0.14, width: 0.45 };

interface Slot {
  group: THREE.Group;
  cls: string;
  height: number;
  /** The roof light bar, while the vehicle runs lights and siren. */
  lights: { red: THREE.Mesh; blue: THREE.Mesh } | null;
  /** Damped pose, so agents glide rather than teleport between frames. */
  x: number;
  z: number;
  heading: number;
  seen: boolean;
}

export class TrafficFleet {
  readonly group = new THREE.Group();
  private readonly material = vehicleMaterial();
  private readonly geometries = new Map<string, THREE.BufferGeometry>();
  private readonly slots = new Map<string, Slot>();
  private readonly free: THREE.Group[] = [];
  private readonly lightGeometry = new THREE.BoxGeometry(LIGHT_SIZE.length, LIGHT_SIZE.height, LIGHT_SIZE.width);
  private readonly red = new THREE.MeshBasicNodeMaterial({ color: '#ff1f2d' });
  private readonly blue = new THREE.MeshBasicNodeMaterial({ color: '#1f5bff' });

  constructor() {
    this.group.name = 'traffic';
  }

  private geometryFor(cls: string): THREE.BufferGeometry {
    let geo = this.geometries.get(cls);
    if (!geo) {
      const size = REFERENCE_SIZE[cls] ?? REFERENCE_SIZE.unknown;
      const style = TRAFFIC_STYLES[cls] ?? TRAFFIC_STYLES.unknown;
      geo = buildVehicleGeometry(size, style);
      this.geometries.set(cls, geo);
    }
    return geo;
  }

  /** `groundFor(heading)` is what a car heading that way stands on (World.groundFor). */
  update(agents: WorldAgent[], dt: number, groundFor: ((heading: number) => HeightFn | null) | null = null): void {
    for (const slot of this.slots.values()) slot.seen = false;

    for (const d of agents) {
      let slot = this.slots.get(d.id);
      if (slot && slot.cls !== d.cls) {
        this.release(d.id, slot);
        slot = undefined;
      }
      if (!slot) {
        const holder = this.free.pop() ?? new THREE.Group();
        holder.clear();
        const mesh = new THREE.Mesh(this.geometryFor(d.cls), this.material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        // The reference geometry is unit-correct for the class; scale to the
        // exact size the simulator reported.
        const ref = REFERENCE_SIZE[d.cls] ?? REFERENCE_SIZE.unknown;
        mesh.scale.set(
          d.size.length / ref.length,
          d.size.height / ref.height,
          d.size.width / ref.width,
        );
        holder.add(mesh);
        holder.visible = true;
        this.group.add(holder);
        slot = {
          group: holder,
          cls: d.cls,
          height: d.size.height,
          lights: null,
          x: d.pose.x,
          z: -d.pose.y,
          heading: d.pose.heading,
          seen: true,
        };
        this.slots.set(d.id, slot);
      }

      // Critically-damped follow; at 60 Hz this is visually instantaneous but
      // removes the stutter when the display runs at 120 Hz.
      const k = 1 - Math.pow(0.0001, dt);
      slot.x += (d.pose.x - slot.x) * k;
      slot.z += (-d.pose.y - slot.z) * k;
      slot.heading = dampAngle(slot.heading, d.pose.heading, 0.0001, dt);
      const a = attitudeOn(groundFor?.(slot.heading) ?? null, slot.x, -slot.z, slot.heading, d.size.length, d.size.width);
      slot.group.position.set(slot.x, a.y, slot.z);
      slot.group.rotation.order = 'YZX';
      slot.group.rotation.set(a.roll, slot.heading, a.pitch);
      slot.seen = true;
    }

    for (const [id, slot] of [...this.slots]) {
      if (!slot.seen) this.release(id, slot);
    }
  }

  /**
   * Light bars on the vehicles in `ids` (the detections marked `emergency`,
   * whose ids are the agents'), flashing on the wall clock `t`. Only
   * ground-truth perception marks them, so under noisy-truth or ML the
   * vehicle still runs but shows no lights.
   */
  setEmergency(ids: ReadonlySet<string>, t: number): void {
    const redOn = (t * LIGHT_HZ) % 1 < 0.5;
    for (const [id, slot] of this.slots) {
      const on = ids.has(id);
      if (on && !slot.lights) {
        const red = new THREE.Mesh(this.lightGeometry, this.red);
        const blue = new THREE.Mesh(this.lightGeometry, this.blue);
        const y = slot.height + LIGHT_SIZE.height / 2;
        red.position.set(0, y, -LIGHT_SIZE.width / 2);
        blue.position.set(0, y, LIGHT_SIZE.width / 2);
        red.name = 'emergency-light-red';
        blue.name = 'emergency-light-blue';
        slot.group.add(red, blue);
        slot.lights = { red, blue };
      } else if (!on && slot.lights) {
        slot.group.remove(slot.lights.red, slot.lights.blue);
        slot.lights = null;
      }
      if (slot.lights) {
        slot.lights.red.visible = redOn;
        slot.lights.blue.visible = !redOn;
      }
    }
  }

  private release(id: string, slot: Slot): void {
    slot.group.clear();
    slot.group.visible = false;
    this.group.remove(slot.group);
    this.free.push(slot.group);
    this.slots.delete(id);
  }

  dispose(): void {
    for (const g of this.geometries.values()) g.dispose();
    this.geometries.clear();
    this.material.dispose();
    this.lightGeometry.dispose();
    this.red.dispose();
    this.blue.dispose();
    this.slots.clear();
    this.group.clear();
  }
}
