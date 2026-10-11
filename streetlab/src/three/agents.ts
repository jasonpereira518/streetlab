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
import { buildCyclist, buildPedestrian, CYCLIST_REF, PEDESTRIAN_REF, type Figure } from './figures';
import { agentMaterial } from './meshKit';
import { attitudeOn, type HeightFn } from './terrain';
import { buildVehicleGeometry, styleFor } from './vehicles';

/** Radians of gait / crank cycle per metre travelled. */
const GAIT_RAD_PER_M = 4.2;
const CRANK_RAD_PER_M = 2.2;

/** Canonical size of the figures' meshes; the frame's own size scales them. */
const FIGURE_REF: Record<string, { length: number; width: number; height: number }> = {
  pedestrian: PEDESTRIAN_REF,
  cyclist: CYCLIST_REF,
};
/** Motorcycles are built at the size the simulator spawns them. */
const MOTORCYCLE_REF = { length: 2.1, width: 0.8, height: 1.3 };

interface Slot {
  group: THREE.Group;
  cls: string;
  /** Pedestrians and cyclists: limbs that move. */
  figure: Figure | null;
  phase: number;
  /** Damped pose, so agents glide rather than teleport between frames. */
  x: number;
  z: number;
  heading: number;
  seen: boolean;
}

export class TrafficFleet {
  readonly group = new THREE.Group();
  private readonly material = agentMaterial();
  /** One merged buffer per (class, paint, size); shared by every agent that matches. */
  private readonly geometries = new Map<string, THREE.BufferGeometry>();
  private readonly slots = new Map<string, Slot>();
  private readonly free: THREE.Group[] = [];

  constructor() {
    this.group.name = 'traffic';
  }

  private geometryFor(d: WorldAgent): THREE.BufferGeometry {
    const style = styleFor(d.cls, d.id);
    const size = d.cls === 'motorcycle' ? MOTORCYCLE_REF : d.size;
    const key = [d.cls, style.body, style.cargo ?? '', size.length, size.width, size.height].join('|');
    let geo = this.geometries.get(key);
    if (!geo) {
      geo = buildVehicleGeometry(size, style);
      this.geometries.set(key, geo);
    }
    return geo;
  }

  /** The mesh (or articulated figure) for a newly seen agent, added to `holder`. */
  private populate(holder: THREE.Group, d: WorldAgent): Figure | null {
    const ref = FIGURE_REF[d.cls];
    if (ref) {
      const figure = d.cls === 'pedestrian' ? buildPedestrian(d.id, this.material) : buildCyclist(d.id, this.material);
      // The reference figure is unit-correct for the class; scale to the size reported.
      figure.group.scale.set(d.size.length / ref.length, d.size.height / ref.height, d.size.width / ref.width);
      holder.add(figure.group);
      return figure;
    }
    const mesh = new THREE.Mesh(this.geometryFor(d), this.material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    if (d.cls === 'motorcycle') {
      mesh.scale.set(d.size.length / MOTORCYCLE_REF.length, d.size.height / MOTORCYCLE_REF.height, d.size.width / MOTORCYCLE_REF.width);
    }
    holder.add(mesh);
    return null;
  }

  update(agents: WorldAgent[], dt: number, ground: HeightFn | null = null): void {
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
        const figure = this.populate(holder, d);
        holder.visible = true;
        this.group.add(holder);
        slot = {
          group: holder,
          cls: d.cls,
          figure,
          phase: 0,
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
      const a = attitudeOn(ground, slot.x, -slot.z, slot.heading, d.size.length, d.size.width);
      slot.group.position.set(slot.x, a.y, slot.z);
      slot.group.rotation.order = 'YZX';
      slot.group.rotation.set(a.roll, slot.heading, a.pitch);
      if (slot.figure) {
        slot.phase += dt * d.speed_mps * (d.cls === 'pedestrian' ? GAIT_RAD_PER_M : CRANK_RAD_PER_M);
        slot.figure.animate(slot.phase, d.speed_mps);
      }
      slot.seen = true;
    }

    for (const [id, slot] of [...this.slots]) {
      if (!slot.seen) this.release(id, slot);
    }
  }

  private release(id: string, slot: Slot): void {
    slot.figure?.dispose();
    slot.group.clear();
    slot.group.visible = false;
    this.group.remove(slot.group);
    this.free.push(slot.group);
    this.slots.delete(id);
  }

  dispose(): void {
    for (const slot of this.slots.values()) slot.figure?.dispose();
    for (const g of this.geometries.values()) g.dispose();
    this.geometries.clear();
    this.material.dispose();
    this.slots.clear();
    this.group.clear();
  }
}
