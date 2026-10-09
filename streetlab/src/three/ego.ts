/**
 * The ego vehicle. Its mesh is the same builder the traffic uses (vehicles.ts);
 * the detector never sees it (layers.ts), the user does.
 */
import * as THREE from 'three/webgpu';
import type { Pose, Size } from '../schema';
import { agentMaterial } from './meshKit';
import { attitudeOn, type HeightFn } from './terrain';
import { buildVehicleGeometry, EGO_STYLE } from './vehicles';

/* ------------------------------------------------------------------ */
/* Ego vehicle                                                         */
/* ------------------------------------------------------------------ */

export class EgoVehicle {
  readonly group: THREE.Group;
  readonly mesh: THREE.Mesh;
  private readonly geometry: THREE.BufferGeometry;
  private readonly material: THREE.MeshStandardNodeMaterial;

  private readonly size: Size;

  constructor(size: Size) {
    this.size = size;
    this.geometry = buildVehicleGeometry(size, EGO_STYLE);
    this.material = agentMaterial();
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.group = new THREE.Group();
    this.group.name = 'ego';
    this.group.add(this.mesh);
  }

  /** World pose -> three.js transform. Heading maps straight to rotation.y. */
  setPose(pose: Pose, ground: HeightFn | null = null): void {
    const a = attitudeOn(ground, pose.x, pose.y, pose.heading, this.size.length, this.size.width);
    this.group.position.set(pose.x, a.y, -pose.y);
    this.group.rotation.order = 'YZX';
    this.group.rotation.set(a.roll, pose.heading, a.pitch);
  }

  /** Subtle body roll and pitch, driven by steering and acceleration. */
  setAttitude(steering: number, accel: number): void {
    this.mesh.rotation.x = THREE.MathUtils.clamp(accel * 0.012, -0.03, 0.03);
    this.mesh.rotation.z = THREE.MathUtils.clamp(-steering * 0.05, -0.04, 0.04);
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
