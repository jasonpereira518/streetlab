/**
 * Translucent blue ribbon laid over the road along `plan.polyline`.
 *
 * Geometry is allocated once for a maximum point count and rewritten in place
 * every frame — no per-frame allocation, no buffer reupload beyond the vertices
 * actually in use. A TSL material adds a flow pulse travelling down the ribbon
 * and soft edges, so it reads as a projected plan rather than a flat decal.
 */
import * as THREE from 'three/webgpu';
import { color, float, mix, smoothstep, time, uniform, uv } from 'three/tsl';
import type { Vec2 } from '../schema';
import { color as tokens } from '../ui/theme';

const MAX_POINTS = 192;
/** Height above the carriageway: clear of lane markings, still hugging it. */
const RIDE_HEIGHT = 0.085;
const HALF_WIDTH = 1.05;

export class PathRibbon {
  readonly mesh: THREE.Mesh;
  private readonly geometry: THREE.BufferGeometry;
  private readonly material: THREE.MeshBasicNodeMaterial;
  private readonly positions: Float32Array;
  private readonly uvs: Float32Array;
  private readonly opacity = uniform(0.55);
  private lastCount = 0;

  constructor() {
    this.geometry = new THREE.BufferGeometry();
    this.positions = new Float32Array(MAX_POINTS * 2 * 3);
    this.uvs = new Float32Array(MAX_POINTS * 2 * 2);

    const index: number[] = [];
    for (let i = 0; i < MAX_POINTS - 1; i++) {
      const a = i * 2;
      index.push(a, a + 1, a + 3, a, a + 3, a + 2);
    }
    this.geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage),
    );
    this.geometry.setAttribute(
      'uv',
      new THREE.BufferAttribute(this.uvs, 2).setUsage(THREE.DynamicDrawUsage),
    );
    this.geometry.setIndex(index);
    this.geometry.setDrawRange(0, 0);
    // The ribbon moves with the car; a stale bounding sphere would cull it.
    this.geometry.boundingSphere = new THREE.Sphere(
      new THREE.Vector3(),
      Number.POSITIVE_INFINITY,
    );

    this.material = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    });

    const u = uv().x;
    const v = uv().y;

    // Soft lateral edges.
    const edge = smoothstep(float(0), float(0.26), v).mul(
      smoothstep(float(1), float(0.74), v),
    );
    // Fade out toward the far end of the plan.
    const tail = smoothstep(float(1.0), float(0.45), u);
    // Flow pulses travelling away from the car.
    const flow = u.mul(4.5).sub(time.mul(0.55)).fract();
    const pulse = smoothstep(float(0), float(0.4), flow).mul(
      smoothstep(float(1), float(0.55), flow),
    );

    this.material.colorNode = mix(
      color(tokens.plan),
      color('#8FC2FF'),
      pulse.mul(0.85),
    );
    this.material.opacityNode = edge
      .mul(tail)
      .mul(this.opacity)
      .mul(pulse.mul(0.35).add(0.72));

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'plan-ribbon';
    this.mesh.renderOrder = 2;
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
  }

  setOpacity(value: number): void {
    this.opacity.value = value;
  }

  /**
   * Rebuild the strip from a world-space polyline. Vertices are offset
   * perpendicular to the local tangent, so the ribbon keeps a constant width
   * through curves.
   */
  update(polyline: Vec2[]): void {
    const n = Math.min(polyline.length, MAX_POINTS);
    if (n < 2) {
      this.mesh.visible = false;
      this.geometry.setDrawRange(0, 0);
      return;
    }

    // Total length first, so the flow pulse has a stable metre-based rate.
    let total = 0;
    for (let i = 1; i < n; i++) {
      total += Math.hypot(
        polyline[i][0] - polyline[i - 1][0],
        polyline[i][1] - polyline[i - 1][1],
      );
    }
    if (total < 1e-3) {
      this.mesh.visible = false;
      this.geometry.setDrawRange(0, 0);
      return;
    }

    let travelled = 0;
    for (let i = 0; i < n; i++) {
      const p = polyline[i];
      const prev = polyline[Math.max(0, i - 1)];
      const next = polyline[Math.min(n - 1, i + 1)];
      let tx = next[0] - prev[0];
      let ty = next[1] - prev[1];
      const len = Math.hypot(tx, ty) || 1;
      tx /= len;
      ty /= len;
      // Left normal in world space.
      const nx = -ty;
      const ny = tx;

      if (i > 0) {
        travelled += Math.hypot(p[0] - polyline[i - 1][0], p[1] - polyline[i - 1][1]);
      }
      const u = travelled / total;

      const lx = p[0] + nx * HALF_WIDTH;
      const ly = p[1] + ny * HALF_WIDTH;
      const rx = p[0] - nx * HALF_WIDTH;
      const ry = p[1] - ny * HALF_WIDTH;

      const a = i * 6;
      this.positions[a] = lx;
      this.positions[a + 1] = RIDE_HEIGHT;
      this.positions[a + 2] = -ly;
      this.positions[a + 3] = rx;
      this.positions[a + 4] = RIDE_HEIGHT;
      this.positions[a + 5] = -ry;

      const b = i * 4;
      this.uvs[b] = u;
      this.uvs[b + 1] = 0;
      this.uvs[b + 2] = u;
      this.uvs[b + 3] = 1;
    }

    this.lastCount = n;
    this.geometry.setDrawRange(0, (n - 1) * 6);
    (this.geometry.attributes.position as THREE.BufferAttribute).addUpdateRange(
      0,
      n * 6,
    );
    (this.geometry.attributes.uv as THREE.BufferAttribute).addUpdateRange(0, n * 4);
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.uv.needsUpdate = true;
    this.mesh.visible = true;
  }

  get pointCount(): number {
    return this.lastCount;
  }

  setVisible(visible: boolean): void {
    this.mesh.visible = visible && this.lastCount >= 2;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
