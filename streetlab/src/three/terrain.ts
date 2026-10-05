/**
 * Ground height, sampled exactly as the backend graded it.
 *
 * A port of `map/elevation.py::Heightfield.sample`, and it must stay one: the
 * backend grades roads level against this interpolation, so any drift between
 * the two puts tarmac under or over the ground it was graded onto.
 */
import type { Terrain } from '../schema';

export type HeightFn = (x: number, y: number) => number;

export class TerrainField {
  readonly originX: number;
  readonly originY: number;
  readonly cell: number;
  readonly cols: number;
  readonly rows: number;
  /** Row-major, row 0 = SOUTH, metres. */
  readonly heights: Float64Array;
  readonly min: number;
  readonly max: number;
  readonly flat: boolean;

  private constructor(
    originX: number,
    originY: number,
    cell: number,
    cols: number,
    rows: number,
    heights: Float64Array,
    flat: boolean,
  ) {
    this.originX = originX;
    this.originY = originY;
    this.cell = cell;
    this.cols = cols;
    this.rows = rows;
    this.heights = heights;
    this.flat = flat;
    let lo = Infinity;
    let hi = -Infinity;
    for (const h of heights) {
      if (h < lo) lo = h;
      if (h > hi) hi = h;
    }
    this.min = heights.length ? lo : 0;
    this.max = heights.length ? hi : 0;
  }

  /** Flat ground at 0 when the scene carries no terrain. */
  static from(terrain: Terrain | null | undefined): TerrainField {
    if (!terrain) return new TerrainField(0, 0, 1, 2, 2, new Float64Array(4), true);
    const bin = atob(terrain.heights_b64);
    const n = terrain.cols * terrain.rows;
    if (bin.length !== n * 2) {
      throw new Error(`terrain: expected ${n * 2} bytes, got ${bin.length}`);
    }
    const heights = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const q = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      heights[i] = terrain.base_m + q * terrain.step_m;
    }
    return new TerrainField(
      terrain.origin[0],
      terrain.origin[1],
      terrain.cell_m,
      terrain.cols,
      terrain.rows,
      heights,
      false,
    );
  }

  /** Bilinear height at world (x, y); clamped to the grid's edge outside it. */
  heightAt = (x: number, y: number): number => {
    if (this.flat) return 0;
    const { cols, rows, cell, heights } = this;
    const fx = Math.min(Math.max((x - this.originX) / cell, 0), cols - 1);
    const fy = Math.min(Math.max((y - this.originY) / cell, 0), rows - 1);
    const c0 = Math.min(Math.floor(fx), Math.max(cols - 2, 0));
    const r0 = Math.min(Math.floor(fy), Math.max(rows - 2, 0));
    const tx = fx - c0;
    const ty = fy - r0;
    const c1 = Math.min(c0 + 1, cols - 1);
    const r1 = Math.min(r0 + 1, rows - 1);
    const south = heights[r0 * cols + c0] * (1 - tx) + heights[r0 * cols + c1] * tx;
    const north = heights[r1 * cols + c0] * (1 - tx) + heights[r1 * cols + c1] * tx;
    return south * (1 - ty) + north * ty;
  };
}

export interface Attitude {
  /** Ground height under the vehicle's centre. */
  y: number;
  /** Nose-up rotation about the vehicle's lateral axis, radians. */
  pitch: number;
  /** Rotation about its long axis; positive drops the right-hand side. */
  roll: number;
}

const LEVEL: Attitude = { y: 0, pitch: 0, roll: 0 };

/**
 * How a vehicle of `length` x `width` sits on the ground at world (x, y),
 * facing `heading`: height from its centre, pitch from its axles (taken at
 * 70% of its length, roughly where the wheels are), roll from its sides.
 *
 * Apply with `rotation.order = 'YZX'` and `rotation.set(roll, heading, pitch)`:
 * yaw outermost, then pitch about the body's own lateral axis, then roll.
 */
export function attitudeOn(
  heightAt: HeightFn | null,
  x: number,
  y: number,
  heading: number,
  length: number,
  width: number,
): Attitude {
  if (!heightAt) return LEVEL;
  const fx = Math.cos(heading);
  const fy = Math.sin(heading);
  const axle = length * 0.35;
  const side = width / 2;
  const front = heightAt(x + fx * axle, y + fy * axle);
  const rear = heightAt(x - fx * axle, y - fy * axle);
  // Left of travel is (-fy, fx).
  const left = heightAt(x - fy * side, y + fx * side);
  const right = heightAt(x + fy * side, y - fx * side);
  return {
    y: (front + rear) / 2,
    pitch: Math.atan2(front - rear, 2 * axle),
    roll: Math.atan2(left - right, 2 * side),
  };
}
