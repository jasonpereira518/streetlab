// @vitest-environment jsdom
/**
 * Painted lines, checked against the mesh the renderer builds.
 *
 * The US rule (MUTCD 3A.05) is about colour before pattern: yellow separates
 * opposing directions and marks the left edge of a one-way roadway; white
 * separates same-direction lanes and marks the right edge.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import * as THREE from 'three/webgpu';
import { buildWorld } from '../src/three/world';
import { DASH_M, GAP_M } from '../src/three/world';
import type { Road, SceneDescription } from '../src/schema';

const osmScene = (): SceneDescription =>
  JSON.parse(readFileSync(resolve(__dirname, 'fixtures/nobHillScene.json'), 'utf8'));

interface Mark {
  x: number;
  y: number;
  yellow: boolean;
  /** The triangle's three corners, for telling a lane line from a stop bar. */
  tri: Array<[number, number]>;
}

/** Every marking triangle's centroid, tagged by the colour it was painted. */
function marks(scene: SceneDescription): Mark[] {
  const world = buildWorld(scene);
  const mesh = world.root.getObjectByName('lane-markings') as THREE.Mesh;
  const pos = mesh.geometry.getAttribute('position');
  const col = mesh.geometry.getAttribute('color');
  const out: Mark[] = [];
  for (let i = 0; i < pos.count; i += 3) {
    let x = 0;
    let y = 0;
    const tri: Array<[number, number]> = [];
    for (let k = 0; k < 3; k++) {
      const vx = pos.getX(i + k);
      const vy = -pos.getZ(i + k);
      tri.push([vx, vy]);
      x += vx / 3;
      y += vy / 3;
    }
    // Yellow paint is much redder than it is blue; white is flat.
    out.push({ x, y, tri, yellow: col.getX(i) - col.getZ(i) > 0.05 });
  }
  return out;
}

const half = (r: Road) => ((r.lanes_forward + r.lanes_backward) * r.lane_width_m) / 2;

/** Span of a triangle projected onto a direction. */
function extent(tri: Array<[number, number]>, ux: number, uy: number): number {
  const ps = tri.map(([x, y]) => x * ux + y * uy);
  return Math.max(...ps) - Math.min(...ps);
}

/** Perpendicular offset of a point from a road, + to the LEFT of travel. */
function offsetFrom(
  road: Road,
  x: number,
  y: number,
): { lateral: number; dist: number; tx: number; ty: number } | null {
  let best: { lateral: number; dist: number; tx: number; ty: number } | null = null;
  for (let i = 1; i < road.centerline.length; i++) {
    const [ax, ay] = road.centerline[i - 1];
    const [bx, by] = road.centerline[i];
    const dx = bx - ax;
    const dy = by - ay;
    const L = Math.hypot(dx, dy);
    if (L < 1e-9) continue;
    const t = ((x - ax) * dx + (y - ay) * dy) / (L * L);
    if (t < 0 || t > 1) continue;
    const px = ax + t * dx;
    const py = ay + t * dy;
    const dist = Math.hypot(x - px, y - py);
    if (!best || dist < best.dist) {
      // `Polyline.offsetAt` puts +lateral at (-ty, +tx) of the tangent.
      best = { lateral: (-(x - px) * dy + (y - py) * dx) / L, dist, tx: dx / L, ty: dy / L };
    }
  }
  return best;
}

/**
 * Each mark paired with the road it belongs to -- the NEAREST one.
 *
 * Attribution matters and a radius does not do it: at a corner, another
 * street's edge line projects onto this one a metre or two off its centreline
 * and reads as a white line down the middle of it. An earlier version of this
 * test flagged 27 roads that way, every one of them correct.
 */
function marksByRoad(scene: SceneDescription): Map<string, Array<Mark & { lateral: number }>> {
  const out = new Map<string, Array<Mark & { lateral: number }>>();
  for (const m of marks(scene)) {
    let bestRoad: Road | null = null;
    let best: { lateral: number; dist: number; tx: number; ty: number } | null = null;
    let second: number | null = null;
    for (const road of scene.roads) {
      const hit = offsetFrom(road, m.x, m.y);
      if (!hit) continue;
      if (!best || hit.dist < best.dist) {
        second = best ? best.dist : second;
        best = hit;
        bestRoad = road;
      } else if (second === null || hit.dist < second) {
        second = hit.dist;
      }
    }
    if (!bestRoad || !best) continue;
    // Skip paint that two roads could equally claim. OSM maps some streets
    // twice (Broadway, California) and splits others into stubs a metre or
    // two apart, and near those a line belonging to one reads as a line on
    // the other. Ambiguous attribution is not evidence of a wrong colour.
    if (second !== null && second - best.dist < 1.5) continue;
    // Keep LONGITUDINAL paint only. A stop bar is white and spans the approach
    // lanes, so it legitimately reaches across the centre of a road -- it is a
    // transverse marking and says nothing about which colour divides the two
    // directions. Judging it as one flagged 27 correct roads.
    //
    // Tested as "is this a thin line" rather than "is it aligned with the
    // road", because the two are only the same question when attribution is
    // right: the last stop bar to slip through was 3.40 m by 0.50 m and had
    // been attributed to an alley running across it, which swapped the two
    // extents and made it look longitudinal. A painted line is <= 0.15 m wide
    // whichever way round you measure it; a stop bar is 0.50 m.
    const thinnest = Math.min(
      extent(m.tri, best.tx, best.ty),
      extent(m.tri, -best.ty, best.tx),
    );
    if (thinnest > 0.2) continue;
    const list = out.get(bestRoad.id) ?? [];
    list.push({ ...m, lateral: best.lateral });
    out.set(bestRoad.id, list);
  }
  return out;
}

describe('lane markings follow the US convention', () => {
  const scene = osmScene();

  it('paints no white line between opposing directions', () => {
    /**
     * Measured before this: 175 of the extract's 264 roads carried a
     * `solid_white` centre line between opposing traffic.
     */
    const byRoad = marksByRoad(scene);
    const offenders: string[] = [];
    for (const road of scene.roads) {
      if (road.lanes_backward === 0 || road.lanes_forward === 0) continue;
      const divide = -half(road) + road.lanes_forward * road.lane_width_m;
      for (const m of byRoad.get(road.id) ?? []) {
        if (m.yellow) continue;
        if (Math.abs(m.lateral - divide) < road.lane_width_m * 0.4) {
          offenders.push(`${road.name} at ${m.lateral.toFixed(2)} m`);
          break;
        }
      }
    }
    expect(offenders).toEqual([]);
  });

  it('keeps a one-way street\'s lane lines between its lanes, not on the kerb', () => {
    /**
     * A one-way road's lanes fill the WHOLE carriageway, so its dividers sit
     * at `-half + k*w`. Laying them out from the centreline, as if the forward
     * lanes occupied only one half of it, put the line on the kerb instead --
     * 37 of the full extract's roads are multi-lane one-ways.
     *
     * Built here rather than fished out of the fixture: the 200 m box happens
     * to contain no multi-lane one-way, so a fixture-driven version of this
     * test would pass without ever exercising the case.
     */
    const w = 3.6;
    const road: Road = {
      id: 'ow',
      name: 'One Way St',
      road_class: 'collector',
      centerline: [
        [0, -60],
        [0, 60],
      ],
      lanes_forward: 3,
      lanes_backward: 0,
      lane_width_m: w,
      speed_limit_mps: 13.4,
      oneway: true,
      center_marking: 'none',
      has_sidewalk: true,
    };
    const painted = marks({ ...osmScene(), roads: [road], crosswalks: [], buildings: [] });
    const h = half(road); // 5.4 m
    // Cluster to 0.1 m: a 0.12 m line puts its two triangle centroids either
    // side of the line's own centre.
    const laterals = [
      ...new Set(
        painted
          .map((m) => offsetFrom(road, m.x, m.y)!.lateral)
          .map((v) => Math.round(v * 10) / 10),
      ),
    ].sort((a, b) => a - b);

    // Two dividers, strictly inside the kerbs, one lane width apart.
    const dividers = laterals.filter((v) => Math.abs(Math.abs(v) - h) > 0.6);
    expect(dividers).toEqual([
      Math.round((-h + w) * 10) / 10,
      Math.round((-h + 2 * w) * 10) / 10,
    ]);
    // And an edge line just inside each kerb -- never ON it.
    for (const v of laterals) expect(Math.abs(v)).toBeLessThan(h);
  });

  it('marks the left edge of a one-way roadway in yellow', () => {
    /** The one place a yellow EDGE line belongs (MUTCD 3A.05). */
    const road: Road = {
      id: 'ow',
      name: 'One Way St',
      road_class: 'collector',
      centerline: [
        [0, -60],
        [0, 60],
      ],
      lanes_forward: 2,
      lanes_backward: 0,
      lane_width_m: 3.6,
      speed_limit_mps: 13.4,
      oneway: true,
      center_marking: 'none',
      has_sidewalk: true,
    };
    const painted = marks({ ...osmScene(), roads: [road], crosswalks: [], buildings: [] });
    const h = half(road);
    // Travel is +y (north), so the driver's left is WEST. `offsetAt` puts
    // +lateral at (-ty, +tx), which for a northbound tangent is -x -- so the
    // left-hand edge line lands at negative x.
    const leftEdge = painted.filter((m) => Math.abs(m.x + (h - 0.28)) < 0.1);
    const rightEdge = painted.filter((m) => Math.abs(m.x - (h - 0.28)) < 0.1);
    expect(leftEdge.length).toBeGreaterThan(0);
    expect(rightEdge.length).toBeGreaterThan(0);
    expect(leftEdge.every((m) => m.yellow)).toBe(true);
    expect(rightEdge.every((m) => !m.yellow)).toBe(true);
  });

  it('uses the standard broken-line pattern', () => {
    // MUTCD 3A.05: 10 ft of line, 30 ft of gap.
    expect(DASH_M).toBeCloseTo(3.05, 2);
    expect(GAP_M).toBeCloseTo(9.14, 2);
  });
});
