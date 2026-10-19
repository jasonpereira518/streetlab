/**
 * Hand-authored San-Francisco-style grid used by the mock simulator.
 *
 * The layout is a 3x3 street grid centred on the origin. The ego route runs
 * clockwise around the north-east block, passing two signalised intersections
 * (Jones x California, Taylor x Pine) and two all-way stops.
 *
 *      y=+80  Pine St        ──────┬──────┬──────
 *                                  │      │
 *      y=0    California St  ──────┼──────┼──────
 *                                  │      │
 *      y=-80  Sacramento St  ──────┴──────┴──────
 *                            x=-80  x=0    x=+80
 *                          Van Ness Jones  Taylor
 */
import { makeRng } from '../units';
import type {
  Building,
  Crosswalk,
  HazardSummary,
  KerbReturn,
  PresetSummary,
  Road,
  SceneDescription,
  ScenarioSummary,
  StopSign,
  StreetSign,
  TrafficLight,
  Tree,
  Vec2,
} from '../schema';
import { PROTOCOL_VERSION } from '../schema';
import { makeRectRoute, Route } from './route';

/* ---------------------------------------------------------------- */
/* Grid definition                                                   */
/* ---------------------------------------------------------------- */

export const MAP_EXTENT = 130;
export const LANE_W = 3.6;
/** Kerb-to-building setback beyond the carriageway edge. */
const SIDEWALK_W = 2.8;

interface StreetSpec {
  id: string;
  name: string;
  axis: 'ns' | 'ew';
  /** x for north-south streets, y for east-west streets. */
  at: number;
  lanes: number; // per direction
  road_class: Road['road_class'];
  speed_mph: number;
}

export const STREETS: StreetSpec[] = [
  { id: 'st_vanness', name: 'Van Ness Ave', axis: 'ns', at: -80, lanes: 2, road_class: 'arterial', speed_mph: 35 },
  { id: 'st_jones', name: 'Jones St', axis: 'ns', at: 0, lanes: 2, road_class: 'collector', speed_mph: 25 },
  { id: 'st_taylor', name: 'Taylor St', axis: 'ns', at: 80, lanes: 2, road_class: 'collector', speed_mph: 25 },
  { id: 'st_sacramento', name: 'Sacramento St', axis: 'ew', at: -80, lanes: 1, road_class: 'residential', speed_mph: 25 },
  { id: 'st_california', name: 'California St', axis: 'ew', at: 0, lanes: 2, road_class: 'collector', speed_mph: 25 },
  { id: 'st_pine', name: 'Pine St', axis: 'ew', at: 80, lanes: 2, road_class: 'collector', speed_mph: 25 },
];

/** Carriageway half-width for a street (kerb to centreline). */
export const halfWidth = (s: StreetSpec): number => s.lanes * LANE_W;

const street = (axis: 'ns' | 'ew', at: number): StreetSpec => {
  const s = STREETS.find((v) => v.axis === axis && v.at === at);
  if (!s) throw new Error(`no ${axis} street at ${at}`);
  return s;
};

/** The block the ego route circles, in street-centreline coordinates. */
export const LOOP_BLOCK = { x0: 0, x1: 80, y0: 0, y1: 80 };

/**
 * The line the ego drives: the inner lane of `LOOP_BLOCK`, filleted at the
 * corners. One definition, used both by the mock simulator to move the car and
 * by `buildScene` to publish `reference_path` — so the line drawn on the road
 * and the car's actual motion cannot disagree.
 */
export function makeEgoRoute(): Route {
  return makeRectRoute(
    LOOP_BLOCK.x0 + EGO_LANE_INSET,
    LOOP_BLOCK.y0 + EGO_LANE_INSET,
    LOOP_BLOCK.x1 - EGO_LANE_INSET,
    LOOP_BLOCK.y1 - EGO_LANE_INSET,
    EGO_TURN_R,
    true,
  );
}

/**
 * The ego drives the KERB lane, 5.4 m out, with the block on its right; a
 * cut-in comes from the lane beside it on its LEFT, 1.8 m from the centreline.
 */
export const EGO_LANE_INSET = LANE_W * 1.5; // 5.4 m from centreline
export const KERB_LANE_INSET = LANE_W * 0.5; // 1.8 m from centreline
/**
 * The ego's right turns, and the kerb returns rounding the loop block's
 * corners concentric with them. A 10 m turn past a square kerb corner cut
 * 1.6 m across the pavement.
 */
const EGO_TURN_R = 6;

const SIGNALISED: Vec2[] = [
  [0, 0],
  [80, 80],
];
const STOP_CONTROLLED: Vec2[] = [
  [0, 80],
  [80, 0],
];

const MPH = 0.44704;

/* ---------------------------------------------------------------- */
/* Palettes                                                          */
/* ---------------------------------------------------------------- */

const FACADES = [
  '#EFE9DE', '#E3DCD1', '#E7EBEE', '#DCE3E9', '#F2EADF',
  '#E6DDD4', '#DEE6E2', '#EBE5EF', '#E4DBD2', '#DAE2E9',
  '#F0E7E1', '#E1E6E1',
];
const ROOFS = ['#A6ADB4', '#99A2AA', '#AEB4B9', '#8F979F'];

/* ---------------------------------------------------------------- */
/* Roads                                                             */
/* ---------------------------------------------------------------- */

function buildRoads(): Road[] {
  return STREETS.map((s) => {
    const centerline: Vec2[] =
      s.axis === 'ns'
        ? [
            [s.at, -MAP_EXTENT],
            [s.at, MAP_EXTENT],
          ]
        : [
            [-MAP_EXTENT, s.at],
            [MAP_EXTENT, s.at],
          ];
    return {
      id: s.id,
      name: s.name,
      road_class: s.road_class,
      centerline,
      lanes_forward: s.lanes,
      lanes_backward: s.lanes,
      lane_width_m: LANE_W,
      speed_limit_mps: s.speed_mph * MPH,
      oneway: false,
      // Yellow divides opposing traffic; broken where passing is permitted,
      // double solid once there is more than one lane each way. Mirrors
      // `_center_marking` in the backend's `map/lanes.py`.
      center_marking: s.lanes > 1 ? 'double_yellow' : 'broken_yellow',
      sidewalk_left: true,
      sidewalk_right: true,
      layer: 0,
      bridge: false,
      tunnel: false,
      edge_m: 0,
      elevation_m: [],
    };
  });
}

/* ---------------------------------------------------------------- */
/* Buildings                                                         */
/* ---------------------------------------------------------------- */

/** Inset of a block edge from the street centreline that bounds it. */
function edgeInset(axis: 'ns' | 'ew', at: number): number {
  if (Math.abs(at) >= MAP_EXTENT) return 0;
  return halfWidth(street(axis, at)) + SIDEWALK_W;
}

function buildBuildings(): Building[] {
  const rng = makeRng(0x5721a3);
  const xs = [-MAP_EXTENT, -80, 0, 80, MAP_EXTENT];
  const ys = [-MAP_EXTENT, -80, 0, 80, MAP_EXTENT];
  const out: Building[] = [];

  for (let bi = 0; bi < xs.length - 1; bi++) {
    for (let bj = 0; bj < ys.length - 1; bj++) {
      const x0 = xs[bi] + edgeInset('ns', xs[bi]);
      const x1 = xs[bi + 1] - edgeInset('ns', xs[bi + 1]);
      const y0 = ys[bj] + edgeInset('ew', ys[bj]);
      const y1 = ys[bj + 1] - edgeInset('ew', ys[bj + 1]);
      if (x1 - x0 < 12 || y1 - y0 < 12) continue;

      // Downtown-ish: the closer a block is to the origin, the taller it gets.
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      const centrality = 1 - Math.min(1, Math.hypot(cx, cy) / 150);

      for (const lot of blockLots(x0, y0, x1, y1, rng)) {
        const [lx0, ly0, lx1, ly1] = lot;
        const base = 9 + centrality * 26;
        const height = base * (0.62 + rng() * 0.85);
        out.push({
          id: `bld_${out.length.toString().padStart(2, '0')}`,
          footprint: lotFootprint(lx0, ly0, lx1, ly1, rng),
          height_m: Math.round(height * 10) / 10,
          color: FACADES[Math.floor(rng() * FACADES.length)],
          roof_color: ROOFS[Math.floor(rng() * ROOFS.length)],
        });
      }
    }
  }
  return out;
}

/** Split a block into lots along its longer axis, leaving light wells between. */
function blockLots(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  rng: () => number,
): Array<[number, number, number, number]> {
  const w = x1 - x0;
  const h = y1 - y0;
  const along = w >= h;
  const span = along ? w : h;
  const n = Math.max(1, Math.min(4, Math.round(span / 30)));
  const gap = 2.4;
  const cell = (span - gap * (n - 1)) / n;
  const lots: Array<[number, number, number, number]> = [];
  for (let i = 0; i < n; i++) {
    const a0 = (along ? x0 : y0) + i * (cell + gap);
    const a1 = a0 + cell;
    // Random depth so the block does not read as one extruded slab.
    const depth = (along ? h : w) * (0.62 + rng() * 0.3);
    const off = ((along ? h : w) - depth) * rng();
    lots.push(
      along
        ? [a0, y0 + off, a1, y0 + off + depth]
        : [x0 + off, a0, x0 + off + depth, a1],
    );
  }
  return lots;
}

/** Mostly rectangles; some get a corner notch so rooflines vary. */
function lotFootprint(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  rng: () => number,
): Vec2[] {
  if (rng() > 0.34) {
    return [
      [x0, y0],
      [x1, y0],
      [x1, y1],
      [x0, y1],
    ];
  }
  const n = Math.min(x1 - x0, y1 - y0) * (0.3 + rng() * 0.2);
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1 - n],
    [x1 - n, y1 - n],
    [x1 - n, y1],
    [x0, y1],
  ];
}

/* ---------------------------------------------------------------- */
/* Crosswalks, signals, signs, trees                                 */
/* ---------------------------------------------------------------- */

const CROSSWALK_DEPTH = 3.6;

function buildCrosswalks(): Crosswalk[] {
  const out: Crosswalk[] = [];
  const intersections = [...SIGNALISED, ...STOP_CONTROLLED];
  for (const [ix, iy] of intersections) {
    const ns = street('ns', ix);
    const ew = street('ew', iy);
    const hx = halfWidth(ns);
    const hy = halfWidth(ew);
    const legs: Array<{ c: Vec2; heading: number; len: number }> = [
      // North and south legs cross the north-south street; walk east-west.
      { c: [ix, iy + hy + CROSSWALK_DEPTH / 2 + 0.4], heading: 0, len: hx * 2 },
      { c: [ix, iy - hy - CROSSWALK_DEPTH / 2 - 0.4], heading: 0, len: hx * 2 },
      // East and west legs cross the east-west street; walk north-south.
      { c: [ix + hx + CROSSWALK_DEPTH / 2 + 0.4, iy], heading: Math.PI / 2, len: hy * 2 },
      { c: [ix - hx - CROSSWALK_DEPTH / 2 - 0.4, iy], heading: Math.PI / 2, len: hy * 2 },
    ];
    legs.forEach((leg, i) => {
      out.push({
        id: `xw_${ix}_${iy}_${i}`,
        center: leg.c,
        heading: leg.heading,
        width_m: CROSSWALK_DEPTH,
        length_m: leg.len,
        style: 'continental',
      });
    });
  }
  return out;
}

/**
 * Four mast-arm heads per signalised intersection, one facing each approach.
 * The pole sits on the far-right corner relative to approaching traffic and the
 * arm reaches back over the carriageway (see `world.ts` for the arm convention:
 * it extends along `heading` rotated -90 degrees).
 */
/**
 * The loop block's four corners, rounded concentric with the ego's turn so
 * its lane keeps the same distance from the kerb all the way round. The
 * other corners stay square: nothing turns at them.
 */
function buildKerbReturns(): KerbReturn[] {
  const out: KerbReturn[] = [];
  for (const [ix, sx] of [[LOOP_BLOCK.x0, 1], [LOOP_BLOCK.x1, -1]] as const) {
    for (const [iy, sy] of [[LOOP_BLOCK.y0, 1], [LOOP_BLOCK.y1, -1]] as const) {
      const hx = halfWidth(street('ns', ix));
      const hy = halfWidth(street('ew', iy));
      // Kerb corner, and the turn's centre; the kerb arc shares that centre.
      const k: Vec2 = [ix + sx * hx, iy + sy * hy];
      const c: Vec2 = [ix + sx * (EGO_LANE_INSET + EGO_TURN_R), iy + sy * (EGO_LANE_INSET + EGO_TURN_R)];
      out.push({
        corner: k,
        tangent_a: [k[0], c[1]],
        center: c,
        tangent_b: [c[0], k[1]],
        radius_m: c[0] * sx - k[0] * sx,
      });
    }
  }
  return out;
}

function buildTrafficLights(): TrafficLight[] {
  const out: TrafficLight[] = [];
  for (const [ix, iy] of SIGNALISED) {
    const hx = halfWidth(street('ns', ix));
    const hy = halfWidth(street('ew', iy));
    // Approach travel directions: north, south, east, west.
    const approaches: Array<{ dir: Vec2; key: string }> = [
      { dir: [0, 1], key: 'n' },
      { dir: [0, -1], key: 's' },
      { dir: [1, 0], key: 'e' },
      { dir: [-1, 0], key: 'w' },
    ];
    for (const a of approaches) {
      const [dx, dy] = a.dir;
      // Right-hand side of travel = travel direction rotated -90 degrees.
      const rx = dy;
      const ry = -dx;
      const alongClear = Math.abs(dx) > 0.5 ? hx : hy;
      const crossClear = Math.abs(dx) > 0.5 ? hy : hx;
      const px = ix + dx * (alongClear + 3.2) + rx * (crossClear + 2.6);
      const py = iy + dy * (alongClear + 3.2) + ry * (crossClear + 2.6);
      out.push({
        id: `tl_${ix}_${iy}_${a.key}`,
        position: [px, py],
        // Lamp faces back toward the approaching traffic.
        heading: Math.atan2(-dy, -dx),
        mast_arm_m: crossClear + 2.0,
        height_m: 5.6,
        stop_bar_extra_m: 0,
      });
    }
  }
  return out;
}

function buildStopSigns(): StopSign[] {
  const out: StopSign[] = [];
  for (const [ix, iy] of STOP_CONTROLLED) {
    const hx = halfWidth(street('ns', ix));
    const hy = halfWidth(street('ew', iy));
    const approaches: Array<{ dir: Vec2; key: string }> = [
      { dir: [0, 1], key: 'n' },
      { dir: [0, -1], key: 's' },
      { dir: [1, 0], key: 'e' },
      { dir: [-1, 0], key: 'w' },
    ];
    for (const a of approaches) {
      const [dx, dy] = a.dir;
      const rx = dy;
      const ry = -dx;
      const alongClear = Math.abs(dx) > 0.5 ? hx : hy;
      const crossClear = Math.abs(dx) > 0.5 ? hy : hx;
      // Near-right corner: back off along the approach, then out to the kerb.
      out.push({
        id: `ss_${ix}_${iy}_${a.key}`,
        position: [
          ix - dx * (alongClear + 2.2) + rx * (crossClear + 2.0),
          iy - dy * (alongClear + 2.2) + ry * (crossClear + 2.0),
        ],
        heading: Math.atan2(-dy, -dx),
        stop_bar_extra_m: 0,
      });
    }
  }
  return out;
}

function buildStreetSigns(): StreetSign[] {
  const out: StreetSign[] = [];
  const corners = [...SIGNALISED, ...STOP_CONTROLLED];
  for (const [ix, iy] of corners) {
    const ns = street('ns', ix);
    const ew = street('ew', iy);
    const hx = halfWidth(ns);
    const hy = halfWidth(ew);
    const cx = ix + hx + 2.4;
    const cy = iy + hy + 2.4;
    // Two blades on one post: one naming each street, each parallel to its own
    // carriageway so it reads from the approaching lane.
    out.push({
      id: `sn_${ix}_${iy}_ns`,
      position: [cx, cy],
      heading: Math.PI / 2,
      text: ns.name,
      kind: 'street_name',
    });
    out.push({
      id: `sn_${ix}_${iy}_ew`,
      position: [cx, cy],
      heading: 0,
      text: ew.name,
      kind: 'street_name',
    });
  }
  // A speed-limit sign on each of the two long approaches.
  out.push({
    id: 'sn_speed_jones',
    position: [halfWidth(street('ns', 0)) + 2.2, 34],
    heading: -Math.PI / 2,
    text: '25',
    kind: 'speed_limit',
  });
  out.push({
    id: 'sn_speed_pine',
    position: [40, 80 - halfWidth(street('ew', 80)) - 2.2],
    heading: Math.PI,
    text: '25',
    kind: 'speed_limit',
  });
  return out;
}

function buildTrees(): Tree[] {
  const rng = makeRng(0x13f7c2);
  const out: Tree[] = [];
  const SPACING = 15;
  const CLEAR = 16; // keep intersections and crosswalks open

  for (const s of STREETS) {
    const kerb = halfWidth(s) + SIDEWALK_W * 0.55;
    const crossAts = STREETS.filter((o) => o.axis !== s.axis).map((o) => o.at);
    for (let t = -MAP_EXTENT + 10; t <= MAP_EXTENT - 10; t += SPACING) {
      if (crossAts.some((c) => Math.abs(t - c) < CLEAR)) continue;
      for (const side of [-1, 1]) {
        const pos: Vec2 =
          s.axis === 'ns' ? [s.at + kerb * side, t] : [t, s.at + kerb * side];
        out.push({
          id: `tr_${out.length}`,
          position: pos,
          height_m: 5.4 + rng() * 3.2,
          canopy_radius_m: 1.9 + rng() * 1.1,
          trunk_radius_m: 0.16 + rng() * 0.08,
          variant: rng(),
        });
      }
    }
  }
  return out;
}

/* ---------------------------------------------------------------- */
/* Scenario catalog                                                  */
/* ---------------------------------------------------------------- */

/**
 * The backend's synthetic catalog, entry for entry (`SyntheticGrid` in
 * `streetlab-backend/map/scene_build.py`, as frozen in
 * `contract/fixtures/scene_description.json`): ids, names, previews and all.
 * `tests/mockParity.test.ts` holds the two together. Thumbnail geometry lives
 * in a 0..100 box; the minimap canvas scales it.
 */
export const SCENARIOS: ScenarioSummary[] = [
  {
    id: 'grid-loop',
    index: 1,
    name: 'Nob Hill Loop',
    location: 'Synthetic Grid',
    description: 'A single block circuit. The gentlest way in. 3 signalised corners, 1 all-way stop; 25-35 mph.',
    duration_s: 59.8910966198699,
    route_length_m: 280.83858188655125,
    bookmarked: true,
    difficulty: 'easy',
    preview_paths: [[[19.231,0.0],[19.231,100.0]],[[50.0,0.0],[50.0,100.0]],[[80.769,0.0],[80.769,100.0]],[[0.0,19.231],[100.0,19.231]],[[0.0,50.0],[100.0,50.0]],[[0.0,80.769],[100.0,80.769]]],
    preview_route: [[54.385,52.077],[52.473,53.094],[52.077,55.266],[52.077,57.516],[52.077,59.767],[52.077,62.017],[52.077,64.267],[52.077,66.518],[52.077,68.768],[52.077,71.018],[52.077,73.269],[52.077,75.519],[52.077,77.769],[53.094,79.681],[55.266,80.077],[57.516,80.077],[59.767,80.077],[62.017,80.077],[64.267,80.077],[66.518,80.077],[68.768,80.077],[71.018,80.077],[73.269,80.077],[75.519,80.077],[77.769,80.077],[79.681,79.06],[80.077,76.888],[80.077,74.637],[80.077,72.387],[80.077,70.137],[80.077,67.886],[80.077,65.636],[80.077,63.386],[80.077,61.136],[80.077,58.885],[80.077,56.635],[80.077,54.385],[79.06,52.473],[76.888,52.077],[74.637,52.077],[72.387,52.077],[70.137,52.077],[67.886,52.077],[65.636,52.077],[63.386,52.077],[61.136,52.077],[58.885,52.077],[56.635,52.077],[54.385,52.077]],
  },
  {
    id: 'grid-arterial',
    index: 2,
    name: 'California Arterial',
    location: 'Synthetic Grid',
    description: 'The long way round the outer streets, with cross traffic. 4 all-way stops; 25 mph.',
    duration_s: 133.21498113430556,
    route_length_m: 615.2385818865512,
    bookmarked: false,
    difficulty: 'moderate',
    preview_paths: [[[19.231,0.0],[19.231,100.0]],[[50.0,0.0],[50.0,100.0]],[[80.769,0.0],[80.769,100.0]],[[0.0,19.231],[100.0,19.231]],[[0.0,50.0],[100.0,50.0]],[[0.0,80.769],[100.0,80.769]]],
    preview_route: [[22.231,19.923],[19.923,23.541],[19.923,28.471],[19.923,33.401],[19.923,38.331],[19.923,43.261],[19.923,48.19],[19.923,53.12],[19.923,58.05],[19.923,62.98],[19.923,67.91],[19.923,72.839],[19.923,77.769],[23.541,80.077],[28.471,80.077],[33.401,80.077],[38.331,80.077],[43.261,80.077],[48.19,80.077],[53.12,80.077],[58.05,80.077],[62.98,80.077],[67.91,80.077],[72.839,80.077],[77.769,80.077],[80.077,76.459],[80.077,71.529],[80.077,66.599],[80.077,61.669],[80.077,56.739],[80.077,51.81],[80.077,46.88],[80.077,41.95],[80.077,37.02],[80.077,32.09],[80.077,27.161],[80.077,22.231],[76.459,19.923],[71.529,19.923],[66.599,19.923],[61.669,19.923],[56.739,19.923],[51.81,19.923],[46.88,19.923],[41.95,19.923],[37.02,19.923],[32.09,19.923],[27.161,19.923],[22.231,19.923]],
  },
  {
    id: 'grid-signals',
    index: 3,
    name: 'Signal Ladder',
    location: 'Synthetic Grid',
    description: 'North-west block, where the light cycle drives. 3 signalised corners, 1 all-way stop; 25-35 mph.',
    duration_s: 59.89109661986989,
    route_length_m: 280.8385818865512,
    bookmarked: false,
    difficulty: 'moderate',
    preview_paths: [[[19.231,0.0],[19.231,100.0]],[[50.0,0.0],[50.0,100.0]],[[80.769,0.0],[80.769,100.0]],[[0.0,19.231],[100.0,19.231]],[[0.0,50.0],[100.0,50.0]],[[0.0,80.769],[100.0,80.769]]],
    preview_route: [[22.231,52.077],[20.319,53.094],[19.923,55.266],[19.923,57.516],[19.923,59.767],[19.923,62.017],[19.923,64.267],[19.923,66.518],[19.923,68.768],[19.923,71.018],[19.923,73.269],[19.923,75.519],[19.923,77.769],[20.94,79.681],[23.112,80.077],[25.363,80.077],[27.613,80.077],[29.863,80.077],[32.114,80.077],[34.364,80.077],[36.614,80.077],[38.864,80.077],[41.115,80.077],[43.365,80.077],[45.615,80.077],[47.527,79.06],[47.923,76.888],[47.923,74.637],[47.923,72.387],[47.923,70.137],[47.923,67.886],[47.923,65.636],[47.923,63.386],[47.923,61.136],[47.923,58.885],[47.923,56.635],[47.923,54.385],[46.906,52.473],[44.734,52.077],[42.484,52.077],[40.233,52.077],[37.983,52.077],[35.733,52.077],[33.482,52.077],[31.232,52.077],[28.982,52.077],[26.731,52.077],[24.481,52.077],[22.231,52.077]],
  },
  {
    id: 'grid-merge',
    index: 4,
    name: 'Hyde Street Merge',
    location: 'Synthetic Grid',
    description: 'The tight block with heavy traffic. Inject a cut-in from the hazard menu. 3 signalised corners, 1 all-way stop; 25-35 mph.',
    duration_s: 59.8910966198699,
    route_length_m: 280.83858188655125,
    bookmarked: true,
    difficulty: 'hard',
    preview_paths: [[[19.231,0.0],[19.231,100.0]],[[50.0,0.0],[50.0,100.0]],[[80.769,0.0],[80.769,100.0]],[[0.0,19.231],[100.0,19.231]],[[0.0,50.0],[100.0,50.0]],[[0.0,80.769],[100.0,80.769]]],
    preview_route: [[54.385,52.077],[52.473,53.094],[52.077,55.266],[52.077,57.516],[52.077,59.767],[52.077,62.017],[52.077,64.267],[52.077,66.518],[52.077,68.768],[52.077,71.018],[52.077,73.269],[52.077,75.519],[52.077,77.769],[53.094,79.681],[55.266,80.077],[57.516,80.077],[59.767,80.077],[62.017,80.077],[64.267,80.077],[66.518,80.077],[68.768,80.077],[71.018,80.077],[73.269,80.077],[75.519,80.077],[77.769,80.077],[79.681,79.06],[80.077,76.888],[80.077,74.637],[80.077,72.387],[80.077,70.137],[80.077,67.886],[80.077,65.636],[80.077,63.386],[80.077,61.136],[80.077,58.885],[80.077,56.635],[80.077,54.385],[79.06,52.473],[76.888,52.077],[74.637,52.077],[72.387,52.077],[70.137,52.077],[67.886,52.077],[65.636,52.077],[63.386,52.077],[61.136,52.077],[58.885,52.077],[56.635,52.077],[54.385,52.077]],
  },
  {
    id: 'grid-night',
    index: 5,
    name: 'Outer Circuit',
    location: 'Synthetic Grid',
    description: 'The full perimeter. Long straights. 4 all-way stops; 25 mph.',
    duration_s: 133.21498113430556,
    route_length_m: 615.2385818865512,
    bookmarked: false,
    difficulty: 'moderate',
    preview_paths: [[[19.231,0.0],[19.231,100.0]],[[50.0,0.0],[50.0,100.0]],[[80.769,0.0],[80.769,100.0]],[[0.0,19.231],[100.0,19.231]],[[0.0,50.0],[100.0,50.0]],[[0.0,80.769],[100.0,80.769]]],
    preview_route: [[22.231,19.923],[19.923,23.541],[19.923,28.471],[19.923,33.401],[19.923,38.331],[19.923,43.261],[19.923,48.19],[19.923,53.12],[19.923,58.05],[19.923,62.98],[19.923,67.91],[19.923,72.839],[19.923,77.769],[23.541,80.077],[28.471,80.077],[33.401,80.077],[38.331,80.077],[43.261,80.077],[48.19,80.077],[53.12,80.077],[58.05,80.077],[62.98,80.077],[67.91,80.077],[72.839,80.077],[77.769,80.077],[80.077,76.459],[80.077,71.529],[80.077,66.599],[80.077,61.669],[80.077,56.739],[80.077,51.81],[80.077,46.88],[80.077,41.95],[80.077,37.02],[80.077,32.09],[80.077,27.161],[80.077,22.231],[76.459,19.923],[71.529,19.923],[66.599,19.923],[61.669,19.923],[56.739,19.923],[51.81,19.923],[46.88,19.923],[41.95,19.923],[37.02,19.923],[32.09,19.923],[27.161,19.923],[22.231,19.923]],
  },
  {
    id: 'grid-overtake',
    index: 6,
    name: 'California Overtake',
    location: 'Synthetic Grid',
    description: 'A two-block circuit whose long leg is four-lane California St: room to pass a slow car honestly. 2 signalised corners, 2 all-way stops; 25-35 mph.',
    duration_s: 94.35303887708776,
    route_length_m: 448.03858188655124,
    bookmarked: false,
    difficulty: 'moderate',
    preview_paths: [[[19.231,0.0],[19.231,100.0]],[[50.0,0.0],[50.0,100.0]],[[80.769,0.0],[80.769,100.0]],[[0.0,19.231],[100.0,19.231]],[[0.0,50.0],[100.0,50.0]],[[0.0,80.769],[100.0,80.769]]],
    preview_route: [[22.231,52.077],[19.926,54.356],[19.923,57.946],[19.923,61.536],[19.923,65.126],[19.923,68.716],[19.923,72.306],[19.923,75.896],[20.54,79.329],[23.918,80.077],[27.508,80.077],[31.099,80.077],[34.689,80.077],[38.279,80.077],[41.869,80.077],[45.459,80.077],[49.049,80.077],[52.639,80.077],[56.229,80.077],[59.819,80.077],[63.409,80.077],[66.999,80.077],[70.589,80.077],[74.179,80.077],[77.769,80.077],[80.074,77.798],[80.077,74.208],[80.077,70.618],[80.077,67.028],[80.077,63.438],[80.077,59.848],[80.077,56.258],[79.46,52.825],[76.082,52.077],[72.492,52.077],[68.901,52.077],[65.311,52.077],[61.721,52.077],[58.131,52.077],[54.541,52.077],[50.951,52.077],[47.361,52.077],[43.771,52.077],[40.181,52.077],[36.591,52.077],[33.001,52.077],[29.411,52.077],[25.821,52.077],[22.231,52.077]],
  },
  {
    id: 'grid-boulevard',
    index: 7,
    name: 'California Boulevard',
    location: 'Synthetic Grid',
    description: "A long loop: under the Bush St viaduct on Larkin, then down the divided boulevard's kerb lane. 2 signalised corners, 2 all-way stops; 25-35 mph.",
    duration_s: 105.44590454160698,
    route_length_m: 435.2385818865512,
    bookmarked: false,
    difficulty: 'moderate',
    preview_paths: [[[26.471,0.0],[26.471,100.0]],[[50.0,0.0],[50.0,100.0]],[[73.529,0.0],[73.529,100.0]],[[0.0,26.471],[100.0,26.471]],[[0.0,38.235],[100.0,38.235]],[[0.0,50.0],[100.0,50.0]],[[0.0,73.529],[100.0,73.529]]],
    preview_route: [[28.765,27.0],[27.01,28.665],[27.0,31.331],[27.0,33.998],[27.0,36.665],[27.0,39.332],[27.0,41.999],[27.0,44.665],[28.566,46.51],[31.232,46.529],[33.899,46.529],[36.566,46.529],[39.232,46.529],[41.899,46.529],[44.566,46.529],[47.233,46.529],[49.9,46.529],[52.567,46.529],[55.234,46.529],[57.901,46.529],[60.568,46.529],[63.235,46.529],[65.901,46.529],[68.568,46.529],[71.235,46.529],[72.99,44.865],[73.0,42.198],[73.0,39.532],[73.0,36.865],[73.0,34.198],[73.0,31.531],[73.0,28.864],[71.434,27.02],[68.768,27.0],[66.101,27.0],[63.434,27.0],[60.768,27.0],[58.101,27.0],[55.434,27.0],[52.767,27.0],[50.1,27.0],[47.433,27.0],[44.766,27.0],[42.099,27.0],[39.432,27.0],[36.765,27.0],[34.099,27.0],[31.432,27.0],[28.765,27.0]],
  },
];

/* ---------------------------------------------------------------- */
/* Assembly                                                          */
/* ---------------------------------------------------------------- */

/**
 * The hazard menu. Mirrors `SCENARIOS` in `streetlab-backend/sim/events.py`
 * entry for entry (`tests/mockServer.test.ts` holds the two together from
 * Task 9); the mock itself only ever stages `cut_in`.
 */
export const HAZARDS: HazardSummary[] = [
  { code: 'sudden_brake', label: 'Sudden brake', level: 'warn', group: 'ahead', ml_limitation: null },
  { code: 'cut_in', label: 'Cut-in', level: 'warn', group: 'ahead', ml_limitation: null },
  { code: 'jaywalker', label: 'Jaywalker', level: 'critical', group: 'crossing', ml_limitation: null },
  {
    code: 'obstacle',
    label: 'Obstacle',
    level: 'warn',
    group: 'ahead',
    ml_limitation: 'The detector has no class for an unclassified obstacle.',
  },
  {
    code: 'emergency_vehicle',
    label: 'Emergency vehicle',
    level: 'info',
    group: 'behind',
    ml_limitation: 'ML perception has no rear camera and cannot see emergency lights.',
  },
  { code: 'stalled_vehicle', label: 'Stalled vehicle', level: 'warn', group: 'ahead', ml_limitation: null },
  { code: 'cyclist_drift', label: 'Cyclist drift', level: 'warn', group: 'ahead', ml_limitation: null },
  {
    code: 'tailgater',
    label: 'Tailgater',
    level: 'info',
    group: 'behind',
    ml_limitation: 'ML perception has no rear camera.',
  },
  { code: 'oncoming_drift', label: 'Oncoming drift', level: 'critical', group: 'ahead', ml_limitation: null },
  { code: 'red_light_runner', label: 'Red-light runner', level: 'critical', group: 'crossing', ml_limitation: null },
];

/**
 * Walkthrough presets, minimal: the backend's catalog (`sim/presets.py`) runs
 * on its own scenario ids, so the mock carries two of its own on
 * `nob-hill-loop`. `timeline` is mock-internal `[at_s, kind]`; only `cut_in`
 * stages, so `jaywalker` exercises the declined path.
 */
export const PRESETS: Array<PresetSummary & { timeline: Array<[number, string]> }> = [
  {
    id: 'mock-cutin-20s',
    title: 'Mock cut-in, pinned',
    blurb: 'A staged cut-in and a jaywalker the mock declines, on a pinned seed.',
    what_to_watch: 'Min TTC and the reaction to the 4 s cut-in.',
    what_varies: 'Nothing: the seed is pinned, so Run and Replay match.',
    scene: 'loop',
    seed: 20,
    perception: 'ground-truth',
    params: {},
    hazards: ['cut_in', 'jaywalker'],
    duration_s: 20,
    timeline: [
      [4, 'cut_in'],
      [10, 'jaywalker'],
    ],
  },
  {
    id: 'mock-cutin-fresh',
    title: 'Mock cut-ins, fresh seed',
    blurb: 'Spontaneous cut-ins at a Poisson mean of 12 s, on a fresh seed per run.',
    what_to_watch: 'How min TTC and hard brakes move from run to run.',
    what_varies: 'The seed, and with it every cut-in interval.',
    scene: 'loop',
    seed: null,
    perception: 'ground-truth',
    params: { cutin_period_s: 12 },
    hazards: [],
    duration_s: 45,
    timeline: [],
  },
];

/** Build the full static scene for a scenario. */
export function buildScene(scenarioId: string): SceneDescription {
  const scenario =
    SCENARIOS.find((s) => s.id === scenarioId) ?? SCENARIOS[0];
  return {
    type: 'scene_description',
    protocol: PROTOCOL_VERSION,
    scene_id: `synthetic-grid:${scenario.id}`,
    scenario_id: scenario.id,
    name: scenario.name,
    location: scenario.location,
    attribution: 'Synthetic scene — no map data',
    origin: { lat: 37.7919, lon: -122.4139 },
    bounds: {
      min_x: -MAP_EXTENT,
      min_y: -MAP_EXTENT,
      max_x: MAP_EXTENT,
      max_y: MAP_EXTENT,
    },
    roads: buildRoads(),
    buildings: buildBuildings(),
    crosswalks: buildCrosswalks(),
    traffic_lights: buildTrafficLights(),
    stop_signs: buildStopSigns(),
    trees: buildTrees(),
    street_signs: buildStreetSigns(),
    kerb_returns: buildKerbReturns(),
    reference_path: (() => {
      const r = makeEgoRoute();
      return r.polyline(0, r.length, 2);
    })(),
    terrain: null,
    catalog: SCENARIOS,
    hazards: HAZARDS,
    presets: PRESETS.map(({ timeline: _timeline, ...summary }) => summary),
    preset_id: null,
    seed: 0,
  };
}

/** Signal group a light belongs to: north-south approaches vs east-west. */
export function signalGroup(lightId: string): 'ns' | 'ew' {
  return lightId.endsWith('_n') || lightId.endsWith('_s') ? 'ns' : 'ew';
}
