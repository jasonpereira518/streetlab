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

/** Ego drives the inner (left) lane; a cut-in comes from the kerb lane. */
export const EGO_LANE_INSET = LANE_W * 1.5; // 5.4 m from centreline
export const KERB_LANE_INSET = LANE_W * 0.5; // 1.8 m from centreline

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
      center_marking: 'double_yellow',
      has_sidewalk: true,
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

/** Thumbnail geometry lives in a 0..100 box; the minimap canvas scales it. */
function previewGrid(xs: number[], ys: number[]): Vec2[][] {
  const paths: Vec2[][] = [];
  for (const x of xs) paths.push([[x, 2], [x, 98]]);
  for (const y of ys) paths.push([[2, y], [98, y]]);
  return paths;
}

function previewRect(x0: number, y0: number, x1: number, y1: number): Vec2[] {
  return [
    [x0, y0],
    [x0, y1],
    [x1, y1],
    [x1, y0],
    [x0, y0],
  ];
}

export const SCENARIOS: ScenarioSummary[] = [
  {
    id: 'nob-hill-loop',
    index: 1,
    name: 'Nob Hill Loop',
    location: 'Nob Hill',
    description: 'Signalised grid circuit with a scripted cut-in',
    duration_s: 180,
    bookmarked: true,
    difficulty: 'moderate',
    preview_paths: previewGrid([22, 55, 84], [20, 52, 82]),
    preview_route: previewRect(28, 26, 78, 76),
  },
  {
    id: 'california-arterial',
    index: 2,
    name: 'California Arterial',
    location: 'Nob Hill',
    description: 'Four-lane straight with dense cross traffic',
    duration_s: 240,
    bookmarked: false,
    difficulty: 'easy',
    preview_paths: previewGrid([18, 44, 70, 92], [50]),
    preview_route: [
      [6, 44],
      [94, 44],
    ],
  },
  {
    id: 'hyde-descent',
    index: 3,
    name: 'Hyde St Descent',
    location: 'Russian Hill',
    description: 'Steep grade, blind crest, parked-car occlusion',
    duration_s: 150,
    bookmarked: false,
    difficulty: 'hard',
    preview_paths: previewGrid([30, 68], [16, 44, 72]),
    preview_route: [
      [30, 92],
      [30, 52],
      [68, 52],
      [68, 10],
    ],
  },
  {
    id: 'union-square-merge',
    index: 4,
    name: 'Union Square Merge',
    location: 'Union Square',
    description: 'Unprotected left across two lanes of oncoming traffic',
    duration_s: 120,
    bookmarked: true,
    difficulty: 'hard',
    preview_paths: previewGrid([26, 60], [30, 66]),
    preview_route: [
      [8, 30],
      [60, 30],
      [60, 92],
    ],
  },
  {
    id: 'embarcadero-night',
    index: 5,
    name: 'Embarcadero Night',
    location: 'Embarcadero',
    description: 'Low-light run with pedestrians and a stalled vehicle',
    duration_s: 200,
    bookmarked: false,
    difficulty: 'moderate',
    preview_paths: previewGrid([40, 74], [24, 58, 88]),
    preview_route: [
      [10, 88],
      [40, 88],
      [40, 24],
      [92, 24],
    ],
  },
];

/* ---------------------------------------------------------------- */
/* Assembly                                                          */
/* ---------------------------------------------------------------- */

/** Build the full static scene for a scenario. */
export function buildScene(scenarioId: string): SceneDescription {
  const scenario =
    SCENARIOS.find((s) => s.id === scenarioId) ?? SCENARIOS[0];
  return {
    type: 'scene_description',
    protocol: PROTOCOL_VERSION,
    scene_id: `scene_${scenario.id}`,
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
    catalog: SCENARIOS,
  };
}

/** Signal group a light belongs to: north-south approaches vs east-west. */
export function signalGroup(lightId: string): 'ns' | 'ew' {
  return lightId.endsWith('_n') || lightId.endsWith('_s') ? 'ns' : 'ew';
}
