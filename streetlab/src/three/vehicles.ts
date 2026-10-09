/**
 * Stylised vehicle meshes.
 *
 * A whole vehicle -- body, glass, tyres, bumpers, lamps, plates -- is merged
 * into one vertex-coloured buffer, so each costs a single draw call. What the
 * detector has to recognise is a COCO car/truck/bus: that means a body with a
 * distinct darker glasshouse, visible tyres, a bumper line, a plate and lamps,
 * not a rounded blob. Truck and bus are separate silhouettes, not one box.
 *
 * Local frame: origin on the ground at the footprint centre, +X forward.
 */
import * as THREE from 'three/webgpu';
import type { Size } from '../schema';
import { buildMotorcycleGeometry, MOTORCYCLE_PAINT } from './figures';
import { boxAt, hashId, mergeParts, paint, slab, wheelAt } from './meshKit';

export type VehicleKind = 'car' | 'truck' | 'bus' | 'motorcycle';

export interface VehicleStyle {
  kind: VehicleKind;
  body: string;
  /** Cargo box (truck) -- a separate paint from the cab. */
  cargo?: string;
  /** Deterministic seed, used by the motorcycle's rider. */
  id?: string;
}

const GLASS = '#2B3947';
const TYRE = '#16171A';
const HUB = '#B9BEC4';
const BUMPER = '#2A2D32';
const PLATE = '#EDEBE0';
const HEAD = '#F4FAFF';
const TAIL = '#A81818';

export const EGO_STYLE: VehicleStyle = { kind: 'car', body: '#FFFFFF' };

const CAR_PAINT = ['#F2F2F0', '#1B1D21', '#B9BDC2', '#6C7078', '#B8282E', '#27508C', '#2F5D45', '#C9B89A', '#E5B43C', '#7A2E2E'];
const TRUCK_PAINT = ['#F2F2F0', '#2F5C9E', '#C23B31', '#E8B03A', '#3A7D55'];
const CARGO_PAINT = ['#F4F4F2', '#D5D9DD', '#9DA5AD', '#E9DFC8'];
const BUS_PAINT = ['#F2F2F0', '#E5B43C', '#2F5C9E', '#D1452F'];

/** Paint for an agent: deterministic in (class, id), so a vehicle never changes colour. */
export function styleFor(cls: string, id: string): VehicleStyle {
  const h = hashId(id);
  switch (cls) {
    case 'truck':
      return { kind: 'truck', body: TRUCK_PAINT[h % TRUCK_PAINT.length], cargo: CARGO_PAINT[(h >>> 4) % CARGO_PAINT.length], id };
    case 'bus':
      return { kind: 'bus', body: BUS_PAINT[h % BUS_PAINT.length], id };
    case 'motorcycle':
      return { kind: 'motorcycle', body: MOTORCYCLE_PAINT[h % MOTORCYCLE_PAINT.length], id };
    default:
      return { kind: 'car', body: CAR_PAINT[h % CAR_PAINT.length], id };
  }
}

/** How many distinct paints a class has -- the fleet caches one geometry per paint. */
export function paintVariants(cls: string): number {
  return cls === 'truck' ? TRUCK_PAINT.length * CARGO_PAINT.length : cls === 'bus' ? BUS_PAINT.length : cls === 'motorcycle' ? MOTORCYCLE_PAINT.length : CAR_PAINT.length;
}

/** Wheel: dark tyre with a lighter hub disc. */
function wheel(parts: THREE.BufferGeometry[], r: number, w: number, x: number, z: number): void {
  parts.push(paint(wheelAt(r, w, x, r, z), TYRE));
  const side = Math.sign(z) || 1;
  parts.push(paint(wheelAt(r * 0.55, w * 0.2, x, r, z + side * w * 0.45), HUB));
}

/** Lamps, bumpers and plates shared by every four-wheeler. */
function ends(parts: THREE.BufferGeometry[], L: number, W: number, y0: number, lampY: number, plateY: number): void {
  for (const sx of [1, -1]) {
    // Full-width bumper bar, then the plate on it.
    parts.push(paint(boxAt(0.14, 0.2, W * 0.98, sx * (L / 2 - 0.02), y0 + 0.12, 0), BUMPER));
    parts.push(paint(boxAt(0.02, 0.12, 0.5, sx * (L / 2 + 0.06), plateY, 0), PLATE));
    for (const sz of [1, -1]) {
      parts.push(paint(boxAt(0.07, 0.1, W * 0.2, sx * (L / 2 - 0.01), lampY, sz * W * 0.36), sx > 0 ? HEAD : TAIL, sx > 0 ? 0.5 : 0.22));
    }
  }
}

/** Side-profile polygon (x forward, y up) extruded across the width, centred on z = 0. */
function profile(points: [number, number][], width: number, bevel = 0.02): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  points.forEach(([x, y], i) => (i === 0 ? shape.moveTo(x, y) : shape.lineTo(x, y)));
  shape.closePath();
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(0.02, width - bevel * 2),
    bevelEnabled: bevel > 0,
    bevelThickness: bevel,
    bevelSize: bevel * 0.5,
    bevelSegments: 1,
  });
  return geo.translate(0, 0, -(width - bevel * 2) / 2);
}

function buildCar(L: number, W: number, H: number, style: VehicleStyle): THREE.BufferGeometry[] {
  const parts: THREE.BufferGeometry[] = [];
  const wheelR = Math.min(0.34, H * 0.24);
  const lower = 0.24;
  const bodyH = H * 0.42;
  const top = lower + bodyH;

  const body = slab(L, W, 0.3, bodyH, 0.07).translate(0, lower, 0);
  parts.push(paint(body, style.body));

  // Glasshouse as a sedan profile: raked windscreen, short roof, sloped rear
  // window. Body-coloured shell, with the same outline scaled down in height
  // as the dark glass, so the roof and pillars stay a frame around it.
  const x0 = -L * 0.33; // base, rear
  const x1 = L * 0.15; // base, front
  const xr0 = -L * 0.22; // roof, rear
  const xr1 = L * 0.0; // roof, front
  const ch = H - top - 0.01;
  const cabin: [number, number][] = [[x0, 0], [x1, 0], [xr1, ch], [xr0, ch]];
  parts.push(paint(profile(cabin.map(([x, y]) => [x, y + top - 0.02] as [number, number]), W * 0.84), style.body));
  const glass = cabin.map(([x, y]) => [x * 1.012 + (x > 0 ? 0.01 : -0.01), y * 0.8] as [number, number]);
  parts.push(paint(profile(glass.map(([x, y]) => [x, y + top - 0.0] as [number, number]), W * 0.865, 0.005), GLASS));
  // Mirrors.
  for (const sz of [1, -1]) {
    parts.push(paint(boxAt(0.1, 0.08, 0.1, x1 - 0.1, top + 0.08, sz * (W * 0.46 + 0.05)), style.body));
  }
  for (const sx of [1, -1]) {
    for (const sz of [1, -1]) {
      const x = sx * L * 0.31;
      const z = sz * (W / 2 - 0.1);
      // Dark wheel arch on the flank, then the tyre proud of it.
      parts.push(paint(wheelAt(wheelR * 1.22, 0.1, x, wheelR * 1.22, sz * (W / 2 - 0.03)), BUMPER));
      wheel(parts, wheelR, 0.24, x, z);
    }
  }
  ends(parts, L, W, lower, lower + bodyH * 0.62, lower + 0.17);
  return parts;
}

function buildTruck(L: number, W: number, H: number, style: VehicleStyle): THREE.BufferGeometry[] {
  const parts: THREE.BufferGeometry[] = [];
  const wheelR = Math.min(0.5, H * 0.16);
  const chassis = 0.62;
  parts.push(paint(boxAt(L * 0.96, 0.18, W * 0.7, 0, chassis - 0.14, 0), BUMPER));
  // Cab (front), windscreen and side glass.
  const cabL = L * 0.26;
  const cabH = H * 0.7;
  const cabX = L / 2 - cabL / 2;
  parts.push(paint(slab(cabL, W * 0.98, 0.22, cabH - chassis, 0.06).translate(cabX, chassis, 0), style.body));
  parts.push(paint(boxAt(0.04, cabH * 0.32, W * 0.84, L / 2 + 0.005, chassis + (cabH - chassis) * 0.7, 0), GLASS));
  parts.push(paint(boxAt(cabL * 0.55, cabH * 0.28, W * 1.0, cabX + cabL * 0.05, chassis + (cabH - chassis) * 0.68, 0), GLASS));
  // Cargo box: taller than the cab, a different paint, rear door seam.
  const boxL = L * 0.68;
  const boxX = -L / 2 + boxL / 2;
  parts.push(paint(slab(boxL, W, 0.1, H - chassis, 0.04).translate(boxX, chassis, 0), style.cargo ?? style.body));
  parts.push(paint(boxAt(0.03, (H - chassis) * 0.9, 0.03, -L / 2 - 0.005, chassis + (H - chassis) * 0.5, 0), BUMPER));
  // Axles: steer, then a tandem pair.
  for (const x of [L * 0.32, -L * 0.3, -L * 0.4]) {
    for (const sz of [1, -1]) wheel(parts, wheelR, 0.3, x, sz * (W / 2 - 0.15));
  }
  ends(parts, L, W, chassis - 0.2, chassis + 0.18, chassis + 0.02);
  return parts;
}

function buildBus(L: number, W: number, H: number, style: VehicleStyle): THREE.BufferGeometry[] {
  const parts: THREE.BufferGeometry[] = [];
  const wheelR = 0.5;
  const floor = 0.42;
  parts.push(paint(slab(L, W, 0.35, H - floor, 0.07).translate(0, floor, 0), style.body));
  // Continuous window band down both sides, large front/rear screens.
  const bandY = floor + (H - floor) * 0.5;
  parts.push(paint(slab(L * 0.93, W * 1.004, 0.3, (H - floor) * 0.34, 0.02).translate(0, bandY, 0), GLASS));
  for (const sx of [1, -1]) {
    parts.push(paint(boxAt(0.04, (H - floor) * 0.4, W * 0.88, sx * (L / 2 + 0.005), bandY + (H - floor) * 0.17, 0), GLASS));
  }
  parts.push(paint(boxAt(0.05, 0.14, W * 0.5, L / 2 + 0.01, H - 0.12, 0), '#FFB02E', 0.7));
  for (const x of [L * 0.3, -L * 0.3]) {
    for (const sz of [1, -1]) wheel(parts, wheelR, 0.32, x, sz * (W / 2 - 0.16));
  }
  ends(parts, L, W, floor - 0.06, floor + 0.4, floor + 0.2);
  return parts;
}

/**
 * Build a complete vehicle as one buffer. Origin is at the centre of the
 * footprint on the ground; +X is forward, +Y up.
 */
export function buildVehicleGeometry(size: Size, style: VehicleStyle): THREE.BufferGeometry {
  const { length: L, width: W, height: H } = size;
  if (style.kind === 'motorcycle') return buildMotorcycleGeometry(style.id ?? 'moto', style.body);
  const parts =
    style.kind === 'truck' ? buildTruck(L, W, H, style) : style.kind === 'bus' ? buildBus(L, W, H, style) : buildCar(L, W, H, style);
  return mergeParts(parts);
}
