import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  cameraParamsFromThree,
  encodeBase64,
  flipRowsInPlace,
  MOUNT_PITCH_RAD,
  shouldFlipRows,
} from '../src/three/detectorCamera';

// Cross-language pin, shared with test_geometry_projection.py's
// test_the_detector_mount_pitch_sign_agrees_with_the_frontend. Not part of
// contract/fixtures/ (that set is wire *messages*, generated from a live
// Simulation) -- a small standalone data fixture instead, so both suites
// assert against the same committed number rather than two independently
// hand-transcribed literals.
const HERE = dirname(fileURLToPath(import.meta.url));
const MOUNT_PITCH_FIXTURE = JSON.parse(
  readFileSync(join(HERE, '..', '..', 'contract', 'mount_pitch_rad.json'), 'utf8'),
) as { mount_pitch_rad: number };

describe('cameraParamsFromThree', () => {
  it('converts Three.js Y-up into wire world coordinates', () => {
    // Three.js: x east, y up, z south. Wire: x east, y north, z up.
    const p = cameraParamsFromThree({ x: 3, y: 1.33, z: -7 }, 0.5, -0.25);
    expect(p.x).toBe(3);
    expect(p.y).toBe(7); // wire north = -three z
    expect(p.z).toBe(1.33); // wire up = three y
    expect(p.yaw).toBe(0.5);
    expect(p.pitch).toBe(-0.25); // reported, not assumed away
  });

  it('reports the configured field of view and aspect', () => {
    const p = cameraParamsFromThree({ x: 0, y: 0, z: 0 }, 0, 0);
    expect(p.fov_y_deg).toBeGreaterThan(0);
    expect(p.aspect).toBeCloseTo(640 / 384, 6);
  });
});

describe('MOUNT_PITCH_RAD', () => {
  // The detector camera is mounted at 1.33 m and aims at a point 40 m ahead
  // of the EGO ORIGIN that sits 0.18 m lower — a deliberate downtilt,
  // inherited from the cockpit view, that keeps ground contact points in
  // frame. `perception/geometry.py` honours whatever pitch the wire carries,
  // so reporting a flat 0 here made every projected range land beyond the
  // object: +3.4 m at a true 30 m, +29.9 m at a true 80 m.
  //
  // The camera itself sits MOUNT_FORWARD (0.15 m) ahead of the origin, so
  // the horizontal run to the aim point is 40 - 0.15 = 39.85 m, not 40.
  const EXPECTED = -Math.atan2(0.18, 40 - 0.15); // -0.0045169078 rad

  it('is negative: the mount tilts down, and the wire calls nose-up positive', () => {
    // schema.ts: "positive tilts the view upward (nose up)". A positive value
    // here would not merely fail to correct the projection — it would double
    // the error, raising a ray that is already too shallow.
    expect(MOUNT_PITCH_RAD).toBeLessThan(0);
  });

  it('has the magnitude the lookAt geometry actually implies', () => {
    // Tight enough that a wrong horizontal run (40 instead of 39.85), a wrong
    // drop, or a hardcoded literal that has drifted from the mount constants
    // fails here rather than surfacing as metres of range error in Phase 3.
    expect(MOUNT_PITCH_RAD).toBeCloseTo(EXPECTED, 12);
    expect(MOUNT_PITCH_RAD).toBeCloseTo(-0.0045169078, 10);
    expect((MOUNT_PITCH_RAD * 180) / Math.PI).toBeCloseTo(-0.2588, 4);
  });

  it('is what the detector frame reports on the wire', () => {
    // The whole point: the derived value must reach CameraParams, not merely
    // exist in the module.
    const p = cameraParamsFromThree({ x: 0, y: 1.33, z: 0 }, 0, MOUNT_PITCH_RAD);
    expect(p.pitch).toBe(MOUNT_PITCH_RAD);
    expect(p.pitch).not.toBe(0);
  });

  it('matches the committed cross-language pitch fixture exactly', () => {
    // `EXPECTED` above is this same file's own recomputation and would pass
    // even if MOUNT_PITCH_RAD had drifted from what the fixture (and
    // test_geometry_projection.py, on the Python side) were pinned to —
    // it's the constant checking itself. This is the one assertion in the
    // suite that ties the *shipped* value to the number Python's test reads
    // from contract/mount_pitch_rad.json: re-parsing that JSON literal in
    // JS reproduces the exact bit pattern MOUNT_PITCH_RAD already is (both
    // are IEEE-754 doubles and decimal<->binary round-tripping is exactly
    // specified), so this is `===`, not `toBeCloseTo`.
    expect(MOUNT_PITCH_RAD).toBe(MOUNT_PITCH_FIXTURE.mount_pitch_rad);
  });
});

describe('flipRowsInPlace', () => {
  it('flips the bottom-up readback into top-down image order', () => {
    // 1x2 image, one pixel per row: row0 = red, row1 = blue.
    const rgba = new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]);
    flipRowsInPlace(rgba, 1, 2);
    expect(Array.from(rgba)).toEqual([0, 0, 255, 255, 255, 0, 0, 255]);
  });

  it('flips an odd height, leaving the middle row untouched', () => {
    // 1x3 image, one pixel per row: row0 red, row1 green (middle), row2 blue.
    const rgba = new Uint8Array([
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 0, 255, 255,
    ]);
    flipRowsInPlace(rgba, 1, 3);
    expect(Array.from(rgba)).toEqual([
      0, 0, 255, 255,
      0, 255, 0, 255,
      255, 0, 0, 255,
    ]);
  });
});

describe('shouldFlipRows', () => {
  it('flips for WebGL2, whose gl.readPixels origin is bottom-left', () => {
    expect(shouldFlipRows('webgl2')).toBe(true);
  });

  it('does not flip for WebGPU, whose readback origin is already top-left', () => {
    expect(shouldFlipRows('webgpu')).toBe(false);
  });
});

describe('encodeBase64', () => {
  it('round-trips through atob', () => {
    const bytes = new Uint8Array([0xff, 0xd8, 0x00, 0x41]);
    const decoded = atob(encodeBase64(bytes));
    expect(decoded.length).toBe(4);
    expect(decoded.charCodeAt(0)).toBe(0xff);
    expect(decoded.charCodeAt(3)).toBe(0x41);
  });

  it('handles payloads larger than one chunk', () => {
    const bytes = new Uint8Array(70_000).fill(7);
    expect(atob(encodeBase64(bytes)).length).toBe(70_000);
  });
});
