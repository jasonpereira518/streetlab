import { describe, expect, it } from 'vitest';
import {
  cameraParamsFromThree,
  encodeBase64,
  flipRowsInPlace,
} from '../src/three/detectorCamera';

describe('cameraParamsFromThree', () => {
  it('converts Three.js Y-up into wire world coordinates', () => {
    // Three.js: x east, y up, z south. Wire: x east, y north, z up.
    const p = cameraParamsFromThree({ x: 3, y: 1.33, z: -7 }, 0.5);
    expect(p.x).toBe(3);
    expect(p.y).toBe(7); // wire north = -three z
    expect(p.z).toBe(1.33); // wire up = three y
    expect(p.yaw).toBe(0.5);
  });

  it('reports the configured field of view and aspect', () => {
    const p = cameraParamsFromThree({ x: 0, y: 0, z: 0 }, 0);
    expect(p.fov_y_deg).toBeGreaterThan(0);
    expect(p.aspect).toBeCloseTo(640 / 384, 6);
  });
});

describe('flipRowsInPlace', () => {
  it('flips the bottom-up readback into top-down image order', () => {
    // 1x2 image, one pixel per row: row0 = red, row1 = blue.
    const rgba = new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]);
    flipRowsInPlace(rgba, 1, 2);
    expect(Array.from(rgba)).toEqual([0, 0, 255, 255, 255, 0, 0, 255]);
  });

  it('is a no-op for a single row', () => {
    const rgba = new Uint8Array([1, 2, 3, 4]);
    flipRowsInPlace(rgba, 1, 1);
    expect(Array.from(rgba)).toEqual([1, 2, 3, 4]);
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
