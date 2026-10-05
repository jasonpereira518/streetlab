import { expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { TerrainSchema } from '../src/schema';
import { TerrainField } from '../src/three/terrain';

/** Shared with streetlab-backend/tests/test_terrain_wire.py. */
const fixture = JSON.parse(
  readFileSync(resolve(__dirname, '../../contract/terrain_samples.json'), 'utf8'),
);

it('samples terrain exactly as the backend graded it', () => {
  const field = TerrainField.from(TerrainSchema.parse(fixture.terrain));
  fixture.points.forEach(([x, y]: [number, number], i: number) => {
    expect(field.heightAt(x, y)).toBeCloseTo(fixture.heights[i], 3);
  });
});

it('is flat ground when a scene has no terrain', () => {
  const field = TerrainField.from(null);
  expect(field.heightAt(123, -456)).toBe(0);
  expect(field.flat).toBe(true);
});
