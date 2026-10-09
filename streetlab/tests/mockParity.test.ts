/**
 * The in-process mock (`?mock=1`) must hand the UI the scene the backend would.
 *
 * `contract/fixtures/scene_description.json` is generated from the real
 * backend's `SyntheticGrid`, so it is the reference. These tests fail when the
 * mock grows or loses a field, a field changes type, or its scenario catalog
 * or hazard menu drifts from the backend's.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { buildScene } from '../src/net/mockCity';

const real = JSON.parse(
  readFileSync(resolve(process.cwd(), '../contract/fixtures/scene_description.json'), 'utf8'),
);

/** Every key path with its JSON type; arrays contribute their first element. */
function shape(v: unknown, path: string, out: Map<string, string>): Map<string, string> {
  if (Array.isArray(v)) {
    out.set(path, 'array');
    if (v.length) shape(v[0], `${path}[]`, out);
  } else if (v && typeof v === 'object') {
    out.set(path, 'object');
    for (const [k, x] of Object.entries(v)) shape(x, `${path}.${k}`, out);
  } else {
    out.set(path, v === null ? 'null' : typeof v);
  }
  return out;
}

describe('mock scene_description parity with the backend', () => {
  const mock = buildScene(real.scenario_id);

  it('has exactly the backend scene_description shape', () => {
    const a = shape(mock, 'scene', new Map());
    const b = shape(real, 'scene', new Map());
    const diff = [
      ...[...b].filter(([k]) => !a.has(k)).map(([k, t]) => `missing in mock: ${k} (${t})`),
      ...[...a].filter(([k]) => !b.has(k)).map(([k, t]) => `extra in mock: ${k} (${t})`),
      ...[...a]
        .filter(([k, t]) => b.has(k) && b.get(k) !== t)
        .map(([k, t]) => `type differs: ${k} mock=${t} real=${b.get(k)}`),
    ];
    expect(diff).toEqual([]);
  });

  it('serves the backend scenario catalog and hazard menu verbatim', () => {
    expect(mock.catalog).toEqual(real.catalog);
    expect(mock.hazards).toEqual(real.hazards);
  });

  it('names its scene the way SyntheticGrid does', () => {
    expect(mock.scene_id).toBe(real.scene_id);
    expect(mock.location).toBe(real.location);
    expect(mock.attribution).toBe(real.attribution);
  });
});
