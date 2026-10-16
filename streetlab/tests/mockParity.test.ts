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

/** Every key path with its JSON type(s); arrays contribute every element, so a key that is null in one preset and a number in another is a number|null, not a mismatch. */
function shape(v: unknown, path: string, out: Map<string, string>): Map<string, string> {
  if (Array.isArray(v)) {
    out.set(path, 'array');
    for (const x of v) shape(x, `${path}[]`, out);
  } else if (v && typeof v === 'object') {
    out.set(path, 'object');
    for (const [k, x] of Object.entries(v)) shape(x, `${path}.${k}`, out);
  } else {
    const t = v === null ? 'null' : typeof v;
    const prev = out.get(path);
    out.set(path, prev && !prev.split('|').includes(t) ? [...prev.split('|'), t].sort().join('|') : (prev ?? t));
  }
  return out;
}

/** Same JSON type in at least one observed element, or one side only ever saw null. */
const overlaps = (a: string, b: string) =>
  a === 'null' || b === 'null' || a.split('|').some((t) => b.split('|').includes(t));

describe('mock scene_description parity with the backend', () => {
  const mock = buildScene(real.scenario_id);

  it('has exactly the backend scene_description shape', () => {
    // A preset's `params` is a sparse override map (only what the preset moves),
    // so its key set legitimately differs per preset and per implementation.
    // The mock deliberately keeps square corners (`kerb_returns: []`, see
    // mockCity.ts), so the backend's per-corner element shape is not mirrored;
    // the array itself must still exist.
    const sparse = ([k]: [string, string]) =>
      !k.startsWith('scene.presets[].params.') && !k.startsWith('scene.kerb_returns[]');
    const a = new Map([...shape(mock, 'scene', new Map())].filter(sparse));
    const b = new Map([...shape(real, 'scene', new Map())].filter(sparse));
    const diff = [
      ...[...b].filter(([k]) => !a.has(k)).map(([k, t]) => `missing in mock: ${k} (${t})`),
      ...[...a].filter(([k]) => !b.has(k)).map(([k, t]) => `extra in mock: ${k} (${t})`),
      ...[...a]
        .filter(([k, t]) => b.has(k) && !overlaps(t, b.get(k)!))
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
