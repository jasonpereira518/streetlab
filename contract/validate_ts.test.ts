/**
 * The Python -> TypeScript half of the wire contract: frames emitted by the
 * real Python `Simulation` (`contract/fixtures/`, generated and kept honest
 * by `validate_py_test.py`) validated by the real zod schema. If schema.py
 * drifts from schema.ts in a way that changes the wire, `parseServerMessage`
 * rejects it here and this suite goes red.
 *
 * Runs under the frontend's own vitest (see `streetlab/vite.config.ts`'s
 * `test.include`), so `zod` and the extensionless import below resolve
 * exactly as they do for the rest of the frontend's tests. Nothing under
 * `streetlab/` is modified — this only reads.
 */
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { parseServerMessage } from '../streetlab/src/schema';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, 'fixtures');
const INVALID = join(FIXTURES, 'invalid');

function jsonFiles(dir: string): string[] {
  return readdirSync(dir)
    .filter((f) => f.endsWith('.json'))
    .sort();
}

function readJson(dir: string, file: string): unknown {
  return JSON.parse(readFileSync(join(dir, file), 'utf8'));
}

describe('contract: Python-emitted frames validate against schema.ts', () => {
  const files = jsonFiles(FIXTURES);
  it('has fixtures to check', () => {
    expect(files.length).toBeGreaterThan(0);
  });
  for (const file of files) {
    it(`accepts ${file}`, () => {
      const res = parseServerMessage(readJson(FIXTURES, file));
      expect(res.ok, res.ok ? undefined : res.error).toBe(true);
    });
  }
});

describe('contract: deliberately broken frames are rejected', () => {
  const files = jsonFiles(INVALID);
  it('has broken variants to check — proves the check has teeth', () => {
    expect(files.length).toBeGreaterThan(0);
  });
  for (const file of files) {
    it(`rejects ${file}`, () => {
      const res = parseServerMessage(readJson(INVALID, file));
      expect(res.ok).toBe(false);
    });
  }
});
