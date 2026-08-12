/**
 * Validate Python-emitted frames against the frontend's real zod schema.
 *
 * This is the half of the contract that pydantic cannot check itself. The
 * fixtures under tests/fixtures/python/ are produced by the actual Simulation,
 * so if schema.py drifts from schema.ts in a way that changes the wire — a
 * renamed field, a wrong type, a dropped key — `parseServerMessage` rejects it
 * here and the build goes red.
 *
 * Two directories, two expectations:
 *   python/          every file MUST validate
 *   python/invalid/  every file MUST be rejected (proving the check has teeth)
 *
 * Run with the frontend's own vite-node so `zod` and the extensionless imports
 * resolve exactly as they do for the frontend's test suite. Nothing under
 * streetlab/ is modified — this only reads.
 */
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseServerMessage } from '../../streetlab/src/schema';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', 'tests', 'fixtures', 'python');
const INVALID = join(ROOT, 'invalid');

function jsonFiles(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith('.json'))
    .sort();
}

let failures = 0;

const valid = jsonFiles(ROOT);
if (valid.length === 0) {
  console.error('no Python fixtures found — run the pytest contract test first');
  process.exit(1);
}

console.log(`validating ${valid.length} Python-emitted frames against schema.ts`);
for (const file of valid) {
  const raw = JSON.parse(readFileSync(join(ROOT, file), 'utf8'));
  const res = parseServerMessage(raw);
  if (res.ok) {
    console.log(`  ok      ${file}  (${res.value.type})`);
  } else {
    console.error(`  REJECT  ${file}: ${res.error}`);
    failures++;
  }
}

const invalid = jsonFiles(INVALID);
if (invalid.length > 0) {
  console.log(`\nchecking ${invalid.length} deliberately broken frames are caught`);
  for (const file of invalid) {
    const raw = JSON.parse(readFileSync(join(INVALID, file), 'utf8'));
    const res = parseServerMessage(raw);
    if (res.ok) {
      // A mutation that slips through means the contract test proves nothing.
      console.error(`  MISSED  ${file} validated but should not have`);
      failures++;
    } else {
      console.log(`  caught  ${file}: ${res.error}`);
    }
  }
}

if (failures > 0) {
  console.error(`\n${failures} contract failure(s)`);
  process.exit(1);
}
console.log('\ncontract holds in the Python -> TypeScript direction');
