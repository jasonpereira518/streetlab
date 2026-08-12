/**
 * Capture canonical wire fixtures from the frontend's TypeScript mock.
 *
 * These fixtures are the ground truth the Python `schema.py` models are held
 * to: whatever the real zod schema emits, pydantic must ingest without loss.
 *
 * Run with the frontend's own vite-node so extensionless TS imports and `zod`
 * resolve exactly as they do for the frontend's test suite. Nothing under
 * `streetlab/` is modified — this only reads.
 *
 *   npm run capture      (see package.json in this directory)
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { MockSim } from '../../streetlab/src/net/mockServer';
import {
  AckSchema,
  PROTOCOL_VERSION,
  SceneDescriptionSchema,
  StateUpdateSchema,
  formatIssues,
} from '../../streetlab/src/schema';
import type { Ack, Command, StateUpdate } from '../../streetlab/src/schema';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, '..', 'tests', 'fixtures');

/** Write `value` only after the real zod schema accepts it. */
function emit(name: string, schema: { safeParse(v: unknown): any }, value: unknown) {
  const res = schema.safeParse(value);
  if (!res.success) {
    throw new Error(`fixture ${name} failed its own schema: ${formatIssues(res.error)}`);
  }
  writeFileSync(join(OUT, `${name}.json`), JSON.stringify(res.data, null, 2) + '\n');
  console.log(`  ${name}.json`);
}

function ackFor(sim: MockSim, command: Command): Ack {
  const res = sim.apply(command);
  return {
    type: 'ack',
    protocol: PROTOCOL_VERSION,
    id: command.id,
    cmd: command.cmd,
    ok: res.ok,
    message: res.message,
    t: Math.round(sim.t * 1000) / 1000,
  };
}

/** Advance until `pred` holds, or give up after `limit` steps. */
function stepUntil(sim: MockSim, pred: (f: StateUpdate) => boolean, limit = 3600): StateUpdate {
  let frame = sim.frame();
  for (let i = 0; i < limit && !pred(frame); i++) {
    sim.step();
    frame = sim.frame();
  }
  return frame;
}

mkdirSync(OUT, { recursive: true });
console.log('capturing fixtures from the TypeScript mock:');

const sim = new MockSim();

emit('scene_description', SceneDescriptionSchema, sim.scene);

// A frame at t=0: nullable fields are overwhelmingly null here.
emit('state_update_initial', StateUpdateSchema, sim.frame());

// A frame once traffic is moving and the lead vehicle is being tracked, so
// `ttc_s`, `lane_offset` and radar returns are populated rather than null.
const moving = stepUntil(sim, (f) => f.detections.length > 0 && f.telemetry.radar.length > 0);
emit('state_update_moving', StateUpdateSchema, moving);

// A frame during a scripted cut-in: exercises `hazard`, `hazard_label`,
// `trajectory.cutin` and `trajectory.cutin_label` in their non-null form.
sim.apply({ id: 'cap-hazard', cmd: 'inject_hazard', kind: 'cut_in' });
const hazardous = stepUntil(sim, (f) => f.detections.some((d) => d.hazard));
emit('state_update_hazard', StateUpdateSchema, hazardous);

emit('ack_ok', AckSchema, ackFor(sim, { id: 'cap-1', cmd: 'set_paused', paused: true }));
emit('ack_error', AckSchema, ackFor(sim, { id: 'cap-2', cmd: 'load_scenario', scenario_id: 'no-such-scenario' }));

const counts = {
  detections: hazardous.detections.length,
  radar: hazardous.telemetry.radar.length,
  hazards: hazardous.detections.filter((d) => d.hazard).length,
  cutin: hazardous.telemetry.trajectory.cutin?.length ?? 0,
};
console.log('captured:', JSON.stringify(counts));
if (counts.hazards === 0) throw new Error('no hazard captured — fixtures would miss non-null hazard fields');
if (counts.cutin === 0) throw new Error('no cut-in trajectory captured — fixtures would miss non-null cutin');
