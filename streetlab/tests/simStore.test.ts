// @vitest-environment jsdom
/**
 * Store-level regression tests for two review findings that straddled the
 * seam between `send()`/`commandLog` and the `state_update` -> React mirror:
 *
 *  - camera_frame commands must not pollute `commandLog` (it is a 50-entry,
 *    newest-first diagnostic log; at 10 Hz a logged camera_frame would own
 *    100% of it within five seconds).
 *  - `perception` being null on both sides of a state_update must not, by
 *    itself, force a React-visible store update — that defeats the
 *    `Object.keys(patch).length` change-gate for every user not running ML.
 */
import { describe, expect, it } from 'vitest';
import { useSimStore } from '../src/store/simStore';
import { createHarness, resetStore } from './harness';

const CAMERA_FRAME_CMD = {
  cmd: 'camera_frame' as const,
  seq: 0,
  t: 0,
  width: 640,
  height: 384,
  format: 'jpeg' as const,
  data: 'AAAA',
  camera: { x: 0, y: 0, z: 1.33, yaw: 0, pitch: 0, roll: 0, fov_y_deg: 50, aspect: 640 / 384 },
};

describe('commandLog', () => {
  it('does not record camera_frame commands', () => {
    resetStore();
    useSimStore.getState().send(CAMERA_FRAME_CMD);
    expect(useSimStore.getState().commandLog).toHaveLength(0);
  });

  it('still records ordinary commands', () => {
    resetStore();
    useSimStore.getState().send({ cmd: 'set_paused', paused: true });
    const log = useSimStore.getState().commandLog;
    expect(log).toHaveLength(1);
    expect(log[0].cmd).toBe('set_paused');
  });

  it('a camera_frame sandwiched between two toggle_layer commands does not evict either', () => {
    resetStore();
    const s = useSimStore.getState();
    s.send({ cmd: 'toggle_layer', layer: 'buildings', visible: false });
    s.send(CAMERA_FRAME_CMD);
    s.send({ cmd: 'toggle_layer', layer: 'trees', visible: false });
    const log = useSimStore.getState().commandLog;
    expect(log.map((c) => c.cmd)).toEqual(['toggle_layer', 'toggle_layer']);
  });
});

describe('perception change-gate', () => {
  it('does not touch the store when perception is null on both sides and nothing else changed', () => {
    const h = createHarness();
    h.emitScene();
    // First frame necessarily changes the store (hasFrames flips false -> true).
    const base = h.emitFrame();
    expect(useSimStore.getState().perception).toBeNull();
    const before = useSimStore.getState();

    // Same paused/assist_active/scenario_id, no events, perception still null:
    // nothing here should be visible to a React subscriber.
    h.emit({ ...base, seq: base.seq + 1, t: base.t + 0.1, events: [] });

    expect(useSimStore.getState()).toBe(before);
  });

  it('still updates when perception goes from null to a real payload', () => {
    const h = createHarness();
    h.emitScene();
    const base = h.emitFrame();
    expect(useSimStore.getState().perception).toBeNull();

    h.emit({
      ...base,
      seq: base.seq + 1,
      t: base.t + 0.1,
      events: [],
      perception: {
        mode: 'ml',
        detector_ms: 1.2,
        server_e2e_ms: 5.6,
        frames_received: 1,
        frames_dropped: 0,
        precision: null,
        recall: null,
        mean_pos_err_m: null,
      },
    });

    expect(useSimStore.getState().perception?.frames_received).toBe(1);
  });
});
