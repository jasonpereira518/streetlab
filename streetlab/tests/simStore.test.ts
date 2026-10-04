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

describe('location error and trip completion', () => {
  it('surfaces a location_failed event as locationError and clears locationPending', () => {
    const h = createHarness();
    h.emitScene();
    const base = h.emitFrame();
    useSimStore.setState({ locationPending: 'Nonexistent Place' });

    h.emit({
      ...base,
      seq: base.seq + 1,
      t: base.t + 0.1,
      events: [
        { t: base.t, level: 'warn', code: 'location_failed', message: "Couldn't find that address." },
      ],
    });

    const s = useSimStore.getState();
    expect(s.locationPending).toBeNull();
    expect(s.locationError).toBe("Couldn't find that address.");
  });

  it('clears a stale locationError as soon as loadLocation is called again', () => {
    const h = createHarness();
    h.emitScene();
    useSimStore.setState({ locationError: 'stale error' });
    useSimStore.getState().loadLocation('Somewhere else');
    expect(useSimStore.getState().locationError).toBeNull();
  });

  it('clears locationError on a successful scene_description', () => {
    const h = createHarness();
    useSimStore.setState({ locationError: 'stale error' });
    h.emitScene();
    expect(useSimStore.getState().locationError).toBeNull();
  });

  it('sets tripComplete when a trip_complete event arrives, and not before', () => {
    const h = createHarness();
    h.emitScene();
    const base = h.emitFrame();
    expect(useSimStore.getState().tripComplete).toBe(false);

    h.emit({
      ...base,
      seq: base.seq + 1,
      t: base.t + 0.1,
      events: [{ t: base.t, level: 'info', code: 'trip_complete', message: 'arrived at destination' }],
    });

    expect(useSimStore.getState().tripComplete).toBe(true);
  });

  it('clears tripComplete on the next loadLocation call', () => {
    const h = createHarness();
    h.emitScene();
    useSimStore.setState({ tripComplete: true });
    useSimStore.getState().loadLocation('Somewhere else');
    expect(useSimStore.getState().tripComplete).toBe(false);
  });

  it('clears tripComplete on the next loadScenario call', () => {
    const h = createHarness();
    h.emitScene();
    useSimStore.setState({ tripComplete: true });
    useSimStore.getState().loadScenario('grid-loop');
    expect(useSimStore.getState().tripComplete).toBe(false);
  });
});

describe('build progress', () => {
  it('sets locationProgress from a location_progress event', () => {
    const h = createHarness();
    h.emitScene();
    const base = h.emitFrame();
    useSimStore.setState({ locationPending: 'Nob Hill' });

    h.emit({
      ...base,
      seq: base.seq + 1,
      t: base.t + 0.1,
      events: [
        { t: base.t, level: 'info', code: 'location_progress', message: 'Geocoding address', progress: 0.1 },
      ],
    });

    expect(useSimStore.getState().locationProgress).toEqual({
      stage: 'Geocoding address',
      fraction: 0.1,
    });
  });

  it('takes the last location_progress event when a batch carries more than one', () => {
    const h = createHarness();
    h.emitScene();
    const base = h.emitFrame();
    useSimStore.setState({ locationPending: 'Nob Hill' });

    h.emit({
      ...base,
      seq: base.seq + 1,
      t: base.t + 0.1,
      events: [
        { t: base.t, level: 'info', code: 'location_progress', message: 'Geocoding address', progress: 0.1 },
        { t: base.t, level: 'info', code: 'location_progress', message: 'Fetching map data', progress: 0.4 },
      ],
    });

    expect(useSimStore.getState().locationProgress).toEqual({
      stage: 'Fetching map data',
      fraction: 0.4,
    });
  });

  it('is null before the first checkpoint and cleared by a fresh loadLocation call', () => {
    const h = createHarness();
    h.emitScene();
    expect(useSimStore.getState().locationProgress).toBeNull();

    useSimStore.setState({ locationProgress: { stage: 'stale', fraction: 0.5 } });
    useSimStore.getState().loadLocation('Somewhere else');
    expect(useSimStore.getState().locationProgress).toBeNull();
  });

  it('is cleared on a successful scene_description', () => {
    const h = createHarness();
    useSimStore.setState({ locationProgress: { stage: 'stale', fraction: 0.5 } });
    h.emitScene();
    expect(useSimStore.getState().locationProgress).toBeNull();
  });

  it('is cleared on a location_failed event, alongside locationPending', () => {
    const h = createHarness();
    h.emitScene();
    const base = h.emitFrame();
    useSimStore.setState({
      locationPending: 'Nonexistent Place',
      locationProgress: { stage: 'Fetching map data', fraction: 0.4 },
    });

    h.emit({
      ...base,
      seq: base.seq + 1,
      t: base.t + 0.1,
      events: [
        { t: base.t, level: 'warn', code: 'location_failed', message: "Couldn't find that address." },
      ],
    });

    expect(useSimStore.getState().locationProgress).toBeNull();
  });

  it('is cleared when the backend rejects load_location at ack time', () => {
    const h = createHarness();
    h.emitScene();
    useSimStore.setState({ locationProgress: { stage: 'stale', fraction: 0.5 } });

    h.emit({
      type: 'ack',
      protocol: 1,
      id: 'whatever',
      cmd: 'load_location',
      ok: false,
      message: 'SyntheticGrid does not support load_location',
      t: 1,
    });

    expect(useSimStore.getState().locationProgress).toBeNull();
  });
});

describe('loadLocation with a destination', () => {
  it('sends both fields and shows a combined pending label', () => {
    const h = createHarness();
    h.emitScene();
    useSimStore.getState().loadLocation('Nob Hill', "Fisherman's Wharf");

    expect(useSimStore.getState().locationPending).toBe("Nob Hill → Fisherman's Wharf");
    const sent = h.sent.find((c) => c.cmd === 'load_location');
    expect(sent).toMatchObject({ query: 'Nob Hill', destination: "Fisherman's Wharf" });
  });

  it('omits destination entirely when left blank, unchanged from before it existed', () => {
    const h = createHarness();
    h.emitScene();
    useSimStore.getState().loadLocation('Nob Hill', '');

    expect(useSimStore.getState().locationPending).toBe('Nob Hill');
    const sent = h.sent.find((c) => c.cmd === 'load_location');
    expect(sent).not.toHaveProperty('destination');
  });
});

describe('suggestAddress', () => {
  it('sends a suggest_address command and does not log it', () => {
    const h = createHarness();
    h.emitScene();
    const id = useSimStore.getState().suggestAddress('nob');

    const sent = h.sent.find((c) => c.cmd === 'suggest_address');
    expect(sent).toMatchObject({ id, query: 'nob' });
    expect(useSimStore.getState().commandLog).toHaveLength(0);
  });

  it('stores a reply keyed by its request id', () => {
    const h = createHarness();
    h.emitScene();
    const id = useSimStore.getState().suggestAddress('nob');

    h.emit({
      type: 'address_suggestions',
      protocol: 1,
      id,
      query: 'nob',
      suggestions: [{ label: 'Nob Hill, San Francisco, CA', lat: 37.79, lon: -122.42 }],
    });

    expect(useSimStore.getState().addressSuggestions[id]).toEqual({
      query: 'nob',
      items: [{ label: 'Nob Hill, San Francisco, CA', lat: 37.79, lon: -122.42 }],
    });
  });

  it('keeps two concurrent requests (start + destination fields) separate', () => {
    const h = createHarness();
    h.emitScene();
    const startId = useSimStore.getState().suggestAddress('nob');
    const destId = useSimStore.getState().suggestAddress('fish');

    h.emit({
      type: 'address_suggestions',
      protocol: 1,
      id: destId,
      query: 'fish',
      suggestions: [{ label: "Fisherman's Wharf, San Francisco, CA", lat: 37.8, lon: -122.4 }],
    });
    h.emit({
      type: 'address_suggestions',
      protocol: 1,
      id: startId,
      query: 'nob',
      suggestions: [{ label: 'Nob Hill, San Francisco, CA', lat: 37.79, lon: -122.42 }],
    });

    const s = useSimStore.getState();
    expect(s.addressSuggestions[startId].items[0].label).toBe('Nob Hill, San Francisco, CA');
    expect(s.addressSuggestions[destId].items[0].label).toBe(
      "Fisherman's Wharf, San Francisco, CA",
    );
  });

  it('evicts the oldest entries once past the cap, keeping recent ones', () => {
    const h = createHarness();
    h.emitScene();
    const ids = Array.from({ length: 10 }, (_, i) =>
      useSimStore.getState().suggestAddress(`q${i}`),
    );
    ids.forEach((id, i) => {
      h.emit({
        type: 'address_suggestions',
        protocol: 1,
        id,
        query: `q${i}`,
        suggestions: [],
      });
    });

    const stored = useSimStore.getState().addressSuggestions;
    expect(Object.keys(stored).length).toBeLessThanOrEqual(8);
    // The most recent request must have survived eviction.
    expect(stored[ids[ids.length - 1]]).toBeDefined();
    expect(stored[ids[0]]).toBeUndefined();
  });
});
