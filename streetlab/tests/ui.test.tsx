// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { TopToolbar } from '../src/ui/TopToolbar';
import { LeftScenarioSidebar } from '../src/ui/LeftScenarioSidebar';
import { RightPanel } from '../src/ui/RightPanel';
import { TelemetryRow } from '../src/ui/TelemetryRow';
import { useSimStore } from '../src/store/simStore';
import { toMph } from '../src/units';
import { createHarness, resetStore } from './harness';
import type { Harness } from './harness';
import { canvasText, clearCanvasCalls, canvasCalls, flushFrames } from './setup';
import type { StateUpdate } from '../src/schema';
import type { Transport, TransportHandlers } from '../src/net/transport';

let harness: Harness | null = null;

afterEach(() => {
  cleanup();
  harness?.detach();
  harness = null;
  resetStore();
});

/** Run enough animation frames for the throttled hooks and canvases to settle. */
function tick(frames = 4): void {
  act(() => flushFrames(frames, 120));
}

describe('TopToolbar', () => {
  it('reflects live values from the frame stream', () => {
    harness = createHarness();
    render(<TopToolbar />);
    harness.emitScene();

    const frame = harness.emitFrame(240);
    tick();

    const expected = String(Math.round(toMph(frame.ego.speed_mps)));
    expect(screen.getByText('mph').previousSibling?.textContent).toBe(expected);
    expect(screen.getByText('Assist active')).toBeTruthy();
    expect(screen.getByText('Full Self-Driving')).toBeTruthy();
  });

  it('updates the speed readout as the simulator advances', () => {
    harness = createHarness();
    render(<TopToolbar />);
    harness.emitScene();

    harness.emitFrame(1);
    tick();
    const first = screen.getByText('mph').previousSibling?.textContent;

    // Several seconds of simulated driving: the speed must have moved.
    harness.emitFrame(300);
    tick();
    const second = screen.getByText('mph').previousSibling?.textContent;
    expect(second).not.toBe(first);
  });

  it('pause emits a set_paused command and the sim honours it', () => {
    harness = createHarness();
    render(<TopToolbar />);
    harness.emitScene();
    harness.emitFrame(60);
    tick();

    fireEvent.click(screen.getByLabelText('Pause simulation'));

    const cmd = harness.sent.find((c) => c.cmd === 'set_paused');
    expect(cmd).toMatchObject({ cmd: 'set_paused', paused: true });
    expect(cmd?.id).toBeTruthy();

    // The frame after the command reports the paused state, and the button flips.
    harness.emitFrame(1);
    tick();
    expect(useSimStore.getState().paused).toBe(true);
    expect(screen.getByLabelText('Resume simulation')).toBeTruthy();

    fireEvent.click(screen.getByLabelText('Resume simulation'));
    expect(harness.sent.filter((c) => c.cmd === 'set_paused')).toHaveLength(2);
  });

  it('switches camera view through the menu', () => {
    harness = createHarness();
    render(<TopToolbar />);
    harness.emitScene();

    fireEvent.click(screen.getByTitle('Camera view'));
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Overhead' }));

    expect(useSimStore.getState().cameraView).toBe('overhead');
    expect(harness.sent).toContainEqual(
      expect.objectContaining({ cmd: 'set_camera', view: 'overhead' }),
    );
  });
});

describe('LeftScenarioSidebar', () => {
  it('lists the catalog the server sent', () => {
    harness = createHarness();
    render(<LeftScenarioSidebar />);
    harness.emitScene();

    expect(screen.getByText('Nob Hill')).toBeTruthy();
    expect(screen.getAllByRole('listitem')).toHaveLength(5);
    expect(screen.getByText('01')).toBeTruthy();
    expect(screen.getByText('05')).toBeTruthy();
    expect(screen.getByText('Hyde St Descent')).toBeTruthy();
  });

  it("a card's play button emits load_scenario", () => {
    harness = createHarness();
    render(<LeftScenarioSidebar />);
    harness.emitScene();

    fireEvent.click(screen.getByLabelText('Load Hyde St Descent'));

    expect(harness.sent).toContainEqual(
      expect.objectContaining({ cmd: 'load_scenario', scenario_id: 'hyde-descent' }),
    );
    // The server answered with a new scene, so the sidebar now shows its location.
    expect(useSimStore.getState().activeScenarioId).toBe('hyde-descent');
    expect(screen.getByText('Russian Hill')).toBeTruthy();
  });

  it('renders a mini-map thumbnail per scenario', () => {
    harness = createHarness();
    const { container } = render(<LeftScenarioSidebar />);
    harness.emitScene();

    const thumbs = container.querySelectorAll('.scenario-thumb canvas');
    expect(thumbs).toHaveLength(5);
    // Each thumbnail actually drew its road skeleton and route.
    for (const t of thumbs) {
      const calls = canvasCalls(t as HTMLCanvasElement);
      expect(calls.some((c) => c.method === 'stroke')).toBe(true);
    }
  });

  it('toggles a bookmark without touching the server', () => {
    harness = createHarness();
    render(<LeftScenarioSidebar />);
    harness.emitScene();

    const before = harness.sent.length;
    fireEvent.click(screen.getByLabelText('Remove bookmark for Nob Hill Loop'));
    expect(screen.getByLabelText('Add bookmark for Nob Hill Loop')).toBeTruthy();
    expect(harness.sent).toHaveLength(before);
  });
});

describe('Location search box', () => {
  it('sends load_location on submit and shows a disabling pending state until the scene arrives', () => {
    harness = createHarness();
    render(<LeftScenarioSidebar />);
    harness.emitScene();

    const box = screen.getByLabelText('Load a location') as HTMLInputElement;
    fireEvent.change(box, { target: { value: 'Nob Hill' } });
    fireEvent.submit(box.closest('form')!);

    expect(harness.sent).toContainEqual(
      expect.objectContaining({ cmd: 'load_location', query: 'Nob Hill' }),
    );
    expect(screen.getByText(/building nob hill/i)).toBeTruthy();
    expect(box.disabled).toBe(true);
    expect(useSimStore.getState().locationPending).toBe('Nob Hill');
    // The field is cleared right away so a stray resubmit can't replay it.
    expect(box.value).toBe('');

    // The scene arrives later, through the ordinary epoch push.
    harness.emitScene();
    expect(useSimStore.getState().locationPending).toBeNull();
    expect(box.disabled).toBe(false);
    expect(screen.queryByText(/building nob hill/i)).toBeNull();
  });

  it('does not send an empty or whitespace-only query, at the component and the store level', () => {
    harness = createHarness();
    render(<LeftScenarioSidebar />);
    harness.emitScene();

    const box = screen.getByLabelText('Load a location') as HTMLInputElement;
    fireEvent.change(box, { target: { value: '   ' } });
    fireEvent.submit(box.closest('form')!);

    expect(harness.sent.filter((c) => c.cmd === 'load_location')).toHaveLength(0);
    expect(useSimStore.getState().locationPending).toBeNull();
    expect(box.disabled).toBe(false);

    // The store action is the reusable surface — a caller that bypasses the
    // component (another widget, a future keyboard shortcut) must get the
    // same guarantee, not just the form's onSubmit.
    act(() => useSimStore.getState().loadLocation('   '));
    expect(harness.sent.filter((c) => c.cmd === 'load_location')).toHaveLength(0);
    expect(useSimStore.getState().locationPending).toBeNull();
  });

  it('clears the pending state on a location_failed event, without waiting for a scene', () => {
    harness = createHarness();
    render(<LeftScenarioSidebar />);
    harness.emitScene();

    const box = screen.getByLabelText('Load a location') as HTMLInputElement;
    fireEvent.change(box, { target: { value: 'Nonexistent Place' } });
    fireEvent.submit(box.closest('form')!);
    expect(useSimStore.getState().locationPending).toBe('Nonexistent Place');
    expect(box.disabled).toBe(true);

    // A real build failure (bad geocode, empty Overpass extract, no
    // drivable roads) surfaces as a warn-level event inside a state_update,
    // per sim/loop.py's `submit_scene` — there is no separate message type.
    const base = harness.emitFrame(1);
    const failureFrame: StateUpdate = {
      ...base,
      events: [
        { t: base.t + 0.05, level: 'warn', code: 'location_failed', message: 'geocode failed' },
      ],
    };
    harness.emit(failureFrame);

    expect(useSimStore.getState().locationPending).toBeNull();
    expect(box.disabled).toBe(false);
    expect(screen.queryByText(/building nonexistent place/i)).toBeNull();
    // Recovery is clean: the field is empty and usable for a fresh query.
    expect(box.value).toBe('');
    fireEvent.change(box, { target: { value: 'Second Try' } });
    fireEvent.submit(box.closest('form')!);
    expect(harness.sent).toContainEqual(
      expect.objectContaining({ cmd: 'load_location', query: 'Second Try' }),
    );
  });

  it('a second submit while already pending does not send a duplicate command', () => {
    harness = createHarness();
    render(<LeftScenarioSidebar />);
    harness.emitScene();

    const box = screen.getByLabelText('Load a location') as HTMLInputElement;
    fireEvent.change(box, { target: { value: 'Nob Hill' } });
    const form = box.closest('form')!;
    fireEvent.submit(form);
    expect(harness.sent.filter((c) => c.cmd === 'load_location')).toHaveLength(1);

    // A stray double-submit (double Enter, a race in the caller) must not
    // replay the request or clobber the pending label with an empty query —
    // the field was already reset to '' after the first submit, and the
    // store's own trim-guard makes a resubmit with that blank value a no-op.
    fireEvent.submit(form);
    expect(harness.sent.filter((c) => c.cmd === 'load_location')).toHaveLength(1);
    expect(useSimStore.getState().locationPending).toBe('Nob Hill');
    expect(screen.getByText(/building nob hill/i)).toBeTruthy();
  });

  it('does not leave the box permanently disabled when the transport drops and never comes back', () => {
    let handlers: TransportHandlers | null = null;
    const transport: Transport = {
      kind: 'ws',
      label: 'test-ws',
      connect: (h) => {
        handlers = h;
        h.onStatus('open', 'test');
      },
      send: vi.fn(),
      close: vi.fn(),
    };
    resetStore();
    const detach = useSimStore.getState().attach(transport);

    act(() => useSimStore.getState().loadLocation('Nob Hill'));
    expect(useSimStore.getState().locationPending).toBe('Nob Hill');

    // A brief reconnect blip must not itself clear pending — the build may
    // still land once the socket comes back, and a real reconnect gets a
    // fresh scene_description from the server (ws_server.py's `_serve`
    // pushes one on every accept), which already clears it through the
    // existing scene_description path.
    act(() => handlers?.onStatus('reconnecting', 'closed (1006) — retrying in 400 ms'));
    expect(useSimStore.getState().locationPending).toBe('Nob Hill');

    // But once the transport gives up for good (`closed`), nothing will
    // ever deliver the scene or the location_failed event that would
    // otherwise clear it — the box must not stay disabled forever.
    act(() => handlers?.onStatus('closed', 'gave up retrying'));
    expect(useSimStore.getState().locationPending).toBeNull();

    detach();
  });
});

describe('Scene attribution', () => {
  it('shows the OpenStreetMap attribution when the scene carries one', () => {
    harness = createHarness();
    render(<LeftScenarioSidebar />);
    harness.emitScene();

    harness.emit({ ...harness.sim.scene, attribution: '© OpenStreetMap contributors' });

    expect(screen.getByText('© OpenStreetMap contributors')).toBeTruthy();
  });

  it("shows the synthetic scene's own notice verbatim, not OSM-branded copy", () => {
    harness = createHarness();
    render(<LeftScenarioSidebar />);
    harness.emitScene();

    // The mock's scene is always synthetic — see mockCity.ts's buildScene —
    // so the attribution the sidebar receives here is literally the string
    // SyntheticGrid uses on the real backend (map/scene_build.py). Anything
    // that dresses it up as an OSM credit would misrepresent the data.
    expect(screen.getByText('Synthetic scene — no map data')).toBeTruthy();
  });

  it('renders no attribution element before any scene has arrived', () => {
    harness = createHarness();
    const { container } = render(<LeftScenarioSidebar />);
    // No harness.emitScene() call — the store's `scene` is still null.

    expect(container.querySelector('.scene-attribution')).toBeNull();
  });

  it('renders no attribution element for an empty (but present) attribution string', () => {
    harness = createHarness();
    const { container } = render(<LeftScenarioSidebar />);
    harness.emitScene();

    harness.emit({ ...harness.sim.scene, attribution: '' });

    // Guards against `{cond && <p>{cond}</p>}` leaving a stray empty <p> —
    // a real risk with falsy-but-defined values like ''.
    expect(container.querySelector('.scene-attribution')).toBeNull();
  });
});

describe('Event log', () => {
  it('renders buffered sim events, newest first, with the level in the className', async () => {
    harness = createHarness();
    render(<RightPanel />);

    act(() =>
      useSimStore.setState({
        events: [
          { t: 1, level: 'info', code: 'location_requested', message: 'building Nob Hill' },
          { t: 2, level: 'warn', code: 'location_failed', message: 'no results' },
        ],
      }),
    );

    fireEvent.click(screen.getByRole('tab', { name: /events/i }));

    const items = screen.getAllByRole('listitem');
    expect(items).toHaveLength(2);
    expect(items[0].textContent).toContain('no results');
    expect(items[0].className).toMatch(/event-warn/);
    expect(items[1].textContent).toContain('building Nob Hill');
    expect(items[1].className).toMatch(/event-info/);

    // Task 10's Playwright specs target this text directly — a
    // `location_failed` event must be discoverable by its code, not just
    // its human-readable message.
    expect(screen.getByText('location_failed')).toBeTruthy();
  });

  it('reads as the normal startup state, not an error, before any event arrives', () => {
    harness = createHarness();
    render(<RightPanel />);

    fireEvent.click(screen.getByRole('tab', { name: /events/i }));

    expect(screen.getByText(/no events yet/i)).toBeTruthy();
    expect(screen.queryByRole('listitem')).toBeNull();
    // Nothing warn/critical-colored should be present for an empty log.
    expect(screen.queryByText(/error|failed/i)).toBeNull();
  });

  it('keeps every list item distinct even when two events share the same t and code', () => {
    // `t` is sim-seconds and resets to 0 on `reset`/`load_scenario`
    // (mockServer.ts's resetDynamics); `reset` does not itself clear the
    // event buffer (no scene_description follows it), so two genuinely
    // different events *can* land in the same 40-entry buffer sharing both
    // `t` and `code`. If the list key ever loses its disambiguating index,
    // React would warn about duplicate keys and (worse) misattribute DOM
    // identity across renders.
    harness = createHarness();
    render(<RightPanel />);
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    act(() =>
      useSimStore.setState({
        events: [
          { t: 0, level: 'info', code: 'location_requested', message: 'first build' },
          { t: 0, level: 'info', code: 'location_requested', message: 'second build, after reset' },
        ],
      }),
    );
    fireEvent.click(screen.getByRole('tab', { name: /events/i }));

    const items = screen.getAllByRole('listitem');
    expect(items).toHaveLength(2);
    expect(items[0].textContent).toContain('second build, after reset');
    expect(items[1].textContent).toContain('first build');

    const duplicateKeyWarning = errorSpy.mock.calls.some((args) =>
      /same key/i.test(String(args[0])),
    );
    expect(duplicateKeyWarning).toBe(false);

    errorSpy.mockRestore();
  });

  it('the 40-event cap can silently evict an older event once enough newer ones arrive', () => {
    // This documents a real, pre-existing limitation of simStore.ts's
    // `.slice(-40)` (Task 3/4), not something Task 7 introduces or fixes:
    // there is no "N events truncated" indicator, so a `location_failed` a
    // user hasn't yet seen can scroll off the end of a long session.
    harness = createHarness();
    render(<RightPanel />);
    harness.emitScene();
    const base = harness.emitFrame(1);

    for (let i = 0; i < 41; i++) {
      harness.emit({
        ...base,
        seq: base.seq + i + 1,
        events: [{ t: base.t + i, level: 'info', code: 'tick', message: `event ${i}` }],
      });
    }

    // Only the newest 40 of the 41 survive; the very first one is gone.
    expect(useSimStore.getState().events).toHaveLength(40);
    expect(useSimStore.getState().events[0].message).toBe('event 1');

    fireEvent.click(screen.getByRole('tab', { name: /events/i }));
    expect(screen.queryByText('event 0')).toBeNull();
    expect(screen.getByText('event 40')).toBeTruthy();
  });
});

describe('RightPanel', () => {
  it('a slider emits set_param and updates the store', () => {
    harness = createHarness();
    render(<RightPanel />);
    harness.emitScene();

    const slider = screen.getByLabelText(/Max speed/);
    fireEvent.change(slider, { target: { value: '62' } });

    expect(useSimStore.getState().params.ego_speed_cap_mph).toBe(62);
    expect(harness.sent).toContainEqual(
      expect.objectContaining({ cmd: 'set_param', key: 'ego_speed_cap_mph', value: 62 }),
    );
  });

  it('render-only params stay on the client', () => {
    harness = createHarness();
    render(<RightPanel />);

    fireEvent.change(screen.getByLabelText(/Plan opacity/), {
      target: { value: '0.9' },
    });

    expect(useSimStore.getState().params.plan_opacity).toBe(0.9);
    expect(harness.sent.filter((c) => c.cmd === 'set_param')).toHaveLength(0);
  });

  it('the colour picker updates the hazard colour', () => {
    harness = createHarness();
    render(<RightPanel />);

    fireEvent.click(screen.getByLabelText('Hazard colour #22C55E'));
    expect(useSimStore.getState().params.hazard_color).toBe('#22C55E');
  });

  it('toggling "detections" hides the hazard overlay layer', () => {
    harness = createHarness();
    render(<RightPanel />);
    fireEvent.click(screen.getByRole('tab', { name: 'Layers' }));

    const toggle = screen.getByRole('switch', { name: 'Detections' });
    expect(useSimStore.getState().layers.detections).toBe(true);

    fireEvent.click(toggle);
    expect(useSimStore.getState().layers.detections).toBe(false);
    expect(harness.sent).toContainEqual(
      expect.objectContaining({
        cmd: 'toggle_layer',
        layer: 'detections',
        visible: false,
      }),
    );

    fireEvent.click(toggle);
    expect(useSimStore.getState().layers.detections).toBe(true);
  });

  it('the "perception only" preset leaves scenery hidden', () => {
    harness = createHarness();
    render(<RightPanel />);
    fireEvent.click(screen.getByRole('tab', { name: 'Layers' }));
    fireEvent.click(screen.getByText('Perception only'));

    const { layers } = useSimStore.getState();
    expect(layers.detections).toBe(true);
    expect(layers.plan_path).toBe(true);
    expect(layers.buildings).toBe(false);
    expect(layers.trees).toBe(false);
  });

  it('the map tab draws the scene and the ego', () => {
    harness = createHarness();
    const { container } = render(<RightPanel />);
    harness.emitScene();
    fireEvent.click(screen.getByRole('tab', { name: 'Map' }));
    harness.emitFrame(120);
    tick();

    const canvas = container.querySelector('.map-canvas canvas') as HTMLCanvasElement;
    expect(canvas).toBeTruthy();
    const methods = new Set(canvasCalls(canvas).map((c) => c.method));
    expect(methods.has('fill')).toBe(true);
    expect(methods.has('stroke')).toBe(true);

    // And the scene facts come from the schema, not from hard-coded strings.
    const facts = container.querySelector('.facts') as HTMLElement;
    expect(within(facts).getByText(String(harness.sim.scene.roads.length))).toBeTruthy();
  });

  it('shows the acknowledgement returned for the last command', () => {
    harness = createHarness();
    render(<RightPanel />);
    fireEvent.change(screen.getByLabelText(/Traffic speed/), {
      target: { value: '0.5' },
    });
    act(() => {});
    expect(screen.getByText('traffic_speed_scale=0.5')).toBeTruthy();
  });
});

describe('Telemetry row', () => {
  it('renders all six widgets', () => {
    harness = createHarness();
    render(<TelemetryRow />);
    for (const title of [
      'Speed',
      'Lane position',
      'Radar',
      'Vehicle',
      'Trajectory',
      'Steering',
    ]) {
      expect(screen.getByText(title)).toBeTruthy();
    }
  });

  it('every widget draws from the mock telemetry each frame', () => {
    harness = createHarness();
    const { container } = render(<TelemetryRow />);
    harness.emitScene();
    harness.emitFrame(200);
    tick(2);

    const canvases = [...container.querySelectorAll('canvas')] as HTMLCanvasElement[];
    expect(canvases).toHaveLength(6);
    for (const c of canvases) clearCanvasCalls(c);

    const frame = harness.emitFrame(30);
    tick(2);

    for (const c of canvases) {
      expect(canvasCalls(c).length).toBeGreaterThan(4);
    }

    const [speed, lane, radar, vehicle, trajectory, steering] = canvases;

    // Speedometer prints the rounded mph and the cruise ceiling.
    const mph = String(Math.round(toMph(frame.ego.speed_mps)));
    expect(canvasText(speed)).toContain(mph);
    expect(canvasText(speed)).toContain(
      `${Math.round(toMph(frame.ego.cruise.set_speed_mps))} MAX`,
    );

    // Lane widget states the lane index that came down the wire.
    expect(canvasText(lane)).toContain(
      `lane ${frame.telemetry.lane.lane_index + 1}/${frame.telemetry.lane.lane_count}`,
    );

    // Radar reports how many returns it plotted.
    expect(canvasText(radar)).toContain(`${frame.telemetry.radar.length} returns`);

    // Vehicle lists subsystems and battery.
    expect(canvasText(vehicle)).toContain('Perception');
    expect(canvasText(vehicle)).toContain(
      `${Math.round(frame.telemetry.vehicle.battery_pct)}%`,
    );

    // Trajectory labels its axes and the "now" divider.
    expect(canvasText(trajectory)).toContain('now');
    expect(canvasText(trajectory)).toContain('plan');

    // Steering prints degrees.
    expect(canvasText(steering)).toMatch(/\d+°/);
  });

  it('animates: consecutive frames produce different speed readouts', () => {
    harness = createHarness();
    const { container } = render(<TelemetryRow />);
    harness.emitScene();

    const speed = container.querySelector('canvas') as HTMLCanvasElement;
    const readings = new Set<string>();
    for (let i = 0; i < 6; i++) {
      clearCanvasCalls(speed);
      harness.emitFrame(45);
      tick(2);
      const shown = canvasText(speed).match(/\b\d+\b/g)?.join(',') ?? '';
      readings.add(shown);
    }
    expect(readings.size).toBeGreaterThan(1);
  });

  it('shows a placeholder before any frame arrives', () => {
    harness = createHarness();
    const { container } = render(<TelemetryRow />);
    tick(2);
    const speed = container.querySelector('canvas') as HTMLCanvasElement;
    expect(canvasText(speed)).toContain('Awaiting telemetry');
  });
});
