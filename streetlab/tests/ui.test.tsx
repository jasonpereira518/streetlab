// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
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
