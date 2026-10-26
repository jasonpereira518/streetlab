// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { useDriveKeys } from '../src/input/driveKeys';
import { DRIVE_HINTS, DriveHud } from '../src/ui/DriveHud';
import { flushFrames } from './setup';
import { useSimStore } from '../src/store/simStore';
import { createHarness, resetStore, type Harness } from './harness';

let harness: Harness | undefined;
afterEach(() => {
  cleanup();
  harness?.detach();
  harness = undefined;
  resetStore();
  vi.useRealTimers();
});

function Drive() {
  useDriveKeys();
  return <DriveHud />;
}

describe('drive HUD', () => {
  it('shows the hint for the current mode, and switches with it', () => {
    harness = createHarness();
    render(<DriveHud />);
    harness.emitScene();
    harness.emitFrame(5);
    act(() => flushFrames(2, 120));
    expect(screen.getByTestId('drive-hint').textContent).toContain(DRIVE_HINTS.fsd);
    act(() => useSimStore.getState().setDriveMode('off'));
    harness.emitFrame(1);
    act(() => flushFrames(4, 120));
    expect(screen.getByTestId('drive-hint').textContent).toContain(DRIVE_HINTS.off);
  });

  it('a pedal key is sent the moment it moves, not on the next 30 Hz tick', () => {
    vi.useFakeTimers();
    harness = createHarness();
    harness.emitScene();
    harness.emitFrame(5);
    render(<Drive />);
    const h = harness;
    const drives = () => h.sent.filter((c) => c.cmd === 'drive_input');
    fireEvent.keyDown(window, { key: 'w' });
    expect(drives().at(-1)).toMatchObject({ throttle: 1, brake: 0 }); // no timer has run
    fireEvent.keyDown(window, { key: 'w' }); // key repeat: no extra send
    const count = drives().length;
    fireEvent.keyUp(window, { key: 'w' });
    expect(drives()).toHaveLength(count + 1);
    expect(drives().at(-1)).toMatchObject({ throttle: 0 });
  });

  it('on-screen Gas drives like a held W key', () => {
    vi.useFakeTimers();
    harness = createHarness();
    harness.emitScene();
    harness.emitFrame(5);
    render(<Drive />);
    act(() => flushFrames(2, 120));
    fireEvent.click(screen.getByRole('button', { name: 'On-screen controls' }));
    const gas = screen.getByRole('button', { name: 'Gas' });
    fireEvent.pointerDown(gas, { pointerId: 1 });
    act(() => vi.advanceTimersByTime(400));
    const h = harness;
    const last = h.sent.filter((c) => c.cmd === 'drive_input').at(-1);
    expect(last).toMatchObject({ throttle: 1, brake: 0 });
    fireEvent.pointerUp(gas, { pointerId: 1 });
    act(() => vi.advanceTimersByTime(400));
    expect(h.sent.filter((c) => c.cmd === 'drive_input').at(-1)).toMatchObject({ throttle: 0 });
  });
  it("sends the driver's indicator at rest, and resets it when FSD engages", () => {
    vi.useFakeTimers();
    harness = createHarness();
    harness.emitScene();
    const manual = harness.sim.frame();
    harness.emit({ ...manual, ego: { ...manual.ego, cruise: { ...manual.ego.cruise, mode: 'off' } } });
    render(<Drive />);
    const h = harness;
    const drives = () => h.sent.filter((c) => c.cmd === 'drive_input');
    act(() => vi.advanceTimersByTime(200));
    expect(drives()).toHaveLength(0); // silent at rest
    act(() => useSimStore.getState().signalLaneChange('right'));
    act(() => vi.advanceTimersByTime(40));
    expect(drives()).toHaveLength(1);
    expect(drives()[0]).toMatchObject({ steer: 0, throttle: 0, brake: 0, signal: 'right' });
    act(() => vi.advanceTimersByTime(200));
    expect(drives()).toHaveLength(1); // once, not every tick
    harness.emit({ ...manual, ego: { ...manual.ego, cruise: { ...manual.ego.cruise, mode: 'fsd' } } });
    act(() => vi.advanceTimersByTime(40));
    expect(useSimStore.getState().driverSignal).toBe('off');
    expect(drives().at(-1)).toMatchObject({ signal: 'off' });
  });

  it('lights the blind-spot pill for a car alongside, and blinks it when signalling into it', () => {
    harness = createHarness();
    render(<DriveHud />);
    harness.emitScene();
    const f = harness.sim.frame();
    const beside = {
      id: 'b',
      cls: 'car' as const,
      pose: { x: f.ego.pose.x - 3.6 * Math.sin(f.ego.pose.heading) - 3 * Math.cos(f.ego.pose.heading), y: f.ego.pose.y + 3.6 * Math.cos(f.ego.pose.heading) - 3 * Math.sin(f.ego.pose.heading), heading: f.ego.pose.heading },
      size: f.ego.size,
      speed_mps: 9,
    };
    harness.emit({ ...f, ego: { ...f.ego, speed_mps: 10 }, world_agents: [beside] });
    act(() => flushFrames(2, 120));
    const pill = screen.getByText(/Blind spot/);
    expect(pill.className).not.toContain('blind-spot--warn');
    harness.emit({ ...f, t: f.t + 0.2, ego: { ...f.ego, speed_mps: 10, turn_signal: 'left' }, world_agents: [beside] });
    act(() => flushFrames(4, 120));
    expect(screen.getByText(/Blind spot/).className).toContain('blind-spot--warn');
  });
});

