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
});

