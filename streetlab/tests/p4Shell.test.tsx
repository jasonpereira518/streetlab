// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { AckToast } from '../src/ui/AckToast';
import { HelpDialog, SHORTCUTS, useShortcuts } from '../src/ui/HelpDialog';
import { ConnectionErrorOverlay } from '../src/ui/ConnectionErrorOverlay';
import { LeftScenarioSidebar } from '../src/ui/LeftScenarioSidebar';
import { RightPanel } from '../src/ui/RightPanel';
import { TelemetryRow } from '../src/ui/TelemetryRow';
import { TopToolbar } from '../src/ui/TopToolbar';
import { useSimStore } from '../src/store/simStore';
import { createHarness, resetStore } from './harness';
import type { Harness } from './harness';
import { canvasText, flushFrames } from './setup';

let harness: Harness | null = null;

beforeEach(() => {
  try {
    localStorage.clear();
  } catch {
    // jsdom always has storage; the guard mirrors the app's own.
  }
});

afterEach(() => {
  cleanup();
  harness?.detach();
  harness = null;
  resetStore();
});

function Keys() {
  useShortcuts();
  return null;
}

describe('no control that does nothing', () => {
  it('has no disabled stub buttons in the toolbar or sidebar', () => {
    harness = createHarness();
    const { container } = render(
      <>
        <TopToolbar />
        <LeftScenarioSidebar />
      </>,
    );
    harness.emitScene();
    for (const name of [/new session/i, /save scenario/i, /^undo$/i, /^new$/i, /^open$/i]) {
      expect(screen.queryByRole('button', { name })).toBeNull();
    }
    // A perception-less backend is the one legitimately disabled toolbar
    // control, and it must say why on hover.
    const toolbar = container.querySelector('.toolbar') as HTMLElement;
    const disabled = within(toolbar).getAllByRole('button').filter((b) => (b as HTMLButtonElement).disabled);
    for (const b of disabled) expect(b.title || b.parentElement?.title, b.outerHTML).toBeTruthy();
  });
});

describe('bookmarks persist', () => {
  it('survives a remount via localStorage', () => {
    harness = createHarness();
    const first = render(<LeftScenarioSidebar />);
    harness.emitScene();
    fireEvent.click(screen.getByLabelText('Remove bookmark for Nob Hill Loop'));
    first.unmount();

    render(<LeftScenarioSidebar />);
    expect(screen.getByLabelText('Add bookmark for Nob Hill Loop')).toBeTruthy();
  });

  it('survives corrupt storage', () => {
    localStorage.setItem('streetlab.bookmarks', '{not json');
    harness = createHarness();
    render(<LeftScenarioSidebar />);
    harness.emitScene();
    expect(screen.getByLabelText('Remove bookmark for Nob Hill Loop')).toBeTruthy();
  });
});

describe('hazard menu', () => {
  it('renders ml_limitation next to the hazard it qualifies', () => {
    harness = createHarness();
    render(<RightPanel />);
    const scene = harness.emitScene();
    const limited = scene.hazards.filter((h) => h.ml_limitation);
    expect(limited.length).toBeGreaterThan(0);
    for (const h of limited) {
      const button = screen.getByRole('button', { name: h.label });
      expect(button.getAttribute('aria-describedby')).toBeTruthy();
      expect(screen.getAllByText(h.ml_limitation!, { exact: false }).length).toBeGreaterThan(0);
    }
  });

  it('shows a hazard ack in the viewport toast, not only the Params tab', () => {
    harness = createHarness();
    render(
      <>
        <AckToast />
        <RightPanel />
      </>,
    );
    harness.emitScene();
    useSimStore.getState().setRightTab('layers');
    act(() => useSimStore.getState().injectHazard('jaywalker'));
    const status = screen.getAllByRole('status')[0];
    expect(within(status).getByText(/only stages cut_in/)).toBeTruthy();
  });

  it('shows a collision where the driver is looking', () => {
    harness = createHarness();
    render(<AckToast />);
    harness.emitScene();
    for (let i = 0; i < 5; i++) harness.sim.step();
    const frame = harness.sim.frame();
    harness.emit({
      ...frame,
      events: [{ t: frame.t, level: 'critical', code: 'collision', message: 'hit a building' }],
    });
    const status = screen.getByRole('status');
    expect(within(status).getByText('Collision')).toBeTruthy();
    expect(within(status).getByText('hit a building')).toBeTruthy();
  });

  it('stays quiet for routine acks', () => {
    harness = createHarness();
    render(<AckToast />);
    harness.emitScene();
    act(() => useSimStore.getState().togglePaused());
    expect(screen.getByRole('status').textContent).toBe('');
  });
});

describe('speedometer', () => {
  it('labels the planner target SET and shows the posted limit', () => {
    harness = createHarness();
    const { container } = render(<TelemetryRow />);
    harness.emitScene();
    harness.emitFrame(10);
    act(() => flushFrames(3, 120));
    const text = canvasText(container.querySelector('canvas') as HTMLCanvasElement);
    expect(text).toContain('SET');
    expect(text).toContain('LIMIT');
    expect(text).not.toContain('MAX');
  });
});

describe('keyboard shortcuts and help', () => {
  it('space pauses, R resets, ? toggles help, Esc closes it', () => {
    harness = createHarness();
    render(
      <>
        <Keys />
        <HelpDialog />
      </>,
    );
    harness.emitScene();

    fireEvent.keyDown(window, { key: ' ' });
    expect(harness.sent.at(-1)).toMatchObject({ cmd: 'set_paused', paused: true });
    fireEvent.keyDown(window, { key: 'r' });
    expect(harness.sent.at(-1)).toMatchObject({ cmd: 'reset' });

    fireEvent.keyDown(window, { key: '?' });
    const dialog = screen.getByRole('dialog', { name: /help/i });
    for (const k of SHORTCUTS) expect(within(dialog).getByText(k.does)).toBeTruthy();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('Q and E signal a lane change, and a refusal reaches the toast', () => {
    harness = createHarness();
    render(
      <>
        <Keys />
        <AckToast />
      </>,
    );
    harness.emitScene();
    fireEvent.keyDown(window, { key: 'e' });
    expect(harness.sent.at(-1)).toMatchObject({ cmd: 'lane_change', direction: 'right' });
    // The ego drives the kerb lane: on its right is the pavement.
    expect(within(screen.getByRole('status')).getByText('no lane to the right')).toBeTruthy();
    fireEvent.keyDown(window, { key: 'Q' });
    expect(harness.sent.at(-1)).toMatchObject({ cmd: 'lane_change', direction: 'left' });
  });

  it('Q and E work the indicator while a person drives, and it rides on drive_input', () => {
    harness = createHarness();
    render(<Keys />);
    harness.emitScene();
    harness.emit({ ...harness.sim.frame(), ego: { ...harness.sim.frame().ego, cruise: { ...harness.sim.frame().ego.cruise, mode: 'off' } } });
    fireEvent.keyDown(window, { key: 'q' });
    expect(useSimStore.getState().driverSignal).toBe('left');
    expect(harness.sent.some((c) => c.cmd === 'lane_change')).toBe(false);
    fireEvent.keyDown(window, { key: 'q' });
    expect(useSimStore.getState().driverSignal).toBe('off');
  });

  it('H toggles the headlights and L is Assist: the two features never share a key', () => {
    harness = createHarness();
    render(<Keys />);
    harness.emitScene();
    const before = useSimStore.getState().params.headlights;
    fireEvent.keyDown(window, { key: 'h' });
    expect(useSimStore.getState().params.headlights).not.toBe(before);
    expect(harness.sent.some((c) => c.cmd === 'set_drive_mode')).toBe(false);
    fireEvent.keyDown(window, { key: 'l' });
    expect(harness.sent.at(-1)).toMatchObject({ cmd: 'set_drive_mode', mode: 'autosteer' });
  });

  it('leaves keys alone while typing or on a focused button', () => {
    harness = createHarness();
    render(
      <>
        <Keys />
        <input aria-label="q" />
        <button type="button">b</button>
      </>,
    );
    harness.emitScene();
    const before = harness.sent.length;
    fireEvent.keyDown(screen.getByLabelText('q'), { key: ' ' });
    fireEvent.keyDown(screen.getByLabelText('q'), { key: 'r' });
    fireEvent.keyDown(screen.getByRole('button', { name: 'b' }), { key: ' ' });
    expect(harness.sent).toHaveLength(before);
  });

  it('ignores every shortcut while the connection-error overlay blocks the UI', () => {
    harness = createHarness();
    render(
      <>
        <Keys />
        <ConnectionErrorOverlay failure={{ kind: 'backend_down', detail: 'x' }} onRetry={() => {}} onUseMock={() => {}} />
      </>,
    );
    harness.emitScene();
    act(() => useSimStore.setState({ failure: { kind: 'backend_down', detail: 'x' } }));
    const before = harness.sent.length;
    const retry = screen.getByRole('button', { name: /retry/i });
    retry.focus();
    for (const key of [' ', 'r', '?']) fireEvent.keyDown(retry, { key });
    fireEvent.keyDown(window, { key: 'r' });
    expect(harness.sent).toHaveLength(before);
    expect(useSimStore.getState().helpOpen).toBe(false);
  });

  it('the toolbar help button opens the dialog', () => {
    harness = createHarness();
    render(
      <>
        <TopToolbar />
        <HelpDialog />
      </>,
    );
    harness.emitScene();
    fireEvent.click(screen.getByRole('button', { name: 'Help and shortcuts' }));
    expect(screen.getByRole('dialog')).toBeTruthy();
  });
});

describe('sliders', () => {
  it('are labelled and expose their unit to assistive tech', () => {
    harness = createHarness();
    render(<RightPanel />);
    harness.emitScene();
    const slider = screen.getByRole('slider', { name: /Follow distance/ });
    expect(slider.getAttribute('aria-valuetext')).toMatch(/s$/);
  });
});
