// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { AckToast } from '../src/ui/AckToast';
import { HelpDialog, SHORTCUTS, useShortcuts } from '../src/ui/HelpDialog';
import { LeftScenarioSidebar } from '../src/ui/LeftScenarioSidebar';
import { RightPanel } from '../src/ui/RightPanel';
import { TelemetryRow } from '../src/ui/TelemetryRow';
import { TopToolbar } from '../src/ui/TopToolbar';
import { PARAM_DEFS, useSimStore } from '../src/store/simStore';
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
  it('has no cut-in interval slider (the backend has no autonomous cut-ins)', () => {
    expect(PARAM_DEFS.map((d) => d.key)).not.toContain('cutin_period_s');
  });

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

  it('stays quiet for routine acks', () => {
    harness = createHarness();
    render(<AckToast />);
    harness.emitScene();
    act(() => useSimStore.getState().togglePaused());
    expect(screen.getByRole('status').textContent).toBe('');
  });
});

describe('speedometer', () => {
  it('labels the planner target as a target and shows the posted limit', () => {
    harness = createHarness();
    const { container } = render(<TelemetryRow />);
    harness.emitScene();
    harness.emitFrame(10);
    act(() => flushFrames(3, 120));
    const text = canvasText(container.querySelector('canvas') as HTMLCanvasElement);
    expect(text).toContain('TARGET');
    expect(text).toContain('LIMIT');
    expect(text).not.toContain('MAX');
  });
});

describe('perception menu', () => {
  it('says why it is disabled on the wrapper too, where webviews still show tooltips', () => {
    harness = createHarness();
    render(<TopToolbar />);
    harness.emitScene();
    harness.emitFrame(1);
    const trigger = screen.getByRole('button', { name: /ground truth/i });
    expect((trigger as HTMLButtonElement).disabled).toBe(true);
    expect(trigger.parentElement?.getAttribute('title')).toMatch(/--perception/);
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
