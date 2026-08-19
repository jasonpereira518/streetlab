// @vitest-environment jsdom
/**
 * App-level fault injection: what the shell does when the sidecar handshake
 * itself fails, before any transport ever attaches — the one case none of
 * wsClient.test.ts's transport-level tests can see, since it's App.tsx's own
 * boot state machine that turns a rejected createTransportFromLocation() into
 * a visible error and a one-click mock fallback.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from '../src/App';
import type { Transport } from '../src/net/transport';
import { useSimStore } from '../src/store/simStore';
import { resetStore } from './harness';

vi.mock('../src/three/Renderer', () => ({
  Renderer: () => null,
}));

vi.mock('../src/net/wsClient', () => ({
  createTransportFromLocation: vi.fn(),
}));

const { createTransportFromLocation } = await import('../src/net/wsClient');
const mockCreateTransport = vi.mocked(createTransportFromLocation);

afterEach(() => {
  cleanup();
  resetStore();
  vi.clearAllMocks();
});

describe('App startup states', () => {
  it('shows a starting overlay while the transport resolves, then hides it', async () => {
    let resolve!: (t: Transport) => void;
    mockCreateTransport.mockReturnValue(
      new Promise((r) => {
        resolve = r;
      }),
    );

    render(<App />);
    expect(screen.getByText('Starting simulator…')).toBeTruthy();

    resolve({
      kind: 'mock',
      label: 'mock',
      connect: vi.fn(),
      send: vi.fn(),
      close: vi.fn(),
    });

    await waitFor(() =>
      expect(screen.queryByText('Starting simulator…')).toBeNull(),
    );
  });

  it('shows an error overlay with the failure reason when the handshake fails', async () => {
    mockCreateTransport.mockRejectedValue(
      new Error('simulator did not start: sidecar exited before READY'),
    );

    render(<App />);

    expect(await screen.findByText("Simulator didn't start")).toBeTruthy();
    expect(
      screen.getByText('simulator did not start: sidecar exited before READY'),
    ).toBeTruthy();
  });

  it('shows a clear message on a protocol mismatch', async () => {
    mockCreateTransport.mockRejectedValue(
      new Error('backend speaks protocol 2, this build expects 1'),
    );

    render(<App />);

    expect(
      await screen.findByText('backend speaks protocol 2, this build expects 1'),
    ).toBeTruthy();
  });

  it('falls back to the mock on one click from the error state', async () => {
    mockCreateTransport.mockRejectedValue(new Error('nope'));

    render(<App />);
    const fallback = await screen.findByText('Use mock instead');
    fireEvent.click(fallback);

    await waitFor(() =>
      expect(screen.queryByText("Simulator didn't start")).toBeNull(),
    );
    expect(useSimStore.getState().sourceKind).toBe('mock');
  });
});

describe('Collapsible shell surfaces', () => {
  /** Boot App with a transport that resolves immediately, past the overlay. */
  async function renderShell() {
    mockCreateTransport.mockResolvedValue({
      kind: 'mock',
      label: 'mock',
      connect: vi.fn(),
      send: vi.fn(),
      close: vi.fn(),
    });
    render(<App />);
    await waitFor(() =>
      expect(screen.queryByText('Starting simulator…')).toBeNull(),
    );
  }

  const PANELS = [
    { id: 'scenarios', label: 'Scenarios', selector: '.sidebar' },
    { id: 'inspector', label: 'Inspector', selector: '.panel' },
    { id: 'telemetry', label: 'Telemetry', selector: '.telemetry' },
  ] as const;

  for (const { id, label, selector } of PANELS) {
    it(`folds the ${id} surface away and back from its handle`, async () => {
      await renderShell();
      const handle = screen.getByTestId(`toggle-${id}`);

      expect(screen.getByLabelText(label)).toBeTruthy();
      expect(handle.getAttribute('aria-expanded')).toBe('true');

      fireEvent.click(handle);
      // Collapsed means unmounted, not hidden: nothing offscreen stays
      // focusable and the widget canvases stop drawing.
      expect(document.querySelector(selector)).toBeNull();
      expect(screen.getByTestId(`toggle-${id}`).getAttribute('aria-expanded')).toBe(
        'false',
      );

      // The handle is anchored to the viewport, so it survives its own panel
      // and is the way back.
      fireEvent.click(screen.getByTestId(`toggle-${id}`));
      expect(screen.getByLabelText(label)).toBeTruthy();
    });
  }

  it('keeps each surface independent', async () => {
    await renderShell();
    fireEvent.click(screen.getByTestId('toggle-scenarios'));
    fireEvent.click(screen.getByTestId('toggle-telemetry'));

    expect(useSimStore.getState().collapsed).toEqual({
      scenarios: true,
      inspector: false,
      telemetry: true,
    });
    expect(screen.getByLabelText('Inspector')).toBeTruthy();
  });

  it('names the action on the handle so the label tracks the state', async () => {
    await renderShell();
    expect(screen.getByTestId('toggle-inspector').getAttribute('aria-label')).toBe(
      'Hide inspector panel',
    );
    fireEvent.click(screen.getByTestId('toggle-inspector'));
    expect(screen.getByTestId('toggle-inspector').getAttribute('aria-label')).toBe(
      'Show inspector panel',
    );
  });
});
