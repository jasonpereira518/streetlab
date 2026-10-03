// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { PerceptionPanel } from '../src/ui/PerceptionPanel';

afterEach(cleanup);

describe('PerceptionPanel', () => {
  it('says nothing is measured when perception is null', () => {
    render(<PerceptionPanel stats={null} />);
    expect(screen.getByText(/not running/i).textContent).toContain('not running');
  });

  it('shows transport numbers and marks quality as pending', () => {
    render(
      <PerceptionPanel
        stats={{
          mode: 'ground-truth', detector_ms: 4.5, server_e2e_ms: 31.2,
          frames_received: 120, frames_dropped: 3,
          precision: null, recall: null, mean_pos_err_m: null,
        }}
      />,
    );
    expect(screen.getByTestId('frames').textContent).toBe('120 received / 3 dropped');
    expect(screen.getByTestId('detector-ms').textContent).toBe('4.5 ms');
    // A null must never render as 0 — that would claim a measurement nobody made.
    expect(screen.getByTestId('precision').textContent).toBe('—');
    expect(screen.getByTestId('recall').textContent).toBe('—');
    expect(screen.getByTestId('mean-pos-err').textContent).toBe('—');
  });

  it('renders a measured mean position error with an m suffix', () => {
    render(
      <PerceptionPanel
        stats={{
          mode: 'ml', detector_ms: 4.5, server_e2e_ms: 31.2,
          frames_received: 120, frames_dropped: 3,
          precision: 0.91, recall: 0.83, mean_pos_err_m: 1.234,
        }}
      />,
    );
    expect(screen.getByTestId('mean-pos-err').textContent).toBe('1.23 m');
  });

  it('distinguishes "not measured" from "measured, and zero" for precision', () => {
    // A genuine 0/0-derived or otherwise-zero score must render as 0.00, never
    // collapse visually into the null/dash case above — that is the whole
    // reason this panel renders null and 0 differently.
    render(
      <PerceptionPanel
        stats={{
          mode: 'ml', detector_ms: 4.5, server_e2e_ms: 31.2,
          frames_received: 120, frames_dropped: 3,
          precision: 0, recall: 0, mean_pos_err_m: 0,
        }}
      />,
    );
    expect(screen.getByTestId('precision').textContent).toBe('0.00');
    expect(screen.getByTestId('recall').textContent).toBe('0.00');
    expect(screen.getByTestId('mean-pos-err').textContent).toBe('0.00 m');
    expect(screen.getByTestId('precision').textContent).not.toBe('—');
    expect(screen.getByTestId('mean-pos-err').textContent).not.toBe('—');
  });
});
