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
          mode: 'ground-truth', detector_ms: 4.5, e2e_ms: 31.2,
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
  });
});
