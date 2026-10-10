// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/react';
import { DemoBanner, RELEASES_URL } from '../src/ui/DemoBanner';

describe('DemoBanner', () => {
  it('says the demo is scripted and links to the releases page', () => {
    const { getByRole, getByText } = render(<DemoBanner />);
    expect(getByRole('note').textContent).toMatch(/scripted data/i);
    expect(getByText('Get the full app').getAttribute('href')).toBe(RELEASES_URL);
  });
});
