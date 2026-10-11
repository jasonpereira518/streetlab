import { describe, expect, it } from 'vitest';
import { formatClock, formatKm } from '../src/units';

describe('trip formatting', () => {
  it('formats distance in km to one decimal', () => {
    expect(formatKm(0)).toBe('0.0');
    expect(formatKm(1849)).toBe('1.8');
    expect(formatKm(12_340)).toBe('12.3');
  });

  it('formats an ETA as m:ss, or h:mm:ss past an hour', () => {
    expect(formatClock(0)).toBe('0:00');
    expect(formatClock(250.4)).toBe('4:10');
    expect(formatClock(3725)).toBe('1:02:05');
  });
});
