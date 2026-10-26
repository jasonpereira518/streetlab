/**
 * Over the road: what the keys do right now, and on-screen driving controls.
 *
 * The hint follows the drive mode, so the one thing a newcomer needs is always
 * in view ("W gas · S brake · A/D steer" in Manual, "steer or brake to take
 * over" under FSD). The on-screen pedals and steering are virtual keys
 * (`pressVirtual`), so touch and mouse get the keyboard's exact feel; they
 * show by default on a touch screen and behind a toggle elsewhere.
 */
import { useState, type PointerEvent as ReactPointerEvent } from 'react';
import { pressVirtual, releaseVirtual, type Control } from '../input/driveKeys';
import type { CruiseMode } from '../schema';
import { useFrameValue } from '../store/hooks';
import { useSimStore } from '../store/simStore';
import { blindSpots } from './warnings';

export const DRIVE_HINTS: Record<CruiseMode, string> = {
  off: 'W gas · S brake (hold at a stop to reverse) · A/D steer · Q/E signal · F FSD',
  cruise: 'A/D steer · W faster · S take over · Q/E signal · [ ] set speed · L Assist',
  autosteer: 'Steer or brake to take over · Q/E change lane · [ ] set speed',
  fsd: 'Steer or brake to take over · Q/E change lane · F to drive yourself',
};

function coarsePointer(): boolean {
  try {
    return typeof window !== 'undefined' && window.matchMedia('(pointer: coarse)').matches;
  } catch {
    return false;
  }
}

function HoldButton({ control, label, className }: { control: Control; label: string; className: string }) {
  const down = (e: ReactPointerEvent<HTMLButtonElement>) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    pressVirtual(control);
  };
  const up = () => releaseVirtual(control);
  return (
    <button
      type="button"
      className={`drive-pad-btn ${className}`}
      aria-label={label}
      onPointerDown={down}
      onPointerUp={up}
      onPointerCancel={up}
      onLostPointerCapture={up}
      onContextMenu={(e) => e.preventDefault()}
    >
      {label}
    </button>
  );
}

export function DriveHud() {
  const mode = useFrameValue((f) => f.ego.cruise.mode, 10);
  const reversing = useFrameValue((f) => f.ego.gear === 'R', 10);
  // Tesla's blind-spot light: steady while a vehicle is there, blinking when
  // the indicator points at it.
  const spots = useFrameValue((f) => {
    const b = blindSpots(f);
    const blink = (side: 'left' | 'right') => (f.ego.turn_signal === side ? '!' : '');
    return `${b.left ? `L${blink('left')}` : ''}|${b.right ? `R${blink('right')}` : ''}`;
  }, 10);
  const toggleDriveMode = useSimStore((s) => s.toggleDriveMode);
  const [pad, setPad] = useState(coarsePointer);
  if (!mode) return null;
  const [left, right] = (spots ?? '|').split('|');
  return (
    <div className="drive-hud">
      {(left || right) && (
        <div className="blind-spots" role="status" aria-label="Blind spot">
          {left && <span className={`blind-spot${left.endsWith('!') ? ' blind-spot--warn' : ''}`}>◀ Blind spot</span>}
          {right && <span className={`blind-spot blind-spot--right${right.endsWith('!') ? ' blind-spot--warn' : ''}`}>Blind spot ▶</span>}
        </div>
      )}
      <p className="drive-hint" data-testid="drive-hint">
        {reversing && <span className="drive-gear">R</span>}
        {DRIVE_HINTS[mode]}
        <button
          type="button"
          className="drive-pad-toggle"
          aria-pressed={pad}
          onClick={() => setPad((p) => !p)}
        >
          On-screen controls
        </button>
      </p>
      {pad && (
        <div className="drive-pad" aria-label="On-screen driving controls">
          <div className="drive-pad-group">
            <HoldButton control="left" label="Steer left" className="drive-pad-btn--steer" />
            <HoldButton control="right" label="Steer right" className="drive-pad-btn--steer" />
          </div>
          <button type="button" className="drive-pad-btn drive-pad-btn--mode" onClick={() => toggleDriveMode('fsd')}>
            {mode === 'fsd' ? 'Take over' : 'FSD'}
          </button>
          <div className="drive-pad-group">
            <HoldButton control="brake" label="Brake" className="drive-pad-btn--brake" />
            <HoldButton control="throttle" label="Gas" className="drive-pad-btn--gas" />
          </div>
        </div>
      )}
    </div>
  );
}
