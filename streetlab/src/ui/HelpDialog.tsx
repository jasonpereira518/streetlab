/**
 * In-app help: the keyboard shortcuts and a legend for what the viewport
 * draws. `SHORTCUTS` is the single list both the key handler and the dialog
 * read, so the visible list cannot drift from what the keys do.
 */
import { useEffect } from 'react';
import { useSimStore } from '../store/simStore';

export const SHORTCUTS: Array<{ keys: string; does: string }> = [
  { keys: 'Space', does: 'Pause / resume the simulation' },
  { keys: 'R', does: 'Reset the scenario' },
  { keys: '?', does: 'Show / hide this help' },
  { keys: 'Esc', does: 'Close help or an open menu' },
];

const LEGEND: Array<{ swatch: string; name: string; does: string }> = [
  { swatch: 'plan', name: 'Plan path', does: "Where the planner intends to drive next; the colour follows its intent." },
  { swatch: 'ref', name: 'Driven line', does: 'The route the ego follows through the scene.' },
  { swatch: 'hazard', name: 'Hazard box', does: 'A detection the planner is reacting to.' },
  { swatch: 'det', name: 'Detection box', does: 'Other road users the perception stack sees.' },
];

/** Keys that mean something else when a control or text field has focus. */
function ownsTheKey(target: EventTarget | null, key: string): boolean {
  const el = target as HTMLElement | null;
  if (!el || !el.tagName) return false;
  if (el.isContentEditable) return true;
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) return true;
  // Space activates a focused button / switch / tab; Enter-style activation
  // must not be hijacked into a pause.
  return key === ' ' && !!el.closest('button, a, [role="switch"], [role="tab"], [role="menuitemradio"]');
}

export function useShortcuts(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
      const s = useSimStore.getState();
      if (e.key === 'Escape') {
        if (s.helpOpen) s.setHelpOpen(false);
        return;
      }
      if (ownsTheKey(e.target, e.key)) return;
      if (e.key === ' ') {
        e.preventDefault();
        s.togglePaused();
      } else if (e.key === 'r' || e.key === 'R') {
        s.resetSim();
      } else if (e.key === '?') {
        s.setHelpOpen(!s.helpOpen);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}

export function HelpDialog() {
  const open = useSimStore((s) => s.helpOpen);
  const setOpen = useSimStore((s) => s.setHelpOpen);
  if (!open) return null;
  return (
    <div className="help-dialog" role="dialog" aria-label="Help and shortcuts">
      <header className="help-head">
        <h2>Help</h2>
        <button type="button" className="help-close" onClick={() => setOpen(false)} aria-label="Close help">
          Close
        </button>
      </header>
      <h3 className="help-sub">Shortcuts</h3>
      <dl className="help-list">
        {SHORTCUTS.map((k) => (
          <div key={k.keys}>
            <dt>
              <kbd>{k.keys}</kbd>
            </dt>
            <dd>{k.does}</dd>
          </div>
        ))}
      </dl>
      <h3 className="help-sub">Legend</h3>
      <dl className="help-list">
        {LEGEND.map((l) => (
          <div key={l.name}>
            <dt>
              <span className={`help-swatch help-swatch--${l.swatch}`} aria-hidden="true" />
              {l.name}
            </dt>
            <dd>{l.does}</dd>
          </div>
        ))}
      </dl>
      <h3 className="help-sub">Perception</h3>
      <p className="help-note">
        Ground truth hands the planner the simulator's exact world and is the default driver. ML
        runs a real detector on rendered frames and drives from what it sees. It is experimental
        and needs the backend started with <code>--perception</code>. Measured, it does not meet
        its gates: car recall 0.150 against 0.70 required, and the closed-loop gate failed
        criteria 1, 3, 4, 5 and 7. Three-camera ML needs about 1.5 to 2.1 times the 100 ms frame
        interval on one CPU worker.
      </p>
    </div>
  );
}
