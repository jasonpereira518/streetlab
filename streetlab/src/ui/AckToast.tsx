/**
 * Shows the backend's answer to a hazard injection (or any refused command)
 * over the viewport, where the user is looking. The Params tab keeps its own
 * copy, but nobody watching the road has that tab open.
 */
import { useEffect, useState } from 'react';
import { useSimStore } from '../store/simStore';

const SHOW_MS = 5000;

export function AckToast() {
  const ack = useSimStore((s) => s.lastAck);
  const [shown, setShown] = useState(ack);

  const worthShowing = ack != null && (ack.cmd === 'inject_hazard' || !ack.ok);
  useEffect(() => {
    if (!worthShowing) return;
    setShown(ack);
    const t = setTimeout(() => setShown(null), SHOW_MS);
    return () => clearTimeout(t);
  }, [ack, worthShowing]);

  // The live region stays mounted while empty so a screen reader registers it
  // before the first message lands.
  const visible = shown != null && shown === ack && worthShowing;
  return (
    <div className="ack-toast-region" role="status" aria-live="polite">
      {visible && (
        <p className={`ack-toast${shown.ok ? '' : ' ack-toast--error'}`}>
          <code>{shown.cmd}</code>
          <span>{shown.message ?? (shown.ok ? 'ok' : 'failed')}</span>
        </p>
      )}
    </div>
  );
}
