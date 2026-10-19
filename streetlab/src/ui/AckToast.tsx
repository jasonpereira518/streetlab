/**
 * Shows the backend's answer to a hazard injection (or any refused command)
 * over the viewport, where the user is looking. The Params tab keeps its own
 * copy, but nobody watching the road has that tab open. Driving incidents
 * (a collision, leaving the road) land here too: without them the car simply
 * stops for no visible reason.
 */
import { useEffect, useState } from 'react';
import { useSimStore } from '../store/simStore';

const SHOW_MS = 5000;
/** Sim events a driver needs to see where they are looking. */
const DRIVING_EVENTS = new Set(['collision', 'off_road']);

interface Toast {
  key: unknown;
  label: string;
  message: string;
  error: boolean;
}

export function AckToast() {
  const ack = useSimStore((s) => s.lastAck);
  const incident = useSimStore((s) => {
    for (let i = s.events.length - 1; i >= 0; i--) {
      if (DRIVING_EVENTS.has(s.events[i].code)) return s.events[i];
    }
    return null;
  });
  const [shown, setShown] = useState<Toast | null>(null);

  const show = (toast: Toast) => {
    setShown(toast);
    const t = setTimeout(() => setShown((cur) => (cur?.key === toast.key ? null : cur)), SHOW_MS);
    return () => clearTimeout(t);
  };

  const ackWorthShowing = ack != null && (ack.cmd === 'inject_hazard' || !ack.ok);
  useEffect(() => {
    if (!ack || !ackWorthShowing) return;
    return show({ key: ack, label: ack.cmd, message: ack.message ?? (ack.ok ? 'ok' : 'failed'), error: !ack.ok });
  }, [ack, ackWorthShowing]);

  useEffect(() => {
    if (!incident) return;
    return show({
      key: incident,
      label: incident.code === 'collision' ? 'Collision' : 'Off road',
      message: incident.message,
      error: true,
    });
  }, [incident]);

  // The live region stays mounted while empty so a screen reader registers it
  // before the first message lands.
  return (
    <div className="ack-toast-region" role="status" aria-live="polite">
      {shown && (
        <p className={`ack-toast${shown.error ? ' ack-toast--error' : ''}`}>
          <code>{shown.label}</code>
          <span>{shown.message}</span>
        </p>
      )}
    </div>
  );
}
