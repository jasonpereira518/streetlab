/**
 * Hosted waitlist UI. The server holds every connection over its session cap
 * in a FIFO line (server/ws_server.py `_wait_in_line`): this is the card they
 * see while waiting, the banner that warns a driver before an idle / time-up
 * close, and the two effects that keep both honest -- a tab-title alert on
 * admission, and heartbeats so camera-only input counts as presence.
 */
import { useEffect, useRef, useState } from 'react';
import { useSimStore } from '../store/simStore';

const HEARTBEAT_MS = 30_000;
const TURN_TITLE = '▶ Your turn! — StreetLab';

function formatWait(etaS: number): string {
  const min = Math.round(etaS / 60);
  return min < 1 ? 'under a minute' : `~${min} min`;
}

export function QueueOverlay({ onUseMock }: { onUseMock: () => void }) {
  const queue = useSimStore((s) => s.queue);
  // The bar measures progress from where this visit joined the line.
  const joinedAt = useRef<number | null>(null);
  if (queue && joinedAt.current === null) joinedAt.current = queue.position;
  if (!queue) {
    joinedAt.current = null;
    return null;
  }
  const start = Math.max(joinedAt.current ?? queue.position, queue.position);
  const progress = start > 1 ? (start - queue.position) / (start - 1) : 0;

  return (
    <div className="startup-overlay" data-testid="queue-overlay">
      <div className="startup-card" role="status" aria-live="polite">
        <p className="startup-title">You're in line</p>
        <p className="queue-position" data-testid="queue-position">
          #{queue.position} <span>of {queue.size}</span>
        </p>
        <p className="startup-body">Estimated wait: {formatWait(queue.eta_s)}</p>
        <div
          className="queue-bar"
          role="progressbar"
          aria-label="Progress through the line"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(progress * 100)}
        >
          <div className="queue-bar__fill" style={{ width: `${Math.max(4, progress * 100)}%` }} />
        </div>
        <p className="startup-reason">
          Two people can drive at once. You'll be connected automatically — keep this tab open.
        </p>
        <button type="button" className="startup-secondary" onClick={onUseMock}>
          Use offline demo instead (leaves the line)
        </button>
      </div>
    </div>
  );
}

export function SessionNoticeBanner() {
  const notice = useSimStore((s) => s.notice);
  const send = useSimStore((s) => s.send);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!notice) return;
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [notice]);
  if (!notice) return null;
  const left = Math.max(0, Math.ceil(notice.seconds_left - (now - notice.at) / 1000));

  return (
    <div className="session-notice" role="alert" data-testid="session-notice" data-kind={notice.kind}>
      {notice.kind === 'idle' ? (
        <>
          <span>Still there? Disconnecting in {left}s.</span>
          <button
            type="button"
            className="startup-fallback"
            onClick={() => {
              send({ cmd: 'heartbeat' });
              // The server's `ok` follows on its next check; clear now so the
              // countdown does not keep running under the user's click.
              useSimStore.setState({ notice: null });
            }}
          >
            I'm here
          </button>
        </>
      ) : (
        <span>Others are waiting — your session ends in {left}s.</span>
      )}
    </div>
  );
}

/** Tab-title alert on admission, and throttled heartbeats from real input. */
export function useWaitlistEffects(): void {
  const queued = useSimStore((s) => s.queue !== null);
  const live = useSimStore((s) => s.sourceKind === 'ws' && s.status === 'open');
  const wasQueued = useRef(false);

  useEffect(() => {
    const admitted = wasQueued.current && !queued;
    wasQueued.current = queued;
    if (!admitted || !document.hidden) return;
    const original = document.title;
    document.title = TURN_TITLE;
    const restore = () => {
      if (document.hidden) return;
      document.title = original;
      document.removeEventListener('visibilitychange', restore);
    };
    document.addEventListener('visibilitychange', restore);
    return () => document.removeEventListener('visibilitychange', restore);
  }, [queued]);

  useEffect(() => {
    if (!live) return;
    let last = 0;
    const beat = () => {
      const t = Date.now();
      if (t - last < HEARTBEAT_MS) return;
      last = t;
      useSimStore.getState().send({ cmd: 'heartbeat' });
    };
    const events = ['pointerdown', 'keydown', 'wheel'] as const;
    for (const e of events) window.addEventListener(e, beat, { passive: true });
    return () => {
      for (const e of events) window.removeEventListener(e, beat);
    };
  }, [live]);
}
