/**
 * Shown when the hosted shell cannot get a simulator: the backend is
 * unreachable, refused this page's origin, is full, or speaks a different
 * protocol. Without it a stranger sees an empty viewport and a small
 * "reconnecting" chip, with no hint that anything is wrong or what to do.
 */
import type { ConnectionFailure, ConnectionFailureKind } from '../net/transport';

const COPY: Record<ConnectionFailureKind, { title: string; body: string }> = {
  backend_down: {
    title: "Can't reach the simulator",
    body: 'The simulation server is not answering. It may be restarting; this page keeps trying in the background.',
  },
  origin_rejected: {
    title: 'The simulator refused this page',
    body: "The server only accepts connections from its own website, and this page's address is not on its allowed list. Retrying will not change that until the server is reconfigured.",
  },
  server_busy: {
    title: 'The simulator is busy',
    body: 'Every simulator slot is in use right now. Try again in a minute.',
  },
  protocol_mismatch: {
    title: 'Version mismatch',
    body: 'This page and the simulator speak different protocol versions. Reload to pick up the latest page; if it persists, the two deployments are out of sync.',
  },
};

interface Props {
  failure: ConnectionFailure;
  onRetry: () => void;
  onUseMock: () => void;
}

export function ConnectionErrorOverlay({ failure, onRetry, onUseMock }: Props) {
  const copy = COPY[failure.kind];
  return (
    <div
      className="startup-overlay"
      role="alert"
      data-testid="connection-error"
      data-kind={failure.kind}
    >
      <div className="startup-card">
        <p className="startup-title startup-title--error">{copy.title}</p>
        <p className="startup-body">{copy.body}</p>
        <p className="startup-reason">{failure.detail}</p>
        <button type="button" className="startup-fallback" onClick={onRetry}>
          Retry
        </button>
        <button type="button" className="startup-secondary" onClick={onUseMock}>
          Use offline demo instead
        </button>
      </div>
    </div>
  );
}
