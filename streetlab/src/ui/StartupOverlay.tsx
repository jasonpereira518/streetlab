/**
 * Covers the shell between mount and the first real transport attach.
 *
 * Most of the time this is invisible — the mock and the plain ?backend= path
 * resolve in a microtask. It only lingers when running inside the packaged
 * app, waiting on the Tauri sidecar handshake, and it is the only place that
 * explains *why* nothing connected if that handshake fails.
 */
interface StartupOverlayProps {
  phase: 'starting' | 'error';
  reason?: string;
  onUseMock: () => void;
}

export function StartupOverlay({ phase, reason, onUseMock }: StartupOverlayProps) {
  return (
    <div className="startup-overlay" role="status" aria-live="polite">
      <div className="startup-card">
        {phase === 'starting' ? (
          <>
            <div className="startup-spinner" aria-hidden="true" />
            <p className="startup-title">Starting simulator…</p>
          </>
        ) : (
          <>
            <p className="startup-title startup-title--error">Simulator didn't start</p>
            {reason && <p className="startup-reason">{reason}</p>}
            <button type="button" className="startup-fallback" onClick={onUseMock}>
              Use mock instead
            </button>
          </>
        )}
      </div>
    </div>
  );
}
