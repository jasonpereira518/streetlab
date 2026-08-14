import { useEffect, useRef, useState } from 'react';
import { createMockTransport } from './net/mockServer';
import { createTransportFromLocation } from './net/wsClient';
import { useSimStore } from './store/simStore';
import { Renderer } from './three/Renderer';
import { LeftScenarioSidebar } from './ui/LeftScenarioSidebar';
import { PerfOverlay } from './ui/PerfOverlay';
import { RightPanel } from './ui/RightPanel';
import { StartupOverlay } from './ui/StartupOverlay';
import { TelemetryRow } from './ui/TelemetryRow';
import { TopToolbar } from './ui/TopToolbar';

type BootPhase = 'starting' | 'ready' | 'error';

export default function App() {
  const attach = useSimStore((s) => s.attach);
  const [boot, setBoot] = useState<BootPhase>('starting');
  const [bootError, setBootError] = useState('');
  const cleanup = useRef<(() => void) | undefined>(undefined);

  // One transport for the life of the app: resolved from the page URL, the
  // Tauri sidecar handshake, or the in-process mock — see
  // createTransportFromLocation for the precedence.
  useEffect(() => {
    let cancelled = false;

    createTransportFromLocation()
      .then((transport) => {
        if (cancelled) return;
        cleanup.current = attach(transport);
        setBoot('ready');
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setBootError(err instanceof Error ? err.message : String(err));
        setBoot('error');
      });

    return () => {
      cancelled = true;
      cleanup.current?.();
    };
  }, [attach]);

  const useMock = () => {
    cleanup.current = attach(createMockTransport());
    setBoot('ready');
  };

  return (
    <div className="app">
      <TopToolbar />
      <div className="stage">
        <LeftScenarioSidebar />
        <main className="stage-main">
          <Renderer />
        </main>
        <RightPanel />
      </div>
      <TelemetryRow />
      <PerfOverlay />
      {boot !== 'ready' && (
        <StartupOverlay
          phase={boot === 'error' ? 'error' : 'starting'}
          reason={bootError}
          onUseMock={useMock}
        />
      )}
    </div>
  );
}
