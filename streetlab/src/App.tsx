import { useEffect, useRef, useState } from 'react';
import { createMockTransport } from './net/mockServer';
import { createTransportFromLocation } from './net/wsClient';
import { useSimStore } from './store/simStore';
import { Renderer } from './three/Renderer';
import { LeftScenarioSidebar } from './ui/LeftScenarioSidebar';
import { PanelHandle } from './ui/PanelHandle';
import { PerfOverlay } from './ui/PerfOverlay';
import { RightPanel } from './ui/RightPanel';
import { StartupOverlay } from './ui/StartupOverlay';
import { TelemetryRow } from './ui/TelemetryRow';
import { TopToolbar } from './ui/TopToolbar';

type BootPhase = 'starting' | 'ready' | 'error';

export default function App() {
  const attach = useSimStore((s) => s.attach);
  const collapsed = useSimStore((s) => s.collapsed);
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

  // Collapsed panels unmount rather than shrink to zero: nothing offscreen
  // stays focusable, and the widget canvases stop drawing entirely. The
  // matching track sizes are zeroed in CSS off these same classes.
  const shell = [
    'app',
    collapsed.scenarios && 'is-scenarios-collapsed',
    collapsed.inspector && 'is-inspector-collapsed',
    collapsed.telemetry && 'is-telemetry-collapsed',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={shell}>
      <TopToolbar />
      <div className="stage">
        {!collapsed.scenarios && <LeftScenarioSidebar />}
        <main className="stage-main">
          <Renderer />
          <PanelHandle panel="scenarios" edge="left" />
          <PanelHandle panel="inspector" edge="right" />
          <PanelHandle panel="telemetry" edge="bottom" />
        </main>
        {!collapsed.inspector && <RightPanel />}
      </div>
      {!collapsed.telemetry && <TelemetryRow />}
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
