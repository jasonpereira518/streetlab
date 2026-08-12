import { useEffect } from 'react';
import { createTransportFromLocation } from './net/wsClient';
import { useSimStore } from './store/simStore';
import { Renderer } from './three/Renderer';
import { LeftScenarioSidebar } from './ui/LeftScenarioSidebar';
import { RightPanel } from './ui/RightPanel';
import { TelemetryRow } from './ui/TelemetryRow';
import { TopToolbar } from './ui/TopToolbar';

export default function App() {
  const attach = useSimStore((s) => s.attach);

  // One transport for the life of the app: the mock by default, or a real
  // simulator when the page was opened with ?backend=ws://host:port.
  useEffect(() => attach(createTransportFromLocation()), [attach]);

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
    </div>
  );
}
