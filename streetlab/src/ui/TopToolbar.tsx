/**
 * Application chrome: identity and file actions on the left, transport control
 * centre-left, the live driving status cluster in the middle, view and settings
 * on the right. Every live number comes through `useFrameValue`, which polls the
 * frame stream at ~10 Hz and only re-renders when the displayed value changes.
 */
import { useEffect, useRef, useState } from 'react';
import type { CameraView } from '../schema';
import { useFrameValue } from '../store/hooks';
import { useSimStore } from '../store/simStore';
import { formatTtc, toMph } from '../units';
import {
  BrandMark,
  CameraIcon,
  ChevronDownIcon,
  FileIcon,
  PauseIcon,
  PlayIcon,
  ResetIcon,
  SaveIcon,
  SettingsIcon,
  UndoIcon,
} from './Icons';
import { IconButton } from './controls';

const CAMERA_LABELS: Record<CameraView, string> = {
  chase: 'Chase',
  overhead: 'Overhead',
  cockpit: 'Cockpit',
  free: 'Free orbit',
};

const CRUISE_LABELS: Record<string, string> = {
  off: 'Manual',
  cruise: 'Cruise',
  autosteer: 'Autosteer',
  fsd: 'Full Self-Driving',
};

const MANEUVER_LABELS: Record<string, string> = {
  keep_lane: 'Keeping lane',
  turn_left: 'Turning left',
  turn_right: 'Turning right',
  lane_change_left: 'Changing lane left',
  lane_change_right: 'Changing lane right',
  stop: 'Stopping',
  yield: 'Yielding',
};

export function TopToolbar() {
  const paused = useSimStore((s) => s.paused);
  const assist = useSimStore((s) => s.assistActive);
  const status = useSimStore((s) => s.status);
  const sourceLabel = useSimStore((s) => s.sourceLabel);
  const cameraView = useSimStore((s) => s.cameraView);
  const scenarioName = useSimStore(
    (s) => s.catalog.find((c) => c.id === s.activeScenarioId)?.name ?? s.scene?.name ?? '—',
  );
  const togglePaused = useSimStore((s) => s.togglePaused);
  const resetSim = useSimStore((s) => s.resetSim);
  const setCameraView = useSimStore((s) => s.setCameraView);
  const setRightTab = useSimStore((s) => s.setRightTab);

  const mph = useFrameValue((f) => Math.round(toMph(f.ego.speed_mps)), 12);
  const ttc = useFrameValue(
    (f) => (f.telemetry.ttc_s == null ? null : Math.round(f.telemetry.ttc_s * 10) / 10),
    8,
  );
  const cruise = useFrameValue((f) => f.ego.cruise.mode, 4);
  const maneuver = useFrameValue((f) => f.plan.maneuver, 4);

  const ttcCritical = ttc != null && ttc < 2.5;

  return (
    <header className="toolbar">
      <div className="toolbar-group toolbar-group--left">
        <div className="brand">
          <BrandMark />
          <div className="brand-text">
            <span className="brand-name">StreetLab</span>
            <span className="brand-sub">{scenarioName}</span>
          </div>
        </div>
        <span className="toolbar-sep" />
        <IconButton label="New session">
          <FileIcon />
        </IconButton>
        <IconButton label="Save scenario">
          <SaveIcon />
        </IconButton>
        <IconButton label="Undo">
          <UndoIcon />
        </IconButton>
      </div>

      <div className="toolbar-group toolbar-group--transport">
        <button
          type="button"
          className={`transport${paused ? ' is-paused' : ''}`}
          onClick={togglePaused}
          aria-label={paused ? 'Resume simulation' : 'Pause simulation'}
          title={paused ? 'Resume' : 'Pause'}
        >
          {paused ? <PlayIcon size={17} /> : <PauseIcon size={17} />}
        </button>
        <IconButton label="Reset scenario" onClick={resetSim}>
          <ResetIcon />
        </IconButton>
      </div>

      <div className="toolbar-group toolbar-group--status">
        <div className={`status-pill${assist ? ' is-on' : ''}`}>
          <span className="status-dot" />
          <span className="status-label">
            {assist ? 'Assist active' : 'Assist off'}
          </span>
        </div>

        <div className="readout">
          <span className="readout-value">{mph ?? '—'}</span>
          <span className="readout-unit">mph</span>
        </div>

        <div className={`readout readout--ttc${ttcCritical ? ' is-critical' : ''}`}>
          <span className="readout-value">{formatTtc(ttc)}</span>
          <span className="readout-unit">TTC</span>
        </div>

        <div className="mode-chip" title={MANEUVER_LABELS[maneuver ?? ''] ?? ''}>
          <span className="mode-chip-title">
            {CRUISE_LABELS[cruise ?? 'off'] ?? 'Manual'}
          </span>
          <span className="mode-chip-sub">
            {MANEUVER_LABELS[maneuver ?? 'keep_lane'] ?? '—'}
          </span>
        </div>
      </div>

      <div className="toolbar-group toolbar-group--right">
        <span className={`link-chip link-chip--${status}`} title={`Source: ${sourceLabel}`}>
          {sourceLabel}
        </span>
        <CameraMenu view={cameraView} onSelect={setCameraView} />
        <IconButton label="Settings" onClick={() => setRightTab('parameters')}>
          <SettingsIcon />
        </IconButton>
      </div>
    </header>
  );
}

/* ------------------------------------------------------------------ */

function CameraMenu({
  view,
  onSelect,
}: {
  view: CameraView;
  onSelect: (v: CameraView) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div className="menu" ref={ref}>
      <button
        type="button"
        className={`menu-trigger${open ? ' is-open' : ''}`}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        title="Camera view"
      >
        <CameraIcon />
        <span>{CAMERA_LABELS[view]}</span>
        <ChevronDownIcon size={14} />
      </button>
      {open && (
        <div className="menu-list" role="menu">
          {(Object.keys(CAMERA_LABELS) as CameraView[]).map((v) => (
            <button
              key={v}
              type="button"
              role="menuitemradio"
              aria-checked={v === view}
              className={`menu-item${v === view ? ' is-active' : ''}`}
              onClick={() => {
                onSelect(v);
                setOpen(false);
              }}
            >
              {CAMERA_LABELS[v]}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
