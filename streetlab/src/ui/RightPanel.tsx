/**
 * Right-hand inspector: planner/traffic/render parameters, a live overhead map,
 * and layer visibility. Every control routes through the store, which turns it
 * into a `set_param` or `toggle_layer` command — the panel never reaches into
 * the renderer or the simulator directly.
 */
import type { LayerKey, ParamValue } from '../schema';
import { LAYER_KEYS } from '../schema';
import { useTelemetryCanvas } from '../store/hooks';
import { PARAM_DEFS, useSimStore } from '../store/simStore';
import type { ParamDef, RightTab } from '../store/simStore';
import { EventLog } from './EventLog';
import { ActivityIcon, LayersIcon, MapIcon, SlidersIcon } from './Icons';
import { ColorPicker, Field, Select, Slider, Toggle } from './controls';
import { alpha, classColor, color } from './theme';

type Tab = RightTab;

const TABS: Array<{ id: Tab; label: string; icon: typeof MapIcon }> = [
  { id: 'parameters', label: 'Parameters', icon: SlidersIcon },
  { id: 'map', label: 'Map', icon: MapIcon },
  { id: 'layers', label: 'Layers', icon: LayersIcon },
  { id: 'events', label: 'Events', icon: ActivityIcon },
];

const LAYER_LABELS: Record<LayerKey, string> = {
  detections: 'Detections',
  plan_path: 'Plan path',
  lane_markings: 'Lane markings',
  crosswalks: 'Crosswalks',
  buildings: 'Buildings',
  trees: 'Trees',
  traffic_lights: 'Signals & signs',
  radar_cone: 'Radar cone',
  labels: 'Labels',
};

const LAYER_HINTS: Partial<Record<LayerKey, string>> = {
  detections: 'Traffic meshes and hazard boxes',
  labels: 'Street blades and warning billboards',
};

const GROUP_TITLES = {
  planner: 'Planner',
  traffic: 'Traffic',
  render: 'Rendering',
} as const;

export function RightPanel() {
  const tab = useSimStore((s) => s.rightTab);
  const setTab = useSimStore((s) => s.setRightTab);

  return (
    <aside className="panel" aria-label="Inspector">
      <div className="panel-tabs" role="tablist">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            className={`panel-tab${tab === id ? ' is-active' : ''}`}
            onClick={() => setTab(id)}
          >
            <Icon size={15} />
            <span>{label}</span>
          </button>
        ))}
      </div>

      <div className="panel-body" role="tabpanel">
        {tab === 'parameters' && <ParametersTab />}
        {tab === 'map' && <MapTab />}
        {tab === 'layers' && <LayersTab />}
        {tab === 'events' && <EventLog />}
      </div>
    </aside>
  );
}

/* ------------------------------------------------------------------ */

function ParametersTab() {
  const params = useSimStore((s) => s.params);
  const setParam = useSimStore((s) => s.setParam);
  const injectHazard = useSimStore((s) => s.injectHazard);
  const lastAck = useSimStore((s) => s.lastAck);

  const groups = (['planner', 'traffic', 'render'] as const).map((g) => ({
    key: g,
    defs: PARAM_DEFS.filter((d) => d.group === g),
  }));

  return (
    <>
      {groups.map(({ key, defs }) => (
        <Field key={key} title={GROUP_TITLES[key]}>
          {defs.map((def) => (
            <ParamControl
              key={def.key}
              def={def}
              value={params[def.key]}
              onChange={(v) => setParam(def.key, v)}
            />
          ))}
        </Field>
      ))}

      <Field title="Actions">
        <button type="button" className="panel-action" onClick={injectHazard}>
          Inject cut-in hazard
        </button>
        {lastAck && (
          <p className={`ack${lastAck.ok ? '' : ' ack--error'}`}>
            <code>{lastAck.cmd}</code>
            <span>{lastAck.message ?? (lastAck.ok ? 'ok' : 'failed')}</span>
          </p>
        )}
      </Field>
    </>
  );
}

function ParamControl({
  def,
  value,
  onChange,
}: {
  def: ParamDef;
  value: ParamValue | undefined;
  onChange: (v: ParamValue) => void;
}) {
  const current = value ?? def.default;
  switch (def.kind) {
    case 'slider':
      return (
        <Slider
          label={def.label}
          value={Number(current)}
          min={def.min ?? 0}
          max={def.max ?? 1}
          step={def.step ?? 0.1}
          unit={def.unit}
          hint={def.hint}
          onChange={onChange}
        />
      );
    case 'toggle':
      return (
        <Toggle
          label={def.label}
          checked={Boolean(current)}
          hint={def.hint}
          onChange={onChange}
        />
      );
    case 'select':
      return (
        <Select
          label={def.label}
          value={String(current)}
          options={def.options ?? []}
          onChange={onChange}
        />
      );
    case 'color':
      return (
        <ColorPicker label={def.label} value={String(current)} onChange={onChange} />
      );
  }
}

/* ------------------------------------------------------------------ */

function LayersTab() {
  const layers = useSimStore((s) => s.layers);
  const setLayer = useSimStore((s) => s.setLayer);
  const commandLog = useSimStore((s) => s.commandLog);
  const lastToggle = commandLog.find((c) => c.cmd === 'toggle_layer');

  return (
    <>
      <Field title="Scene layers">
        {LAYER_KEYS.map((key) => (
          <Toggle
            key={key}
            label={LAYER_LABELS[key]}
            hint={LAYER_HINTS[key]}
            checked={layers[key]}
            onChange={(v) => setLayer(key, v)}
          />
        ))}
      </Field>
      <Field title="Presets">
        <div className="preset-row">
          <button
            type="button"
            className="panel-action panel-action--sm"
            onClick={() => {
              for (const k of LAYER_KEYS) setLayer(k, true);
            }}
          >
            Show all
          </button>
          <button
            type="button"
            className="panel-action panel-action--sm"
            onClick={() => {
              for (const k of LAYER_KEYS) {
                setLayer(k, k === 'detections' || k === 'plan_path' || k === 'lane_markings');
              }
            }}
          >
            Perception only
          </button>
        </div>
        {lastToggle && (
          <p className="ack">
            <code>toggle_layer</code>
            <span>sent as {lastToggle.id}</span>
          </p>
        )}
      </Field>
    </>
  );
}

/* ------------------------------------------------------------------ */

function MapTab() {
  const scene = useSimStore((s) => s.scene);

  const ref = useTelemetryCanvas(({ ctx, width, height, frame }) => {
    if (!scene) return;
    const { bounds } = scene;
    const M = 8;
    const spanX = bounds.max_x - bounds.min_x;
    const spanY = bounds.max_y - bounds.min_y;
    const scale = Math.min((width - M * 2) / spanX, (height - M * 2) / spanY);
    const ox = (width - spanX * scale) / 2;
    const oy = (height - spanY * scale) / 2;
    const px = (x: number) => ox + (x - bounds.min_x) * scale;
    const py = (y: number) => oy + (bounds.max_y - y) * scale;

    ctx.fillStyle = color.surfaceSunken;
    ctx.fillRect(0, 0, width, height);

    // Building footprints.
    ctx.fillStyle = alpha(color.borderStrong, 0.75);
    for (const b of scene.buildings) {
      ctx.beginPath();
      b.footprint.forEach(([x, y], i) =>
        i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y)),
      );
      ctx.closePath();
      ctx.fill();
    }

    // Carriageways.
    ctx.strokeStyle = color.surface;
    ctx.lineCap = 'butt';
    for (const r of scene.roads) {
      ctx.lineWidth = Math.max(
        2,
        (r.lanes_forward + r.lanes_backward) * r.lane_width_m * scale,
      );
      ctx.beginPath();
      r.centerline.forEach(([x, y], i) =>
        i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y)),
      );
      ctx.stroke();
    }

    if (!frame) return;

    // Plan.
    ctx.strokeStyle = color.plan;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    frame.plan.polyline.forEach(([x, y], i) =>
      i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y)),
    );
    ctx.stroke();

    // Detections.
    for (const d of frame.detections) {
      ctx.beginPath();
      ctx.arc(px(d.pose.x), py(d.pose.y), d.hazard ? 4 : 3, 0, Math.PI * 2);
      ctx.fillStyle = d.hazard ? color.warn : classColor[d.cls] ?? color.textMuted;
      ctx.fill();
    }

    // Ego, drawn as a heading triangle.
    const ex = px(frame.ego.pose.x);
    const ey = py(frame.ego.pose.y);
    ctx.save();
    ctx.translate(ex, ey);
    ctx.rotate(-frame.ego.pose.heading);
    ctx.beginPath();
    ctx.moveTo(7, 0);
    ctx.lineTo(-4, 4.5);
    ctx.lineTo(-4, -4.5);
    ctx.closePath();
    ctx.fillStyle = color.accent;
    ctx.fill();
    ctx.strokeStyle = color.surface;
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.restore();
  });

  return (
    <>
      <Field title="Overhead">
        <div className="map-canvas">
          <canvas ref={ref} aria-label="Overhead scenario map" role="img" />
        </div>
      </Field>
      <Field title="Scene">
        <dl className="facts">
          <div>
            <dt>Location</dt>
            <dd>{scene?.location ?? '—'}</dd>
          </div>
          <div>
            <dt>Origin</dt>
            <dd>
              {scene ? `${scene.origin.lat.toFixed(4)}, ${scene.origin.lon.toFixed(4)}` : '—'}
            </dd>
          </div>
          <div>
            <dt>Roads</dt>
            <dd>{scene?.roads.length ?? 0}</dd>
          </div>
          <div>
            <dt>Buildings</dt>
            <dd>{scene?.buildings.length ?? 0}</dd>
          </div>
          <div>
            <dt>Signals</dt>
            <dd>{scene?.traffic_lights.length ?? 0}</dd>
          </div>
          <div>
            <dt>Trees</dt>
            <dd>{scene?.trees.length ?? 0}</dd>
          </div>
        </dl>
      </Field>
    </>
  );
}
