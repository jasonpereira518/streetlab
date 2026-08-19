/**
 * Application store.
 *
 * Split into two halves on purpose:
 *
 *  - `frameBus` carries the 60 Hz `StateUpdate` stream. It is a plain
 *    publish/subscribe object, deliberately *outside* React, because pushing
 *    60 new objects a second through component state would re-render the whole
 *    UI every frame. The renderer and the canvas telemetry widgets read it
 *    imperatively from their own animation loop.
 *
 *  - `useSimStore` (zustand) holds everything that changes rarely: the scene,
 *    connection status, layer visibility, parameters, scenario catalog. Frame
 *    fields that the DOM genuinely needs (paused, assist, scenario) are mirrored
 *    into it, but only when the value actually changes.
 *
 * Nothing above this file touches the transport directly.
 */
import { create } from 'zustand';
import type {
  Ack,
  CameraView,
  Command,
  CommandInput,
  LayerKey,
  ParamValue,
  PerceptionStats,
  SceneDescription,
  ScenarioSummary,
  ServerMessage,
  SimEvent,
  StateUpdate,
} from '../schema';
import { LAYER_KEYS } from '../schema';
import type { ConnectionStatus, Transport } from '../net/transport';
import { httpUrlForWsLabel, perfMetrics } from '../perf/perfMetrics';

/* ------------------------------------------------------------------ */
/* Frame bus                                                           */
/* ------------------------------------------------------------------ */

type FrameListener = (frame: StateUpdate) => void;

class FrameBus {
  latest: StateUpdate | null = null;
  /** Frames received since the last reset; used for the FPS/rate readout. */
  received = 0;
  private listeners = new Set<FrameListener>();

  publish(frame: StateUpdate): void {
    this.latest = frame;
    this.received++;
    for (const l of this.listeners) l(frame);
  }

  subscribe(listener: FrameListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  reset(): void {
    this.latest = null;
    this.received = 0;
  }
}

export const frameBus = new FrameBus();

/* ------------------------------------------------------------------ */
/* Parameter registry                                                  */
/* ------------------------------------------------------------------ */

export type ParamKind = 'slider' | 'toggle' | 'select' | 'color';

export interface ParamDef {
  key: string;
  label: string;
  kind: ParamKind;
  group: 'planner' | 'traffic' | 'render';
  default: ParamValue;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: Array<{ value: string; label: string }>;
  /** Render-only params never leave the client. */
  clientOnly?: boolean;
  hint?: string;
}

export const PARAM_DEFS: ParamDef[] = [
  {
    key: 'ego_speed_cap_mph',
    label: 'Max speed',
    kind: 'slider',
    group: 'planner',
    default: 45,
    min: 15,
    max: 75,
    step: 1,
    unit: 'mph',
    hint: 'Upper bound the planner will target',
  },
  {
    key: 'follow_distance_s',
    label: 'Follow distance',
    kind: 'slider',
    group: 'planner',
    default: 1.5,
    min: 0.6,
    max: 3,
    step: 0.1,
    unit: 's',
  },
  {
    key: 'assist_enabled',
    label: 'Assist engaged',
    kind: 'toggle',
    group: 'planner',
    default: true,
  },
  {
    key: 'traffic_speed_scale',
    label: 'Traffic speed',
    kind: 'slider',
    group: 'traffic',
    default: 1,
    min: 0.4,
    max: 1.6,
    step: 0.05,
    unit: '×',
  },
  {
    key: 'cutin_period_s',
    label: 'Cut-in interval',
    kind: 'slider',
    group: 'traffic',
    default: 22,
    min: 6,
    max: 60,
    step: 1,
    unit: 's',
  },
  {
    key: 'plan_opacity',
    label: 'Plan opacity',
    kind: 'slider',
    group: 'render',
    default: 0.55,
    min: 0.1,
    max: 1,
    step: 0.05,
    clientOnly: true,
  },
  {
    key: 'label_scale',
    label: 'Label size',
    kind: 'slider',
    group: 'render',
    default: 1,
    min: 0.7,
    max: 1.6,
    step: 0.05,
    clientOnly: true,
  },
  {
    key: 'hazard_color',
    label: 'Hazard colour',
    kind: 'color',
    group: 'render',
    default: '#FF7A1A',
    clientOnly: true,
  },
  {
    key: 'time_of_day',
    label: 'Lighting',
    kind: 'select',
    group: 'render',
    default: 'midday',
    clientOnly: true,
    options: [
      { value: 'morning', label: 'Morning' },
      { value: 'midday', label: 'Midday' },
      { value: 'golden', label: 'Golden hour' },
      { value: 'overcast', label: 'Overcast' },
    ],
  },
];

const DEFAULT_PARAMS: Record<string, ParamValue> = Object.fromEntries(
  PARAM_DEFS.map((d) => [d.key, d.default]),
);

const DEFAULT_LAYERS = Object.fromEntries(
  LAYER_KEYS.map((k) => [k, true]),
) as Record<LayerKey, boolean>;

/* ------------------------------------------------------------------ */
/* Store                                                               */
/* ------------------------------------------------------------------ */

/**
 * Right-panel tab identifiers. Defined once here — rather than as a
 * separately-typed literal union duplicated in `RightPanel.tsx` — so a tab
 * added to one side and forgotten on the other is a type error, not a
 * runtime surprise (`setRightTab('events')` compiling while the panel has
 * nothing registered for it, or vice versa).
 */
export type RightTab = 'parameters' | 'map' | 'layers' | 'events';

export interface SimStoreState {
  /* connection */
  status: ConnectionStatus;
  statusDetail: string;
  sourceKind: 'mock' | 'ws';
  sourceLabel: string;

  /* world */
  scene: SceneDescription | null;
  /** Bumped whenever a new scene arrives, so the renderer can rebuild. */
  sceneEpoch: number;
  catalog: ScenarioSummary[];
  activeScenarioId: string | null;
  /**
   * The trimmed query text of an in-flight `load_location` build, or `null`
   * when none is outstanding. Set the instant the command is sent; cleared
   * either by the eventual `scene_description` or by a `location_failed`
   * event — whichever arrives first. Never carries a request id (the wire
   * protocol has none for this), so it can only ever track "is *something*
   * building", not which specific query a late event belongs to.
   */
  locationPending: string | null;

  /* mirrored frame fields (only updated on change) */
  paused: boolean;
  assistActive: boolean;
  hasFrames: boolean;
  /** Null when no ML perception is running — distinct from "measured, and
   * zero"; see PerceptionPanel. Updated on every frame, unlike the fields
   * above, since its counters are expected to change every tick. */
  perception: PerceptionStats | null;

  /* UI state */
  layers: Record<LayerKey, boolean>;
  params: Record<string, ParamValue>;
  cameraView: CameraView;
  rightTab: RightTab;
  perfOverlayVisible: boolean;

  /* diagnostics */
  events: SimEvent[];
  lastAck: Ack | null;
  invalidCount: number;
  lastInvalid: string | null;
  /** Commands sent this session, newest first — used by tests and the log. */
  commandLog: Array<{ id: string; cmd: string; at: number }>;

  /* actions */
  attach(transport: Transport): () => void;
  send(command: CommandInput): string;
  togglePaused(): void;
  loadScenario(scenarioId: string): void;
  loadLocation(query: string): void;
  setParam(key: string, value: ParamValue): void;
  setLayer(layer: LayerKey, visible: boolean): void;
  setCameraView(view: CameraView): void;
  setRightTab(tab: RightTab): void;
  togglePerfOverlay(): void;
  resetSim(): void;
  injectHazard(): void;
}

let transportRef: Transport | null = null;
let commandSeq = 0;

export const useSimStore = create<SimStoreState>((set, get) => ({
  status: 'idle',
  statusDetail: '',
  sourceKind: 'mock',
  sourceLabel: 'mock',

  scene: null,
  sceneEpoch: 0,
  catalog: [],
  activeScenarioId: null,
  locationPending: null,

  paused: false,
  assistActive: false,
  hasFrames: false,
  perception: null,

  layers: { ...DEFAULT_LAYERS },
  params: { ...DEFAULT_PARAMS },
  cameraView: 'chase',
  rightTab: 'parameters',
  perfOverlayVisible: false,

  events: [],
  lastAck: null,
  invalidCount: 0,
  lastInvalid: null,
  commandLog: [],

  attach(transport) {
    transportRef = transport;
    frameBus.reset();
    perfMetrics.reset();
    perfMetrics.watchHealth(httpUrlForWsLabel(transport.label));
    set({
      sourceKind: transport.kind,
      sourceLabel: transport.label,
      status: 'connecting',
      hasFrames: false,
    });

    transport.connect({
      onMessage: (msg) => applyServerMessage(msg, set, get),
      onStatus: (status, detail) =>
        set((s) => ({
          status,
          statusDetail: detail ?? '',
          // A transport that has given up for good (`closed`) will never
          // deliver the scene_description or location_failed event that
          // would otherwise clear this. `closed` isn't reachable today from
          // a flaky connection — wsClient.ts's scheduleRetry() retries
          // forever with capped backoff, it never gives up on its own;
          // `closed` only happens via an intentional close() (app teardown)
          // or a `reconnect: false` config. This is future-proofing for
          // either of those, not a fix for a currently-reachable stuck
          // state. A transient `reconnecting` blip is left alone: the build
          // may still land once the socket comes back, and a fresh
          // connection gets its own scene_description (ws_server.py pushes
          // one on every accept), which already clears this through the
          // ordinary path.
          locationPending: status === 'closed' ? null : s.locationPending,
        })),
      onInvalid: (error, raw) => {
        console.warn('[streetlab] dropped invalid frame:', error, raw);
        set((s) => ({ invalidCount: s.invalidCount + 1, lastInvalid: error }));
      },
      onRawFrame: (bytes) => perfMetrics.reportFrameBytes(bytes),
    });

    return () => {
      transport.close();
      perfMetrics.watchHealth(null);
      if (transportRef === transport) transportRef = null;
    };
  },

  send(partial) {
    const id = partial.id ?? `c${++commandSeq}`;
    const command = { ...partial, id } as Command;
    transportRef?.send(command);
    // `camera_frame` is excluded from the log the same way it is excluded
    // from the ack path everywhere else (wsClient.ts, ws_server.py,
    // harness.tsx): at 10 Hz it would be 100% of a 50-entry log within five
    // seconds, permanently hiding diagnostics like LayersTab's last-toggle
    // readout, and would force every commandLog subscriber to re-render at
    // 10 Hz forever since `send()` allocates a new array on every call.
    if (command.cmd === 'camera_frame') return id;
    set((s) => ({
      commandLog: [
        { id, cmd: command.cmd, at: Date.now() },
        ...s.commandLog,
      ].slice(0, 50),
    }));
    return id;
  },

  togglePaused() {
    get().send({ cmd: 'set_paused', paused: !get().paused });
  },

  loadScenario(scenarioId) {
    set({ activeScenarioId: scenarioId });
    get().send({ cmd: 'load_scenario', scenario_id: scenarioId });
  },

  loadLocation(query) {
    const trimmed = query.trim();
    if (!trimmed) return;
    set({ locationPending: trimmed });
    get().send({ cmd: 'load_location', query: trimmed });
  },

  setParam(key, value) {
    set((s) => ({ params: { ...s.params, [key]: value } }));
    const def = PARAM_DEFS.find((d) => d.key === key);
    if (def?.clientOnly) return;
    get().send({ cmd: 'set_param', key, value });
  },

  setLayer(layer, visible) {
    set((s) => ({ layers: { ...s.layers, [layer]: visible } }));
    get().send({ cmd: 'toggle_layer', layer, visible });
  },

  setCameraView(view) {
    set({ cameraView: view });
    get().send({ cmd: 'set_camera', view });
  },

  setRightTab(tab) {
    set({ rightTab: tab });
  },

  togglePerfOverlay() {
    set((s) => ({ perfOverlayVisible: !s.perfOverlayVisible }));
  },

  resetSim() {
    get().send({ cmd: 'reset' });
  },

  injectHazard() {
    // `cut_in` is the backend's own name for this scenario
    // (`streetlab-backend/sim/events.py`). This shipped as `cutin`, which cost
    // nothing while every kind produced the identical hard-brake and would
    // cost the button its ack now that they do not. The backend still accepts
    // the old spelling as an alias, so an older build of this app keeps
    // working against a newer sidecar.
    get().send({ cmd: 'inject_hazard', kind: 'cut_in' });
  },
}));

/* ------------------------------------------------------------------ */
/* Message routing                                                     */
/* ------------------------------------------------------------------ */

type Setter = (
  partial:
    | Partial<SimStoreState>
    | ((s: SimStoreState) => Partial<SimStoreState>),
) => void;

function applyServerMessage(
  msg: ServerMessage,
  set: Setter,
  get: () => SimStoreState,
): void {
  switch (msg.type) {
    case 'scene_description':
      frameBus.reset();
      set((s) => ({
        scene: msg,
        sceneEpoch: s.sceneEpoch + 1,
        catalog: msg.catalog,
        activeScenarioId: msg.scenario_id,
        hasFrames: false,
        events: [],
        locationPending: null,
      }));
      return;

    case 'state_update': {
      // Hot path. Publish first so the renderer sees the newest frame as early
      // as possible, then mirror only what actually changed into React state.
      frameBus.publish(msg);
      perfMetrics.reportTick(performance.now());
      const s = get();
      const patch: Partial<SimStoreState> = {};
      // Unlike the other mirrored fields, perception is not gated on strict
      // equality: its counters (frames_received, frames_dropped) are expected
      // to move on essentially every tick while ML perception is running, and
      // a fresh object reference would defeat a `!==` check every time
      // anyway. But it must still be gated on *something*, or `patch` is
      // never empty and `set()` below fires 60 times a second even when
      // perception is null and nothing else changed. Both sides null is the
      // one case guaranteed not to be a change.
      if (s.perception !== null || msg.perception !== null) {
        patch.perception = msg.perception;
      }
      if (s.paused !== msg.paused) patch.paused = msg.paused;
      if (s.assistActive !== msg.assist_active) {
        patch.assistActive = msg.assist_active;
      }
      if (!s.hasFrames) patch.hasFrames = true;
      if (s.activeScenarioId !== msg.scenario_id) {
        patch.activeScenarioId = msg.scenario_id;
      }
      if (msg.events.length) {
        patch.events = [...s.events, ...msg.events].slice(-40);
        // A geocode/Overpass failure (or a query with no drivable roads)
        // never produces a new scene — it surfaces here instead, per
        // sim/loop.py's `submit_scene`. Without this the box would stay
        // disabled forever on any bad address, the single most likely thing
        // a first-time user types.
        if (
          s.locationPending !== null &&
          msg.events.some((e) => e.code === 'location_failed')
        ) {
          patch.locationPending = null;
        }
      }
      if (Object.keys(patch).length) set(patch);
      return;
    }

    case 'ack':
      // A `load_location` that fails AT ACK TIME never reaches the executor,
      // so it never emits `location_failed` and never produces a scene — the
      // two things above that clear `locationPending`. Without this the
      // sidebar locks: the search box AND every scenario play button stay
      // disabled until a reload or a dropped connection. It is not an exotic
      // path — `_cmd_load_location` (sim/loop.py) acks `ok=false` whenever the
      // source has no `build_location` at all, which is every backend started
      // as plain `streetlab serve` (the CLI's `--source` default is
      // `synthetic`). Typing an address there is the documented behaviour in
      // DEMO.md, and it used to brick the sidebar.
      if (msg.cmd === 'load_location' && !msg.ok) {
        set({ lastAck: msg, locationPending: null });
        return;
      }
      set({ lastAck: msg });
      return;
  }
}

/** Convenience for non-React consumers (the renderer) that need the scene. */
export const getScene = (): SceneDescription | null => useSimStore.getState().scene;
