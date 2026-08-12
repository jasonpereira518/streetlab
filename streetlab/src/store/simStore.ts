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
  SceneDescription,
  ScenarioSummary,
  ServerMessage,
  SimEvent,
  StateUpdate,
} from '../schema';
import { LAYER_KEYS } from '../schema';
import type { ConnectionStatus, Transport } from '../net/transport';

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

  /* mirrored frame fields (only updated on change) */
  paused: boolean;
  assistActive: boolean;
  hasFrames: boolean;

  /* UI state */
  layers: Record<LayerKey, boolean>;
  params: Record<string, ParamValue>;
  cameraView: CameraView;
  rightTab: 'parameters' | 'map' | 'layers';

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
  setParam(key: string, value: ParamValue): void;
  setLayer(layer: LayerKey, visible: boolean): void;
  setCameraView(view: CameraView): void;
  setRightTab(tab: 'parameters' | 'map' | 'layers'): void;
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

  paused: false,
  assistActive: false,
  hasFrames: false,

  layers: { ...DEFAULT_LAYERS },
  params: { ...DEFAULT_PARAMS },
  cameraView: 'chase',
  rightTab: 'parameters',

  events: [],
  lastAck: null,
  invalidCount: 0,
  lastInvalid: null,
  commandLog: [],

  attach(transport) {
    transportRef = transport;
    frameBus.reset();
    set({
      sourceKind: transport.kind,
      sourceLabel: transport.label,
      status: 'connecting',
      hasFrames: false,
    });

    transport.connect({
      onMessage: (msg) => applyServerMessage(msg, set, get),
      onStatus: (status, detail) => set({ status, statusDetail: detail ?? '' }),
      onInvalid: (error, raw) => {
        console.warn('[streetlab] dropped invalid frame:', error, raw);
        set((s) => ({ invalidCount: s.invalidCount + 1, lastInvalid: error }));
      },
    });

    return () => {
      transport.close();
      if (transportRef === transport) transportRef = null;
    };
  },

  send(partial) {
    const id = partial.id ?? `c${++commandSeq}`;
    const command = { ...partial, id } as Command;
    transportRef?.send(command);
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

  resetSim() {
    get().send({ cmd: 'reset' });
  },

  injectHazard() {
    get().send({ cmd: 'inject_hazard', kind: 'cutin' });
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
      }));
      return;

    case 'state_update': {
      // Hot path. Publish first so the renderer sees the newest frame as early
      // as possible, then mirror only what actually changed into React state.
      frameBus.publish(msg);
      const s = get();
      const patch: Partial<SimStoreState> = {};
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
      }
      if (Object.keys(patch).length) set(patch);
      return;
    }

    case 'ack':
      set({ lastAck: msg });
      return;
  }
}

/** Convenience for non-React consumers (the renderer) that need the scene. */
export const getScene = (): SceneDescription | null => useSimStore.getState().scene;
