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
  AddressSuggestion,
  CameraView,
  Command,
  CommandInput,
  LayerKey,
  ParamValue,
  PerceptionMode,
  PerceptionStats,
  RunSummary,
  SceneDescription,
  ScenarioSummary,
  ServerMessage,
  SimEvent,
  StateUpdate,
} from '../schema';
import { LAYER_KEYS } from '../schema';
import type { ConnectionStatus, Transport } from '../net/transport';
import { httpUrlForWsLabel, perfMetrics } from '../perf/perfMetrics';

/** Whether the server has an ML perception pipeline. Not `perception !==
 * null`: noisy truth sends stats without one. Gates camera-frame uploads
 * (Renderer) and the ML perception option (TopToolbar). */
export const hasPipeline = (p: PerceptionStats | null): boolean => p?.pipeline === true;

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
    label: 'Mean cut-in interval',
    kind: 'slider',
    group: 'traffic',
    default: 0,
    min: 0,
    max: 60,
    step: 1,
    unit: 's',
    hint: '0 = off; Poisson mean seconds between spontaneous cut-ins',
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
/* Session refresh                                                     */
/* ------------------------------------------------------------------ */

/**
 * How long `refreshAll` waits for the backend to confirm the reset before
 * reloading regardless. Generous next to a local socket round trip, short
 * enough that a wedged or already-dead backend never leaves the button stuck.
 */
export const RESET_ACK_TIMEOUT_MS = 600;

/**
 * Navigating is the one thing jsdom will not do, so the reload lives in the
 * store as a swappable field rather than a bare `window.location.reload()`
 * inside the action. That keeps the ordering this feature exists for —
 * reset first, reload second — testable, which is the only part of it that
 * can silently be wrong.
 */
export const DEFAULT_RELOAD_PAGE = (): void => {
  window.location.reload();
};

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
export type RightTab = 'parameters' | 'run' | 'map' | 'layers' | 'events';

/** Scorecards kept per preset (newest first) — what the Run tab compares. */
export const RUN_HISTORY_CAP = 5;

/**
 * The three shell surfaces the user can fold away to give the viewport more
 * room. Keyed rather than three booleans so `togglePanel` stays one action and
 * a new surface cannot be added without the reducer seeing it.
 */
export type PanelId = 'scenarios' | 'inspector' | 'telemetry';

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
  /**
   * The most recent `location_progress` event for the in-flight build, or
   * `null` while none has arrived yet (the ack-to-first-checkpoint gap, or a
   * cache-hit build that finishes before ever reporting one). Cleared
   * whenever `locationPending` is — a fresh `loadLocation` call, the
   * eventual `scene_description`, or a `location_failed` event.
   */
  locationProgress: { stage: string; fraction: number } | null;
  /**
   * Plain, user-facing text from the most recent `location_failed` event, or
   * `null` when nothing has failed since the last attempt. Cleared the
   * instant a new `loadLocation` call goes out, and on a successful
   * `scene_description` — same lifecycle as `locationPending`, just carrying
   * the failure text rather than only a boolean.
   */
  locationError: string | null;
  /**
   * True once a `trip_complete` event has arrived for the currently-loaded
   * scene (a point-to-point route the ego has actually stopped at the end
   * of). Cleared on every `loadLocation`/`loadScenario` call and on a fresh
   * `scene_description`, so it never carries over from a previous trip.
   */
  tripComplete: boolean;
  /**
   * Replies to in-flight `suggest_address` requests, keyed by the request's
   * own command id — not by query text, so two fields typing similar
   * addresses at once (start + destination) never clobber each other's
   * results. Callers look up their own id and ignore the rest; entries are
   * pruned oldest-first past `MAX_ADDRESS_SUGGESTIONS` so a long session
   * typing many addresses doesn't grow this without bound.
   */
  addressSuggestions: Record<string, { query: string; items: AddressSuggestion[] }>;

  /* mirrored frame fields (only updated on change) */
  paused: boolean;
  assistActive: boolean;
  hasFrames: boolean;
  /** Null when no ML perception is running — distinct from "measured, and
   * zero"; see PerceptionPanel. Updated on every frame, unlike the fields
   * above, since its counters are expected to change every tick. */
  perception: PerceptionStats | null;
  /** What the planner drives on. Optimistic on `setPerceptionMode`; corrected
   * from `perception.mode` whenever a frame carries it, and from the preset's
   * own mode when a preset's scene arrives. */
  perceptionMode: PerceptionMode;

  /* walkthrough presets */
  /** The preset the current scene was loaded by; server-authoritative. */
  activePresetId: string | null;
  /** The seed the current scene is running on, from `scene.seed`. */
  runSeed: number | null;
  /** Last seed each preset ran on, so Replay can re-run it exactly. */
  lastSeedByPreset: Record<string, number>;
  /** Scorecards by `preset_id ?? 'ad-hoc'`, newest first, capped. Harvested
   * the frame they arrive so the 40-entry `events` cap can never evict one,
   * and kept across scene loads so runs can be compared. */
  runHistory: Record<string, RunSummary[]>;

  /* UI state */
  layers: Record<LayerKey, boolean>;
  params: Record<string, ParamValue>;
  cameraView: CameraView;
  rightTab: RightTab;
  /** Purely local chrome state — collapsing a panel sends no command. */
  collapsed: Record<PanelId, boolean>;
  perfOverlayVisible: boolean;
  /** A `refreshAll` is in flight; the button that starts one is disabled. */
  refreshPending: boolean;
  /** See `DEFAULT_RELOAD_PAGE` — swapped by tests, never at runtime. */
  reloadPage: () => void;

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
  loadLocation(query: string, destination?: string): void;
  /** Run a preset: no seed follows its policy; a seed replays that run. */
  loadPreset(presetId: string, seed?: number): void;
  /** Fire off a `suggest_address` request and return its command id, so the
   * caller can look its result up in `addressSuggestions` once it arrives. */
  suggestAddress(query: string): string;
  setParam(key: string, value: ParamValue): void;
  setLayer(layer: LayerKey, visible: boolean): void;
  setCameraView(view: CameraView): void;
  setPerceptionMode(mode: PerceptionMode): void;
  setRightTab(tab: RightTab): void;
  togglePanel(panel: PanelId): void;
  togglePerfOverlay(): void;
  resetSim(): void;
  refreshAll(): Promise<void>;
  injectHazard(kind: string): void;
}

let transportRef: Transport | null = null;
let commandSeq = 0;

/** Bounds `addressSuggestions` the same way `commandLog`'s `.slice(0, 50)`
 * bounds itself — small, since only the two search fields ever populate it. */
const MAX_ADDRESS_SUGGESTIONS = 8;

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
  locationProgress: null,
  locationError: null,
  tripComplete: false,
  addressSuggestions: {},

  paused: false,
  assistActive: false,
  hasFrames: false,
  perception: null,
  perceptionMode: 'ground-truth',

  activePresetId: null,
  runSeed: null,
  lastSeedByPreset: {},
  runHistory: {},

  layers: { ...DEFAULT_LAYERS },
  params: { ...DEFAULT_PARAMS },
  cameraView: 'chase',
  rightTab: 'parameters',
  collapsed: { scenarios: false, inspector: false, telemetry: false },
  perfOverlayVisible: false,
  refreshPending: false,
  reloadPage: DEFAULT_RELOAD_PAGE,

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
    // `suggest_address` is excluded for the same reason at a smaller
    // scale: it fires on every debounced keystroke in an address field and
    // never gets an ack, so it would just be noise among real commands.
    if (command.cmd === 'camera_frame' || command.cmd === 'suggest_address') return id;
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
    set({
      activeScenarioId: scenarioId,
      activePresetId: null,
      locationProgress: null,
      locationError: null,
      tripComplete: false,
    });
    get().send({ cmd: 'load_scenario', scenario_id: scenarioId });
  },

  loadLocation(query, destination) {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;
    const trimmedDest = destination?.trim() || undefined;
    set({
      locationPending: trimmedDest ? `${trimmedQuery} → ${trimmedDest}` : trimmedQuery,
      activePresetId: null,
      locationProgress: null,
      locationError: null,
      tripComplete: false,
    });
    get().send({
      cmd: 'load_location',
      query: trimmedQuery,
      ...(trimmedDest ? { destination: trimmedDest } : {}),
    });
  },

  loadPreset(presetId, seed) {
    set({ activePresetId: presetId, rightTab: 'run' });
    get().send({
      cmd: 'load_preset',
      preset_id: presetId,
      ...(seed === undefined ? {} : { seed }),
    });
  },

  suggestAddress(query) {
    return get().send({ cmd: 'suggest_address', query });
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

  setPerceptionMode(mode) {
    // Optimistic local update, same shape as setCameraView above: the
    // backend's next frame carries the confirmed `perception.mode` (non-null
    // whenever noisy-truth or ML is driving), which corrects a refused switch.
    set((s) => ({
      perceptionMode: mode,
      ...(s.perception ? { perception: { ...s.perception, mode } } : {}),
    }));
    get().send({ cmd: 'set_perception', mode });
  },

  setRightTab(tab) {
    set({ rightTab: tab });
  },

  togglePanel(panel) {
    set((s) => ({ collapsed: { ...s.collapsed, [panel]: !s.collapsed[panel] } }));
  },

  togglePerfOverlay() {
    set((s) => ({ perfOverlayVisible: !s.perfOverlayVisible }));
  },

  resetSim() {
    get().send({ cmd: 'reset' });
  },

  /**
   * Restart the whole session: reset the simulator, then reload the app.
   *
   * The wait between the two is the point. `reloadPage` tears down the
   * websocket, and a `send` issued into a socket that is about to close is
   * not guaranteed to be flushed — so reloading without waiting would
   * intermittently come back to a simulation that never reset, which is the
   * failure mode a "refresh everything" button most needs not to have.
   */
  async refreshAll() {
    if (get().refreshPending) return;
    set({ refreshPending: true });

    await new Promise<void>((resolve) => {
      let id = '';
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        unsubscribe();
        resolve();
      };
      const unsubscribe = useSimStore.subscribe((s, prev) => {
        if (s.lastAck !== prev.lastAck && s.lastAck?.id === id) finish();
      });
      const timer = setTimeout(finish, RESET_ACK_TIMEOUT_MS);

      id = get().send({ cmd: 'reset' });
      // A transport that acks synchronously inside `send` — the in-process
      // mock does — already ran the subscriber above, back when `id` was
      // still the empty string it could not match. Check for that ack here,
      // where the id finally exists.
      if (get().lastAck?.id === id) finish();
    });

    // Only observable when `reloadPage` is a no-op, i.e. under test. In the
    // app the navigation below ends this document.
    set({ refreshPending: false });
    get().reloadPage();
  },

  injectHazard(kind) {
    // `kind` is a `HazardSummary.code` from the scene's `hazards`; the
    // backend declines, by name, one the scene cannot host.
    get().send({ cmd: 'inject_hazard', kind });
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
      set((s) => {
        // Server-authoritative: on the shared hosted world another client's
        // load_preset swaps this scene too, and preset_id says so.
        const preset = msg.presets.find((p) => p.id === msg.preset_id);
        return {
          scene: msg,
          sceneEpoch: s.sceneEpoch + 1,
          catalog: msg.catalog,
          activeScenarioId: msg.scenario_id,
          hasFrames: false,
          events: [],
          locationPending: null,
          locationProgress: null,
          locationError: null,
          tripComplete: false,
          activePresetId: msg.preset_id,
          runSeed: msg.seed,
          // runHistory is deliberately untouched: comparing runs is the point.
          ...(msg.preset_id !== null
            ? { lastSeedByPreset: { ...s.lastSeedByPreset, [msg.preset_id]: msg.seed } }
            : {}),
          // Mirror what the server applied, so the sliders and the perception
          // control stay honest. Not re-sent: the server already has them.
          ...(preset
            ? { params: { ...s.params, ...preset.params }, perceptionMode: preset.perception }
            : {}),
        };
      });
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
      // Null perception happens in exactly one backend state: ground truth
      // with no pipeline (a pipeline always sends stats; so does noisy truth).
      const wireMode = msg.perception?.mode ?? 'ground-truth';
      if (wireMode !== s.perceptionMode) patch.perceptionMode = wireMode;
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
        const summaries = msg.events.flatMap((e) => (e.summary ? [e.summary] : []));
        if (summaries.length) {
          const history = { ...s.runHistory };
          for (const summary of summaries) {
            const key = summary.preset_id ?? 'ad-hoc';
            history[key] = [summary, ...(history[key] ?? [])].slice(0, RUN_HISTORY_CAP);
          }
          patch.runHistory = history;
        }
        // A geocode/Overpass failure (or a query with no drivable roads)
        // never produces a new scene — it surfaces here instead, per
        // sim/loop.py's `submit_scene`. Without this the box would stay
        // disabled forever on any bad address, the single most likely thing
        // a first-time user types.
        const failure = msg.events.find((e) => e.code === 'location_failed');
        if (failure) {
          if (s.locationPending !== null) patch.locationPending = null;
          patch.locationProgress = null;
          patch.locationError = failure.message;
        }
        // A point-to-point trip's own arrival, surfaced next to the search
        // box the same way a failure is — see LeftScenarioSidebar.tsx.
        if (!s.tripComplete && msg.events.some((e) => e.code === 'trip_complete')) {
          patch.tripComplete = true;
        }
        // The build's own checkpoints, for a live progress bar. Last one in
        // this batch wins — events land in the order the backend emitted
        // them (see sim/loop.py's `submit_scene`), so the last is the
        // farthest along. A `location_progress` event always carries a
        // `progress` fraction (sim/loop.py's `emit_progress` never omits
        // it); the `?? 0` only guards a hand-built test fixture that didn't.
        const progress = [...msg.events].reverse().find((e) => e.code === 'location_progress');
        if (progress) {
          patch.locationProgress = { stage: progress.message, fraction: progress.progress ?? 0 };
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
        set({
          lastAck: msg,
          locationPending: null,
          locationProgress: null,
          locationError: msg.message,
        });
        return;
      }
      // A refused load_preset never swaps the scene, so the optimistic
      // activePresetId from `loadPreset` falls back to what is running.
      if (msg.cmd === 'load_preset' && !msg.ok) {
        set((s) => ({ lastAck: msg, activePresetId: s.scene?.preset_id ?? null }));
        return;
      }
      set({ lastAck: msg });
      return;

    case 'address_suggestions':
      set((s) => {
        const ids = Object.keys(s.addressSuggestions);
        const evicted =
          ids.length >= MAX_ADDRESS_SUGGESTIONS
            ? Object.fromEntries(
                ids.slice(ids.length - MAX_ADDRESS_SUGGESTIONS + 1).map((id) => [id, s.addressSuggestions[id]]),
              )
            : s.addressSuggestions;
        return {
          addressSuggestions: {
            ...evicted,
            [msg.id]: { query: msg.query, items: msg.suggestions },
          },
        };
      });
      return;
  }
}

/** Convenience for non-React consumers (the renderer) that need the scene. */
export const getScene = (): SceneDescription | null => useSimStore.getState().scene;
