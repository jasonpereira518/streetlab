/**
 * Test harness: a controllable transport plus helpers to drive the store with
 * real frames from `MockSim`. Nothing here reaches into component internals —
 * tests interact through the same store and schema the UI uses.
 */
import { act } from '@testing-library/react';
import type { Command, SceneDescription, ServerMessage, StateUpdate } from '../src/schema';
import type { Transport, TransportHandlers } from '../src/net/transport';
import { MockSim } from '../src/net/mockServer';
import { PARAM_DEFS, frameBus, useSimStore } from '../src/store/simStore';
import { LAYER_KEYS } from '../src/schema';

export interface Harness {
  transport: Transport;
  sent: Command[];
  sim: MockSim;
  /** Push the scene the sim built. */
  emitScene(): SceneDescription;
  /** Advance the sim by `steps` and publish the resulting frame. */
  emitFrame(steps?: number): StateUpdate;
  emit(message: ServerMessage): void;
  detach(): void;
}

const INITIAL = {
  status: 'idle' as const,
  statusDetail: '',
  scene: null,
  catalog: [],
  activeScenarioId: null,
  locationPending: null,
  paused: false,
  assistActive: false,
  hasFrames: false,
  perception: null,
  cameraView: 'chase' as const,
  rightTab: 'parameters' as const,
  events: [],
  lastAck: null,
  invalidCount: 0,
  lastInvalid: null,
  commandLog: [],
};

/** Reset the singleton store so tests do not leak into one another. */
export function resetStore(): void {
  useSimStore.setState({
    ...INITIAL,
    sceneEpoch: useSimStore.getState().sceneEpoch,
    layers: Object.fromEntries(LAYER_KEYS.map((k) => [k, true])) as Record<
      (typeof LAYER_KEYS)[number],
      boolean
    >,
    params: Object.fromEntries(PARAM_DEFS.map((d) => [d.key, d.default])),
  });
  frameBus.reset();
}

export function createHarness(scenarioId?: string): Harness {
  const sent: Command[] = [];
  const sim = new MockSim(scenarioId);
  let handlers: TransportHandlers | null = null;

  const transport: Transport = {
    kind: 'mock',
    label: 'test',
    connect(h) {
      handlers = h;
      h.onStatus('open', 'test');
    },
    send(command) {
      sent.push(command);
      // Mirror the real mock: commands actually drive the simulator, so a test
      // that pauses sees `paused: true` on the next frame. `camera_frame` is
      // the one exception — like the real backend (ws_server.py `_handle`)
      // and createMockTransport, it bypasses the command/ack path entirely.
      if (command.cmd === 'camera_frame') return;
      const res = sim.apply(command);
      handlers?.onMessage({
        type: 'ack',
        protocol: 1,
        id: command.id,
        cmd: command.cmd,
        ok: res.ok,
        message: res.message,
        t: sim.t,
      });
      if (res.scene) handlers?.onMessage(res.scene);
    },
    close() {
      handlers = null;
    },
    pendingCount() {
      return 0;
    },
  };

  resetStore();
  const detach = useSimStore.getState().attach(transport);

  return {
    transport,
    sent,
    sim,
    emitScene() {
      const scene = sim.scene;
      act(() => handlers?.onMessage(scene));
      return scene;
    },
    emitFrame(steps = 1) {
      for (let i = 0; i < steps; i++) sim.step();
      const frame = sim.frame();
      act(() => handlers?.onMessage(frame));
      return frame;
    },
    emit(message) {
      act(() => handlers?.onMessage(message));
    },
    detach,
  };
}
