// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { clearMocks, mockIPC } from '@tauri-apps/api/mocks';
import {
  createTransportFromLocation,
  createWebSocketTransport,
} from '../src/net/wsClient';
import type { ConnectionStatus } from '../src/net/transport';
import { MockSim } from '../src/net/mockServer';
import { PROTOCOL_VERSION } from '../src/schema';
import type { ServerMessage } from '../src/schema';

/* ------------------------------------------------------------------ */
/* A WebSocket stand-in the test drives by hand                        */
/* ------------------------------------------------------------------ */

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readyState = FakeWebSocket.CONNECTING;
  sent: string[] = [];

  onopen: (() => void) | null = null;
  onclose: ((e: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = FakeWebSocket.CLOSED;
  }

  /* -- test controls -- */
  open(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }
  receive(value: unknown): void {
    this.onmessage?.({
      data: typeof value === 'string' ? value : JSON.stringify(value),
    });
  }
  serverClose(code = 1006): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code });
  }
}

const original = globalThis.WebSocket;

beforeEach(() => {
  FakeWebSocket.instances = [];
  globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket;
});

afterEach(() => {
  globalThis.WebSocket = original;
});

function collector() {
  const messages: ServerMessage[] = [];
  const invalid: string[] = [];
  const statuses: ConnectionStatus[] = [];
  return {
    messages,
    invalid,
    statuses,
    handlers: {
      onMessage: (m: ServerMessage) => messages.push(m),
      onInvalid: (e: string) => invalid.push(e),
      onStatus: (s: ConnectionStatus) => statuses.push(s),
    },
  };
}

/* ------------------------------------------------------------------ */

describe('transport selection', () => {
  afterEach(() => {
    clearMocks();
    delete (globalThis as { isTauri?: boolean }).isTauri;
  });

  it('uses the mock for ?mock=1, explicitly', async () => {
    expect((await createTransportFromLocation('?mock=1')).kind).toBe('mock');
  });

  it('switches to a WebSocket for ?backend=ws://…', async () => {
    const t = await createTransportFromLocation('?backend=ws://localhost:8765');
    expect(t.kind).toBe('ws');
    expect(t.label).toBe('ws://localhost:8765');
  });

  it('accepts wss:// too', async () => {
    expect((await createTransportFromLocation('?backend=wss://sim.example/ws')).kind).toBe(
      'ws',
    );
  });

  it('falls back to the mock for a non-WebSocket URL', async () => {
    expect(
      (await createTransportFromLocation('?backend=http://localhost:8765')).kind,
    ).toBe('mock');
    expect((await createTransportFromLocation('?backend=')).kind).toBe('mock');
  });

  it('defaults to the browser-dev backend when no param and no Tauri IPC', async () => {
    const t = await createTransportFromLocation('');
    expect(t.kind).toBe('ws');
    expect(t.label).toBe('ws://127.0.0.1:8765');

    const t2 = await createTransportFromLocation('?scenario=x');
    expect(t2.kind).toBe('ws');
    expect(t2.label).toBe('ws://127.0.0.1:8765');
  });

  it('?mock=1 wins even when Tauri IPC is present', async () => {
    (globalThis as { isTauri?: boolean }).isTauri = true;
    mockIPC(() => ({
      ws: 'ws://127.0.0.1:54321',
      http: 'http://127.0.0.1:54321',
      pid: 1,
      protocol: PROTOCOL_VERSION,
    }));
    expect((await createTransportFromLocation('?mock=1')).kind).toBe('mock');
  });

  it('uses backend_url() from the sidecar handshake when Tauri IPC is present', async () => {
    (globalThis as { isTauri?: boolean }).isTauri = true;
    mockIPC((cmd) => {
      if (cmd === 'backend_url') {
        return {
          ws: 'ws://127.0.0.1:54321',
          http: 'http://127.0.0.1:54321',
          pid: 4242,
          protocol: PROTOCOL_VERSION,
        };
      }
      throw new Error(`unexpected command: ${cmd}`);
    });

    const t = await createTransportFromLocation('');
    expect(t.kind).toBe('ws');
    expect(t.label).toBe('ws://127.0.0.1:54321');
  });

  it('rejects with a clear message when the sidecar handshake fails', async () => {
    (globalThis as { isTauri?: boolean }).isTauri = true;
    mockIPC(() => {
      throw new Error('sidecar exited before READY');
    });

    await expect(createTransportFromLocation('')).rejects.toThrow(
      /simulator did not start/,
    );
  });

  it('rejects with a clear message on a protocol mismatch', async () => {
    (globalThis as { isTauri?: boolean }).isTauri = true;
    mockIPC(() => ({
      ws: 'ws://127.0.0.1:54321',
      http: 'http://127.0.0.1:54321',
      pid: 1,
      protocol: PROTOCOL_VERSION + 1,
    }));

    await expect(createTransportFromLocation('')).rejects.toThrow(/protocol/);
  });
});

describe('WebSocket transport', () => {
  it('validates inbound frames and forwards only good ones', () => {
    const t = createWebSocketTransport({ url: 'ws://x/1', reconnect: false });
    const c = collector();
    t.connect(c.handlers);
    const socket = FakeWebSocket.instances[0];
    socket.open();

    const sim = new MockSim();
    sim.step();
    socket.receive(sim.scene);
    socket.receive(sim.frame());

    expect(c.messages.map((m) => m.type)).toEqual([
      'scene_description',
      'state_update',
    ]);
    expect(c.invalid).toEqual([]);
    expect(c.statuses).toContain('open');
  });

  it('reports malformed frames without dropping the connection', () => {
    const t = createWebSocketTransport({ url: 'ws://x/2', reconnect: false });
    const c = collector();
    t.connect(c.handlers);
    const socket = FakeWebSocket.instances[0];
    socket.open();

    socket.receive('not json at all');
    socket.receive({ type: 'state_update', seq: 'nope' });
    socket.receive({ type: 'who_knows' });

    expect(c.invalid).toHaveLength(3);
    expect(c.messages).toHaveLength(0);

    // Still usable afterwards.
    const sim = new MockSim();
    socket.receive(sim.scene);
    expect(c.messages).toHaveLength(1);
  });

  it('validates outbound commands and queues them until the socket opens', () => {
    const t = createWebSocketTransport({ url: 'ws://x/3', reconnect: false });
    const c = collector();
    t.connect(c.handlers);
    const socket = FakeWebSocket.instances[0];

    t.send({ id: 'a', cmd: 'set_paused', paused: true });
    expect(socket.sent).toHaveLength(0);

    socket.open();
    expect(socket.sent).toHaveLength(1);
    expect(JSON.parse(socket.sent[0])).toMatchObject({ cmd: 'set_paused' });

    t.send({ id: 'b', cmd: 'reset' });
    expect(socket.sent).toHaveLength(2);

    // A command that fails validation never reaches the wire.
    t.send({ id: 'c', cmd: 'toggle_layer', layer: 'bogus', visible: true } as never);
    expect(socket.sent).toHaveLength(2);
    expect(c.invalid.at(-1)).toContain('outbound command rejected');
  });

  it('reconnects with backoff after an unexpected close', async () => {
    const t = createWebSocketTransport({ url: 'ws://x/4', maxBackoffMs: 10 });
    const c = collector();
    t.connect(c.handlers);
    FakeWebSocket.instances[0].open();
    FakeWebSocket.instances[0].serverClose();

    expect(c.statuses).toContain('reconnecting');
    await new Promise((r) => setTimeout(r, 30));
    expect(FakeWebSocket.instances.length).toBeGreaterThan(1);
    t.close();
  });

  it('does not reconnect after the client closes it', async () => {
    const t = createWebSocketTransport({ url: 'ws://x/5' });
    const c = collector();
    t.connect(c.handlers);
    FakeWebSocket.instances[0].open();
    t.close();
    await new Promise((r) => setTimeout(r, 30));
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(c.statuses.at(-1)).toBe('closed');
  });

  it('round-trips an ack from the server', () => {
    const t = createWebSocketTransport({ url: 'ws://x/6', reconnect: false });
    const c = collector();
    t.connect(c.handlers);
    const socket = FakeWebSocket.instances[0];
    socket.open();
    socket.receive({
      type: 'ack',
      protocol: PROTOCOL_VERSION,
      id: 'a',
      cmd: 'set_paused',
      ok: true,
      message: 'paused',
      t: 1.5,
    });
    expect(c.messages[0]).toMatchObject({ type: 'ack', id: 'a', ok: true });
  });
});
