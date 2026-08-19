/**
 * Typed WebSocket transport for the real Python simulator.
 *
 * Every inbound frame is validated against the shared zod schema before it
 * reaches the store, so a backend that drifts from the contract produces a
 * visible warning rather than a corrupted scene. Outbound commands are
 * validated too, and queued while the socket is down.
 */
import { invoke, isTauri } from '@tauri-apps/api/core';
import { PROTOCOL_VERSION, parseCommand, parseServerMessage } from '../schema';
import type { Command } from '../schema';
import type { Transport, TransportHandlers } from './transport';
import { createMockTransport } from './mockServer';

/** The CLI's own default port — used so `npm run dev` + `streetlab serve`
 * works with no arguments on either side. */
const BROWSER_DEV_DEFAULT_URL = 'ws://127.0.0.1:8765';

interface BackendHandshake {
  ws: string;
  http: string;
  pid: number;
  protocol: number;
}

function isBackendHandshake(v: unknown): v is BackendHandshake {
  return (
    typeof v === 'object' &&
    v !== null &&
    typeof (v as Record<string, unknown>).ws === 'string' &&
    typeof (v as Record<string, unknown>).protocol === 'number'
  );
}

export interface WebSocketTransportOptions {
  url: string;
  /** Reconnect with exponential backoff after an unexpected close. */
  reconnect?: boolean;
  maxBackoffMs?: number;
  /** Commands buffered while disconnected. Older ones are dropped. */
  queueLimit?: number;
}

export function createWebSocketTransport(
  opts: WebSocketTransportOptions,
): Transport {
  const {
    url,
    reconnect = true,
    maxBackoffMs = 8000,
    queueLimit = 32,
  } = opts;

  let handlers: TransportHandlers | null = null;
  let socket: WebSocket | null = null;
  let closedByUs = false;
  let attempt = 0;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  const queue: Command[] = [];

  const flushQueue = () => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    while (queue.length) {
      socket.send(JSON.stringify(queue.shift()));
    }
  };

  const open = () => {
    if (closedByUs) return;
    handlers?.onStatus(attempt === 0 ? 'connecting' : 'reconnecting', url);
    try {
      socket = new WebSocket(url);
    } catch (err) {
      scheduleRetry(String(err));
      return;
    }

    socket.onopen = () => {
      attempt = 0;
      handlers?.onStatus('open', url);
      flushQueue();
    };

    socket.onmessage = (ev) => {
      if (!handlers) return;
      const text = typeof ev.data === 'string' ? ev.data : '';
      handlers.onRawFrame?.(text.length);
      let raw: unknown;
      try {
        raw = JSON.parse(text);
      } catch {
        handlers.onInvalid('message was not valid JSON', ev.data);
        return;
      }
      const res = parseServerMessage(raw);
      if (!res.ok) {
        handlers.onInvalid(res.error, raw);
        return;
      }
      handlers.onMessage(res.value);
    };

    socket.onerror = () => {
      // `onerror` carries no useful detail in browsers; `onclose` follows and
      // carries the code, so the retry decision is made there.
      handlers?.onStatus('error', `socket error on ${url}`);
    };

    socket.onclose = (ev) => {
      socket = null;
      if (closedByUs) {
        handlers?.onStatus('closed', url);
        return;
      }
      scheduleRetry(`closed (${ev.code})`);
    };
  };

  const scheduleRetry = (detail: string) => {
    if (!reconnect || closedByUs) {
      handlers?.onStatus('closed', detail);
      return;
    }
    const delay = Math.min(maxBackoffMs, 400 * Math.pow(2, attempt));
    attempt++;
    handlers?.onStatus('reconnecting', `${detail} — retrying in ${delay} ms`);
    retryTimer = setTimeout(open, delay);
  };

  return {
    kind: 'ws',
    label: url,
    connect(h) {
      handlers = h;
      closedByUs = false;
      attempt = 0;
      open();
    },
    send(command) {
      const res = parseCommand(command);
      if (!res.ok) {
        handlers?.onInvalid(`outbound command rejected: ${res.error}`, command);
        return;
      }
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(res.value));
        return;
      }
      // Camera frames are worthless once stale, and 32 queued frames is ~2 MB
      // of imagery describing a world that has already moved on. Drop them.
      if (res.value.cmd === 'camera_frame') return;
      queue.push(res.value);
      while (queue.length > queueLimit) queue.shift();
    },
    close() {
      closedByUs = true;
      if (retryTimer) clearTimeout(retryTimer);
      socket?.close();
      socket = null;
      handlers?.onStatus('closed');
      handlers = null;
    },
    pendingCount() {
      return queue.length;
    },
  };
}

/**
 * Pick a transport from the page URL, with the sidecar handshake as the
 * default inside the packaged app. Precedence:
 *
 *   ?mock=1            in-process mock, explicit — offline dev, and what the
 *                       test suite and Playwright specs use.
 *   ?backend=ws://…     explicit override, for dev against a hand-started
 *                       server.
 *   Tauri IPC present   `backend_url()` from the sidecar handshake.
 *   otherwise           the CLI's own default (`ws://127.0.0.1:8765`), so
 *                       `npm run dev` + `streetlab serve` works with no
 *                       arguments on either side.
 *
 * Throws when Tauri IPC is present but the handshake fails or reports a
 * protocol this build doesn't speak — the caller turns that into a startup
 * error state with a one-click mock fallback, rather than a confusing stream
 * of connection failures.
 */
export async function createTransportFromLocation(
  search: string = typeof window === 'undefined' ? '' : window.location.search,
): Promise<Transport> {
  const params = new URLSearchParams(search);

  if (params.get('mock') === '1') return createMockTransport();

  if (params.has('backend')) {
    const backend = params.get('backend') ?? '';
    if (!/^wss?:\/\//i.test(backend)) {
      console.warn(
        `[streetlab] ignoring ?backend=${backend} — expected a ws:// or wss:// URL`,
      );
      return createMockTransport();
    }
    return createWebSocketTransport({ url: backend });
  }

  if (isTauri()) {
    let handshake: unknown;
    try {
      handshake = await invoke('backend_url');
    } catch (err) {
      throw new Error(`simulator did not start: ${String(err)}`);
    }
    if (!isBackendHandshake(handshake)) {
      throw new Error('simulator returned a malformed handshake');
    }
    if (handshake.protocol !== PROTOCOL_VERSION) {
      throw new Error(
        `backend speaks protocol ${handshake.protocol}, this build expects ${PROTOCOL_VERSION}`,
      );
    }
    return createWebSocketTransport({ url: handshake.ws });
  }

  return createWebSocketTransport({ url: BROWSER_DEV_DEFAULT_URL });
}
