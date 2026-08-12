/**
 * Typed WebSocket transport for the real Python simulator.
 *
 * Every inbound frame is validated against the shared zod schema before it
 * reaches the store, so a backend that drifts from the contract produces a
 * visible warning rather than a corrupted scene. Outbound commands are
 * validated too, and queued while the socket is down.
 */
import { parseCommand, parseServerMessage } from '../schema';
import type { Command } from '../schema';
import type { Transport, TransportHandlers } from './transport';
import { createMockTransport } from './mockServer';

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
      let raw: unknown;
      try {
        raw = JSON.parse(typeof ev.data === 'string' ? ev.data : '');
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
  };
}

/**
 * Pick a transport from the page URL. `?backend=ws://host:port` switches to the
 * real simulator; anything else keeps the in-process mock, so the app runs
 * standalone with no backend present.
 */
export function createTransportFromLocation(
  search: string = typeof window === 'undefined' ? '' : window.location.search,
): Transport {
  const params = new URLSearchParams(search);
  const backend = params.get('backend');
  if (!backend) return createMockTransport();

  if (!/^wss?:\/\//i.test(backend)) {
    console.warn(
      `[streetlab] ignoring ?backend=${backend} — expected a ws:// or wss:// URL`,
    );
    return createMockTransport();
  }
  return createWebSocketTransport({ url: backend });
}
