/**
 * The seam between the UI and whatever is producing frames. The in-process mock
 * and the real WebSocket client both implement `Transport`, so nothing above
 * this line knows which one it is talking to.
 */
import type { Command, ServerMessage } from '../schema';

export type ConnectionStatus =
  | 'idle'
  | 'connecting'
  | 'open'
  | 'reconnecting'
  | 'closed'
  | 'error';

/**
 * Why a transport has stopped working in a way the user must hear about.
 * `backend_down` keeps retrying in the background; the other three are verdicts
 * from a server that answered, so retrying on a timer would only repeat them
 * and the transport halts until the user asks (`Transport.retry`).
 */
export type ConnectionFailureKind =
  | 'backend_down'
  | 'origin_rejected'
  | 'server_busy'
  | 'protocol_mismatch';

export interface ConnectionFailure {
  kind: ConnectionFailureKind;
  detail: string;
}

/** WebSocket close codes the backend uses on purpose (server/ws_server.py). */
export const CLOSE_ORIGIN_REJECTED = 1008;
export const CLOSE_SERVER_BUSY = 4429;

export interface TransportHandlers {
  /** A schema-valid message arrived. */
  onMessage(msg: ServerMessage): void;
  /** Connection lifecycle changed. `detail` is a human-readable reason. */
  onStatus(status: ConnectionStatus, detail?: string): void;
  /** The connection is failing for a reason worth showing; `null` clears it. */
  onFailure?(failure: ConnectionFailure | null): void;
  /** A frame arrived but failed validation. Logged, never fatal. */
  onInvalid(error: string, raw: unknown): void;
  /** Wire size of an inbound message, in bytes, before parsing. Optional —
   * feeds the perf overlay only; nothing else depends on it. */
  onRawFrame?(bytes: number): void;
}

export interface Transport {
  readonly kind: 'mock' | 'ws';
  /** Shown in the UI, e.g. "mock" or "ws://localhost:8765". */
  readonly label: string;
  connect(handlers: TransportHandlers): void;
  send(cmd: Command): void;
  close(): void;
  /** Reconnect now, resuming a transport that halted on a failure. */
  retry?(): void;
  /** Commands currently buffered while disconnected. */
  pendingCount(): number;
}
