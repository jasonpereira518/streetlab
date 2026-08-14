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

export interface TransportHandlers {
  /** A schema-valid message arrived. */
  onMessage(msg: ServerMessage): void;
  /** Connection lifecycle changed. `detail` is a human-readable reason. */
  onStatus(status: ConnectionStatus, detail?: string): void;
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
}
