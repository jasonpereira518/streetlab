/**
 * Vitest global setup.
 *
 * Node tests need a `performance` shim. jsdom tests additionally need three
 * things jsdom does not implement: a 2D canvas backend, `ResizeObserver`, and
 * elements with a non-zero layout box. The canvas stub records every call so a
 * test can assert what a widget actually drew, and animation frames are put
 * under manual control via `flushFrames` so nothing depends on wall clock.
 */
import { vi } from 'vitest';

if (typeof globalThis.performance === 'undefined') {
  // @ts-expect-error - minimal shim
  globalThis.performance = { now: () => Date.now() };
}

export interface RecordedCall {
  method: string;
  args: unknown[];
}

const contexts = new WeakMap<HTMLCanvasElement, RecordedContext>();

type RecordedContext = CanvasRenderingContext2D & { calls: RecordedCall[] };

/** Every drawing call made on a canvas so far, in order. */
export function canvasCalls(canvas: HTMLCanvasElement): RecordedCall[] {
  return contexts.get(canvas)?.calls ?? [];
}

/** Concatenated text drawn into a canvas, for readable assertions. */
export function canvasText(canvas: HTMLCanvasElement): string {
  return canvasCalls(canvas)
    .filter((c) => c.method === 'fillText' || c.method === 'strokeText')
    .map((c) => String(c.args[0]))
    .join(' ');
}

export function clearCanvasCalls(canvas: HTMLCanvasElement): void {
  const ctx = contexts.get(canvas);
  if (ctx) ctx.calls.length = 0;
}

/* ------------------------------------------------------------------ */
/* Manual animation frames                                             */
/* ------------------------------------------------------------------ */

type FrameCallback = (t: number) => void;
const frameQueue = new Map<number, FrameCallback>();
let frameId = 0;
let clock = 0;

/** Run `count` animation frames, advancing the clock by `stepMs` each time. */
export function flushFrames(count = 1, stepMs = 16.7): void {
  for (let i = 0; i < count; i++) {
    clock += stepMs;
    const due = [...frameQueue.entries()];
    frameQueue.clear();
    for (const [, cb] of due) cb(clock);
  }
}

if (typeof globalThis.document !== 'undefined') {
  globalThis.requestAnimationFrame = ((cb: FrameCallback) => {
    const id = ++frameId;
    frameQueue.set(id, cb);
    return id;
  }) as typeof requestAnimationFrame;

  globalThis.cancelAnimationFrame = ((id: number) => {
    frameQueue.delete(id);
  }) as typeof cancelAnimationFrame;

  globalThis.performance.now = () => clock;

  /* ---- ResizeObserver ---- */
  class StubResizeObserver {
    constructor(private readonly cb: ResizeObserverCallback) {}
    observe(): void {
      // Fire once so consumers pick up the stubbed layout box immediately.
      this.cb([], this as unknown as ResizeObserver);
    }
    unobserve(): void {}
    disconnect(): void {}
  }
  globalThis.ResizeObserver =
    StubResizeObserver as unknown as typeof ResizeObserver;

  /* ---- layout boxes ---- */
  const BOX = { width: 260, height: 132 };
  Element.prototype.getBoundingClientRect = function getBoundingClientRect() {
    return {
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: BOX.width,
      bottom: BOX.height,
      width: BOX.width,
      height: BOX.height,
      toJSON: () => ({}),
    } as DOMRect;
  };

  /* ---- canvas 2D ---- */
  const makeContext = (canvas: HTMLCanvasElement): RecordedContext => {
    const calls: RecordedCall[] = [];
    const state: Record<string, unknown> = {
      canvas,
      calls,
      font: '',
      fillStyle: '#000',
      strokeStyle: '#000',
      lineWidth: 1,
      globalAlpha: 1,
    };
    const gradient = { addColorStop: () => {} };
    const fns: Record<string, (...a: unknown[]) => unknown> = {
      measureText: (t: unknown) => ({ width: String(t).length * 6 }),
      createLinearGradient: () => gradient,
      createRadialGradient: () => gradient,
      getImageData: () => ({ data: new Uint8ClampedArray(4) }),
    };

    return new Proxy(state, {
      get(target, prop: string) {
        if (prop in target) return target[prop];
        return (...args: unknown[]) => {
          calls.push({ method: prop, args });
          return fns[prop]?.(...args);
        };
      },
      set(target, prop: string, value) {
        target[prop] = value;
        return true;
      },
    }) as unknown as RecordedContext;
  };

  HTMLCanvasElement.prototype.getContext = vi.fn(function (
    this: HTMLCanvasElement,
    kind: string,
  ) {
    if (kind !== '2d') return null;
    let ctx = contexts.get(this);
    if (!ctx) {
      ctx = makeContext(this);
      contexts.set(this, ctx);
    }
    return ctx;
  }) as unknown as HTMLCanvasElement['getContext'];
}
