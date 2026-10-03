/**
 * React bindings for the 60 Hz frame stream.
 *
 * The rule this file exists to enforce: nothing re-renders at frame rate.
 * Text readouts poll a throttled value; canvas widgets never re-render at all
 * and instead redraw themselves from a single shared animation loop.
 */
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { StateUpdate } from '../schema';
import { frameBus } from './simStore';

/* ------------------------------------------------------------------ */
/* One animation loop for the whole DOM layer                          */
/* ------------------------------------------------------------------ */

type Tick = (dt: number, now: number) => void;

const ticks = new Set<Tick>();
let rafId = 0;
let lastNow = 0;

function pump(now: number): void {
  const dt = lastNow ? Math.min(0.1, (now - lastNow) / 1000) : 1 / 60;
  lastNow = now;
  for (const t of ticks) t(dt, now);
  rafId = ticks.size ? requestAnimationFrame(pump) : 0;
}

function addTick(t: Tick): () => void {
  ticks.add(t);
  if (!rafId) {
    lastNow = 0;
    rafId = requestAnimationFrame(pump);
  }
  return () => {
    ticks.delete(t);
    if (!ticks.size && rafId) {
      cancelAnimationFrame(rafId);
      rafId = 0;
    }
  };
}

/** Register a callback on the shared loop for the lifetime of a component. */
export function useAnimationFrame(cb: Tick, enabled = true): void {
  const ref = useRef(cb);
  useLayoutEffect(() => {
    ref.current = cb;
  });
  useEffect(() => {
    if (!enabled) return;
    return addTick((dt, now) => ref.current(dt, now));
  }, [enabled]);
}

/* ------------------------------------------------------------------ */
/* Throttled frame reads for DOM text                                  */
/* ------------------------------------------------------------------ */

/**
 * Poll a value out of the frame stream at `hz`, re-rendering only when the
 * selected value actually changes. Use for numbers rendered as text; use
 * `useFrameRef` plus a canvas for anything that should move every frame.
 */
export function useFrameValue<T>(
  select: (frame: StateUpdate) => T,
  hz = 10,
  equals: (a: T, b: T) => boolean = Object.is,
): T | null {
  const [value, setValue] = useState<T | null>(() =>
    frameBus.latest ? select(frameBus.latest) : null,
  );
  const selectRef = useRef(select);
  const equalsRef = useRef(equals);
  const valueRef = useRef(value);
  useLayoutEffect(() => {
    selectRef.current = select;
    equalsRef.current = equals;
    valueRef.current = value;
  });

  useEffect(() => {
    const interval = 1000 / hz;
    // -Infinity, not 0: the first frame after subscribing must be published
    // immediately rather than waiting out one throttle window.
    let last = -Infinity;
    return frameBus.subscribe((frame) => {
      const now = performance.now();
      if (now - last < interval) return;
      last = now;
      const next = selectRef.current(frame);
      const prev = valueRef.current;
      if (prev !== null && equalsRef.current(prev as T, next)) return;
      valueRef.current = next;
      setValue(next);
    });
  }, [hz]);

  return value;
}

/** Live handle on the newest frame, for imperative draw code. */
export function useFrameRef(): { current: StateUpdate | null } {
  const ref = useRef<StateUpdate | null>(frameBus.latest);
  useEffect(() => {
    ref.current = frameBus.latest;
    return frameBus.subscribe((f) => {
      ref.current = f;
    });
  }, []);
  return ref;
}

/* ------------------------------------------------------------------ */
/* Canvas widgets                                                      */
/* ------------------------------------------------------------------ */

export interface CanvasDrawArgs {
  ctx: CanvasRenderingContext2D;
  /** CSS pixels — the context is already scaled by devicePixelRatio. */
  width: number;
  height: number;
  frame: StateUpdate | null;
  /** Seconds since the widget mounted; for idle animation. */
  time: number;
  dt: number;
}

/**
 * Mount a device-pixel-ratio-correct 2D canvas that redraws every animation
 * frame from the newest simulator frame. The component itself never re-renders.
 */
export function useTelemetryCanvas(
  draw: (args: CanvasDrawArgs) => void,
): React.RefObject<HTMLCanvasElement> {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  const sizeRef = useRef({ w: 0, h: 0, dpr: 1 });
  const timeRef = useRef(0);
  const frameRef = useFrameRef();

  useLayoutEffect(() => {
    drawRef.current = draw;
  });

  // Keep the backing store in step with the element's layout box.
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement ?? canvas;

    const resize = () => {
      // The content box, not the border box: `inset: 0` lays the canvas out
      // inside its frame's border, so measuring with getBoundingClientRect
      // oversizes the backing store by the border and the overflow clip eats
      // the last pixel of the drawing. jsdom reports 0 here, hence the rect
      // fallback — that path is what the unit tests measure through.
      const rect = parent.getBoundingClientRect();
      const dpr = Math.min(3, window.devicePixelRatio || 1);
      const w = Math.max(1, Math.round(parent.clientWidth || rect.width));
      const h = Math.max(1, Math.round(parent.clientHeight || rect.height));
      if (
        sizeRef.current.w === w &&
        sizeRef.current.h === h &&
        sizeRef.current.dpr === dpr
      ) {
        return;
      }
      sizeRef.current = { w, h, dpr };
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(parent);
    return () => ro.disconnect();
  }, []);

  useAnimationFrame((dt) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { w, h, dpr } = sizeRef.current;
    if (w === 0 || h === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    timeRef.current += dt;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    drawRef.current({
      ctx,
      width: w,
      height: h,
      frame: frameRef.current,
      time: timeRef.current,
      dt,
    });
  });

  return canvasRef;
}
