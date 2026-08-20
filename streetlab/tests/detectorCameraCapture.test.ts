import { beforeAll, describe, expect, it, vi } from 'vitest';
import type * as THREE from 'three/webgpu';
import { createDetectorCamera, DETECTOR_FRAME } from '../src/three/detectorCamera';
import type { DetectorCamera } from '../src/three/detectorCamera';

// `createDetectorCamera` unconditionally constructs an OffscreenCanvas. Neither
// vitest's default `node` environment nor `jsdom` implement it, and the render
// target restore path under test here never reaches the canvas at all (the stub
// renderer below rejects before capture() gets that far) — so a minimal
// stand-in that only needs to exist and return a truthy context is enough. It
// never rasterizes anything.
beforeAll(() => {
  if (typeof (globalThis as { OffscreenCanvas?: unknown }).OffscreenCanvas === 'undefined') {
    (globalThis as { OffscreenCanvas?: unknown }).OffscreenCanvas = class {
      constructor(_width: number, _height: number) {}
      getContext() {
        return {};
      }
    };
  }
});

describe('capture() render target restore', () => {
  it('restores the previous render target even when readback rejects', async () => {
    // The renderer is shared with the main view. If capture() ever leaves it
    // pointed at the detector's offscreen target after a failure, the next
    // main-view render draws into the wrong place instead of the visible canvas.
    const originalTarget = { name: 'main-view-target' };
    const setCalls: unknown[] = [];

    const renderer = {
      getRenderTarget: () => originalTarget,
      setRenderTarget: (t: unknown) => {
        setCalls.push(t);
      },
      renderAsync: async () => {},
      readRenderTargetPixelsAsync: async () => {
        throw new Error('simulated GPU readback failure');
      },
    } as unknown as THREE.WebGPURenderer;

    const scene = {} as THREE.Scene;
    const detector = createDetectorCamera(scene, renderer, 'webgpu');

    await expect(detector.capture()).rejects.toThrow('simulated GPU readback failure');

    // Switched to the detector's own target, then back — never left stuck.
    expect(setCalls.length).toBe(2);
    expect(setCalls[0]).not.toBe(originalTarget);
    expect(setCalls[1]).toBe(originalTarget);
  });

  it('does not permanently wedge capture() when getRenderTarget itself throws', async () => {
    // getRenderTarget is called outside of anything that would have produced a
    // value to restore. If that failure leaves `busy` stuck true, every later
    // call is silently dead for the rest of the session — frame capture just
    // stops, with nothing to see in the logs.
    let getRenderTargetCalls = 0;
    const originalTarget = { name: 'main-view-target' };

    const renderer = {
      getRenderTarget: () => {
        getRenderTargetCalls += 1;
        if (getRenderTargetCalls === 1) {
          throw new Error('simulated getRenderTarget failure');
        }
        return originalTarget;
      },
      setRenderTarget: () => {},
      renderAsync: async () => {},
      readRenderTargetPixelsAsync: async () => {
        throw new Error('simulated GPU readback failure');
      },
    } as unknown as THREE.WebGPURenderer;

    const scene = {} as THREE.Scene;
    const detector = createDetectorCamera(scene, renderer, 'webgpu');

    // First call: getRenderTarget throws before anything is captured to restore.
    await expect(detector.capture()).rejects.toThrow('simulated getRenderTarget failure');

    // The property that matters is not the first call's failure — it's that a
    // SECOND call isn't permanently blocked by a stuck `busy` guard. A stuck
    // guard makes every later call resolve to `null` immediately without
    // touching the renderer at all. Reaching (and failing at) readback instead
    // proves `busy` was released.
    await expect(detector.capture()).rejects.toThrow('simulated GPU readback failure');
  });
});

describe('capture() renderTargetBusy timing', () => {
  // These two tests exist because of a real bug: the render loop (Renderer.tsx)
  // calls `renderer.render(scene, cam.camera)` once per display frame, and it
  // shares a renderer with this offscreen capture. `renderTargetBusy()` is what
  // that loop gates the main render call on, so its timing IS the correctness
  // property — if it reads false while the target is still switched, or true
  // after it's been restored, the main view can silently render into the
  // detector's offscreen buffer (or the readback can silently capture the
  // user's view). Neither of the existing tests above exercises the flag at
  // all. What this does NOT (and cannot, without a live WebGPU context) prove
  // is that Renderer.tsx's loop actually checks the flag every tick before
  // calling render() — that half of the guarantee is a single `if` at the call
  // site, verified by reading the code, not by this test.
  it('is true only while the render target is actually switched away from the main view', async () => {
    const originalTarget = { name: 'main-view-target' };
    let detector!: DetectorCamera;
    let busyDuringRenderPass = false;

    const renderer = {
      getRenderTarget: () => originalTarget,
      setRenderTarget: () => {},
      renderAsync: async () => {
        // capture() switches the target synchronously before its first
        // `await`, so by the time this runs, the flag the render loop reads
        // must already be true.
        busyDuringRenderPass = detector.renderTargetBusy();
      },
      readRenderTargetPixelsAsync: async () => {
        throw new Error('simulated GPU readback failure');
      },
    } as unknown as THREE.WebGPURenderer;

    const scene = {} as THREE.Scene;
    detector = createDetectorCamera(scene, renderer, 'webgpu');

    expect(detector.renderTargetBusy()).toBe(false);
    await expect(detector.capture()).rejects.toThrow('simulated GPU readback failure');

    expect(busyDuringRenderPass).toBe(true);
    // Released even though capture() went on to fail at readback — a stuck
    // `true` here would wedge the main render loop into skipping every frame
    // for the rest of the session, not just the failed capture.
    expect(detector.renderTargetBusy()).toBe(false);
  });

  it('is already false once GPU readback starts, not just once capture() finishes', async () => {
    // This is the actual fix under test: restoring the render target right
    // after the render pass, instead of in `finally` after readback and JPEG
    // encoding, shrinks the window the render loop must treat as unsafe down
    // to just the render pass. Before that change, this would read `true` —
    // proving the render loop's gate would otherwise stay closed for the
    // entire GPU round trip, not just the render.
    const originalTarget = { name: 'main-view-target' };
    let detector!: DetectorCamera;
    const busyAtReadbackStart: boolean[] = [];

    const renderer = {
      getRenderTarget: () => originalTarget,
      setRenderTarget: () => {},
      renderAsync: async () => {},
      readRenderTargetPixelsAsync: async () => {
        busyAtReadbackStart.push(detector.renderTargetBusy());
        throw new Error('simulated GPU readback failure');
      },
    } as unknown as THREE.WebGPURenderer;

    const scene = {} as THREE.Scene;
    detector = createDetectorCamera(scene, renderer, 'webgpu');

    await expect(detector.capture()).rejects.toThrow('simulated GPU readback failure');

    expect(busyAtReadbackStart).toEqual([false]);
  });

  it('releases both guards even when the restore itself throws (e.g. a lost GPU device)', async () => {
    // The double-fault this guards against: the early restore throws, so
    // `restoredEarly` never gets set, and `finally`'s fallback retries the
    // identical setRenderTarget(previous) call — which, on a genuinely lost
    // device, throws again. If that second throw were allowed to escape
    // `finally` before the resets below it ran, `targetBusy` would stay
    // `true` forever, and since the render loop gates the visible canvas on
    // that flag, the canvas would freeze for the rest of the session — a far
    // worse outcome than the one frame that might draw into the wrong target
    // while the GPU is already failing.
    const originalTarget = { name: 'main-view-target' };
    let setRenderTargetCalls = 0;
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const renderer = {
      getRenderTarget: () => originalTarget,
      setRenderTarget: () => {
        setRenderTargetCalls += 1;
        // The first call switches to the detector's own target and succeeds.
        // Every call after that is a restore attempt (early, then finally's
        // fallback if reached) — all fail, simulating a lost device.
        if (setRenderTargetCalls > 1) {
          throw new Error('simulated device-lost on restore');
        }
      },
      renderAsync: async () => {},
      // Distinct from the restore failure on purpose: if some future change
      // ever let capture() reach readback despite the restore throwing, this
      // would surface as an assertion failure below (wrong error message)
      // rather than an accidentally-passing test.
      readRenderTargetPixelsAsync: async () => {
        throw new Error('should not be reached — restore fails first');
      },
    } as unknown as THREE.WebGPURenderer;

    const scene = {} as THREE.Scene;
    const detector = createDetectorCamera(scene, renderer, 'webgpu');

    await expect(detector.capture()).rejects.toThrow('simulated device-lost on restore');

    // The property that matters: neither guard is left stuck true.
    expect(detector.renderTargetBusy()).toBe(false);
    expect(warnSpy).toHaveBeenCalled();

    // `busy` released too — a second call must reach the renderer again
    // (and fail the same way) rather than short-circuiting to `null`, which
    // is what a stuck `busy` guard would do instead.
    setRenderTargetCalls = 0;
    await expect(detector.capture()).rejects.toThrow('simulated device-lost on restore');
    expect(detector.renderTargetBusy()).toBe(false);

    warnSpy.mockRestore();
  });
});

describe('capture() readback timeout', () => {
  it('releases its guards when the readback never settles', async () => {
    // The observed failure, not a rejection: renderAsync and the render
    // target restore both succeed normally, but the GPU readback itself
    // never calls back. Without a timeout, `busy` would stay stuck forever
    // and perception would silently die for the rest of the session.
    const originalTarget = { name: 'main-view-target' };
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const renderer = {
      getRenderTarget: () => originalTarget,
      setRenderTarget: () => {},
      renderAsync: async () => {},
      readRenderTargetPixelsAsync: () => new Promise<never>(() => {}),
    } as unknown as THREE.WebGPURenderer;

    const scene = {} as THREE.Scene;
    const detector = createDetectorCamera(scene, renderer, 'webgpu');

    vi.useFakeTimers();
    try {
      const result = detector.capture();
      await vi.advanceTimersByTimeAsync(DETECTOR_FRAME.captureTimeoutMs);
      await expect(result).resolves.toBeNull();
      expect(detector.renderTargetBusy()).toBe(false);

      // The guard must be free for the next tick, or perception is dead for
      // good. The stub's readback still never settles, so this also times
      // out — the point is that it is reached at all, rather than being
      // short-circuited by a `busy` guard the first call never released.
      const second = detector.capture();
      await vi.advanceTimersByTimeAsync(DETECTOR_FRAME.captureTimeoutMs);
      await expect(second).resolves.toBeDefined();
      expect(detector.renderTargetBusy()).toBe(false);

      // One warning for the whole session, not one per timeout.
      expect(warnSpy).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
      warnSpy.mockRestore();
    }
  });

  it('does not let a late-settling readback corrupt a later capture', async () => {
    // The hard part of this fix: the abandoned readback promise from a timed
    // out capture can still resolve afterwards. If that resolution reached
    // into shared state, it could restore a stale render target, release a
    // guard a newer capture now owns, or otherwise disturb work already in
    // progress. This proves none of that happens: the first capture's
    // readback resolves *after* a second capture has already run to
    // completion, and the second capture's outcome is unaffected.
    //
    // The second capture is made to fail at readback too (rather than
    // succeed) because this suite's OffscreenCanvas stand-in (see the
    // `beforeAll` above) has no real 2D context — only the timeout and
    // rejection paths are exercisable here, not a full successful encode.
    const originalTarget = { name: 'main-view-target' };
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const setRenderTargetCalls: unknown[] = [];
    let releaseFirstReadback: (pixels: Uint8Array) => void = () => {
      throw new Error('readRenderTargetPixelsAsync was never called for the first capture');
    };
    let readbackCallCount = 0;

    const renderer = {
      getRenderTarget: () => originalTarget,
      setRenderTarget: (t: unknown) => {
        setRenderTargetCalls.push(t);
      },
      renderAsync: async () => {},
      readRenderTargetPixelsAsync: () => {
        readbackCallCount += 1;
        if (readbackCallCount === 1) {
          // Never resolves within this test's timeout window, but is kept
          // alive so it can be resolved *after* the second capture finishes.
          return new Promise<Uint8Array>((resolve) => {
            releaseFirstReadback = resolve;
          });
        }
        return Promise.reject(new Error('simulated GPU readback failure for second capture'));
      },
    } as unknown as THREE.WebGPURenderer;

    const scene = {} as THREE.Scene;
    const detector = createDetectorCamera(scene, renderer, 'webgpu');

    vi.useFakeTimers();
    try {
      const first = detector.capture();
      await vi.advanceTimersByTimeAsync(DETECTOR_FRAME.captureTimeoutMs);
      await expect(first).resolves.toBeNull();

      const second = detector.capture();
      await expect(second).rejects.toThrow('simulated GPU readback failure for second capture');
      expect(detector.renderTargetBusy()).toBe(false);
      const setRenderTargetCallsAfterSecond = setRenderTargetCalls.length;

      // Now let the first capture's long-abandoned readback finally settle —
      // well after the second capture has already run to completion.
      expect(readbackCallCount).toBeGreaterThanOrEqual(1);
      releaseFirstReadback(new Uint8Array(4));
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();

      // Nothing changed as a result: no extra render-target switches beyond
      // what the second capture itself already made, and the guard is still
      // free for a third capture.
      expect(setRenderTargetCalls.length).toBe(setRenderTargetCallsAfterSecond);
      expect(detector.renderTargetBusy()).toBe(false);
    } finally {
      vi.useRealTimers();
      warnSpy.mockRestore();
    }
  });
});
