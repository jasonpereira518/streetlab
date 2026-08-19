import { beforeAll, describe, expect, it } from 'vitest';
import type * as THREE from 'three/webgpu';
import { createDetectorCamera } from '../src/three/detectorCamera';

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
    const detector = createDetectorCamera(scene, renderer);

    await expect(detector.capture()).rejects.toThrow('simulated GPU readback failure');

    // Switched to the detector's own target, then back — never left stuck.
    expect(setCalls.length).toBe(2);
    expect(setCalls[0]).not.toBe(originalTarget);
    expect(setCalls[1]).toBe(originalTarget);
  });
});
