/**
 * The camera perception sees.
 *
 * Deliberately NOT the camera the user sees: they must be able to orbit to
 * overhead or free view without changing what the detector is looking at.
 * Rigidly mounted at the ego windshield, same mount as the `cockpit` view.
 *
 * Rendered to an offscreen target at a fixed size and a fixed rate, so the
 * frames the backend scores are independent of display resolution and FPS.
 */

import * as THREE from 'three/webgpu';
import type { CameraParams } from '../schema';

export const DETECTOR_FRAME = {
  width: 640,
  height: 384,
  fovYDeg: 50,
  /** ~10 Hz. Independent of render FPS. */
  intervalMs: 100,
  /** JPEG quality: the wire cost is roughly linear in this. */
  quality: 0.6,
} as const;

/** Mount height and forward offset, matching the cockpit view. */
const MOUNT_HEIGHT = 1.33;
const MOUNT_FORWARD = 0.15;

/**
 * Three.js is Y-up with `+x` east and `+z` south. The wire is `+x` east,
 * `+y` north, `+z` up. Converting here means the backend never learns that a
 * renderer convention exists.
 */
export function cameraParamsFromThree(
  position: { x: number; y: number; z: number },
  headingRad: number,
): CameraParams {
  return {
    x: position.x,
    y: -position.z,
    z: position.y,
    yaw: headingRad,
    pitch: 0,
    roll: 0,
    fov_y_deg: DETECTOR_FRAME.fovYDeg,
    aspect: DETECTOR_FRAME.width / DETECTOR_FRAME.height,
  };
}

/**
 * GPU readback returns rows bottom-up; images are top-down. Without this the
 * detector sees an upside-down world and every projection is wrong.
 */
export function flipRowsInPlace(rgba: Uint8Array, width: number, height: number): void {
  const stride = width * 4;
  const row = new Uint8Array(stride);
  for (let y = 0; y < Math.floor(height / 2); y++) {
    const top = y * stride;
    const bottom = (height - 1 - y) * stride;
    row.set(rgba.subarray(top, top + stride));
    rgba.copyWithin(top, bottom, bottom + stride);
    rgba.set(row, bottom);
  }
}

/** btoa in chunks: spreading 60 KB into String.fromCharCode blows the stack. */
export function encodeBase64(bytes: Uint8Array): string {
  const CHUNK = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

export interface DetectorCamera {
  update(pose: { x: number; z: number; heading: number }): void;
  capture(): Promise<{ data: string; camera: CameraParams } | null>;
  dispose(): void;
}

export function createDetectorCamera(
  scene: THREE.Scene,
  renderer: THREE.WebGPURenderer,
): DetectorCamera {
  const { width, height, fovYDeg, quality } = DETECTOR_FRAME;
  const camera = new THREE.PerspectiveCamera(fovYDeg, width / height, 0.1, 400);
  // Defaults to UnsignedByteType. `capture()` reinterprets the readback's raw
  // bytes as a Uint8Array directly (no per-channel conversion) — switching this
  // to FloatType/HalfFloatType would make that reinterpretation silent garbage.
  const target = new THREE.RenderTarget(width, height);
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  let busy = false;
  // Remembered from `update` rather than re-derived from the camera's matrix in
  // `capture`: the heading is known exactly here, and reading it back out of
  // matrixWorld columns is sign-error bait for no benefit.
  let heading = 0;

  return {
    update(pose) {
      heading = pose.heading;
      const fx = Math.cos(pose.heading);
      const fz = -Math.sin(pose.heading);
      camera.position.set(
        pose.x + fx * MOUNT_FORWARD,
        MOUNT_HEIGHT,
        pose.z + fz * MOUNT_FORWARD,
      );
      camera.lookAt(pose.x + fx * 40, MOUNT_HEIGHT - 0.18, pose.z + fz * 40);
    },

    async capture() {
      // One capture in flight at a time. Readback is async; overlapping calls
      // would interleave GPU work for frames nobody is waiting for.
      if (busy || !ctx) return null;
      busy = true;
      // Captured before the try so it is always available to restore, even if
      // setRenderTarget(target) itself is what throws.
      const previous = renderer.getRenderTarget();
      try {
        renderer.setRenderTarget(target);
        await renderer.renderAsync(scene, camera);
        const pixels = await renderer.readRenderTargetPixelsAsync(
          target, 0, 0, width, height,
        );

        const rgba = new Uint8Array(
          pixels.buffer, pixels.byteOffset, pixels.byteLength,
        );
        flipRowsInPlace(rgba, width, height);
        ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0);
        const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
        const buffer = new Uint8Array(await blob.arrayBuffer());

        return {
          data: encodeBase64(buffer),
          camera: cameraParamsFromThree(camera.position, heading),
        };
      } finally {
        // Always reapply, success or failure: the renderer is shared with the
        // main view, and a perception hiccup must never leave it pointed at
        // the detector's offscreen target instead of the visible canvas.
        renderer.setRenderTarget(previous);
        busy = false;
      }
    },

    dispose() {
      target.dispose();
    },
  };
}
