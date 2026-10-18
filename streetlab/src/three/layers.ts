/**
 * Which camera sees what.
 *
 * Two cameras share one scene: the user's view and the detector's. Everything
 * the detector photographs must be independent of UI state, so visibility is
 * decided by THREE.Layers channels rather than `Object3D.visible`:
 *
 *  - WORLD (0, Three's default): geometry that exists in the simulated world --
 *    ground, roads, sky, the traffic fleet. Both cameras see it, always.
 *  - Toggleable world categories (buildings, trees, ...): also real geometry,
 *    each on its own channel so the *main* camera can drop one while the
 *    detector still sees it. A UI toggle changes the main camera's mask only.
 *  - OVERLAY: annotations a human reads (hazard billboards, perception
 *    outlines, shadow boxes, plan ribbon, radar cone). Never in a detector
 *    frame -- a purple box drawn around a car must not become the car's pixels.
 *  - EGO: the ego mesh. The detector sits inside it.
 *  - CABIN: the ego's interior. Only the main camera in cockpit view sees it,
 *    and that view drops EGO instead: the exterior shell would wrap the eye.
 */
import type { Object3D, Camera } from 'three/webgpu';
import type { CameraView, LayerKey } from '../schema';

export const CH = {
  WORLD: 0,
  OVERLAY: 1,
  EGO: 2,
  LANE_MARKINGS: 3,
  CROSSWALKS: 4,
  BUILDINGS: 5,
  TREES: 6,
  TRAFFIC_LIGHTS: 7,
  LABELS: 8,
  CABIN: 9,
} as const;

/** UI toggle -> the world channel it hides from the main camera. */
export const TOGGLE_CHANNEL: Partial<Record<LayerKey, number>> = {
  lane_markings: CH.LANE_MARKINGS,
  crosswalks: CH.CROSSWALKS,
  buildings: CH.BUILDINGS,
  trees: CH.TREES,
  traffic_lights: CH.TRAFFIC_LIGHTS,
  labels: CH.LABELS,
};

const bit = (channel: number): number => 1 << channel;

/** Every channel that is real world geometry. The detector enables exactly these. */
export const DETECTOR_LAYER_MASK =
  bit(CH.WORLD) |
  bit(CH.LANE_MARKINGS) |
  bit(CH.CROSSWALKS) |
  bit(CH.BUILDINGS) |
  bit(CH.TREES) |
  bit(CH.TRAFFIC_LIGHTS) |
  bit(CH.LABELS);

/** Put an object and everything under it on one channel. */
export function assignChannel(root: Object3D, channel: number): void {
  root.traverse((o) => o.layers.set(channel));
}

/** The detector camera sees world geometry only, whatever the UI toggles say. */
export function setDetectorLayers(camera: Camera): void {
  camera.layers.mask = DETECTOR_LAYER_MASK;
}

/**
 * The main camera: world, overlay and either the ego (outside it) or its cabin
 * (cockpit view), minus any world category toggled off.
 */
export function applyMainLayers(
  camera: Camera,
  layers: Record<LayerKey, boolean>,
  view: CameraView = 'chase',
): void {
  camera.layers.mask =
    bit(CH.WORLD) | bit(CH.OVERLAY) | bit(view === 'cockpit' ? CH.CABIN : CH.EGO);
  for (const [key, channel] of Object.entries(TOGGLE_CHANNEL)) {
    if (layers[key as LayerKey]) camera.layers.enable(channel as number);
  }
}
