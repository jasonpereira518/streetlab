/**
 * Orange bounding boxes and warning billboards for hazardous detections.
 *
 * The box is a single translucent cube whose edges are drawn in the shader from
 * the face UVs — that gives a crisp outline of controllable thickness without
 * the 1-pixel line-width limit of `LineSegments`. Labels are sprites scaled
 * every frame to hold a constant share of the viewport height, so a hazard
 * 60 m away is as readable as one at 8 m.
 */
import * as THREE from 'three/webgpu';
import { float, min, smoothstep, time, uniform, uv } from 'three/tsl';
import type { Detection } from '../schema';
import { clamp } from '../units';
import { hazardLabelTexture } from './labels';

const MAX_HAZARDS = 12;
/** Label height as a fraction of viewport height, before `labelScale`. */
const LABEL_SCREEN_FRACTION = 0.052;
/**
 * TTC is quantised before it reaches the label. Rasterising a canvas and
 * uploading a texture is far too expensive to do per frame, and a continuously
 * counting number on a 3D billboard is unreadable anyway.
 */
const TTC_QUANTUM = 0.5;
/** Cap on cached label textures; oldest are disposed first. */
const LABEL_CACHE_LIMIT = 48;

interface Slot {
  box: THREE.Mesh;
  sprite: THREE.Sprite;
  spriteMat: THREE.SpriteNodeMaterial;
  labelKey: string;
  aspect: number;
}

export class HazardOverlay {
  readonly group = new THREE.Group();

  private readonly boxGeo: THREE.BoxGeometry;
  private readonly boxMat: THREE.MeshBasicNodeMaterial;
  private readonly accent = uniform(new THREE.Color('#FF7A1A'));
  private readonly slots: Slot[] = [];
  private readonly textures = new Map<
    string,
    { texture: THREE.Texture; aspect: number }
  >();
  private labelScale = 1;
  private showLabels = true;

  constructor() {
    this.group.name = 'hazards';
    this.boxGeo = new THREE.BoxGeometry(1, 1, 1);

    this.boxMat = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    });

    // Distance to the nearest edge of each face, in UV space.
    const d = min(
      min(uv().x, uv().x.oneMinus()),
      min(uv().y, uv().y.oneMinus()),
    );
    const edge = smoothstep(float(0.055), float(0.012), d);
    // Slow breathing pulse so a live hazard reads as active, not painted on.
    const pulse = time.mul(3.4).sin().mul(0.5).add(0.5).mul(0.28).add(0.72);

    this.boxMat.colorNode = this.accent;
    this.boxMat.opacityNode = edge.mul(0.92).add(float(0.1)).mul(pulse);
  }

  setAccent(hex: string): void {
    this.accent.value.set(hex);
  }

  setLabelScale(scale: number): void {
    this.labelScale = scale;
  }

  setLabelsVisible(visible: boolean): void {
    this.showLabels = visible;
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  private textureFor(key: string, text: string, detail: string | null) {
    const entry = this.textures.get(key);
    if (entry) return entry;

    const made = hazardLabelTexture({
      text,
      detail,
      accent: `#${this.accent.value.getHexString()}`,
    });
    const created = { texture: made.texture, aspect: made.aspect };
    this.textures.set(key, created);

    // Map iteration order is insertion order, so the first key is the oldest.
    while (this.textures.size > LABEL_CACHE_LIMIT) {
      const oldest = this.textures.keys().next();
      if (oldest.done) break;
      this.textures.get(oldest.value)?.texture.dispose();
      this.textures.delete(oldest.value);
    }
    return created;
  }

  private slotAt(i: number): Slot {
    let slot = this.slots[i];
    if (slot) return slot;

    const box = new THREE.Mesh(this.boxGeo, this.boxMat);
    box.renderOrder = 5;
    box.frustumCulled = false;

    const spriteMat = new THREE.SpriteNodeMaterial({
      transparent: true,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.renderOrder = 20;
    sprite.frustumCulled = false;

    this.group.add(box, sprite);
    slot = { box, sprite, spriteMat, labelKey: '', aspect: 4 };
    this.slots[i] = slot;
    return slot;
  }

  /**
   * Point the overlay at the current frame's hazards. `camera` is needed to
   * keep label size constant on screen.
   */
  update(detections: Detection[], camera: THREE.PerspectiveCamera): void {
    const hazards = detections.filter((d) => d.hazard).slice(0, MAX_HAZARDS);

    // Vertical half-extent of the view frustum at unit distance.
    const tanHalfFov = Math.tan((camera.fov * Math.PI) / 360);

    hazards.forEach((d, i) => {
      const slot = this.slotAt(i);
      const x = d.pose.x;
      const z = -d.pose.y;

      // Boxes are padded slightly so they read as an annotation around the
      // vehicle rather than coplanar with its bodywork.
      const pad = 0.18;
      slot.box.visible = true;
      slot.box.position.set(x, d.size.height / 2, z);
      slot.box.rotation.y = d.pose.heading;
      slot.box.scale.set(
        d.size.length + pad * 2,
        d.size.height + pad,
        d.size.width + pad * 2,
      );

      const text = d.hazard_label ?? 'Hazard';
      const detail =
        d.ttc_s != null
          ? `TTC ${(Math.round(d.ttc_s / TTC_QUANTUM) * TTC_QUANTUM).toFixed(1)} s`
          : null;
      // The accent is part of the key: recolouring must not reuse stale plates.
      const key = `${this.accent.value.getHexString()}|${text}|${detail ?? ''}`;
      if (slot.labelKey !== key) {
        const { texture, aspect } = this.textureFor(key, text, detail);
        slot.spriteMat.map = texture;
        slot.spriteMat.needsUpdate = true;
        slot.labelKey = key;
        slot.aspect = aspect;
      }

      const labelY = d.size.height + 1.05;
      slot.sprite.position.set(x, labelY, z);
      const dist = camera.position.distanceTo(slot.sprite.position);
      const worldH = clamp(
        2 * dist * tanHalfFov * LABEL_SCREEN_FRACTION * this.labelScale,
        0.62,
        6.5,
      );
      slot.sprite.scale.set(worldH * slot.aspect, worldH, 1);
      slot.sprite.visible = this.showLabels;
    });

    for (let i = hazards.length; i < this.slots.length; i++) {
      this.slots[i].box.visible = false;
      this.slots[i].sprite.visible = false;
    }
  }

  dispose(): void {
    this.boxGeo.dispose();
    this.boxMat.dispose();
    for (const s of this.slots) s.spriteMat.dispose();
    for (const t of this.textures.values()) t.texture.dispose();
    this.textures.clear();
    this.slots.length = 0;
    this.group.clear();
  }
}
