/**
 * The cabin's centre screen: a live third-person view of the car and the road
 * around it (rendered into a texture), with a 2D readout drawn over its left
 * side -- speed, gear, limit, autopilot state, blinkers, battery.
 *
 * The screen camera never sees the cabin (CH.CABIN is off its mask), so the
 * screen never draws itself; buildings, trees and labels are dropped too, for
 * the clean "what the car perceives" look.
 *
 * Render-target slot: the renderer has one, shared with the visible canvas
 * and the detector capture. Callers must only call `update()` while
 * `detectorCamera.renderTargetBusy()` is false, and before the main render of
 * the same tick. The pass is synchronous, and the slot is restored before it
 * returns.
 */
import * as THREE from 'three/webgpu';
import { texture, uv, vec2 } from 'three/tsl';
import type { StateUpdate } from '../schema';
import { toMph } from '../units';
import { type Cabin, type TurnSignal, SCREEN, blinkOn } from './cabin';
import { CH, assignChannel } from './layers';

const VIEW_HZ = 15;
const HUD_HZ = 10; // fast enough for the 1.5 Hz blinker's edges
const VIEW_W = 640;
const VIEW_H = Math.round((VIEW_W * SCREEN.h) / SCREEN.w);

const bit = (c: number) => 1 << c;
export const SCREEN_LAYER_MASK =
  bit(CH.WORLD) |
  bit(CH.EGO) |
  bit(CH.OVERLAY) |
  bit(CH.LANE_MARKINGS) |
  bit(CH.CROSSWALKS) |
  bit(CH.TRAFFIC_LIGHTS);

const MANEUVER_TEXT: Record<StateUpdate['plan']['maneuver'], string> = {
  keep_lane: 'Following route',
  turn_left: 'Turning left',
  turn_right: 'Turning right',
  lane_change_left: 'Changing lanes left',
  lane_change_right: 'Changing lanes right',
  stop: 'Stopping',
  yield: 'Yielding',
  arrived: 'Arrived',
  emergency_brake: 'EMERGENCY BRAKING',
  pull_over: 'Pulling over',
};

const CRUISE_TEXT = { off: 'Manual', cruise: 'Cruise', autosteer: 'Assist', fsd: 'FSD (Supervised)' };

export class FsdScreen {
  readonly camera = new THREE.PerspectiveCamera(48, SCREEN.w / SCREEN.h, 0.5, 400);
  private readonly target = new THREE.RenderTarget(VIEW_W, VIEW_H);
  private readonly hudCanvas: HTMLCanvasElement;
  private readonly hudCtx: CanvasRenderingContext2D | null;
  private readonly hudTex: THREE.CanvasTexture;
  private readonly hudMat: THREE.MeshBasicNodeMaterial;
  private sinceView = Infinity;
  private sinceHud = Infinity;
  private readonly eye = new THREE.Vector3();
  private readonly aim = new THREE.Vector3();

  constructor(cabin: Cabin) {
    this.camera.layers.mask = SCREEN_LAYER_MASK;

    const mat = cabin.screen.material;
    // A render target comes back top-down relative to a loaded image, so the
    // view is sampled with v flipped; through `map` it showed upside down.
    mat.colorNode = texture(this.target.texture, vec2(uv().x, uv().y.oneMinus()));
    mat.needsUpdate = true;

    this.hudCanvas = document.createElement('canvas');
    this.hudCanvas.width = VIEW_W;
    this.hudCanvas.height = VIEW_H;
    this.hudCtx = this.hudCanvas.getContext('2d');
    this.hudTex = new THREE.CanvasTexture(this.hudCanvas);
    this.hudTex.colorSpace = THREE.SRGBColorSpace;
    this.hudMat = new THREE.MeshBasicNodeMaterial({
      map: this.hudTex,
      transparent: true,
      toneMapped: false,
      depthWrite: false,
    });
    // A hair in front of the screen surface, sharing its plane and tilt.
    const hud = new THREE.Mesh(cabin.screen.geometry, this.hudMat);
    hud.position.x = -0.001;
    hud.renderOrder = 1;
    cabin.screen.add(hud);
    assignChannel(hud, CH.CABIN);
  }

  /**
   * Advance clocks and, when due, redraw the readout and re-render the view.
   * `egoRoot` is the ego group (heading + terrain attitude, no body roll).
   */
  update(
    renderer: THREE.WebGPURenderer,
    scene: THREE.Scene,
    egoRoot: THREE.Object3D,
    frame: StateUpdate,
    signal: TurnSignal,
    t: number,
    dt: number,
  ): void {
    this.sinceHud += dt;
    if (this.sinceHud >= 1 / HUD_HZ) {
      this.sinceHud = 0;
      this.drawHud(frame, signal, t);
    }

    this.sinceView += dt;
    if (this.sinceView < 1 / VIEW_HZ) return;
    this.sinceView = 0;

    // Behind and above the car, looking past it down the road.
    egoRoot.updateWorldMatrix(true, false);
    egoRoot.localToWorld(this.eye.set(-9, 6, 0));
    egoRoot.localToWorld(this.aim.set(15, 0, 0));
    this.camera.position.copy(this.eye);
    this.camera.lookAt(this.aim);
    this.camera.updateMatrixWorld();

    const previous = renderer.getRenderTarget();
    renderer.setRenderTarget(this.target);
    try {
      renderer.render(scene, this.camera);
    } finally {
      renderer.setRenderTarget(previous);
    }
  }

  private drawHud(frame: StateUpdate, signal: TurnSignal, t: number): void {
    const ctx = this.hudCtx;
    if (!ctx) return; // headless (tests)
    const W = VIEW_W;
    const H = VIEW_H;
    const ego = frame.ego;
    const font = (w: number, px: number) => `${w} ${px}px Inter, system-ui, sans-serif`;
    ctx.clearRect(0, 0, W, H);

    // Left panel, darkened so the readout holds over a bright road.
    const panelW = W * 0.3;
    const grad = ctx.createLinearGradient(0, 0, panelW * 1.3, 0);
    grad.addColorStop(0, 'rgba(10,12,15,0.86)');
    grad.addColorStop(0.75, 'rgba(10,12,15,0.6)');
    grad.addColorStop(1, 'rgba(10,12,15,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, panelW * 1.3, H);

    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'center';
    const cx = panelW / 2;

    // Gear row.
    ctx.font = font(600, 18);
    'PRND'.split('').forEach((g, i) => {
      ctx.fillStyle = g === ego.gear ? '#ffffff' : 'rgba(255,255,255,0.32)';
      ctx.fillText(g, cx + (i - 1.5) * 22, 40);
    });

    // Speed.
    ctx.fillStyle = '#ffffff';
    ctx.font = font(300, 84);
    ctx.fillText(String(Math.round(toMph(Math.abs(ego.speed_mps)))), cx, 128);
    ctx.font = font(500, 16);
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText('MPH', cx, 152);

    // Speed-limit sign.
    const sx = cx - 26;
    const sy = 172;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(sx, sy, 52, 62);
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    ctx.strokeRect(sx + 3, sy + 3, 46, 56);
    ctx.fillStyle = '#000000';
    ctx.font = font(700, 9);
    ctx.fillText('SPEED', cx, sy + 16);
    ctx.fillText('LIMIT', cx, sy + 26);
    ctx.font = font(700, 24);
    ctx.fillText(String(Math.round(toMph(ego.speed_limit_mps))), cx, sy + 52);

    // Autopilot state and set speed: blue when engaged.
    const engaged = ego.cruise.mode !== 'off';
    ctx.font = font(600, 15);
    ctx.fillStyle = engaged ? '#3E8BFF' : 'rgba(255,255,255,0.45)';
    ctx.fillText(CRUISE_TEXT[ego.cruise.mode], cx, 262);
    if (engaged) {
      ctx.font = font(500, 13);
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.fillText(`Set ${Math.round(toMph(ego.cruise.set_speed_mps))} mph`, cx, 280);
    }

    // Battery.
    const pct = frame.telemetry.vehicle.battery_pct;
    const bx = cx - 30;
    const by = H - 34;
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(bx, by, 34, 14);
    ctx.fillStyle = pct < 20 ? '#E5484D' : '#46C46E';
    ctx.fillRect(bx + 2, by + 2, (30 * pct) / 100, 10);
    ctx.font = font(500, 12);
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.fillText(`${Math.round(pct)}%`, bx + 40, by + 12);

    // Blinker arrows, top corners of the view.
    const lit = signal !== 'off' && blinkOn(t);
    const arrow = (x: number, dir: -1 | 1, on: boolean) => {
      ctx.fillStyle = on ? '#46C46E' : 'rgba(255,255,255,0.12)';
      ctx.beginPath();
      ctx.moveTo(x + dir * 18, 26);
      ctx.lineTo(x, 12);
      ctx.lineTo(x, 20);
      ctx.lineTo(x - dir * 14, 20);
      ctx.lineTo(x - dir * 14, 32);
      ctx.lineTo(x, 32);
      ctx.lineTo(x, 40);
      ctx.closePath();
      ctx.fill();
    };
    arrow(panelW + 40, -1, lit && signal === 'left');
    arrow(W - 40, 1, lit && signal === 'right');

    // Current manoeuvre, bottom of the view.
    const m = frame.plan.maneuver;
    const msg = MANEUVER_TEXT[m];
    const mx = panelW + (W - panelW) / 2;
    ctx.textAlign = 'center';
    ctx.font = font(600, 16);
    const tw = ctx.measureText(msg).width + 28;
    ctx.fillStyle = m === 'emergency_brake' ? 'rgba(229,72,77,0.92)' : 'rgba(10,12,15,0.7)';
    ctx.fillRect(mx - tw / 2, H - 44, tw, 30);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(msg, mx, H - 23);

    // A red bar along the bottom while braking.
    if (ego.brake > 0.05) {
      ctx.fillStyle = `rgba(229,72,77,${Math.min(0.9, 0.3 + ego.brake)})`;
      ctx.fillRect(0, H - 4, W, 4);
    }

    this.hudTex.needsUpdate = true;
  }

  dispose(): void {
    this.target.dispose();
    this.hudTex.dispose();
    this.hudMat.dispose();
  }
}
