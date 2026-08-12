/**
 * Canvas-backed textures for anything in the 3D scene that carries text:
 * green street-name blades, STOP faces, speed-limit roundels and the orange
 * hazard billboards.
 *
 * All of them are drawn at a fixed high resolution and downsampled with
 * mipmaps + anisotropy, which is what keeps small text readable at distance.
 */
import * as THREE from 'three/webgpu';
import { color as tokens } from '../ui/theme';

const FONT = `-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif`;

function makeCanvas(w: number, h: number): {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
} {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas context unavailable');
  return { canvas, ctx };
}

function toTexture(canvas: HTMLCanvasElement): THREE.Texture {
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Green MUTCD-style street-name blade. Aspect is 4:1. */
export function streetNameTexture(text: string): THREE.Texture {
  const W = 1024;
  const H = 256;
  const { canvas, ctx } = makeCanvas(W, H);
  ctx.fillStyle = '#0B6B45';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#FFFFFF';
  ctx.lineWidth = 10;
  ctx.strokeRect(16, 16, W - 32, H - 32);

  ctx.fillStyle = '#FFFFFF';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  let size = 118;
  ctx.font = `600 ${size}px ${FONT}`;
  while (ctx.measureText(text).width > W - 90 && size > 40) {
    size -= 4;
    ctx.font = `600 ${size}px ${FONT}`;
  }
  ctx.fillText(text.toUpperCase(), W / 2, H / 2 + 4);
  return toTexture(canvas);
}

/** White regulatory speed-limit plate. Aspect is 2:3 (w:h). */
export function speedLimitTexture(mph: string): THREE.Texture {
  const W = 384;
  const H = 576;
  const { canvas, ctx } = makeCanvas(W, H);
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#1A1A1A';
  ctx.lineWidth = 16;
  ctx.strokeRect(22, 22, W - 44, H - 44);

  ctx.fillStyle = '#1A1A1A';
  ctx.textAlign = 'center';
  ctx.font = `700 74px ${FONT}`;
  ctx.fillText('SPEED', W / 2, 150);
  ctx.fillText('LIMIT', W / 2, 226);
  ctx.font = `800 210px ${FONT}`;
  ctx.fillText(mph, W / 2, 440);
  return toTexture(canvas);
}

/** White "STOP" lettering on transparent ground, laid over the red octagon. */
export function stopFaceTexture(): THREE.Texture {
  const S = 512;
  const { canvas, ctx } = makeCanvas(S, S);
  ctx.clearRect(0, 0, S, S);
  ctx.strokeStyle = '#FFFFFF';
  ctx.lineWidth = 14;
  // Inner octagon outline.
  ctx.beginPath();
  for (let i = 0; i < 8; i++) {
    const a = (Math.PI / 4) * i + Math.PI / 8;
    const r = S * 0.41;
    const x = S / 2 + Math.cos(a) * r;
    const y = S / 2 + Math.sin(a) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();

  ctx.fillStyle = '#FFFFFF';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `800 150px ${FONT}`;
  ctx.fillText('STOP', S / 2, S / 2 + 6);
  return toTexture(canvas);
}

export interface HazardLabelOptions {
  text: string;
  accent?: string;
  /** Optional second line, e.g. "TTC 2.4 s". */
  detail?: string | null;
}

/** Rounded white billboard with a warning triangle. Aspect is 4:1 (w:h). */
export function hazardLabelTexture(
  opts: HazardLabelOptions,
): { texture: THREE.Texture; aspect: number } {
  const H = 256;
  const PAD = 34;
  const ICON = 116;
  const accent = opts.accent ?? tokens.warn;

  // Measure first so the plate hugs the text instead of floating in whitespace.
  const probe = makeCanvas(8, 8).ctx;
  probe.font = `700 96px ${FONT}`;
  const titleW = probe.measureText(opts.text).width;
  probe.font = `600 62px ${FONT}`;
  const detailW = opts.detail ? probe.measureText(opts.detail).width : 0;
  const textW = Math.max(titleW, detailW);
  const W = Math.ceil(PAD * 2 + ICON + 26 + textW);

  const { canvas, ctx } = makeCanvas(W, H);
  ctx.clearRect(0, 0, W, H);

  // Plate.
  ctx.fillStyle = 'rgba(255, 255, 255, 0.97)';
  roundRect(ctx, 6, 6, W - 12, H - 12, 40);
  ctx.fill();
  ctx.strokeStyle = accent;
  ctx.lineWidth = 7;
  ctx.stroke();

  // Warning triangle.
  const cx = PAD + ICON / 2;
  const cy = H / 2;
  ctx.fillStyle = accent;
  ctx.beginPath();
  const r = ICON / 2;
  ctx.moveTo(cx, cy - r);
  ctx.lineTo(cx + r * 0.92, cy + r * 0.72);
  ctx.lineTo(cx - r * 0.92, cy + r * 0.72);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(cx - 7, cy - r * 0.36, 14, r * 0.86);
  ctx.beginPath();
  ctx.arc(cx, cy + r * 0.56, 8.5, 0, Math.PI * 2);
  ctx.fill();

  // Text.
  const textX = PAD + ICON + 26;
  ctx.textAlign = 'left';
  ctx.fillStyle = tokens.text;
  if (opts.detail) {
    ctx.textBaseline = 'alphabetic';
    ctx.font = `700 92px ${FONT}`;
    ctx.fillText(opts.text, textX, H / 2 - 8);
    ctx.font = `600 58px ${FONT}`;
    ctx.fillStyle = accent;
    ctx.fillText(opts.detail, textX, H / 2 + 66);
  } else {
    ctx.textBaseline = 'middle';
    ctx.font = `700 96px ${FONT}`;
    ctx.fillText(opts.text, textX, H / 2 + 2);
  }

  return { texture: toTexture(canvas), aspect: W / H };
}
