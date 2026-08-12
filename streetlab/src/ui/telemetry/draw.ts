/** Canvas drawing helpers shared by the six telemetry widgets. */
import { color, font } from '../theme';

export const SANS = font.sans;
export const MONO = font.mono;

export interface TextOptions {
  size?: number;
  weight?: number;
  color?: string;
  align?: CanvasTextAlign;
  baseline?: CanvasTextBaseline;
  mono?: boolean;
  tracking?: number;
}

export function text(
  ctx: CanvasRenderingContext2D,
  value: string,
  x: number,
  y: number,
  opts: TextOptions = {},
): void {
  const {
    size = 11,
    weight = 500,
    color: fill = color.textMuted,
    align = 'left',
    baseline = 'alphabetic',
    mono = false,
    tracking = 0,
  } = opts;
  ctx.save();
  ctx.font = `${weight} ${size}px ${mono ? MONO : SANS}`;
  ctx.fillStyle = fill;
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  if (tracking) {
    // Canvas has no letter-spacing in every engine we target; step manually.
    const chars = [...value];
    const widths = chars.map((c) => ctx.measureText(c).width);
    const total =
      widths.reduce((a, b) => a + b, 0) + tracking * (chars.length - 1);
    let cx = align === 'center' ? x - total / 2 : align === 'right' ? x - total : x;
    ctx.textAlign = 'left';
    chars.forEach((c, i) => {
      ctx.fillText(c, cx, y);
      cx += widths[i] + tracking;
    });
  } else {
    ctx.fillText(value, x, y);
  }
  ctx.restore();
}

export function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const rad = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

export function arc(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  a0: number,
  a1: number,
  width: number,
  stroke: string | CanvasGradient,
  cap: CanvasLineCap = 'round',
): void {
  if (a1 === a0) return;
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, a0, a1);
  ctx.lineWidth = width;
  ctx.lineCap = cap;
  ctx.strokeStyle = stroke;
  ctx.stroke();
  ctx.restore();
}

export function line(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  stroke: string,
  width = 1,
  dash: number[] = [],
): void {
  ctx.save();
  ctx.beginPath();
  ctx.setLineDash(dash);
  ctx.lineWidth = width;
  ctx.strokeStyle = stroke;
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.restore();
}

export function dot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  fill: string,
): void {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.restore();
}

/** Centred "waiting for data" state, so an empty card never looks broken. */
export function placeholder(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  message = 'Awaiting telemetry',
): void {
  text(ctx, message, w / 2, h / 2, {
    size: 11,
    color: color.textFaint,
    align: 'center',
    baseline: 'middle',
  });
}

/** Smooth polyline through points, using midpoint quadratics. */
export function smoothPath(
  ctx: CanvasRenderingContext2D,
  pts: Array<[number, number]>,
): void {
  if (pts.length === 0) return;
  ctx.moveTo(pts[0][0], pts[0][1]);
  if (pts.length < 3) {
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    return;
  }
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (pts[i][0] + pts[i + 1][0]) / 2;
    const my = (pts[i][1] + pts[i + 1][1]) / 2;
    ctx.quadraticCurveTo(pts[i][0], pts[i][1], mx, my);
  }
  ctx.lineTo(pts[pts.length - 1][0], pts[pts.length - 1][1]);
}
