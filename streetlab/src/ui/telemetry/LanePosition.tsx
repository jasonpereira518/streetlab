/**
 * Bird's-eye lane strip: the ego in its lane plus every neighbour the planner
 * is tracking, placed by longitudinal distance and lateral offset.
 *
 * The view is always three lanes wide and centred on the ego's own lane, so the
 * car stays put and the world moves around it. Lanes that do not exist on the
 * real carriageway are greyed out rather than omitted, which keeps the widget
 * from jumping when the lane count changes.
 */
import { useTelemetryCanvas } from '../../store/hooks';
import { TelemetryCard } from '../controls';
import { alpha, classColor, color } from '../theme';
import { line, placeholder, roundRect, text } from './draw';

/** Longitudinal window, metres: behind the ego .. ahead of it. */
const BEHIND = 16;
const AHEAD = 52;
/** Lanes drawn either side of the ego's lane. */
const SIDE_LANES = 1;

export function LanePosition() {
  const ref = useTelemetryCanvas(({ ctx, width, height, frame }) => {
    if (!frame) return placeholder(ctx, width, height);
    const lane = frame.telemetry.lane;

    const padY = 5;
    const usableH = height - padY * 2 - 12;
    const lanes = SIDE_LANES * 2 + 1;
    const laneW = Math.min((width - 34) / lanes, 46);
    const cx = width / 2 - 6;
    const half = (lanes * laneW) / 2;
    const bottom = padY + usableH;

    /** Metres ahead -> y (ahead is up the card). */
    const toY = (m: number) => bottom - ((m + BEHIND) / (BEHIND + AHEAD)) * usableH;
    /** Metres left of the ego lane centre -> x. */
    const toX = (m: number) => cx - (m / lane.lane_width_m) * laneW;

    // Carriageway.
    ctx.save();
    roundRect(ctx, cx - half, padY, half * 2, usableH, 5);
    ctx.fillStyle = color.surfaceSunken;
    ctx.fill();
    ctx.restore();

    // Grey out neighbouring lanes that do not exist on this road.
    const leftExists = lane.lane_index > 0;
    const rightExists = lane.lane_index < lane.lane_count - 1;
    ctx.save();
    ctx.fillStyle = alpha(color.borderStrong, 0.45);
    if (!leftExists) ctx.fillRect(cx - half, padY, laneW, usableH);
    if (!rightExists) ctx.fillRect(cx + half - laneW, padY, laneW, usableH);
    ctx.restore();

    // Ego lane boundaries use the markings the planner actually reported.
    const markStyle = (m: string) =>
      m === 'double_yellow'
        ? { c: '#D9B441', dash: [] as number[], n: 2 }
        : m === 'solid_white'
          ? { c: color.borderStrong, dash: [] as number[], n: 1 }
          : m === 'dashed_white'
            ? { c: color.laneMark, dash: [5, 5], n: 1 }
            : { c: color.border, dash: [2, 6], n: 1 };

    for (const side of [-1, 1] as const) {
      const style = markStyle(side < 0 ? lane.left_marking : lane.right_marking);
      const x = cx + side * (laneW / 2);
      for (let k = 0; k < style.n; k++) {
        const off = style.n === 2 ? (k === 0 ? -1.6 : 1.6) : 0;
        line(ctx, x + off, padY, x + off, bottom, style.c, 1.3, style.dash);
      }
    }
    // Outer edges.
    for (const side of [-1, 1]) {
      const x = cx + side * half;
      line(ctx, x, padY, x, bottom, color.borderStrong, 1.4);
    }

    // Distance rungs every 20 m.
    for (let m = 0; m <= AHEAD; m += 20) {
      const y = toY(m);
      line(ctx, cx - half, y, cx + half, y, color.border, 1, [2, 4]);
      if (m > 0) {
        text(ctx, `${m}`, cx + half + 4, y, {
          size: 8,
          color: color.textFaint,
          baseline: 'middle',
        });
      }
    }

    const carW = laneW * 0.56;
    const carH = Math.max(9, usableH * 0.085);

    const drawCar = (
      x: number,
      y: number,
      fill: string,
      stroke: string,
      hazard: boolean,
    ) => {
      if (hazard) {
        ctx.save();
        roundRect(ctx, x - carW / 2 - 3, y - carH / 2 - 3, carW + 6, carH + 6, 5);
        ctx.fillStyle = alpha(color.warn, 0.18);
        ctx.fill();
        ctx.restore();
      }
      ctx.save();
      roundRect(ctx, x - carW / 2, y - carH / 2, carW, carH, 3);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = stroke;
      ctx.stroke();
      // Nose marker so heading is unambiguous.
      ctx.fillStyle = stroke;
      ctx.fillRect(x - carW * 0.22, y - carH / 2 + 1.5, carW * 0.44, 1.6);
      ctx.restore();
    };

    // Neighbours first so the ego always sits on top.
    const maxLateral = (SIDE_LANES + 0.5) * lane.lane_width_m;
    for (const n of lane.neighbors) {
      if (n.longitudinal_m < -BEHIND || n.longitudinal_m > AHEAD) continue;
      if (Math.abs(n.lateral_m) > maxLateral) continue;
      drawCar(
        toX(n.lateral_m),
        toY(n.longitudinal_m),
        n.hazard ? color.warnSoft : color.surface,
        n.hazard ? color.warn : classColor[n.cls] ?? color.textMuted,
        n.hazard,
      );
    }

    // Ego, offset laterally by its in-lane error.
    drawCar(toX(lane.offset_m), toY(0), color.surface, color.accent, false);

    /* ---- footer readout ---- */
    const off = lane.offset_m;
    const centred = Math.abs(off) < 0.06;
    text(
      ctx,
      centred ? 'centred in lane' : `${Math.abs(off).toFixed(2)} m ${off > 0 ? 'left' : 'right'}`,
      6,
      height - 6,
      {
        size: 9.5,
        weight: 600,
        color: Math.abs(off) > 0.5 ? color.warn : color.textMuted,
        baseline: 'bottom',
      },
    );
    text(
      ctx,
      `lane ${lane.lane_index + 1}/${lane.lane_count}`,
      width - 6,
      height - 6,
      {
        size: 9,
        color: color.textFaint,
        align: 'right',
        baseline: 'bottom',
      },
    );
  });

  return (
    <TelemetryCard title="Lane position">
      <canvas ref={ref} />
    </TelemetryCard>
  );
}
