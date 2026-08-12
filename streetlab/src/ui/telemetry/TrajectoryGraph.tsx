/**
 * Lateral offset over time: the planner's trajectory against the predicted
 * path of whatever is cutting in. Left of the "now" line is observed history,
 * right of it is prediction — which is where the two curves diverge during a
 * cut-in and the planner's evasive nudge becomes visible.
 */
import { useRef } from 'react';
import { useTelemetryCanvas } from '../../store/hooks';
import { clamp, damp } from '../../units';
import { TelemetryCard } from '../controls';
import { alpha, color } from '../theme';
import { line, placeholder, smoothPath, text } from './draw';

const T_MIN = -3;
/**
 * The vertical scale adapts to the data. Normal lane keeping wanders by ~0.1 m
 * and would be a flat line on a fixed lane-width axis; a cut-in swings a full
 * 3.6 m and would clip. The window tracks the largest value on screen, damped
 * so it eases rather than snaps, and never closes tighter than +/-0.8 m.
 */
const MIN_RANGE = 0.8;
const MAX_RANGE = 5;

export function TrajectoryGraph() {
  const rangeRef = useRef(MIN_RANGE);

  const ref = useTelemetryCanvas(({ ctx, width, height, frame, dt }) => {
    if (!frame) return placeholder(ctx, width, height);
    const traj = frame.telemetry.trajectory;
    const tMax = traj.horizon_s;

    let peak = 0;
    for (const s of traj.planned) peak = Math.max(peak, Math.abs(s.lateral_m));
    for (const s of traj.cutin ?? []) peak = Math.max(peak, Math.abs(s.lateral_m));
    const wanted = clamp(peak * 1.3, MIN_RANGE, MAX_RANGE);
    rangeRef.current = damp(rangeRef.current, wanted, 0.02, dt);
    const LATERAL_RANGE = rangeRef.current;

    const padL = 26;
    const padR = 8;
    const padT = 10;
    const padB = 15;
    const w = width - padL - padR;
    const h = height - padT - padB;

    const toX = (t: number) => padL + ((t - T_MIN) / (tMax - T_MIN)) * w;
    const toY = (m: number) =>
      padT + h / 2 - (clamp(m, -LATERAL_RANGE, LATERAL_RANGE) / LATERAL_RANGE) * (h / 2);

    // Gridlines at 0 and +/- half and full scale, labelled to match the range.
    const decimals = LATERAL_RANGE < 2.5 ? 1 : 0;
    for (const m of [-LATERAL_RANGE, -LATERAL_RANGE / 2, 0, LATERAL_RANGE / 2, LATERAL_RANGE]) {
      const y = toY(m);
      line(ctx, padL, y, padL + w, y, m === 0 ? color.borderStrong : color.border, 1, m === 0 ? [] : [2, 4]);
      text(
        ctx,
        m === 0 ? '0' : `${m > 0 ? '+' : '-'}${Math.abs(m).toFixed(decimals)}`,
        padL - 5,
        y,
        { size: 8, color: color.textFaint, align: 'right', baseline: 'middle' },
      );
    }

    // Prediction half gets a faint wash so past and future read apart.
    const nowX = toX(0);
    ctx.save();
    ctx.fillStyle = alpha(color.accent, 0.04);
    ctx.fillRect(nowX, padT, padL + w - nowX, h);
    ctx.restore();
    line(ctx, nowX, padT, nowX, padT + h, color.accent, 1.2, [3, 3]);
    text(ctx, 'now', nowX + 3, padT + 1, {
      size: 8.5,
      weight: 600,
      color: color.accent,
      baseline: 'top',
    });

    const series = (
      samples: Array<{ t: number; lateral_m: number }>,
      stroke: string,
      dash: number[],
      fill?: string,
    ) => {
      const pts = samples
        .filter((s) => s.t >= T_MIN && s.t <= tMax)
        .map((s): [number, number] => [toX(s.t), toY(s.lateral_m)]);
      if (pts.length < 2) return;

      if (fill) {
        ctx.save();
        ctx.beginPath();
        smoothPath(ctx, pts);
        ctx.lineTo(pts[pts.length - 1][0], toY(0));
        ctx.lineTo(pts[0][0], toY(0));
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.restore();
      }

      ctx.save();
      ctx.beginPath();
      smoothPath(ctx, pts);
      ctx.setLineDash(dash);
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.strokeStyle = stroke;
      ctx.stroke();
      ctx.restore();

      // Head marker at the far end of the prediction.
      const last = pts[pts.length - 1];
      ctx.save();
      ctx.beginPath();
      ctx.arc(last[0], last[1], 2.6, 0, Math.PI * 2);
      ctx.fillStyle = stroke;
      ctx.fill();
      ctx.restore();
    };

    if (traj.cutin) series(traj.cutin, color.warn, [5, 4]);
    series(traj.planned, color.plan, [], alpha(color.plan, 0.1));

    // Time axis labels.
    for (const t of [T_MIN, 0, tMax]) {
      text(ctx, `${t > 0 ? '+' : ''}${t}s`, toX(t), height - 4, {
        size: 8.5,
        color: color.textFaint,
        align: t === T_MIN ? 'left' : t === 0 ? 'center' : 'right',
        baseline: 'bottom',
      });
    }

    // Legend.
    const legendY = padT + 2;
    line(ctx, padL + w - 58, legendY, padL + w - 46, legendY, color.plan, 2);
    text(ctx, 'plan', padL + w - 43, legendY, {
      size: 8.5,
      weight: 600,
      color: color.plan,
      baseline: 'middle',
    });
    if (traj.cutin) {
      line(ctx, padL + w - 58, legendY + 11, padL + w - 46, legendY + 11, color.warn, 2, [4, 3]);
      text(ctx, 'cut-in', padL + w - 43, legendY + 11, {
        size: 8.5,
        weight: 600,
        color: color.warn,
        baseline: 'middle',
      });
    }
  });

  return (
    <TelemetryCard title="Trajectory" badge="lateral m">
      <canvas ref={ref} />
    </TelemetryCard>
  );
}
