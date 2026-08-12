/**
 * Forward radar returns on a polar plot. Tracked returns are teal and sized by
 * radar cross-section; unassociated clutter is a faint grey. Anything closing
 * on the ego is tinted orange, which makes a cut-in visible here before the
 * hazard box appears in the 3D view.
 */
import { useTelemetryCanvas } from '../../store/hooks';
import { clamp } from '../../units';
import { TelemetryCard } from '../controls';
import { alpha, color } from '../theme';
import { arc, line, placeholder, text } from './draw';

const MAX_RANGE = 70;
const HALF_FOV = Math.PI / 2.6;

export function RadarPolar() {
  const ref = useTelemetryCanvas(({ ctx, width, height, frame, time }) => {
    if (!frame) return placeholder(ctx, width, height);
    const points = frame.telemetry.radar;

    const cx = width / 2;
    const cy = height - 12;
    const r = Math.min(width / 2 - 16, height - 22);

    // Range rings.
    for (const ring of [1 / 3, 2 / 3, 1]) {
      arc(
        ctx,
        cx,
        cy,
        r * ring,
        -Math.PI / 2 - HALF_FOV,
        -Math.PI / 2 + HALF_FOV,
        1,
        ring === 1 ? color.borderStrong : color.border,
        'butt',
      );
    }

    // Bearing spokes at -30, 0, +30 degrees.
    for (const a of [-HALF_FOV, -HALF_FOV / 2, 0, HALF_FOV / 2, HALF_FOV]) {
      const ang = -Math.PI / 2 + a;
      line(
        ctx,
        cx,
        cy,
        cx + Math.cos(ang) * r,
        cy + Math.sin(ang) * r,
        a === 0 ? color.borderStrong : color.border,
        1,
        a === 0 ? [3, 4] : [],
      );
    }

    // Sweep highlight, purely decorative but it sells "live sensor".
    const sweep = -HALF_FOV + ((time * 0.55) % 1) * HALF_FOV * 2;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, -Math.PI / 2 + sweep - 0.14, -Math.PI / 2 + sweep);
    ctx.closePath();
    ctx.fillStyle = alpha(color.accent, 0.07);
    ctx.fill();
    ctx.restore();

    // Returns.
    let closing = 0;
    for (const p of points) {
      if (Math.abs(p.azimuth) > HALF_FOV || p.range_m > MAX_RANGE) continue;
      // Screen bearing: azimuth is + to the left, so subtract it from "up".
      const ang = -Math.PI / 2 - p.azimuth;
      const rad = (p.range_m / MAX_RANGE) * r;
      const x = cx + Math.cos(ang) * rad;
      const y = cy + Math.sin(ang) * rad;
      const approaching = p.range_rate_mps < -0.5;
      if (approaching && p.tracked) closing++;

      const size = p.tracked
        ? clamp(2.2 + p.rcs_db * 0.08, 2.4, 5)
        : clamp(1.2 + p.rcs_db * 0.06, 1, 2.2);
      const fill = !p.tracked
        ? alpha(color.textFaint, 0.5)
        : approaching
          ? color.warn
          : color.accent;

      if (p.tracked) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, size * 2.4, 0, Math.PI * 2);
        ctx.fillStyle = alpha(approaching ? color.warn : color.accent, 0.16);
        ctx.fill();
        ctx.restore();
      }
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.restore();
    }

    // Ego marker.
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(cx, cy - 6);
    ctx.lineTo(cx + 4.5, cy + 3);
    ctx.lineTo(cx - 4.5, cy + 3);
    ctx.closePath();
    ctx.fillStyle = color.text;
    ctx.fill();
    ctx.restore();

    text(ctx, `${MAX_RANGE} m`, width - 6, 10, {
      size: 8.5,
      color: color.textFaint,
      align: 'right',
    });
    text(ctx, `${points.length} returns`, 6, 10, {
      size: 8.5,
      color: color.textFaint,
    });
    if (closing > 0) {
      text(ctx, `${closing} closing`, 6, height - 6, {
        size: 9,
        weight: 600,
        color: color.warn,
      });
    }
  });

  return (
    <TelemetryCard title="Radar">
      <canvas ref={ref} />
    </TelemetryCard>
  );
}
