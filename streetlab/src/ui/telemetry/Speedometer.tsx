/**
 * Speed gauge: a 240-degree arc with the live speed in the middle and the
 * cruise set-speed marked on the rim as "45 MAX".
 */
import { useTelemetryCanvas } from '../../store/hooks';
import { clamp, toMph } from '../../units';
import { TelemetryCard } from '../controls';
import { color } from '../theme';
import { arc, dot, placeholder, text } from './draw';

const START = Math.PI * 0.72;
const SWEEP = Math.PI * 1.56;

export function Speedometer() {
  const ref = useTelemetryCanvas(({ ctx, width, height, frame }) => {
    if (!frame) return placeholder(ctx, width, height);

    const mph = toMph(frame.ego.speed_mps);
    const setMph = toMph(frame.ego.cruise.set_speed_mps);
    const limitMph = toMph(frame.ego.speed_limit_mps);
    // Round the ceiling up to the next 10 so the needle never pins.
    const ceiling = Math.max(60, Math.ceil((Math.max(setMph, mph) + 10) / 10) * 10);

    const cx = width / 2;
    // The arc spans 280 degrees starting at 130, so it reaches 0.77r below the
    // centre. Placing the centre at 0.56h keeps both ends inside the card.
    const cy = height * 0.56;
    const r = Math.min(width * 0.33, height * 0.42);
    const w = Math.max(6, r * 0.19);

    // Track.
    arc(ctx, cx, cy, r, START, START + SWEEP, w, color.surfaceSunken);

    // Value.
    const frac = clamp(mph / ceiling, 0, 1);
    if (frac > 0.001) {
      const g = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
      g.addColorStop(0, color.accent);
      g.addColorStop(1, color.accentDark);
      arc(ctx, cx, cy, r, START, START + SWEEP * frac, w, g);
    }

    // Set-speed marker.
    const setFrac = clamp(setMph / ceiling, 0, 1);
    const setAngle = START + SWEEP * setFrac;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(setAngle);
    ctx.fillStyle = color.warn;
    ctx.fillRect(r - w / 2 - 2, -1.4, w + 4, 2.8);
    ctx.restore();

    // Speed-limit tick, drawn thinner so it reads as secondary.
    const limitAngle = START + SWEEP * clamp(limitMph / ceiling, 0, 1);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(limitAngle);
    ctx.fillStyle = color.textFaint;
    ctx.fillRect(r + w / 2 + 2, -0.9, 4, 1.8);
    ctx.restore();

    // Readout.
    const big = Math.min(34, r * 0.95);
    text(ctx, String(Math.round(mph)), cx, cy - big * 0.06, {
      size: big,
      weight: 700,
      color: color.text,
      align: 'center',
      baseline: 'alphabetic',
    });
    text(ctx, 'mph', cx, cy + big * 0.34, {
      size: 10,
      weight: 600,
      color: color.textMuted,
      align: 'center',
      baseline: 'top',
      tracking: 0.6,
    });

    // Rim labels, tucked just inside the open bottom of the arc.
    text(ctx, '0', cx - r * 0.72, cy + r * 0.86, {
      size: 8.5,
      color: color.textFaint,
      align: 'center',
      baseline: 'middle',
    });
    text(ctx, String(ceiling), cx + r * 0.72, cy + r * 0.86, {
      size: 8.5,
      color: color.textFaint,
      align: 'center',
      baseline: 'middle',
    });

    // Set-speed legend, top-right where the arc leaves the corner free.
    dot(ctx, width - 52, 9, 3, color.warn);
    text(ctx, `${Math.round(setMph)} MAX`, width - 45, 9, {
      size: 9.5,
      weight: 700,
      color: color.textMuted,
      baseline: 'middle',
    });
  });

  return (
    <TelemetryCard title="Speed">
      <canvas ref={ref} />
    </TelemetryCard>
  );
}
