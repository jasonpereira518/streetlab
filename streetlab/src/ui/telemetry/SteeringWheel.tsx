/**
 * Steering angle. The rim rotates with the wheel and a green arc sweeps from
 * top-dead-centre to the current angle, so both the magnitude and the direction
 * of the input are readable at a glance.
 */
import { useTelemetryCanvas } from '../../store/hooks';
import { RAD_TO_DEG, clamp } from '../../units';
import { TelemetryCard } from '../controls';
import { color } from '../theme';
import { arc, placeholder, text } from './draw';

/** Road-wheel angle to hand-wheel angle. */
const STEERING_RATIO = 14;
const MAX_WHEEL_DEG = 450;

export function SteeringWheel() {
  const ref = useTelemetryCanvas(({ ctx, width, height, frame }) => {
    if (!frame) return placeholder(ctx, width, height);

    // Positive steering is left; on screen, left is counter-clockwise.
    const wheelDeg = clamp(
      frame.ego.steering_angle * RAD_TO_DEG * STEERING_RATIO,
      -MAX_WHEEL_DEG,
      MAX_WHEEL_DEG,
    );
    const rot = -wheelDeg * (Math.PI / 180);

    const cx = width / 2;
    // Leave a 26 px band at the bottom for the numeric readout.
    const cy = (height - 26) / 2;
    const r = Math.min(width * 0.28, (height - 26) * 0.36);

    // Sweep arc from 12 o'clock.
    const top = -Math.PI / 2;
    const sweep = clamp(rot, -Math.PI * 1.35, Math.PI * 1.35);
    arc(ctx, cx, cy, r + 8, top, top + sweep, 4, color.surfaceSunken, 'butt');
    if (Math.abs(sweep) > 0.01) {
      arc(
        ctx,
        cx,
        cy,
        r + 8,
        top,
        top + sweep,
        4,
        Math.abs(wheelDeg) > 300 ? color.warn : color.ok,
      );
    }
    // Centre reference pip.
    ctx.save();
    ctx.fillStyle = color.textFaint;
    ctx.fillRect(cx - 0.9, cy - r - 13, 1.8, 5);
    ctx.restore();

    /* ---- wheel ---- */
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(rot);

    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.lineWidth = Math.max(4, r * 0.2);
    ctx.strokeStyle = color.text;
    ctx.stroke();

    // Spokes: two at 9 and 3 o'clock, one down.
    ctx.lineWidth = Math.max(3, r * 0.16);
    ctx.lineCap = 'round';
    ctx.strokeStyle = color.text;
    ctx.beginPath();
    ctx.moveTo(-r * 0.86, 0);
    ctx.lineTo(r * 0.86, 0);
    ctx.moveTo(0, 0);
    ctx.lineTo(0, r * 0.86);
    ctx.stroke();

    // Hub.
    ctx.beginPath();
    ctx.arc(0, 0, r * 0.24, 0, Math.PI * 2);
    ctx.fillStyle = color.surface;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color.text;
    ctx.stroke();

    // Top-of-wheel index mark.
    ctx.beginPath();
    ctx.strokeStyle = color.accent;
    ctx.lineWidth = 2.4;
    ctx.moveTo(0, -r * 1.02);
    ctx.lineTo(0, -r * 0.72);
    ctx.stroke();
    ctx.restore();

    /* ---- readout ---- */
    const abs = Math.abs(Math.round(wheelDeg));
    text(ctx, `${abs}°`, cx, height - 14, {
      size: 16,
      weight: 700,
      color: color.text,
      align: 'center',
      baseline: 'bottom',
    });
    const dir = abs < 2 ? 'centred' : wheelDeg > 0 ? 'left' : 'right';
    text(ctx, dir, cx, height - 12, {
      size: 8.5,
      weight: 600,
      color: color.textFaint,
      align: 'center',
      baseline: 'top',
      tracking: 0.4,
    });

    text(ctx, 'L', 8, cy, { size: 9, color: color.textFaint, baseline: 'middle' });
    text(ctx, 'R', width - 8, cy, {
      size: 9,
      color: color.textFaint,
      align: 'right',
      baseline: 'middle',
    });
  });

  return (
    <TelemetryCard title="Steering">
      <canvas ref={ref} />
    </TelemetryCard>
  );
}
