/**
 * Vehicle schematic with a per-corner tyre readout and a subsystem list. The
 * overall verdict is a green tick badge on the car when everything is nominal.
 */
import { useTelemetryCanvas } from '../../store/hooks';
import { TelemetryCard } from '../controls';
import { color } from '../theme';
import { dot, line, placeholder, roundRect, text } from './draw';

const STATUS_COLOR = {
  ok: color.ok,
  warn: color.warn,
  fault: color.danger,
} as const;

export function VehicleStatus() {
  const ref = useTelemetryCanvas(({ ctx, width, height, frame }) => {
    if (!frame) return placeholder(ctx, width, height);
    const v = frame.telemetry.vehicle;

    /* ---- car schematic, left third ---- */
    const carW = Math.min(38, width * 0.24);
    const carH = carW * 2.05;
    const cx = Math.max(carW * 0.9, width * 0.2);
    const cy = height / 2 - 3;

    ctx.save();
    roundRect(ctx, cx - carW / 2, cy - carH / 2, carW, carH, carW * 0.34);
    ctx.fillStyle = color.surfaceSunken;
    ctx.fill();
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = color.borderStrong;
    ctx.stroke();
    ctx.restore();

    // Windscreen hint.
    ctx.save();
    roundRect(ctx, cx - carW * 0.3, cy - carH * 0.26, carW * 0.6, carH * 0.3, 4);
    ctx.fillStyle = color.border;
    ctx.fill();
    ctx.restore();

    // Tyres, coloured by pressure deviation from 250 kPa.
    const corners: Array<[number, number]> = [
      [cx - carW / 2, cy - carH * 0.27],
      [cx + carW / 2, cy - carH * 0.27],
      [cx - carW / 2, cy + carH * 0.27],
      [cx + carW / 2, cy + carH * 0.27],
    ];
    v.tire_pressure_kpa.forEach((kpa, i) => {
      const bad = Math.abs(kpa - 250) > 25;
      const [tx, ty] = corners[i];
      ctx.save();
      roundRect(ctx, tx - 3, ty - 7, 6, 14, 3);
      ctx.fillStyle = bad ? color.warn : color.textMuted;
      ctx.fill();
      ctx.restore();
    });

    // Verdict badge.
    const bx = cx + carW * 0.42;
    const by = cy + carH * 0.42;
    const tone = STATUS_COLOR[v.overall];
    ctx.save();
    ctx.beginPath();
    ctx.arc(bx, by, 9.5, 0, Math.PI * 2);
    ctx.fillStyle = tone;
    ctx.fill();
    ctx.strokeStyle = color.surface;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (v.overall === 'ok') {
      ctx.moveTo(bx - 4, by);
      ctx.lineTo(bx - 1.2, by + 3);
      ctx.lineTo(bx + 4.2, by - 3.4);
    } else {
      ctx.moveTo(bx, by - 4);
      ctx.lineTo(bx, by + 0.6);
      ctx.moveTo(bx, by + 3.4);
      ctx.lineTo(bx, by + 3.6);
    }
    ctx.stroke();
    ctx.restore();

    /* ---- subsystem list, right side ---- */
    const listX = cx + carW * 0.9 + 10;
    const rows = v.subsystems.slice(0, 5);
    const rowH = Math.min(15, (height - 26) / Math.max(1, rows.length));
    const top = (height - rowH * rows.length) / 2 + rowH / 2 - 6;

    rows.forEach((s, i) => {
      const y = top + i * rowH;
      dot(ctx, listX + 3, y, 3, STATUS_COLOR[s.status]);
      text(ctx, s.label, listX + 11, y, {
        size: 10,
        weight: 500,
        color: s.status === 'ok' ? color.textMuted : STATUS_COLOR[s.status],
        baseline: 'middle',
      });
    });

    /* ---- battery bar ---- */
    const barY = height - 11;
    const barX = listX + 3;
    const barW = width - barX - 10;
    line(ctx, barX, barY, barX + barW, barY, color.surfaceSunken, 5);
    line(
      ctx,
      barX,
      barY,
      barX + (barW * v.battery_pct) / 100,
      barY,
      v.battery_pct < 15 ? color.warn : color.ok,
      5,
    );
    text(ctx, `${Math.round(v.battery_pct)}%`, barX, barY - 8, {
      size: 9,
      weight: 600,
      color: color.textMuted,
      baseline: 'bottom',
    });
    text(ctx, `${Math.round(v.range_km)} km`, barX + barW, barY - 8, {
      size: 9,
      color: color.textFaint,
      align: 'right',
      baseline: 'bottom',
    });
  });

  return (
    <TelemetryCard title="Vehicle">
      <canvas ref={ref} />
    </TelemetryCard>
  );
}
