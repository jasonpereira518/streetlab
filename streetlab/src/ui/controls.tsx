/**
 * Small styled primitives shared by the toolbar and the right-hand panel.
 * Every control is uncontrolled-free: value in, change out, no local state.
 */
import type { ReactNode } from 'react';
import { useId } from 'react';

/* ------------------------------------------------------------------ */

export function IconButton({
  label,
  onClick,
  active,
  disabled,
  tone = 'neutral',
  children,
}: {
  label: string;
  onClick?: () => void;
  active?: boolean;
  disabled?: boolean;
  tone?: 'neutral' | 'accent';
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      className={`icon-btn icon-btn--${tone}${active ? ' is-active' : ''}`}
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      aria-pressed={active}
      title={label}
    >
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------ */

export function Toggle({
  label,
  checked,
  onChange,
  hint,
}: {
  label: string;
  checked: boolean;
  onChange: (next: boolean) => void;
  hint?: string;
}) {
  const id = useId();
  return (
    <div className="ctl ctl--toggle">
      <label className="ctl-label" htmlFor={id}>
        {label}
        {hint && <span className="ctl-hint">{hint}</span>}
      </label>
      <button
        type="button"
        id={id}
        role="switch"
        aria-checked={checked}
        aria-label={label}
        className={`switch${checked ? ' is-on' : ''}`}
        onClick={() => onChange(!checked)}
      >
        <span className="switch-knob" />
      </button>
    </div>
  );
}

/* ------------------------------------------------------------------ */

export function Slider({
  label,
  value,
  min,
  max,
  step,
  unit,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  hint?: string;
  onChange: (next: number) => void;
}) {
  const id = useId();
  const pct = ((value - min) / (max - min)) * 100;
  const decimals = step < 1 ? String(step).split('.')[1]?.length ?? 1 : 0;
  return (
    <div className="ctl ctl--slider">
      <div className="ctl-row">
        <label className="ctl-label" htmlFor={id}>
          {label}
          {hint && <span className="ctl-hint">{hint}</span>}
        </label>
        <output className="ctl-value" htmlFor={id}>
          {value.toFixed(decimals)}
          {unit && <span className="ctl-unit">{unit}</span>}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ ['--fill' as string]: `${pct}%` }}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ */

export function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (next: string) => void;
}) {
  const id = useId();
  return (
    <div className="ctl ctl--select">
      <label className="ctl-label" htmlFor={id}>
        {label}
      </label>
      <div className="select-wrap">
        <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

const SWATCHES = ['#FF7A1A', '#E5484D', '#F5A524', '#22C55E', '#0FB5C9', '#2F80ED'];

export function ColorPicker({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (next: string) => void;
}) {
  const id = useId();
  return (
    <div className="ctl ctl--color">
      <div className="ctl-row">
        <label className="ctl-label" htmlFor={id}>
          {label}
        </label>
        <span className="ctl-value ctl-value--mono">{value.toUpperCase()}</span>
      </div>
      <div className="color-row">
        <input
          id={id}
          type="color"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          aria-label={`${label} custom colour`}
        />
        {SWATCHES.map((c) => (
          <button
            key={c}
            type="button"
            className={`swatch${c.toLowerCase() === value.toLowerCase() ? ' is-active' : ''}`}
            style={{ background: c }}
            onClick={() => onChange(c)}
            aria-label={`${label} ${c}`}
            title={c}
          />
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

export function Field({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="field">
      <h3 className="field-title">{title}</h3>
      <div className="field-body">{children}</div>
    </section>
  );
}

/** Bottom-row telemetry card shell: title, optional badge, canvas area. */
export function TelemetryCard({
  title,
  badge,
  badgeTone = 'neutral',
  children,
}: {
  title: string;
  badge?: ReactNode;
  badgeTone?: 'neutral' | 'ok' | 'warn' | 'accent';
  children: ReactNode;
}) {
  return (
    <article className="tcard">
      <header className="tcard-head">
        <h4>{title}</h4>
        {badge != null && <span className={`tbadge tbadge--${badgeTone}`}>{badge}</span>}
      </header>
      <div className="tcard-body">{children}</div>
    </article>
  );
}
