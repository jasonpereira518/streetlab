/**
 * The fold-away control for one shell surface. It lives on the seam between
 * the viewport and the panel it governs — the same spot whether that panel is
 * open or collapsed — so the way back is always where the way out was. The
 * chevron points the direction the panel will travel.
 */
import type { PanelId } from '../store/simStore';
import { useSimStore } from '../store/simStore';
import {
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronUpIcon,
} from './Icons';

type Edge = 'left' | 'right' | 'bottom';

const LABELS: Record<PanelId, string> = {
  scenarios: 'scenario sidebar',
  inspector: 'inspector panel',
  telemetry: 'telemetry row',
};

export function PanelHandle({ panel, edge }: { panel: PanelId; edge: Edge }) {
  const collapsed = useSimStore((s) => s.collapsed[panel]);
  const toggle = useSimStore((s) => s.togglePanel);

  // Open, the chevron points outward (fold it away); collapsed, inward.
  const Icon =
    edge === 'bottom'
      ? collapsed
        ? ChevronUpIcon
        : ChevronDownIcon
      : (edge === 'left') === !collapsed
        ? ChevronLeftIcon
        : ChevronRightIcon;

  const label = `${collapsed ? 'Show' : 'Hide'} ${LABELS[panel]}`;

  return (
    <button
      type="button"
      className={`rail rail--${edge}`}
      onClick={() => toggle(panel)}
      aria-expanded={!collapsed}
      aria-label={label}
      title={label}
      data-testid={`toggle-${panel}`}
    >
      <Icon size={14} />
    </button>
  );
}
