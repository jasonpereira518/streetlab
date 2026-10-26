/**
 * The driver-attention prompt (`attention.ts`), over the viewport where the
 * driver is looking. Tesla's wording; flashing once escalated, as Tesla's
 * banner does, unless the user prefers reduced motion.
 */
import { useSimStore } from '../store/simStore';

const MESSAGES = {
  prompt: 'Please pay attention to the road',
  escalated: 'Please pay attention to the road',
  stopping: 'Driver attention warnings ignored. Steer or brake to take over.',
} as const;

export function AttentionBanner() {
  const stage = useSimStore((s) => s.attention.stage);
  if (stage === 'ok') return null;
  return (
    <div className={`attention-banner attention-banner--${stage}`} role="alert" data-testid="attention-banner">
      {MESSAGES[stage]}
    </div>
  );
}
