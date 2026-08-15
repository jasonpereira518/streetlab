/**
 * The first consumer of `events[]`.
 *
 * `simStore` has buffered the last 40 `SimEvent`s since Cycle 1
 * (`simStore.ts`'s `applyServerMessage`, `state_update` case) with nothing
 * reading them — which meant a failed `load_location` build (bad geocode,
 * empty Overpass extract, no drivable roads — see the backend's
 * `sim/loop.py` `submit_scene`) was invisible. The search box just quietly
 * re-enables itself on `location_failed` (Task 6); this is where a user
 * finds out *why*.
 *
 * Newest first, because the interesting event is the one that just
 * happened — but note this is arrival order, not a sort by `t`. `t` is
 * sim-seconds and resets to 0 on `reset`/`load_scenario` (and a bare
 * `reset` does not itself clear this buffer — only a fresh
 * `scene_description` does, per `applyServerMessage`'s `scene_description`
 * case), so two unrelated events can legitimately share the same `t` and
 * `code`. That's harmless here only because the `.map()` index is folded
 * into the key below; drop it and duplicate keys become possible.
 *
 * The buffer itself is capped at 40 entries with no truncation indicator —
 * a long enough run of later events can push an unseen one out before a
 * user opens this tab. That's an existing property of the buffer (Task 3/4),
 * not something introduced or fixed here.
 */
import { useSimStore } from '../store/simStore';
import { Field } from './controls';

export function EventLog() {
  const events = useSimStore((s) => s.events);

  return (
    <Field title="Events">
      {events.length === 0 ? (
        <p className="panel-empty">No events yet.</p>
      ) : (
        <ul className="event-log" role="list">
          {[...events].reverse().map((e, i) => (
            <li key={`${e.t}-${e.code}-${i}`} className={`event event-${e.level}`}>
              <span className="event-t">{e.t.toFixed(1)}s</span>
              <span className="event-code">{e.code}</span>
              <span className="event-msg">{e.message}</span>
            </li>
          ))}
        </ul>
      )}
    </Field>
  );
}
