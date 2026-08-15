/**
 * Saved-scenario browser. Each card carries a 2D mini-map thumbnail rendered
 * from the scenario's own `preview_paths`/`preview_route` — the catalog is data
 * from the server, so the sidebar has no knowledge of the mock.
 */
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { ScenarioSummary } from '../schema';
import { useSimStore } from '../store/simStore';
import { BookmarkIcon, FolderIcon, PlayIcon, PlusIcon } from './Icons';
import { alpha, color } from './theme';

export function LeftScenarioSidebar() {
  const catalog = useSimStore((s) => s.catalog);
  const activeId = useSimStore((s) => s.activeScenarioId);
  const location = useSimStore((s) => s.scene?.location ?? '—');
  const attribution = useSimStore((s) => s.scene?.attribution);
  const loadScenario = useSimStore((s) => s.loadScenario);
  const loadLocation = useSimStore((s) => s.loadLocation);
  const locationPending = useSimStore((s) => s.locationPending);
  const [bookmarks, setBookmarks] = useState<Record<string, boolean>>({});
  const [query, setQuery] = useState('');

  const isBookmarked = (s: ScenarioSummary) => bookmarks[s.id] ?? s.bookmarked;

  return (
    <aside className="sidebar" aria-label="Scenarios">
      <header className="sidebar-head">
        <span className="sidebar-eyebrow">Location</span>
        <h2 className="sidebar-title">{location}</h2>
        <p className="sidebar-sub">
          {catalog.length} saved scenario{catalog.length === 1 ? '' : 's'}
        </p>
        {/*
          ODbL requires crediting OpenStreetMap wherever its data is shown;
          `attribution` is never omitted from the wire shape (SyntheticGrid
          fills it with "Synthetic scene — no map data" for scenes that
          carry no real map data at all), so this shows verbatim rather than
          being wrapped in OSM-specific copy that would misdescribe a
          synthetic scene. The `&&` guard also keeps an empty string (a
          defensive case the schema technically allows) from leaving a
          stray, blank <p> in the DOM.
        */}
        {attribution && <p className="scene-attribution">{attribution}</p>}
      </header>

      <form
        className="location-search"
        onSubmit={(e) => {
          e.preventDefault();
          loadLocation(query);
          setQuery('');
        }}
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Address or place…"
          aria-label="Load a location"
          disabled={locationPending !== null}
        />
        {locationPending !== null && (
          <p className="location-pending">Building {locationPending}…</p>
        )}
      </form>

      <div className="scenario-list" role="list">
        {catalog.length === 0 && (
          <p className="sidebar-empty">Waiting for the scenario catalog…</p>
        )}
        {catalog.map((s) => (
          <article
            key={s.id}
            role="listitem"
            className={`scenario${s.id === activeId ? ' is-active' : ''}`}
          >
            <ScenarioThumb scenario={s} active={s.id === activeId} />
            <div className="scenario-body">
              <div className="scenario-line">
                <span className="scenario-index">
                  {String(s.index).padStart(2, '0')}
                </span>
                <h3 className="scenario-name" title={s.name}>
                  {s.name}
                </h3>
              </div>
              <p className="scenario-desc" title={s.description}>
                {s.description}
              </p>
              <div className="scenario-meta">
                <span className={`difficulty difficulty--${s.difficulty}`}>
                  {s.difficulty}
                </span>
                <span className="scenario-dur">
                  {Math.round(s.duration_s / 60)} min
                </span>
              </div>
            </div>
            <div className="scenario-actions">
              <button
                type="button"
                className={`ghost-btn${isBookmarked(s) ? ' is-marked' : ''}`}
                onClick={() =>
                  setBookmarks((b) => ({ ...b, [s.id]: !isBookmarked(s) }))
                }
                aria-label={`${isBookmarked(s) ? 'Remove' : 'Add'} bookmark for ${s.name}`}
                aria-pressed={isBookmarked(s)}
                title="Bookmark"
              >
                <BookmarkIcon size={15} filled={isBookmarked(s)} />
              </button>
              <button
                type="button"
                className="play-btn"
                onClick={() => loadScenario(s.id)}
                aria-label={`Load ${s.name}`}
                title="Load scenario"
              >
                <PlayIcon size={13} />
              </button>
            </div>
          </article>
        ))}
      </div>

      <footer className="sidebar-foot">
        <button type="button" className="foot-btn" title="New scenario">
          <PlusIcon size={16} />
          <span>New</span>
        </button>
        <button type="button" className="foot-btn" title="Open folder">
          <FolderIcon size={16} />
          <span>Open</span>
        </button>
      </footer>
    </aside>
  );
}

/* ------------------------------------------------------------------ */

/**
 * Static mini-map. Preview geometry lives in a 0..100 box; the canvas fits it
 * with a small margin, so scenarios of different shapes all read at the same
 * visual weight.
 */
function ScenarioThumb({
  scenario,
  active,
}: {
  scenario: ScenarioSummary;
  active: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useLayoutEffect(() => {
    const canvas = ref.current;
    const parent = canvas?.parentElement;
    if (!canvas || !parent) return;
    const measure = () => {
      const r = parent.getBoundingClientRect();
      setSize({ w: Math.round(r.width), h: Math.round(r.height) });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(parent);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || size.w === 0 || size.h === 0) return;
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    canvas.width = Math.round(size.w * dpr);
    canvas.height = Math.round(size.h * dpr);
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${size.h}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const M = 7;
    const scale = Math.min((size.w - M * 2) / 100, (size.h - M * 2) / 100);
    const ox = (size.w - 100 * scale) / 2;
    const oy = (size.h - 100 * scale) / 2;
    const px = (x: number) => ox + x * scale;
    const py = (y: number) => oy + (100 - y) * scale;

    ctx.fillStyle = active ? color.accentSoft : color.surfaceSunken;
    ctx.fillRect(0, 0, size.w, size.h);

    ctx.lineCap = 'round';
    ctx.strokeStyle = active ? alpha(color.accentDark, 0.35) : color.borderStrong;
    ctx.lineWidth = 2.4;
    for (const path of scenario.preview_paths) {
      if (path.length < 2) continue;
      ctx.beginPath();
      path.forEach(([x, y], i) => (i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))));
      ctx.stroke();
    }

    const route = scenario.preview_route;
    if (route.length >= 2) {
      ctx.strokeStyle = active ? color.accent : color.plan;
      ctx.lineWidth = 2.2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      route.forEach(([x, y], i) => (i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))));
      ctx.stroke();

      const [sx, sy] = route[0];
      ctx.beginPath();
      ctx.arc(px(sx), py(sy), 2.6, 0, Math.PI * 2);
      ctx.fillStyle = color.ok;
      ctx.fill();
    }
  }, [scenario, active, size]);

  return (
    <div className="scenario-thumb">
      <canvas ref={ref} aria-label={`${scenario.name} mini-map`} role="img" />
    </div>
  );
}
