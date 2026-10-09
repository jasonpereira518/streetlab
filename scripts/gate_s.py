#!/usr/bin/env python3
"""Gate S of the ML driving spec (section 5e): run the matrix, apply the criteria mechanically.

    docs/superpowers/specs/2026-10-04-streetlab-ml-driving-design.md

Two steps, so a long run can be resumed and a report re-cut without re-running:

    cd streetlab-backend
    nohup uv run python ../scripts/gate_s.py run --out /path/cells.jsonl --jobs 6 &
    uv run python ../scripts/gate_s.py report --cells /path/cells.jsonl --write

`run`: for scene in {nobhill, grid, grid_slow} x seed 1-5 x {hazard-free, each Cycle 6
hazard} x {ground truth, noisy-truth nominal, noisy-truth stress}. Hazard-free runs
last the budgets' own length (340 / 150 / 200 s); hazard runs last 120 s, the hazard
asked for from 15 s while the ego is doing 4 m/s, until 105 s (a declined staging is
retried). Every cell is one JSON line; a cell already in the file is skipped.

`report` applies the seven criteria exactly as the spec words them. Where the wording
leaves a reading open, the reading is fixed HERE, before any result was looked at, and
the strict one is taken:

  1. zero collisions in every nominal-noise run (all scenes, seeds, hazards). Each
     collision is listed with whether the paired ground-truth run also had one and the
     bearing of the other vehicle -- attribution is reported, the criterion is not softened.
  2. per hazard type, over pairs where GT staged it AND reacted (M0's definition: a named
     reaction source or a stop/yield/AEB maneuver within 8 s): ML reacted on >= 80 %.
     Pairs where the ML run never got the hazard staged are excluded and counted.
  3. zero AEB activations in every hazard-free nominal run.
  4. for every hazard-free pair: each budget (`budget_failures`, the suite's own checks)
     that GT meets, the noisy run meets.
  5. for every pair (hazard runs included): noisy 5th-percentile time gap >= GT's - 0.3 s.
     A pair where either run never had a lead while moving has no gap to compare.
  6. degraded share <= 5 % of frames in every nominal run.
  7. noisy route progress >= 90 % of GT's in every pair (same duration by construction).

At stress noise only criterion 1 is binding; the others are reported.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "streetlab-backend"))
sys.path.insert(0, str(REPO / "scripts"))

KEYS = ("nobhill", "grid", "nobhill_slow", "grid_slow")
SEEDS = (1, 2, 3, 4, 5)
VARIANTS = ("gt", "noisy", "stress")
HAZARD_RUN_S = 120.0
STAGE_FROM_S, STAGE_UNTIL_S = 15.0, 105.0
STANDARD_S = {"nobhill": 340.0, "nobhill_slow": 340.0, "grid": 150.0, "grid_slow": 200.0}
GAP_SLACK_S = 0.3
REACT_FRACTION = 0.80
DEGRADED_MAX = 0.05
PROGRESS_MIN = 0.90

_NOB = None


def _scene():
    global _NOB
    if _NOB is None:
        from driving_baseline import _nob_hill_scene

        _NOB = _nob_hill_scene()
    return _NOB


def hazards() -> list[str]:
    from sim.events import SCENARIOS

    return list(SCENARIOS)


def run_cell(cell: tuple[str, int, str | None, str]) -> dict:
    from evaluation.driving_metrics import budget_failures, closed_loop_summary, make_sim, record
    from perception.noisy_truth import NOMINAL, STRESS, NoisyTruthPerception

    key, seed, hazard, variant = cell
    kwargs = {}
    if variant != "gt":
        npt = NoisyTruthPerception(NOMINAL if variant == "noisy" else STRESS, seed)
        kwargs = {"perception_pipeline": npt.pipeline, "ml_perception": npt}
    sim = make_sim(key, _scene(), seed=seed, **kwargs)
    if variant != "gt":
        sim.perception_mode = "ml"
    label = f"{key}/seed{seed}/{hazard or 'free'}/{variant}"
    run = record(
        sim,
        HAZARD_RUN_S if hazard else STANDARD_S[key],
        label,
        stage=(hazard, STAGE_FROM_S, STAGE_UNTIL_S) if hazard else None,
    )
    out = {
        "key": key, "seed": seed, "hazard": hazard, "variant": variant,
        "staged": bool(run.hazards), "staged_at": run.hazards[0][0] if run.hazards else None,
        **closed_loop_summary(run),
    }
    if hazard is None:
        out["budget"] = budget_failures(run, key)
    return out


def cmd_run(args) -> None:
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["key"], r["seed"], r["hazard"], r["variant"]))
    cells = [
        (k, s, h, v)
        for k in args.keys for s in args.seeds for h in [None, *hazards()] for v in VARIANTS
        if (k, s, h, v) not in done
    ]
    # The binding variants first (stress is only ever checked for collisions), and
    # longest first within each so the pool does not end on a lone Nob Hill run.
    cells.sort(key=lambda c: (c[3] == "stress",
                              -(STANDARD_S[c[0]] if c[2] is None else HAZARD_RUN_S) * (3 if c[0].startswith("nobhill") else 1)))
    print(f"{len(cells)} cells to run, {len(done)} done", flush=True)
    with ProcessPoolExecutor(args.jobs) as pool, out.open("a") as fh:
        futures = {pool.submit(run_cell, c): c for c in cells}
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[{n}/{len(cells)}] {row['label']}", flush=True)


# -- the criteria --------------------------------------------------------------- #


def _load(path: str) -> dict[tuple, dict]:
    rows = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        rows[(r["key"], r["seed"], r["hazard"], r["variant"])] = r
    return rows


def _pairs(rows, variant):
    for (k, s, h, v), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")):
        if v == variant and (k, s, h, "gt") in rows:
            yield r, rows[(k, s, h, "gt")]


def _reacted(r: dict, hazard: str) -> bool | None:
    """Whether the run staged `hazard` and reacted to it; None if it was never staged."""
    if not r["staged"]:
        return None
    hz = r["hazard_reactions"].get(hazard)
    return bool(hz and hz["reacted"])


def c1(rows, variant) -> dict:
    items = []
    for r, g in _pairs(rows, variant):
        for ev in r["collision_events"]:
            items.append(
                {
                    "run": r["label"], "t": ev["t"], "agent": ev["agent"],
                    "gt_paired_collisions": g["collisions"],
                    "bearing_deg": ev["bearing_deg"], "bearing_1s_before_deg": ev["bearing_1s_before_deg"],
                    "ego_speed_mps": ev["ego_speed_mps"],
                }
            )
    total = sum(r["collisions"] for r, _ in _pairs(rows, variant))
    return {"pass": total == 0, "total": total, "events": items,
            "gt_total": sum(g["collisions"] for _, g in _pairs(rows, variant))}


def c2(rows) -> dict:
    per = {}
    for r, g in _pairs(rows, "noisy"):
        h = r["hazard"]
        if h is None:
            continue
        gt_r = _reacted(g, h)
        if not gt_r:
            continue
        e = per.setdefault(h, {"gt_reacted": 0, "ml_reacted": 0, "ml_not_staged": 0, "misses": []})
        e["gt_reacted"] += 1
        ml = _reacted(r, h)
        if ml is None:
            e["ml_not_staged"] += 1
        elif ml:
            e["ml_reacted"] += 1
        else:
            e["misses"].append(f"{r['key']}/seed{r['seed']}")
    ok = True
    for h, e in per.items():
        n = e["gt_reacted"] - e["ml_not_staged"]
        e["fraction"] = e["ml_reacted"] / n if n else None
        e["pass"] = n == 0 or e["fraction"] >= REACT_FRACTION
        ok &= e["pass"]
    return {"pass": ok, "per_hazard": per}


def c3(rows) -> dict:
    items = [(r["label"], r["aeb_activations"]) for r, _ in _pairs(rows, "noisy") if r["hazard"] is None]
    gt = {r["label"]: r["aeb_activations"] for (_, _, h, v), r in rows.items() if v == "gt" and h is None}
    return {"pass": all(n == 0 for _, n in items), "per_run": items, "gt_total": sum(gt.values())}


def c4(rows, variant="noisy") -> dict:
    fails = []
    n = 0
    for r, g in _pairs(rows, variant):
        if r["hazard"] is not None:
            continue
        for name, why in g["budget"].items():
            if why is None:
                n += 1
                if r["budget"].get(name) is not None:
                    fails.append(f"{r['key']}/seed{r['seed']}: {name}: {r['budget'][name]}")
    return {"pass": not fails, "checked": n, "failures": fails}


def c5(rows) -> dict:
    items, ok = [], True
    for r, g in _pairs(rows, "noisy"):
        a, b = r["time_gap_p5_s"], g["time_gap_p5_s"]
        if a is None or b is None:
            items.append((r["label"], a, b, None))
            continue
        passed = a >= b - GAP_SLACK_S
        ok &= passed
        items.append((r["label"], a, b, passed))
    return {"pass": ok, "items": items}


def c6(rows) -> dict:
    items = [(r["label"], r["degraded_share"]) for r, _ in _pairs(rows, "noisy")]
    return {"pass": all(s <= DEGRADED_MAX for _, s in items), "items": items}


def c7(rows) -> dict:
    items, ok = [], True
    for r, g in _pairs(rows, "noisy"):
        if g["route_progress_m"] <= 0:
            continue
        ratio = r["route_progress_m"] / g["route_progress_m"]
        passed = ratio >= PROGRESS_MIN
        ok &= passed
        items.append((r["label"], r["route_progress_m"], g["route_progress_m"], ratio, passed))
    return {"pass": ok, "items": items}


def verdict(rows) -> dict:
    res = {"1": c1(rows, "noisy"), "2": c2(rows), "3": c3(rows), "4": c4(rows),
           "5": c5(rows), "6": c6(rows), "7": c7(rows)}
    res["stress_1"] = c1(rows, "stress")
    res["pass"] = all(res[k]["pass"] for k in "1234567") and res["stress_1"]["pass"]
    return res


# -- report -------------------------------------------------------------------- #


def _f(v, n=2):
    return "-" if v is None else f"{v:.{n}f}"


def render(rows, res, meta) -> str:
    L = []
    P = lambda b: "PASS" if b else "**FAIL**"  # noqa: E731
    L += [f"# Gate S: noisy-truth perception against ground truth ({meta['date']})", ""]
    L += [
        f"Run by `scripts/gate_s.py` at `{meta['sha']}`. {len(rows)} runs: {len(KEYS)} scenes x "
        f"{len(SEEDS)} seeds x (1 hazard-free + {len(hazards())} hazards) x (ground truth, noisy nominal, noisy stress).",
        "",
        "Criteria are the spec's (section 5e), fixed before any run; readings that the wording leaves open "
        "are fixed in the script's docstring and are the strict ones. Nothing here was tuned to the gate.",
        "",
        f"## Verdict: {'PASS' if res['pass'] else 'FAIL'}", "",
        "| # | criterion | result | detail |", "|---|---|---|---|",
        f"| 1 | zero collisions, nominal noise | {P(res['1']['pass'])} | {res['1']['total']} collisions "
        f"(ground-truth runs on the same seeds: {res['1']['gt_total']}) |",
        f"| 2 | ML reacts on >= 80 % of injections GT reacts to, per hazard | {P(res['2']['pass'])} | "
        f"{sum(1 for e in res['2']['per_hazard'].values() if not e['pass'])} of {len(res['2']['per_hazard'])} hazard types below 80 % |",
        f"| 3 | zero AEB in hazard-free runs | {P(res['3']['pass'])} | {sum(n for _, n in res['3']['per_run'])} activations "
        f"(GT: {res['3']['gt_total']}) |",
        f"| 4 | every budget GT meets, noisy meets | {P(res['4']['pass'])} | {len(res['4']['failures'])} failures of {res['4']['checked']} checks |",
        f"| 5 | p5 time gap >= GT - 0.3 s | {P(res['5']['pass'])} | {sum(1 for i in res['5']['items'] if i[3] is False)} of "
        f"{sum(1 for i in res['5']['items'] if i[3] is not None)} comparable pairs fail |",
        f"| 6 | degraded <= 5 % of time | {P(res['6']['pass'])} | max {_f(max(s for _, s in res['6']['items']), 3)} |",
        f"| 7 | route progress >= 90 % of GT | {P(res['7']['pass'])} | min ratio {_f(min(i[3] for i in res['7']['items']), 3)} |",
        f"| stress | zero collisions at stress noise (only binding stress criterion) | {P(res['stress_1']['pass'])} | "
        f"{res['stress_1']['total']} collisions (GT {res['stress_1']['gt_total']}) |",
        "",
    ]

    L += ["## Criterion 1: collisions", ""]
    for name, key in (("nominal", "1"), ("stress", "stress_1")):
        ev = res[key]["events"]
        L += [f"### {name}: {res[key]['total']} collisions", ""]
        if ev:
            L += ["| run | t (s) | other | GT run collides too | other's bearing at contact | 1 s earlier | ego m/s |",
                  "|---|---|---|---|---|---|---|"]
            for e in ev:
                L.append(
                    f"| {e['run']} | {e['t']:.1f} | {e['agent']} | {'yes (' + str(e['gt_paired_collisions']) + ')' if e['gt_paired_collisions'] else 'no'} "
                    f"| {_f(e['bearing_deg'], 0)} | {_f(e['bearing_1s_before_deg'], 0)} | {e['ego_speed_mps']:.1f} |"
                )
        L.append("")
    L += ["Bearing: the other vehicle's direction from the ego, relative to the ego's heading (0 ahead, + left, 180 behind). "
          "The ground-truth driving feed (`perception/driver_view.py`) sees 75 deg either side ahead and 40 deg behind within 45 m.", ""]

    L += ["## Criterion 2: hazard reactions", "",
          "| hazard | GT staged and reacted | ML staged and reacted | ML never staged (excluded) | fraction | result | ML misses |",
          "|---|---|---|---|---|---|---|"]
    for h, e in sorted(res["2"]["per_hazard"].items()):
        L.append(f"| {h} | {e['gt_reacted']} | {e['ml_reacted']} | {e['ml_not_staged']} | {_f(e['fraction'])} | {P(e['pass'])} | {', '.join(e['misses']) or '-'} |")
    L.append("")

    L += ["## Criterion 3: AEB in hazard-free runs", "", "| run | AEB activations (noisy) | (GT) |", "|---|---|---|"]
    for (lab, n) in res["3"]["per_run"]:
        g = rows[next(k for k in rows if rows[k]["label"] == lab.replace("/noisy", "/gt"))]
        L.append(f"| {lab} | {n} | {g['aeb_activations']} |")
    L.append("")

    L += ["## Criterion 4: budgets GT meets, noisy meets", ""]
    if res["4"]["failures"]:
        L += [f"- {f}" for f in res["4"]["failures"]]
    else:
        L.append(f"None of the {res['4']['checked']} checks failed.")
    L += ["", "Budgets each GT run fails (so are not required of the noisy run):", ""]
    for (k, s, h, v), r in sorted(rows.items()):
        if v == "gt" and h is None:
            bad = sorted(n for n, w in r["budget"].items() if w is not None)
            L.append(f"- {k}/seed{s}: {', '.join(bad) or 'none'}")
    L.append("")

    L += ["## Criterion 5: 5th-percentile time gap to lead (s)", "",
          "| run | noisy | GT | result |", "|---|---|---|---|"]
    for lab, a, b, p in res["5"]["items"]:
        L.append(f"| {lab} | {_f(a)} | {_f(b)} | {'n/a (no lead while moving)' if p is None else P(p)} |")
    L.append("")

    L += ["## Criterion 6: degraded share of frames", "", "| run | share |", "|---|---|"]
    L += [f"| {lab} | {s:.3f} |" for lab, s in res["6"]["items"]]
    L.append("")

    L += ["## Criterion 7: route progress (m)", "", "| run | noisy | GT | ratio | result |", "|---|---|---|---|---|"]
    L += [f"| {lab} | {a:.0f} | {b:.0f} | {r:.3f} | {P(p)} |" for lab, a, b, r, p in res["7"]["items"]]
    L.append("")

    L += ["## Staging", "", "| scene | hazard | GT staged | noisy staged | stress staged |", "|---|---|---|---|---|"]
    for k in KEYS:
        for h in hazards():
            cells = [[rows[(k, s, h, v)]["staged"] for s in SEEDS if (k, s, h, v) in rows] for v in VARIANTS]
            L.append(f"| {k} | {h} | " + " | ".join(f"{sum(c)}/{len(c)}" for c in cells) + " |")
    L.append("")
    return "\n".join(L)


def cmd_report(args) -> None:
    rows = _load(args.cells)
    res = verdict(rows)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    meta = {"date": datetime.date.today().isoformat(), "sha": sha or "unknown"}
    md = render(rows, res, meta)
    print(md)
    if args.write:
        d = REPO / "docs" / "measurements"
        stem = d / f"{meta['date']}-ml-gate-s"
        stem.with_suffix(".md").write_text(md + "\n")
        stem.with_suffix(".json").write_text(json.dumps({"meta": meta, "verdict": res, "runs": list(rows.values())}, indent=1) + "\n")
        print(f"wrote {stem.name}.{{md,json}}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--jobs", type=int, default=6)
    r.add_argument("--keys", nargs="+", default=list(KEYS), choices=KEYS)
    r.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--cells", required=True)
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
