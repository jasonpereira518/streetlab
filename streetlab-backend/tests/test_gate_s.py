"""The Gate S criteria, applied to hand-built rows with known answers.

The gate is mechanical: these tests pin that each criterion fails when (and only
when) the spec's condition is violated, so a result cannot pass by a reading
nobody wrote down.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import gate_s  # noqa: E402


def row(variant, *, hazard=None, key="grid", seed=1, collisions=0, aeb=0, p5=1.5, degraded=0.0,
        progress=100.0, staged=None, reacted=False, budget=None, events=()):
    staged = (hazard is not None) if staged is None else staged
    return {
        "key": key, "seed": seed, "hazard": hazard, "variant": variant,
        "label": f"{key}/seed{seed}/{hazard or 'free'}/{variant}",
        "staged": staged,
        "collisions": collisions, "collision_events": list(events), "aeb_activations": aeb,
        "time_gap_p5_s": p5, "degraded_share": degraded, "route_progress_m": progress,
        "hazard_reactions": {hazard: {"injected": 1, "reacted": int(reacted)}} if staged and hazard else {},
        "budget": budget if budget is not None else {"ego_jerk": None, "nose_gap": "fails"},
    }


def rows_of(*rs):
    return {(r["key"], r["seed"], r["hazard"], r["variant"]): r for r in rs}


def pair(**kw):
    gt = {k: v for k, v in kw.pop("gt", {}).items()}
    return [row("gt", **{**kw, **gt}), row("noisy", **kw), row("stress", **kw)]


def test_a_clean_matrix_passes_everything():
    rows = rows_of(*pair(), *pair(hazard="cut_in", reacted=True))
    assert gate_s.verdict(rows)["pass"]


def test_one_collision_fails_criterion_1_even_if_ground_truth_has_it_too():
    ev = {"t": 5.0, "agent": "a1", "ego_speed_mps": 3.0, "bearing_deg": 100.0,
          "bearing_1s_before_deg": 105.0, "dist_1s_before_m": 4.0}
    rows = rows_of(*pair(hazard="cut_in", reacted=True), *[
        row("gt", hazard="tailgater", reacted=True, collisions=1, events=[ev]),
        row("noisy", hazard="tailgater", reacted=True, collisions=1, events=[ev]),
        row("stress", hazard="tailgater", reacted=True),
    ])
    res = gate_s.verdict(rows)
    assert not res["1"]["pass"] and res["1"]["total"] == 1 and res["1"]["gt_total"] == 1
    assert res["1"]["events"][0]["gt_paired_collisions"] == 1
    assert not res["pass"]


def test_stress_collisions_fail_the_gate_but_no_other_stress_number_does():
    rows = rows_of(*[
        row("gt"), row("noisy"), row("stress", collisions=1, aeb=9, degraded=0.9, progress=1.0),
    ])
    res = gate_s.verdict(rows)
    assert all(res[k]["pass"] for k in "1234567")
    assert not res["stress_1"]["pass"] and not res["pass"]


def test_reactions_are_judged_only_where_ground_truth_reacted():
    never = rows_of(*[row("gt", hazard="cut_in", reacted=False), row("noisy", hazard="cut_in", reacted=False),
                      row("stress", hazard="cut_in")])
    assert gate_s.c2(never)["pass"], "GT did not react, so ML owes nothing"

    cells = []
    for seed in range(1, 6):  # GT reacts 5/5, ML 3/5 = 0.6
        cells += [row("gt", hazard="cut_in", seed=seed, reacted=True),
                  row("noisy", hazard="cut_in", seed=seed, reacted=seed <= 3)]
    res = gate_s.c2(rows_of(*cells))
    assert not res["pass"] and res["per_hazard"]["cut_in"]["fraction"] == 0.6
    assert res["per_hazard"]["cut_in"]["misses"] == ["grid/seed4", "grid/seed5"]


def test_four_of_five_is_exactly_the_threshold():
    cells = []
    for seed in range(1, 6):
        cells += [row("gt", hazard="cut_in", seed=seed, reacted=True),
                  row("noisy", hazard="cut_in", seed=seed, reacted=seed != 5)]
    assert gate_s.c2(rows_of(*cells))["pass"]


def test_an_ml_run_that_never_staged_the_hazard_is_excluded_and_counted():
    cells = [row("gt", hazard="cut_in", reacted=True), row("noisy", hazard="cut_in", staged=False)]
    res = gate_s.c2(rows_of(*cells))
    assert res["pass"] and res["per_hazard"]["cut_in"]["ml_not_staged"] == 1


def test_any_aeb_in_a_hazard_free_run_fails_but_in_a_hazard_run_does_not():
    assert not gate_s.c3(rows_of(*[row("gt"), row("noisy", aeb=1)]))["pass"]
    assert gate_s.c3(rows_of(*[row("gt", hazard="cut_in", reacted=True),
                               row("noisy", hazard="cut_in", reacted=True, aeb=3)]))["pass"]


def test_a_budget_the_noisy_run_breaks_that_gt_meets_fails_criterion_4():
    gt = row("gt", budget={"ego_jerk": None, "nose_gap": "x"})
    ok = row("noisy", budget={"ego_jerk": None, "nose_gap": "y"})
    bad = row("noisy", budget={"ego_jerk": "jerk p99 9", "nose_gap": "y"})
    assert gate_s.c4(rows_of(gt, ok))["pass"], "GT fails nose_gap, so it is not required"
    res = gate_s.c4(rows_of(gt, bad))
    assert not res["pass"] and "ego_jerk" in res["failures"][0]


def test_time_gap_slack_is_three_tenths_of_a_second():
    assert gate_s.c5(rows_of(*[row("gt", p5=1.5), row("noisy", p5=1.21)]))["pass"]
    assert not gate_s.c5(rows_of(*[row("gt", p5=1.5), row("noisy", p5=1.19)]))["pass"]
    assert gate_s.c5(rows_of(*[row("gt", p5=None), row("noisy", p5=0.1)]))["pass"], "no GT gap, nothing to compare"


def test_degraded_share_is_capped_at_five_percent_per_run():
    assert gate_s.c6(rows_of(*[row("gt"), row("noisy", degraded=0.05)]))["pass"]
    assert not gate_s.c6(rows_of(*[row("gt"), row("noisy", degraded=0.051)]))["pass"]


def test_crawling_fails_the_progress_criterion():
    assert gate_s.c7(rows_of(*[row("gt", progress=100), row("noisy", progress=90)]))["pass"]
    assert not gate_s.c7(rows_of(*[row("gt", progress=100), row("noisy", progress=89)]))["pass"]
