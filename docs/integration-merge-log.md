# Integration merge log (claude/integration-all)

A combined view of the stacked, still-open PRs, built from `origin/main` (70472b1, after #34).
Nothing here is merged to main. Merge order and what each step needed:

| # | PR / branch | Result |
|---|---|---|
| 1 | #24 `p0-green-baseline` | clean |
| 2 | #25 `p2-location-routing` | clean |
| 3 | #29 `p3-map-render` | clean |
| 4 | #26 `p5-hosted-deploy` | conflicts in `fly.toml` and `server/ws_server.py` (below) |
| 5 | #28 `m0-ml-foundation` | clean (moves `tests/driving_metrics.py` to `evaluation/`) |
| 6 | #27 `p4-ui-shell` | conflict in `App.tsx`: kept both the connection-error overlay and the shortcuts/toast/help |
| 7 | #30 `h-hazard-reactions` | contract fixtures `state_update_events/hazard.json` conflicted: regenerated at the end |
| 8 | #33 `p1-driving-realism-p3` | `driving_metrics` conflicts (below); imports/comments repointed to `evaluation/` |
| 9 | #31 `m2-renderer-gate1` | clean |
| 10 | #32 `m1-ml-stack` | `driving_metrics.py` leftovers (took m1's version, the emergency field is set once), fixtures regenerated |
| 11 | `portfolio-credibility-plan-cbca8d` (CI, release, README, ARCHITECTURE) | merged rather than rebased (a rebase onto an unmerged stack would rewrite the branch). Conflicts: `CLAUDE.md` (kept main's, added four traps), `README.md` (took the branch's front door, then rewrote Results/Hosted/roadmap), `App.tsx` and `styles.css` (kept both). CI: removed the temporary feature-branch trigger. |

`PROTOCOL_VERSION` is 11 on both sides (from #32); contract fixtures regenerated with
`uv run pytest ../contract --update-fixtures`, and the contract suite passes on the Python and TypeScript sides.

## Resolutions that needed a judgement call

- **`streetlab-backend/fly.toml`** (Jason to confirm). Main's #34 chose scale-to-zero
  (`auto_stop_machines = "suspend"`, `min_machines_running = 0`); #26 chose an always-on machine
  because a stopped machine drops live sessions. I kept main's merged cost decision and kept #26's
  `STREETLAB_MAX_SESSIONS = "2"`; the comment now says so. If you want #26's always-on policy, flip those
  three lines back.
- **`server/ws_server.py`**: took #26's per-connection `SimLoop` and `_run_connection`, and carried #25's
  `conn._suggest_task.cancel()` into `_run_connection` so a pending address-suggest task is cancelled on disconnect.
- **`evaluation/driving_metrics.py`**: #33 added the `emergency` field and a `nobhill_slow` run; #28 moved the file and
  added `make_sim`/`DEFAULT_SEEDS` and the closed-loop fields. Merged: both field sets, `RUN_KEYS` has four keys,
  `make_sim` handles `nobhill_slow` (traffic scale 0.4), `DEFAULT_SEEDS` has it at seed 1.
- **`tests/test_driving_budgets.py`**: imports `Simulation` and `evaluation.driving_metrics`.

## Added on top of the merges

- `tests/test_detector_cameras.py::test_default_ground_truth_app_builds_no_detector_pipeline_so_no_extra_cameras`.
- README Results, DEMO.md, `docs/ARCHITECTURE.md` roadmap and the in-app help text now state the ML gate results.
