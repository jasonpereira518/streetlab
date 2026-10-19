# Closing the realism plan's known gaps

**Date:** 2026-10-19
**Branch:** `claude/realism-gaps` (stacked on 8b, PR #69)

The runner is written up separately (`2026-10-19-red-light-runner-fair.md`).

## Fixed

| Gap | Cause | Now | Measured |
|---|---|---|---|
| Ego drove onto a crossing with a walker on its far half | It only braked for a walker in its own path | Crossings are ego control points (kind `crosswalk`, with `extent`). The ego holds short of the near edge while any walker is on the crossing. | Ego stops before x = 42 for walkers at y = 0 and y = 5; drives through when the crossing is empty or the walker is on the pavement beyond. Trip length unchanged (grid-loop 0.51 vs 0.52 km). |
| Some traffic right turns skipped right of way | A 6 m fillet passes about 5 m from the junction centre, outside the 4.6 m box, so the turn got no passage | A turn within 4 m of the box counts as passing through it. A straight route that close only passes beside it and still doesn't. | Of the near misses, 1 was a real turn (Nob Hill ambient) and it now has a passage. 6 were straight routes beside a box and are correctly left out. Grid: none. |
| Pavement walkers never crossed a street | Strollers were cut at junctions and came round at their ends | At a block's end, a stroller takes a crossing that starts within 16 m, on the same terms as a crossing walker, then walks on along the far pavement. | Over 240 s: grid-loop 14 strollers crossed, Nob Hill 13. None stood on a carriageway away from a crossing, apart from the kerb step: the paint stops about 0.75 m short of the kerb. |
| No bus on the boulevard loop | In 7b, a rigid bus turning off the middle lane swung 0.6 m into the car beside it | The bus is back on the loop. It runs in the kerb lane (7c), and two approaches are set back 7.5 m. | Dense, seeds 1, 3 and 7: every pair of vehicles stays at least 0.88 m apart. Nothing goes over a kerb or into a building or prop. |
| Crosswalks invisible from the overhead camera | Their quads were wound clockwise, so they faced down and were culled. This affected every camera, not just overhead. | Wound counter-clockwise. A render test checks that roads, paint and every crossing style face up. | Overhead screenshot: all four legs show their stripes. |
| Tunnels, chase camera, route line on bridges | No tunnel interior; the camera rose above the hill; the route line was flat on decks | Tunnels have a floor and bore. The camera follows the car underground. The route line is draped on decks. | Frontend 393 tests green. |

## Not done, and why

- **Parked cars on the ego loop (per-side carriageway).** None of the 18 OSM ways on the Nob Hill ego loop has a parking tag: all are `parking 0/0` with `edge_m` 0. A per-side carriageway, a wire change, would place no car there. It is skipped until a loop street carries parking tags.
