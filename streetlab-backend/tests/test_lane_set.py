"""Lanes derived from the ego route and the road network.

The ego's own lane is `ego_route` itself, and a neighbour is that route offset
by one lane width either way. Which of those neighbours the car may actually
enter at a given arc length is a separate question, answered by fitting the
target lane inside the forward carriageway -- NOT by how many lanes were
geometrically constructed, and NOT by `Road.lanes_forward` alone. Both shipped
scenes park the ego in the LEFTMOST forward lane wherever two run its way, so a
count of 2 says nothing about whether the lane beside it is road or oncoming
traffic (docs/superpowers/plans/2026-08-16-cycle3-phase2-revision.md).
"""

import math
from collections import Counter

import pytest

from map.lanes import (
    LANE_FIT_TOL_M,
    LANE_W,
    derive_lanes,
    lane_change_is_legal,
    nearest_road_along,
)
from map.scene_build import SyntheticGrid
from schema import Road
from sim.route import EGO_LANE_ID, Route


@pytest.fixture(scope="module")
def grid_loop():
    return SyntheticGrid().build("grid-loop")


def test_the_ego_lane_is_the_ego_route(grid_loop):
    lanes = grid_loop.lanes
    assert lanes.ego.route.points == grid_loop.ego_route.points
    # Identity, not just equal points: the ego's lane must be the SAME Route
    # object as `ego_route`, not a reconstructed lookalike. A rebuilt one can
    # silently diverge on anything `ego_route` carries beyond geometry --
    # `segment_limits` in particular, which `posted_limit()` (`sim/loop.py`)
    # reads directly off `ego_route` and nothing would keep a copy in sync with.
    assert lanes.ego.route is grid_loop.ego_route
    assert lanes.ego.offset_m == 0.0


def test_a_lane_set_without_an_ego_lane_refuses_to_designate_one():
    """`legal_along` defaults to `()` on a refuse-by-default principle; `ego`
    used to do the opposite, falling back to `lanes[0]` -- which in
    `derive_lanes` order is `lane_right`, one lane width off the route the car
    is actually tracking. `_segment_at`, `count_at`, `legal_at`, `road_at`,
    `ego_offset_at` and `neighbour` all key off `ego`, so that fallback keys
    every per-station answer to the wrong route rather than saying so.

    No live path reaches it -- `derive_lanes` always builds the ego lane, and
    every hand-built `LaneSet` in the suite names one -- which is exactly why
    it has to raise: a `LaneSet` that lost its ego lane is a construction bug,
    and the only useful thing it can do is fail where it was built.
    """
    from sim.route import Lane, LaneSet

    orphan = LaneSet(
        lanes=(Lane("lane_right", -LANE_W, Route([(0.0, 0.0), (10.0, 0.0)], closed=False), None, None),),
        count_along=(1,),
    )
    with pytest.raises(ValueError, match="no ego lane"):
        orphan.ego


def test_every_derived_lane_carries_segment_limits(grid_loop):
    """`Route.offset` deliberately drops them (`sim/route.py:27-34`), so a lane
    that forgot to re-attach would silently drive at the scene-wide figure.

    The ego's own lane is excluded: it IS `ego_route`, and `SyntheticGrid`'s
    `ego_route` deliberately carries no per-segment limits at all
    (`sim/loop.py`'s `posted_limit()` falls back to the scene-wide figure for
    exactly this route) -- asserting limits on it here would demand
    `derive_lanes` invent a second, disagreeing answer for the same route,
    which is worse than not having one.
    """
    for lane in grid_loop.lanes.lanes:
        if lane.id == EGO_LANE_ID:
            continue
        assert lane.route.segment_limits is not None, f"{lane.id} has no limits"
        assert len(lane.route.segment_limits) == len(lane.route.points)


def test_neighbour_handles_link_the_lanes_in_order(grid_loop):
    """The set is anchored on the ego's lane, with one neighbour either side."""
    lanes = grid_loop.lanes
    assert [lane.offset_m for lane in lanes.lanes] == [-LANE_W, 0.0, LANE_W]
    for a, b in zip(lanes.lanes, lanes.lanes[1:]):
        assert a.left_id == b.id
        assert b.right_id == a.id
    assert lanes.lanes[0].right_id is None
    assert lanes.lanes[-1].left_id is None
    assert lanes.neighbour(+1) is lanes.by_id(lanes.ego.left_id)
    assert lanes.neighbour(-1) is lanes.by_id(lanes.ego.right_id)


def test_the_lane_count_varies_along_the_route(grid_loop):
    """grid-loop runs Hyde St and California St (2-lane arterials) plus
    Leavenworth and Sacramento (1 lane each), so the count must change.
    """
    counts = {grid_loop.lanes.count_at(s) for s in range(0, int(grid_loop.ego_route.length_m), 5)}
    assert counts == {1, 2}, f"expected both counts, saw {counts}"


def test_count_at_never_reports_fewer_than_one(grid_loop):
    for s in range(0, int(grid_loop.ego_route.length_m)):
        assert grid_loop.lanes.count_at(float(s)) >= 1


def test_the_lane_set_keeps_the_road_and_the_offset_it_judged_from(grid_loop):
    """The two inputs the carriageway model is built from, kept per segment.

    Without them a caller that has to place the ego INSIDE the carriageway --
    `sim/loop.py`'s `_lane_state`, for the index and the markings the wire
    carries -- has no way to do it but search for the governing road a second
    time, and a second search is a second chance to match a different one.

    Pinned here rather than only through the wire because neither shipped scene
    can tell a correct `ego_offset_along` from an all-zero one by the reported
    lane index alone: zero reads as "on the centreline", which clamps to the
    same leftmost forward lane the ego genuinely occupies on both.
    """
    lanes, route = grid_loop.lanes, grid_loop.ego_route
    assert len(lanes.road_along) == len(lanes.count_along)
    assert len(lanes.ego_offset_along) == len(lanes.count_along)

    for s, road in _segments(grid_loop):
        assert lanes.road_at(s) is road
        # The count and the marking a segment reports have to come from ONE
        # road, or the wire draws a divider belonging to a different street.
        assert lanes.road_at(s).lanes_forward == lanes.count_at(s)

    # Measured on grid-loop: `EGO_LANE_INSET` puts the ego route half a lane
    # RIGHT of the centreline down every straight, and `Route.offset`'s mitre
    # scaling carries it to 2.46 m through the filleted corners. Never zero and
    # never positive -- a positive offset here would mean the ego route had
    # crossed to the far side of its own road's centreline.
    assert all(off < 0.0 for off in lanes.ego_offset_along)
    assert max(lanes.ego_offset_along) == pytest.approx(-1.800, abs=1e-3)
    assert min(lanes.ego_offset_along) == pytest.approx(-2.461, abs=1e-3)


@pytest.mark.parametrize("direction", [+1, -1])
def test_each_neighbour_sits_one_lane_width_to_its_own_side(grid_loop, direction):
    """Positive lateral offset is left of travel (`sim/route.py:133-141`)."""
    lanes = grid_loop.lanes
    ego = grid_loop.ego_route
    s = ego.length_m * 0.5
    lane = lanes.neighbour(direction)
    point = lane.route.point_at(lane.route.project(ego.point_at(s)))
    assert ego.lateral_offset(point) == pytest.approx(direction * LANE_W, abs=0.6)
    assert lane.offset_m == direction * LANE_W


def test_the_osm_scene_derives_lanes_too(nob_hill_scene):
    lanes = nob_hill_scene.lanes
    assert lanes is not None and lanes.lanes
    assert lanes.ego.route.points == nob_hill_scene.ego_route.points
    # Same identity requirement as the grid case (see
    # test_the_ego_lane_is_the_ego_route). Here it also confirms the OSM side of
    # the asymmetry: `OsmSceneSource` attaches `segment_limits` to `ego_route`
    # BEFORE calling `derive_lanes`, so the ego's lane, being that same object,
    # must carry them too.
    assert lanes.ego.route is nob_hill_scene.ego_route
    assert lanes.ego.route.segment_limits is not None


def test_most_of_the_nob_hill_loop_is_a_single_lane(nob_hill_scene):
    """The number Phase 2's acceptance design turns on."""
    route = nob_hill_scene.ego_route
    step = route.length_m / 400
    counts = [nob_hill_scene.lanes.count_at(i * step) for i in range(400)]
    single = sum(1 for c in counts if c < 2)
    assert single / len(counts) > 0.7, f"only {single}/400 samples single-lane"


# --------------------------------------------------------------------------- #
# Containment legality                                                         #
# --------------------------------------------------------------------------- #


def _offset_from(road, point, heading):
    """Signed distance from `road`'s centreline to `point`, + = left of `heading`.

    Re-derived here rather than imported from `map.lanes`: the safety property
    below is only evidence if the thing it measures with is independent of the
    code that decided the answer.
    """
    nearest, best = None, math.inf
    for a, b in zip(road.centerline, road.centerline[1:]):
        vx, vy = b[0] - a[0], b[1] - a[1]
        wx, wy = point[0] - a[0], point[1] - a[1]
        leg2 = vx * vx + vy * vy
        t = 0.0 if leg2 <= 0 else min(max((wx * vx + wy * vy) / leg2, 0.0), 1.0)
        candidate = (a[0] + vx * t, a[1] + vy * t)
        d = math.dist(point, candidate)
        if d < best:
            nearest, best = candidate, d
    return (
        -(point[0] - nearest[0]) * math.sin(heading)
        + (point[1] - nearest[1]) * math.cos(heading)
    )


def _segments(scene):
    """Each ego-route segment as `(arc length of its midpoint, governing road)`.

    The midpoint, because that is the point `nearest_road_along` matched the
    road on -- asking about a segment's start would sometimes land the query
    one segment earlier than the road it is being compared against.
    """
    route, roads = scene.ego_route, scene.description.roads
    for k, i in enumerate(nearest_road_along(route, roads)):
        if i is not None:
            yield (route._cum[k] + route._cum[k + 1]) / 2.0, roads[i]


@pytest.mark.parametrize("scene_name", ["grid_loop", "nob_hill_scene"])
def test_no_legal_change_targets_a_lane_left_of_a_two_way_centreline(
    scene_name, request
):
    """The safety property of the whole phase, asserted rather than observed.

    On a two-way road the centreline IS the divider, so a target lane whose
    centre sits left of it is the oncoming carriageway. This walks the legality
    answer back onto the geometry it authorises -- the actual neighbour `Route`
    reached through `left_id`/`right_id` -- so building the neighbours on the
    wrong side of the ego fails here even though the containment arithmetic is
    untouched.
    """
    scene = request.getfixturevalue(scene_name)
    lanes, route = scene.lanes, scene.ego_route
    judged, crossings = 0, []
    for s, road in _segments(scene):
        if road.oneway:
            continue  # a oneway centreline has no oncoming traffic beyond it
        for direction in lanes.legal_at(s):
            target = lanes.neighbour(direction)
            centre = target.route.point_at(target.route.project(route.point_at(s)))
            offset = _offset_from(road, centre, route.heading_at(s))
            judged += 1
            if offset > 0.0:
                crossings.append((round(s, 1), road.name, direction, round(offset, 2)))
    assert judged, "no change was ruled legal on a two-way road; nothing was judged"
    assert not crossings, (
        f"{len(crossings)} of {judged} legal targets sit left of the centreline: "
        f"{crossings[:5]}"
    )


@pytest.mark.parametrize("scene_name", ["grid_loop", "nob_hill_scene"])
def test_every_two_way_road_the_ego_drives_is_a_symmetric_carriageway(
    scene_name, request
):
    """`_legal_directions_along`'s load-bearing premise, asserted not narrated.

    That function reads `lanes_forward`/`lanes_backward` as the ROAD stores
    them and never swaps them when the ego runs against the storage direction
    -- which it does on 18/36 grid-loop and 90/339 Nob Hill segments. Reading
    them unswapped is sound exactly where the two counts are EQUAL, and today
    they are on every two-way road either route touches. That is a property of
    two routes, not of either scene: the Nob Hill extract holds 264 roads, 7 of
    them asymmetric two-way (five Powell Street ways at 2/1 and 1/2, Wetmore
    Street at 1/0, and a California Street way at 2/1), and runtime address
    search has been able to route onto them since Cycle 2. The day one is
    matched, this fails instead of the car silently reading the oncoming
    carriageway's lane count as its own.

    Oneway roads are excluded rather than overlooked. Sacramento, Washington
    and Clay Street are matched at 2/0 and 1/0, and a swap there would report
    ZERO forward lanes at the junction corners where the ego is turning ACROSS
    them -- a worse answer than the one containment already gives, which is to
    refuse both directions (see test_sacramento_street_refuses_both_directions).

    Handling an asymmetric two-way road CORRECTLY is deliberately not attempted
    here; the assertion is the whole deliverable.
    """
    scene = request.getfixturevalue(scene_name)
    two_way = {
        road.id: road for _, road in _segments(scene) if not road.oneway
    }
    assert two_way, "no two-way road was matched; the premise was not exercised"
    assert any(r.lanes_forward >= 2 for r in two_way.values()), (
        "no matched two-way road runs two lanes the ego's way, so the premise "
        "this asserts carries no legality decision on this scene"
    )
    asymmetric = [
        (r.id, r.name, r.lanes_forward, r.lanes_backward)
        for r in two_way.values()
        if r.lanes_forward != r.lanes_backward
    ]
    assert not asymmetric, (
        f"{len(asymmetric)} of {len(two_way)} matched two-way roads are "
        f"asymmetric, so `_legal_directions_along` may be reading the oncoming "
        f"carriageway's lane count as forward: {asymmetric}"
    )


@pytest.mark.parametrize(
    "scene_name, expected",
    [("grid_loop", {-1: 18}), ("nob_hill_scene", {-1: 33})],
)
def test_each_scene_admits_exactly_the_measured_changes(scene_name, expected, request):
    """Non-vacuousness, and the shape of the answer, in one assertion.

    Without this the safety property above passes on a model that rules
    everything illegal. Measured: grid-loop admits a RIGHT change on the 9
    California St and 9 Hyde St segments (2 lanes each way, ego at -1.79 m); Nob
    Hill on the 33 California Street segments. Neither scene admits a left
    change anywhere, which is the direct consequence of the ego sitting in the
    inner forward lane -- it passes on the right and returns left.
    """
    scene = request.getfixturevalue(scene_name)
    seen = Counter(d for s, _ in _segments(scene) for d in scene.lanes.legal_at(s))
    assert dict(seen) == expected


def test_sacramento_street_refuses_both_directions(nob_hill_scene):
    """Where the ego's own placement is ambiguous, the rule refuses rather than guesses.

    Sacramento Street is oneway with two forward lanes, and the ego route
    crosses its centreline (measured `ego_off` 0.00 m on all 16 matched
    segments -- these are the fillet vertices of the turn across it, not a
    stretch driven along it). A lane 3.6 m either way needs 1.80 m of slack to
    fit inside a 7.2 m carriageway from there, well past `LANE_FIT_TOL_M`, so
    both directions are refused. A count-based rule says 2 here and would admit
    the change.
    """
    matched = [s for s, road in _segments(nob_hill_scene) if road.name == "Sacramento Street"]
    assert len(matched) == 16, f"the fixture no longer matches Sacramento Street 16 times: {len(matched)}"
    assert nob_hill_scene.lanes.count_at(matched[0]) == 2, "not the two-lane case any more"
    for s in matched:
        assert nob_hill_scene.lanes.legal_at(s) == ()


def test_the_containment_predicate_holds_at_its_tolerance_boundary():
    """`LANE_FIT_TOL_M` pinned from BOTH sides by measured literals.

    A two-way road with two lanes each way is 14.4 m wide, so the forward half
    is `[-7.2, 0.0]` signed left-positive from the centreline. From the inner
    forward lane (-1.8 m, where both scenes put the ego) a right change fits
    exactly, flush against the kerb edge, and needs no slack at all.

    The two boundary literals are measured, NOT derived from the constant. An
    earlier version of this test wrote them as `-1.8 - LANE_FIT_TOL_M` and
    `-1.8 - LANE_FIT_TOL_M - 1e-9`, which is an algebraic exact tie for every
    tolerance -- both sides of the `>=` reduce to `-7.2 - tol` -- so it moved
    with the constant and pinned nothing. Measured: it passed at 0.6 (a
    tolerance that BREAKS grid-loop) and failed at 0.9/1.0/1.4/1.75 only on
    which side of the tie IEEE-754 rounding landed.

    -2.4615 m is grid-loop's worst ego offset (`Route.offset`'s mitre scaling
    through a filleted corner, min of `ego_offset_along` -- see
    test_the_lane_set_keeps_the_road_and_the_offset_it_judged_from, which pins
    it at -2.461); a right change from there needs 0.6615 m of slack, so the
    tolerance may not drop below that without refusing a change the shipped
    grid-loop replay makes. -2.56 m needs 0.76 m and is not a case either
    scene produces: it is the tripwire on the other side, so the tolerance
    cannot be loosened past 0.76 m without a fresh measurement replacing this
    literal. Together they pin `LANE_FIT_TOL_M` to [0.6615, 0.76).
    """
    assert lane_change_is_legal(-1.8, 2, 2, -1)
    assert not lane_change_is_legal(-1.8, 2, 2, +1)
    assert lane_change_is_legal(-2.4615, 2, 2, -1), "grid-loop's worst corner is refused"
    assert not lane_change_is_legal(-2.56, 2, 2, -1), "the tolerance has been loosened"


def test_a_route_matched_to_no_road_at_all_authorises_nothing():
    """Every table `derive_lanes` fills is fed by one nearest-road pass, and
    when that pass matches nothing `_fill_forward` has no value to fill from.

    The count falls back to 1 -- a road the car is driving has at least one lane,
    and the wire has to report something -- while legality, the governing road
    and the offset all stay empty, so `legal_at` refuses, `road_at` says None
    and `ego_offset_at` reads as the centreline. Reached only here: both shipped
    scenes match all 36 / 339 of their segments. This branch used to be covered
    through `lanes_forward_along`, which R1-FIX removed as dead code.
    """
    from map.lanes import derive_lanes

    lanes = derive_lanes(Route([(0.0, 0.0), (100.0, 0.0)], closed=False), [])
    assert lanes.count_along == (1,)
    assert lanes.legal_along == ((),)
    assert lanes.road_along == ()
    assert lanes.ego_offset_along == ()
    assert lanes.legal_at(50.0) == ()
    assert lanes.road_at(50.0) is None
    assert lanes.ego_offset_at(50.0) == 0.0


def test_the_kerbside_lane_of_the_same_road_may_change_left():
    """The rule is the carriageway, not a hard-coded "right only".

    Same 2+2 road, ego in the KERBSIDE forward lane at -5.4 m: now the left
    change is the one that fits and the right one runs off the kerb. Nothing in
    either shipped scene reaches this today (ruling Q19 keeps the ego half a
    lane off the centreline), so without this the model would pass its scene
    tests while quietly encoding the answer instead of computing it.
    """
    assert lane_change_is_legal(-5.4, 2, 2, +1)
    assert not lane_change_is_legal(-5.4, 2, 2, -1)


@pytest.mark.parametrize(
    "ego_y, expected",
    [(-5.4, (1,)), (-1.8, (-1,)), (1.8, (-1,)), (5.4, ())],
)
def test_the_ego_offset_is_measured_against_its_own_direction_of_travel(ego_y, expected):
    """`derive_lanes` on a hand-placed ego, where the sign of the offset shows.

    Neither shipped scene can catch a sign error here. On a road with two lanes
    each way the RIGHT-change test is symmetric in the ego's offset -- it admits
    anything within `LANE_W/2 + LANE_FIT_TOL_M` of the centreline on either side
    -- and a right change on such a road is the only thing either scene ever
    rules legal, so negating the offset changes no answer on any of their 375
    segments. The outer rows below are not symmetric: a 14.4 m carriageway
    running east along y=0, with the ego driving east in each of its four lanes
    in turn. From the kerbside forward lane (-5.4 m) only a left change fits;
    from the far oncoming lane (+5.4 m) nothing does. The two inner rows agree
    with each other, and are here to record exactly that symmetry rather than to
    discriminate -- from +1.8 m the car is on the wrong side of the road and a
    right change is genuinely the move that puts it back on its own carriageway.
    """
    road = Road(
        id="r", name="Test St", road_class="arterial",
        centerline=[(0.0, 0.0), (400.0, 0.0)],
        lanes_forward=2, lanes_backward=2, lane_width_m=LANE_W,
        speed_limit_mps=15.0, oneway=False, center_marking="double_yellow",
        has_sidewalk=True,
    )
    route = Route([(10.0, ego_y), (390.0, ego_y)], closed=False)
    lanes = derive_lanes(route, [road])
    assert lanes.legal_at(route.length_m / 2) == expected


def test_a_single_forward_lane_refuses_what_containment_alone_would_admit():
    """`lanes_forward >= 2` is a precondition, not a restatement of the fit.

    A one-lane oneway is 3.6 m wide, `[-1.8, +1.8]`. An ego 3.6 m LEFT of that
    centreline -- outside its own carriageway entirely -- has a right-change
    target that lands back inside it, so containment alone would wave it
    through. Measured, this changes no answer on either shipped scene (every
    single-lane segment is refused by containment too; the worst case is Clay
    Street, whose closest left change still needs 1.79 m of slack), so this
    guards a future `EGO_LANE_INSET` or scene source rather than today's.
    """
    assert lane_change_is_legal(3.6, 1, 0, -1) is False
    # Sanity: it is the precondition doing that, not the fit.
    assert 3.6 - LANE_W + LANE_W / 2 <= 1.8 + LANE_FIT_TOL_M
