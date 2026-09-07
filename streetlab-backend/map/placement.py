"""Where a control device stands relative to the traffic it governs.

One convention, stated once, because three scene sources were each inventing
their own and getting it wrong in different directions.

A device's `heading` is the way its FACE points -- back at the driver reading
it -- so the traffic it governs travels `heading + pi`. Everything else follows
from that travel direction:

* the driver's right-hand kerb is the travel direction turned clockwise
  (US right-hand traffic; see `right_of`),
* a stop sign stands on that kerb, set BACK along the approach so the driver
  reads it before the stop line,
* a signal pole stands on the same kerb but set FORWARD, past the junction, so
  its mast arm reaches back over the approaching lanes.

`sim` and the wire schema share the world frame this assumes: +x east, +y
north, heading 0 at +x, CCW positive.
"""

from __future__ import annotations

import math

Vec2 = tuple[float, float]

#: How far past the kerb a sign or pole is planted. Enough that the post is
#: clearly on the pavement rather than balanced on the kerb edge, and small
#: enough that it does not read as standing in a front garden.
KERB_CLEARANCE_M = 1.2


def right_of(travel: Vec2) -> Vec2:
    """The driver's right-hand side, for traffic travelling `travel`.

    The travel direction turned clockwise: a northbound driver's kerb is to
    the east. This is the one place the right-hand-traffic assumption lives --
    a left-hand-traffic scene would negate exactly this function.
    """
    dx, dy = travel
    return dy, -dx


def facing(travel: Vec2) -> float:
    """The `heading` a device needs to face traffic travelling `travel`.

    Normalised into `(-pi, pi]` so two devices governing the same approach
    compare equal rather than differing by a full turn.
    """
    return math.remainder(math.atan2(travel[1], travel[0]) + math.pi, math.tau)


def kerb_offset(
    at: Vec2,
    travel: Vec2,
    half_width_m: float,
    clearance_m: float = KERB_CLEARANCE_M,
    *,
    along: float = 0.0,
) -> Vec2:
    """A point on the right-hand kerb of an approach passing through `at`.

    `half_width_m` is the approach carriageway's centreline-to-kerb distance,
    so the result clears the road surface by `clearance_m`. `along` shifts the
    point up (+) or back (-) the approach: negative for a stop sign, which the
    driver must read before the line; positive for a signal pole, which stands
    beyond the junction.
    """
    rx, ry = right_of(travel)
    side = half_width_m + clearance_m
    return (
        at[0] + rx * side + travel[0] * along,
        at[1] + ry * side + travel[1] * along,
    )


#: How far a device may be shifted along its approach to get out of a crossing
#: carriageway, and in what increments. A stop sign at a tight corner can land
#: in the cross street even though it clears its own; backing it off toward the
#: driver is the move that fixes it without inventing a new lateral offset --
#: a sign further from the junction is still a sign on the right approach,
#: whereas one pushed sideways ends up in a front garden.
PUSH_STEP_M = 1.0
PUSH_LIMIT_M = 12.0


def push_steps(direction: float = -1.0) -> tuple[float, ...]:
    """Longitudinal offsets to try, nearest first. -1 backs off, +1 goes on."""
    count = int(PUSH_LIMIT_M / PUSH_STEP_M)
    return tuple(direction * PUSH_STEP_M * i for i in range(count + 1))


#: How far sideways a device may additionally be pushed, away from the road.
#: Generous, because the number it is escaping is not always the real kerb:
#: OSM maps Broadway on the Nob Hill extract TWICE -- once as a two-way
#: `residential` way and again as a pair of oneway carriageways lying over it
#: -- so a sign offset correctly from the narrow representation still lands
#: inside the wide one. California Street at Polk is mapped three times over
#: (twice oneway at a 3.6 m half-width, once two-way at 7.2 m), and its worst
#: signal pole needs 8.5 m. Measured escapes across the whole extract: four
#: stop signs at 4.5-7.5 m, one pole at 8.5 m. The search is nearest-first, so
#: a well-formed corner resolves at 0-1 m and never sees these values; only a
#: multiply-mapped street reaches them.
SIDE_LIMIT_M = 10.0


def escape_offsets(direction: float = -1.0) -> tuple[tuple[float, float], ...]:
    """`(along, side)` shifts to try, nearest first, for a device in the road.

    Sliding along the approach alone is not enough on real data. Two cases on
    the Nob Hill extract defeat it: Broadway is mapped as a DUAL carriageway,
    two separate oneway ways, so a device on one of them is pushed the length
    of the other and never leaves tarmac; and a `traffic_signals` node is not
    always at its junction's centre, so "one junction further on" can land
    square on the crossing street's centreline.

    Both directions are one-way. `along` only ever moves the way the caller
    asks -- back toward the driver for a sign, on past the junction for a
    signal pole -- and `side` only ever moves further from the carriageway.
    Letting either reverse would "fix" a placement by sliding it across the
    road it is standing in.
    """
    alongs = push_steps(direction)
    count = int(SIDE_LIMIT_M / PUSH_STEP_M)
    sides = tuple(PUSH_STEP_M * i for i in range(count + 1))
    pairs = [(along, side) for along in alongs for side in sides]
    # Nearest first, with a deterministic tiebreak so the same extract always
    # resolves a crowded corner the same way.
    pairs.sort(key=lambda p: (math.hypot(p[0], p[1]), abs(p[0]), p[1]))
    return tuple(pairs)


def push_clear(offer, occupied, steps) -> tuple[Vec2, float]:
    """The first offered position that is not `occupied`, and its offset.

    `offer(along)` builds a candidate position for a shift of `along` metres;
    `occupied(point)` says whether that point is still on a road surface.
    `steps` is tried in order, so the caller controls both the direction and
    how far the search goes.

    Falls back to the FIRST offer when nothing is clear, rather than returning
    nothing: a junction can genuinely have no clear corner (a tiny triangular
    island, a slip road) and a device placed imperfectly is still better than
    a scene that silently loses its stop sign. The caller is expected to count
    these, not to treat the result as verified.
    """
    for along in steps:
        at = offer(along)
        if not occupied(at):
            return at, along
    return offer(steps[0]), steps[0]


#: How closely a head's approach direction must agree with the route heading
#: for that head to be the one governing the ego. Generous, because the route
#: is filleted through a junction and its heading there is not the street's.
HEAD_TOL_RAD = math.radians(60.0)


def faces_the_route(route, heading: float, centre: Vec2, setback_m: float) -> bool:
    """True if a device at `centre` faces traffic going the ego's way.

    Four heads govern each crossroads, in two opposing phase groups. Taking
    all four would put the ego at one stop line facing a group that is red
    whenever the other is green -- it would never move. The head that governs
    a driver is the one whose face turns back at them, so `heading + pi` is
    the direction that driver travels, and the route heading picks it out.

    Evaluated at the STOP LINE, `setback_m` back from the junction, not at the
    junction itself: the centre sits mid-turn on a filleted corner, where the
    route heading is neither the entry nor the exit street's, so no head
    matches it well and more than one can pass a generous tolerance. The stop
    line is where the car is still on its approach and the route heading is
    the real street heading -- and it is the same point `project_control_points`
    measures from, which is what makes this an exact match rather than a coin
    flip between two heads sharing a phase group.
    """
    stop_s = route.normalise(route.project(centre) - setback_m)
    travel = heading + math.pi
    return abs(math.remainder(route.heading_at(stop_s) - travel, math.tau)) < HEAD_TOL_RAD
