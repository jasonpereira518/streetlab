"""Is this object actually visible, or is a building in the way?

Occlusion inverts when labels stop being scored and start being trained on.
A box on a vehicle hidden behind a building merely capped recall when
scoring -- the benchmark's documented ~0.55 ceiling -- but in a training set
the same box teaches a detector to predict vehicles it cannot see. This
module is what tells the two apart.

**Buildings only.** Vehicle-vehicle occlusion is not modelled: buildings are
the dominant occluder in these scenes and are static, while vehicles
occluding vehicles is a smaller effect for considerably more work. Saying
"not modelled" is the honest description; do not let this module's existence
be read as "occlusion is solved".

**The fraction is the product; the boolean is a convenience.** Callers store
`visible_fraction` and derive `visible` from it, never the reverse. Storing
a derived value and discarding its input is exactly how the per-class size
prior stayed invisible for two phases (`contract/benchmark/README.md`), and
a stored fraction means a consumer who disagrees with `MIN_VISIBLE_FRACTION`
can re-threshold committed labels without re-capturing anything.
"""

from __future__ import annotations

from typing import Sequence

from perception.projection import box_corners
from schema import Building, CameraParams, Size

# A box is `visible` when at least this share of its 9 sample points is
# unoccluded -- 3 of 9. A default, not a value derived from data: one visible
# corner out of nine is a sliver no detector should be taught to find, while
# demanding a majority would discard genuinely half-visible vehicles a
# detector can and should see. Because `visible_fraction` is what gets
# stored, this constant is re-derivable downstream and is not a commitment.
MIN_VISIBLE_FRACTION: float = 0.25


def _blocked_at(
    camera: CameraParams, sx: float, sy: float, sz: float, building: Building
) -> bool:
    """Does `building` block the sight line from `camera` to `(sx, sy, sz)`?

    Two dimensional first: walk the footprint ring's edges and find where the
    camera-to-sample segment crosses one. Buildings are extruded prisms, so a
    crossing only occludes if the sight line is still *below* the roof where
    it crosses -- hence the height check at the crossing parameter rather
    than a bare 2D intersection test. A kerb between camera and car crosses
    the footprint and blocks nothing.

    The intersection parameter is required to lie strictly inside the
    segment (`0 < t < 1`), which is what keeps a building *behind* the object
    from being counted as an occluder.
    """
    ax, ay = camera.x, camera.y
    bx, by = sx, sy
    r_x, r_y = bx - ax, by - ay

    ring = building.footprint
    n = len(ring)
    for i in range(n):
        cx, cy = ring[i]
        dx, dy = ring[(i + 1) % n]
        s_x, s_y = dx - cx, dy - cy
        denom = r_x * s_y - r_y * s_x
        if denom == 0.0:
            continue  # parallel or collinear: no single crossing point
        t = ((cx - ax) * s_y - (cy - ay) * s_x) / denom
        u = ((cx - ax) * r_y - (cy - ay) * r_x) / denom
        if not (0.0 < t < 1.0 and 0.0 <= u <= 1.0):
            continue
        z_at_crossing = camera.z + t * (sz - camera.z)
        if z_at_crossing < building.height_m:
            return True
    return False


def visible_fraction(
    x: float,
    y: float,
    heading: float,
    size: Size,
    camera: CameraParams,
    buildings: Sequence[Building],
) -> float:
    """Share of the object's 9 sample points with an unobstructed sight line.

    The samples are the 8 corners `projection.box_corners` builds -- the same
    corners `project_box` projects into the box this fraction describes --
    plus the object's centre at half height. Sharing `box_corners` is
    deliberate: a visibility flag computed against different corners than the
    box it annotates would be quietly meaningless.

    Returns 1.0 when `buildings` is empty. That is the correct answer for the
    occluder set supplied, not a claim that nothing occludes -- callers that
    forget to pass buildings get an all-visible dataset, which is why
    `CaptureSink` records the occluder count per frame.
    """
    samples = box_corners(x, y, heading, size)
    samples.append((x, y, size.height / 2.0))
    unblocked = sum(
        1
        for (sx, sy, sz) in samples
        if not any(_blocked_at(camera, sx, sy, sz, b) for b in buildings)
    )
    return unblocked / len(samples)


def is_visible(fraction: float) -> bool:
    """Whether `fraction` clears `MIN_VISIBLE_FRACTION`."""
    return fraction >= MIN_VISIBLE_FRACTION
