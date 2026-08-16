"""Time-to-collision and hazard inference.

Extracted from `perception/service.py`, whose module docstring said in Cycle 1
that this is inference rather than sensing and belongs here. It is imported
back by perception -- `Detection.ttc_s` and `.hazard` are wire fields the
frontend's TTC readout needs from frame one -- so the wire is unchanged and
there is exactly one implementation rather than two that can drift.
"""

from __future__ import annotations

#: Below this closing speed a gap is not meaningfully shrinking, so TTC is
#: reported as None rather than as an enormous number.
MIN_CLOSING_MPS = 0.25

#: A detection at or under this TTC is flagged for the frontend's hazard
#: overlay.
HAZARD_TTC_S = 4.0

_LABELS = {
    "pedestrian": "Pedestrian in path",
    "cyclist": "Cyclist in path",
}


def time_to_collision(
    gap: float | None, lane_offset: int, ego_speed: float, other_speed: float
) -> float | None:
    """Seconds until ego reaches `gap`, or None when the question is undefined.

    None rather than infinity for the zero-closing-speed case: a caller ranking
    leads by TTC must be able to say "this one does not apply" without a
    sentinel that sorts.
    """
    if gap is None or lane_offset != 0 or gap <= 0:
        return None
    closing = ego_speed - other_speed
    if closing < MIN_CLOSING_MPS:
        return None
    return gap / closing


def is_hazard(ttc: float | None) -> bool:
    return ttc is not None and ttc <= HAZARD_TTC_S


def hazard_label(cls: str) -> str:
    return _LABELS.get(cls, "Closing on lead vehicle")
