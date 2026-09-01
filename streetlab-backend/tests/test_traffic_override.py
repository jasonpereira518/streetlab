"""Raising agent density for capture, without changing what a scenario means.

Phase 3a measured that capture yield is governed by agent spacing --
`route_length / (traffic + 1)` -- and that at shipped densities two of the
four planned training scenarios yield 5 and 0 usable boxes. This override
is how Phase 3b closes that spacing deliberately.
"""

from __future__ import annotations

import pytest

from map.scene_build import SCENARIOS, SyntheticGrid


def test_without_an_override_every_scenario_keeps_its_shipped_count():
    """The flag must be inert when absent, or it changes the demo and the
    packaged app for the trainer's convenience."""
    plain = SyntheticGrid()
    for scenario in SCENARIOS:
        built = plain.build(scenario.id)
        assert built.traffic_count == scenario.traffic
        assert len(built.agent_routes) == scenario.traffic


def test_an_override_moves_both_consumers_together():
    """`traffic_count` rides on the wire in SceneDescription while
    `_agent_routes` builds the actual agent list. If an override reaches one
    and not the other, the frontend is told one number and the sim runs
    another -- a silent disagreement no existing test would catch."""
    built = SyntheticGrid(traffic_override=11).build("grid-loop")
    assert built.traffic_count == 11
    assert len(built.agent_routes) == 11
    assert built.traffic_count == len(built.agent_routes)


def test_an_override_of_zero_is_legal_and_empties_the_road():
    """An empty road is a real capture condition -- `PoseHistory.at` keeps
    `()` and `None` apart precisely so a zero-truth frame is a measurement
    rather than a dropped frame."""
    built = SyntheticGrid(traffic_override=0).build("grid-loop")
    assert built.traffic_count == 0
    assert built.agent_routes == []


def test_a_negative_override_is_refused():
    with pytest.raises(ValueError, match="traffic"):
        SyntheticGrid(traffic_override=-1)


def test_the_override_does_not_change_the_ego_route_or_buildings():
    """Density is the only variable the checkpoint may move. Buildings are
    seeded from the scenario id alone (`Random(_seed(scenario.id))`), and the
    ego route is derived from the block rectangle -- neither should shift."""
    plain = SyntheticGrid().build("grid-loop")
    dense = SyntheticGrid(traffic_override=11).build("grid-loop")
    assert dense.ego_route.length_m == plain.ego_route.length_m
    assert len(dense.description.buildings) == len(plain.description.buildings)
    assert [b.id for b in dense.description.buildings] == [
        b.id for b in plain.description.buildings
    ]
