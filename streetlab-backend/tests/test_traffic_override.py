"""Raising agent density for capture, without changing what a scenario means.

Phase 3a measured that capture yield is governed by agent spacing --
`route_length / (traffic + 1)` -- and that at shipped densities two of the
four planned training scenarios yield 5 and 0 usable boxes. This override
is how Phase 3b closes that spacing deliberately.
"""

from __future__ import annotations

import pytest

from map.scene_build import SCENARIOS, SyntheticGrid, TrafficOverrideError
from server import cli


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


def test_the_refusal_is_a_valueerror_so_existing_callers_still_read_it():
    """`TrafficOverrideError` exists so `cli._SOURCE_ERRORS` can name this
    rejection without catching every unrelated `ValueError` under a scene
    build. It must stay a `ValueError` subclass: anything that only knows the
    stdlib type -- including the test above -- must keep working."""
    with pytest.raises(TrafficOverrideError):
        SyntheticGrid(traffic_override=-1)
    assert issubclass(TrafficOverrideError, ValueError)


def test_serve_refuses_a_negative_override_without_a_traceback(capsys):
    """The sibling rejection -- `--traffic` with `--source osm` -- exits
    cleanly through `parser.error`. This one reached `SyntheticGrid.__init__`
    and raised out of `main()`, printing a raw traceback at a user. Fails if
    the exception escapes at all, because an escaping exception never reaches
    these assertions."""
    code = cli.main(["serve", "--traffic", "-1", "--port", "0"])
    out = capsys.readouterr().out

    assert code == 1
    assert out.strip().startswith("error:")
    assert "traffic override must be >= 0, got -1" in out
    assert "Traceback" not in out


def test_serve_shuts_down_the_perception_pipeline_when_the_override_is_refused(
    capsys, monkeypatch
):
    """The half of this that is not cosmetic. `perception_pipeline_for` runs
    BEFORE `Simulation(...)`, so by the time the override is rejected a live
    `ThreadPoolExecutor` already exists. The teardown that guarantees it is
    reclaimed lives on the `except _SOURCE_ERRORS` path -- an exception that
    is not in that tuple walks straight past it and leaks the pool.

    Reuses `test_cli_osm.py`'s spy rather than a second copy: it wraps the
    real `shutdown`, so a spied run still leaves no live thread behind."""
    from tests.test_cli_osm import _spy_on_pipeline_shutdown

    shutdown_calls = _spy_on_pipeline_shutdown(monkeypatch)

    code = cli.main(
        ["serve", "--traffic", "-1", "--port", "0", "--perception", "ml"]
    )
    out = capsys.readouterr().out

    assert code == 1
    assert out.strip().startswith("error:")
    assert len(shutdown_calls) == 1, (
        "the pipeline's ThreadPoolExecutor was never shut down on the "
        "--traffic rejection path"
    )


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
