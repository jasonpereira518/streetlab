"""The camera layouts the wire names are pinned to contract/detector_cameras.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from perception.noisy_truth import CAMERA_SETS
from perception.pipeline import camera_count

PIN = json.loads((Path(__file__).resolve().parents[2] / "contract" / "detector_cameras.json").read_text())["sets"]


@pytest.mark.parametrize("name", sorted(PIN))
def test_each_wire_layout_matches_the_pin(name):
    ours = CAMERA_SETS[name]
    assert len(ours) == len(PIN[name]) == camera_count(name)
    for cam, pinned in zip(ours, PIN[name]):
        assert cam.name == pinned["name"]
        assert math.degrees(cam.yaw_rad) == pytest.approx(pinned["yaw_deg"], abs=1e-9)
        assert cam.fov_y_deg == pytest.approx(pinned["fov_y_deg"], abs=1e-9)
        assert (cam.width, cam.height) == (pinned["width"], pinned["height"])


def test_the_schema_literal_names_exactly_the_pinned_layouts():
    from typing import get_args

    from schema import CameraSet

    assert set(get_args(CameraSet)) == set(PIN)


def test_default_ground_truth_app_builds_no_detector_pipeline_so_no_extra_cameras():
    """`--cameras` defaults to front+sides100 (3 captures per frame time), but that only
    reaches the client through `PerceptionStats.camera_set`, and the client renders and sends
    detector frames only while stats exist. Ground truth (the default) builds no pipeline, so
    the default app pays for zero detector captures, not three."""
    from server.cli import build_parser, perception_pipeline_for

    for cmd in ("serve", "run"):
        args = build_parser().parse_args([cmd])
        assert args.perception == "ground-truth"
        assert perception_pipeline_for(args) is None
