"""The wire terrain decodes, and samples, to the numbers the renderer checks.

`contract/terrain_samples.json` is read by both this and
`streetlab/tests/terrain.test.ts`: the backend grades roads level against this
interpolation, so the two ports must agree to the millimetre.
"""

import json
from pathlib import Path

import numpy as np

from map.terrain import from_wire
from schema import Terrain

FIXTURE = Path(__file__).resolve().parents[2] / "contract" / "terrain_samples.json"


def test_wire_terrain_samples_match_the_shared_fixture():
    data = json.loads(FIXTURE.read_text())
    field = from_wire(Terrain.model_validate(data["terrain"]))
    got = field.sample_many(np.asarray(data["points"]))
    assert np.allclose(got, data["heights"], atol=1e-3)
