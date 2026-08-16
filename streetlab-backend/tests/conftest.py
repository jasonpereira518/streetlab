import json
import math
from pathlib import Path

import pytest

# The canonical, committed fixture set lives at the git root and is shared
# with the TypeScript validator (../../contract/validate_ts.test.ts). It is
# generated from the real Simulation and kept honest by
# ../../contract/validate_py_test.py — see that file for how to regenerate it.
FIXTURES = Path(__file__).resolve().parents[2] / "contract" / "fixtures"


def load_fixture(name: str) -> dict:
    """Read a wire fixture from the canonical contract fixture set."""
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture
def fixture():
    return load_fixture


def assert_heading(actual: float, expected: float, tol: float = 1e-6) -> None:
    """Compare headings as angles, not as numbers.

    `math.atan2` returns -pi for a due-west chord whose dy is negative zero and
    +pi when it is positive zero -- one direction, two representatives. Every
    consumer feeds a heading to cos/sin, so the shortest angular difference is
    the property that actually matters.
    """
    diff = (actual - expected + math.pi) % (2 * math.pi) - math.pi
    assert abs(diff) <= tol, f"{actual} is not {expected} (mod 2pi)"
