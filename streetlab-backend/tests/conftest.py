import json
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
