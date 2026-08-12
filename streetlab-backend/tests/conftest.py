import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Read a wire fixture captured from the frontend's TypeScript mock."""
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.fixture
def fixture():
    return load_fixture
