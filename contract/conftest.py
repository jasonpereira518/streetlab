"""conftest for the cross-language contract suite.

Registers ``--update-fixtures`` so an intentional schema change becomes a
reviewable git diff in ``contract/fixtures/`` rather than a silent rewrite.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-fixtures",
        action="store_true",
        default=False,
        help=(
            "regenerate contract/fixtures/ from the live simulation instead "
            "of diffing against what's committed"
        ),
    )


@pytest.fixture
def update_fixtures(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-fixtures"))
