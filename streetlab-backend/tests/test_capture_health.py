"""The health report must not invent numbers when a capture cannot supply them."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from capture_health import _quantiles  # noqa: E402


def test_quantiles_of_a_short_series_fall_back_to_min_median_max():
    lo, mid, hi = _quantiles([1.0, 2.0, 3.0])
    assert (lo, mid, hi) == (1.0, 2.0, 3.0)


def test_quantiles_of_a_long_series_are_real_quartiles():
    lo, mid, hi = _quantiles([float(i) for i in range(1, 101)])
    assert lo < mid < hi
    assert 20.0 < lo < 30.0 and 45.0 < mid < 55.0 and 70.0 < hi < 80.0
