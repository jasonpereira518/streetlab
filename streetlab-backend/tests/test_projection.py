import math

import pytest

from map.projection import EARTH_R, LatLon, to_latlon, to_local

NOB_HILL = LatLon(lat=37.7945, lon=-122.4156)


def test_origin_maps_to_zero():
    assert to_local(NOB_HILL.lat, NOB_HILL.lon, NOB_HILL) == pytest.approx((0.0, 0.0))


def test_one_degree_north_is_one_meridian_degree():
    _, y = to_local(NOB_HILL.lat + 1.0, NOB_HILL.lon, NOB_HILL)
    assert y == pytest.approx(math.radians(1.0) * EARTH_R, rel=1e-9)


def test_east_is_positive_x_and_north_is_positive_y():
    x, _ = to_local(NOB_HILL.lat, NOB_HILL.lon + 0.001, NOB_HILL)
    _, y = to_local(NOB_HILL.lat + 0.001, NOB_HILL.lon, NOB_HILL)
    assert x > 0
    assert y > 0


def test_longitude_is_compressed_by_latitude():
    """A degree of longitude at 37.8N is about cos(37.8) of a degree of latitude."""
    x, _ = to_local(NOB_HILL.lat, NOB_HILL.lon + 1.0, NOB_HILL)
    _, y = to_local(NOB_HILL.lat + 1.0, NOB_HILL.lon, NOB_HILL)
    assert x / y == pytest.approx(math.cos(math.radians(NOB_HILL.lat)), rel=1e-9)


@pytest.mark.parametrize(
    "dlat,dlon",
    [(0.0, 0.0), (0.001, 0.002), (-0.004, 0.003), (0.009, -0.009)],
)
def test_round_trip_within_a_millimetre(dlat, dlon):
    lat, lon = NOB_HILL.lat + dlat, NOB_HILL.lon + dlon
    x, y = to_local(lat, lon, NOB_HILL)
    back_lat, back_lon = to_latlon(x, y, NOB_HILL)
    # 1e-8 degrees is roughly a millimetre — far below lane-width significance.
    assert back_lat == pytest.approx(lat, abs=1e-8)
    assert back_lon == pytest.approx(lon, abs=1e-8)
