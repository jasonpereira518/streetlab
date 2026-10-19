"""Overpass: payload validation, Retry-After / 429, mirror fallback (spec 2026-10-05 section 3.C, amended)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from map.cache import DiskCache
from map.osm_source import describe_build_failure
from map.overpass import (
    BBox,
    HttpxFetcher,
    OverpassClient,
    OverpassError,
    OverpassRateLimited,
    parse_retry_after,
)
from map.useragent import USER_AGENT

GOOD = json.loads((Path(__file__).parent / "fixtures" / "overpass_nob_hill.json").read_text())
BBX = BBox.around(37.7945, -122.4156, 300.0)


class Scripted:
    """Returns / raises each scripted item in turn, recording calls."""

    def __init__(self, *items):
        self.items = list(items)
        self.calls = 0

    def fetch(self, query):
        self.calls += 1
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, Exception):
            raise item
        return item


def client(tmp_path, primary, *mirrors, sleeps=None, retries=3):
    c = OverpassClient(primary, DiskCache(tmp_path), retries=retries, backoff_s=1.0, mirrors=list(mirrors))
    if sleeps is not None:
        c._sleep = sleeps.append
    return c


# --- payload validation ------------------------------------------------------ #


@pytest.mark.parametrize(
    "bad",
    [
        {"elements": [], "remark": "runtime error: Query timed out in \"query\" at line 1"},
        {"elements": [], "remark": "runtime error: Query ran out of memory"},
        {"version": 0.6},
        {"elements": "nope"},
        [],
    ],
)
def test_a_bad_payload_is_retried_and_never_cached(tmp_path, bad):
    f = Scripted(bad)
    c = client(tmp_path, f, sleeps=[])
    with pytest.raises(OverpassError):
        c.graph(BBX)
    assert f.calls == 3
    assert not list(tmp_path.glob("*.json")), "a bad payload must never reach the cache"


def test_a_good_payload_after_a_bad_one_is_cached(tmp_path):
    f = Scripted({"elements": [], "remark": "runtime error: timeout"}, GOOD)
    sleeps: list[float] = []
    c = client(tmp_path, f, sleeps=sleeps)
    c.graph(BBX)
    assert f.calls == 2
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_a_harmless_remark_is_not_an_error(tmp_path):
    f = Scripted({**GOOD, "remark": "some note"})
    client(tmp_path, f, sleeps=[]).graph(BBX)
    assert f.calls == 1


# --- Retry-After ------------------------------------------------------------- #


def test_parse_retry_after():
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after("999") == 30.0  # capped
    assert parse_retry_after("-3") == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None


def test_rate_limited_waits_the_retry_after_not_the_default_backoff(tmp_path):
    f = Scripted(OverpassRateLimited("429", retry_after_s=12.0), GOOD)
    sleeps: list[float] = []
    client(tmp_path, f, sleeps=sleeps).graph(BBX)
    assert sleeps == [12.0]


def test_rate_limited_without_a_header_uses_exponential_backoff(tmp_path):
    f = Scripted(OverpassRateLimited("429"), OverpassRateLimited("429"), GOOD)
    sleeps: list[float] = []
    client(tmp_path, f, sleeps=sleeps).graph(BBX)
    assert sleeps == [1.0, 2.0]


def test_httpx_fetcher_turns_429_into_rate_limited_with_the_header(monkeypatch):
    class R:
        status_code = 429
        headers = {"Retry-After": "9"}

        def raise_for_status(self):
            raise RuntimeError("429")

        def json(self):
            return {}

    seen = {}

    def post(url, content=None, headers=None, timeout=None):
        seen["headers"] = headers
        return R()

    monkeypatch.setattr("httpx.post", post)
    with pytest.raises(OverpassRateLimited) as exc:
        HttpxFetcher().fetch("q")
    assert exc.value.retry_after_s == 9.0
    assert seen["headers"]["User-Agent"] == USER_AGENT


def test_httpx_fetcher_504_is_also_retry_after_aware(monkeypatch):
    class R:
        status_code = 504
        headers = {}

    monkeypatch.setattr("httpx.post", lambda *a, **k: R())
    with pytest.raises(OverpassRateLimited) as exc:
        HttpxFetcher().fetch("q")
    assert exc.value.retry_after_s is None


# --- mirrors ----------------------------------------------------------------- #


def test_falls_over_to_the_mirror_without_waiting(tmp_path):
    primary = Scripted(OverpassRateLimited("429", retry_after_s=30.0))
    mirror = Scripted(GOOD)
    sleeps: list[float] = []
    client(tmp_path, primary, mirror, sleeps=sleeps).graph(BBX)
    assert (primary.calls, mirror.calls) == (1, 1)
    assert sleeps == []  # a different host is not bound by the first one's Retry-After


def test_every_endpoint_is_tried_before_waiting(tmp_path):
    a, b, c3 = Scripted(OverpassError("x")), Scripted(OverpassError("y")), Scripted(GOOD)
    sleeps: list[float] = []
    client(tmp_path, a, b, c3, sleeps=sleeps).graph(BBX)
    assert (a.calls, b.calls, c3.calls) == (1, 1, 1)


def test_all_endpoints_down_raises_after_the_attempts(tmp_path):
    a, b = Scripted(OverpassError("x")), Scripted(OverpassError("y"))
    with pytest.raises(OverpassError):
        client(tmp_path, a, b, sleeps=[]).graph(BBX)
    assert a.calls + b.calls == 3


def test_a_cache_hit_touches_no_endpoint(tmp_path):
    f = Scripted(GOOD)
    c = client(tmp_path, f, sleeps=[])
    c.graph(BBX)
    c.graph(BBX)
    assert f.calls == 1


# --- user-facing messages ---------------------------------------------------- #


def test_distinct_user_messages_for_busy_and_unreachable():
    busy = describe_build_failure(OverpassRateLimited("429"))
    down = describe_build_failure(OverpassError("boom"))
    assert busy != down
    assert "busy" in busy.lower()
    for m in (busy, down):
        assert "429" not in m and "boom" not in m
