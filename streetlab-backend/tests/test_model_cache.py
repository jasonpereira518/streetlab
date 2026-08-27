"""Weights resolve through a content-addressed cache, fetched once.

Mirrors `map/cache.py`'s shape: hash-named files under a root, an LRU budget,
and a fetch seam so tests stay offline.
"""

from __future__ import annotations

import hashlib

import pytest

from perception.model_cache import ModelCache, ModelSpec

PAYLOAD = b"pretend onnx bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
SPEC = ModelSpec(name="rtdetr-test", url="https://example.invalid/m.onnx",
                 sha256=DIGEST, size_bytes=len(PAYLOAD))


def writer(payload: bytes = PAYLOAD):
    calls: list[str] = []

    def fetch(url: str, dest) -> None:
        calls.append(url)
        dest.write_bytes(payload)

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def test_first_ensure_fetches_and_caches(tmp_path):
    cache = ModelCache(tmp_path, budget_bytes=1_000)
    fetch = writer()
    path = cache.ensure(SPEC, fetch)
    assert path.exists()
    assert path.read_bytes() == PAYLOAD
    assert len(fetch.calls) == 1


def test_second_ensure_does_not_refetch(tmp_path):
    cache = ModelCache(tmp_path, budget_bytes=1_000)
    fetch = writer()
    cache.ensure(SPEC, fetch)
    cache.ensure(SPEC, fetch)
    # The whole point: second launch needs no network.
    assert len(fetch.calls) == 1


def test_a_corrupt_cached_file_is_refetched(tmp_path):
    cache = ModelCache(tmp_path, budget_bytes=1_000)
    fetch = writer()
    path = cache.ensure(SPEC, fetch)
    path.write_bytes(b"truncated")
    cache.ensure(SPEC, fetch)
    assert len(fetch.calls) == 2
    assert path.read_bytes() == PAYLOAD


def test_a_hash_mismatch_from_the_fetcher_raises_and_leaves_no_file(tmp_path):
    cache = ModelCache(tmp_path, budget_bytes=1_000)
    bad = writer(b"not what was promised")
    with pytest.raises(ValueError):
        cache.ensure(SPEC, bad)
    # A file that failed verification must not be left where a later run
    # would trust it.
    assert not cache.path_for(SPEC).exists()


def test_evicting_to_budget_removes_the_least_recently_used(tmp_path):
    cache = ModelCache(tmp_path, budget_bytes=len(PAYLOAD))
    other = ModelSpec(name="other", url="https://example.invalid/o.onnx",
                      sha256=hashlib.sha256(b"other bytes").hexdigest(),
                      size_bytes=len(b"other bytes"))
    cache.ensure(SPEC, writer())
    cache.ensure(other, writer(b"other bytes"))
    removed = cache.evict_to_budget()
    assert cache.path_for(SPEC) in removed
    assert cache.path_for(other).exists()


def test_the_fp32_spec_is_the_same_architecture_at_full_precision():
    """Phase 2 measures whether int8 quantization is what blinds the detector.

    That comparison is only meaningful if the two checkpoints differ in
    precision and nothing else -- a different architecture would confound
    the one variable this cell exists to isolate.
    """
    from perception.model_cache import DEFAULT_MODEL, FP32_MODEL

    assert FP32_MODEL.name != DEFAULT_MODEL.name
    assert "rtdetr_r18vd" in FP32_MODEL.url, "must be the same checkpoint family"
    assert FP32_MODEL.url.endswith("model.onnx"), "the unquantized file"
    assert len(FP32_MODEL.sha256) == 64
    assert FP32_MODEL.size_bytes > DEFAULT_MODEL.size_bytes, (
        "fp32 weights are larger than int8; if this fails the URLs are swapped"
    )


def test_the_default_model_is_still_the_quantized_one():
    """Phase 2 ships nothing. Adding a second spec must not change which
    checkpoint the packaged app resolves.
    """
    from perception.model_cache import DEFAULT_MODEL

    assert DEFAULT_MODEL.name == "rtdetr_r18vd_quantized"
    assert DEFAULT_MODEL.sha256 == (
        "85703b0f56dbaceb89b21122e580fd11e11a879111fd727d0e9abdaf0e3620bf"
    )
    assert DEFAULT_MODEL.url == (
        "https://huggingface.co/onnx-community/rtdetr_r18vd/resolve/main/onnx/model_quantized.onnx"
    )
