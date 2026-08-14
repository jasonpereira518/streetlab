"""Tests for map.cache.DiskCache.

Beyond the brief's happy-path/eviction tests, this file probes the
filesystem-adjacent edges a content-addressed cache actually meets in
practice: crash-orphaned temp files, a cache directory deleted out from
under a live process, and a budget too small to hold even one entry.
"""

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from map.cache import DiskCache, default_cache_dir


def test_put_then_get_round_trips(tmp_path):
    cache = DiskCache(tmp_path)
    cache.put("some-key", {"elements": [1, 2, 3]})
    assert cache.get("some-key") == {"elements": [1, 2, 3]}


def test_missing_key_returns_none(tmp_path):
    assert DiskCache(tmp_path).get("never-written") is None


def test_keys_are_hashed_so_hostile_keys_cannot_escape(tmp_path):
    cache = DiskCache(tmp_path)
    cache.put("../../etc/passwd", {"ok": True})
    written = list(tmp_path.iterdir())
    assert len(written) == 1
    assert written[0].parent == tmp_path
    assert cache.get("../../etc/passwd") == {"ok": True}


def test_corrupt_entry_is_treated_as_a_miss(tmp_path):
    cache = DiskCache(tmp_path)
    cache.put("k", {"a": 1})
    next(tmp_path.iterdir()).write_text("{not json")
    assert cache.get("k") is None


def test_eviction_keeps_total_under_budget(tmp_path):
    payload = {"blob": "x" * 2000}
    size = len(json.dumps(payload).encode())
    cache = DiskCache(tmp_path, budget_bytes=size * 3)
    for i in range(6):
        cache.put(f"key-{i}", payload)
    assert cache.total_bytes() <= size * 3


def test_eviction_drops_least_recently_used_first(tmp_path):
    payload = {"blob": "x" * 2000}
    size = len(json.dumps(payload).encode())
    cache = DiskCache(tmp_path, budget_bytes=size * 2)
    cache.put("old", payload)
    cache.put("new", payload)
    # Touch "old" so "new" becomes the least recently used.
    assert cache.get("old") == payload
    cache.put("newest", payload)
    assert cache.get("old") == payload
    assert cache.get("new") is None


def test_default_cache_dir_is_under_the_users_cache_root():
    assert "StreetLab" in str(default_cache_dir())


# --- Adversarial probing beyond the brief -----------------------------------


def test_written_entries_only_ever_use_json_suffix(tmp_path):
    """The on-disk footprint must be exactly the cache entries, nothing else.

    A writer temp file that survives under a name the eviction glob cannot
    see (e.g. a bare `.tmp` swap of the `.json` suffix) would be invisible to
    total_bytes()/eviction and leak forever. After a normal put(), no such
    file should remain.
    """
    cache = DiskCache(tmp_path)
    cache.put("k", {"a": 1})
    names = [p.name for p in tmp_path.iterdir()]
    assert len(names) == 1
    assert names[0].endswith(".json")


def test_orphaned_tmp_file_from_a_crashed_writer_is_swept_on_open(tmp_path):
    """Simulate a process killed between write and rename: a stray temp file.

    It must not survive across a fresh DiskCache being opened on the same
    directory (e.g. the app restarting), or it leaks disk space forever
    without ever counting against the budget.
    """
    tmp_path.mkdir(exist_ok=True)
    orphan = tmp_path / "deadbeef.json.31337.orphan.tmp"
    orphan.write_text('{"a": 1}')
    assert orphan.exists()

    DiskCache(tmp_path)  # reopening the cache should clean up after a crash

    assert not orphan.exists()


def _race_writers(monkeypatch, put_fn, long_payload, short_payload):
    """Run put_fn(long_payload) and put_fn(short_payload) on separate
    threads, with Path.write_text patched so:

    1. Both writers' target files are fully open (and, per O_TRUNC,
       truncated) before either one writes any content.
    2. The writer with the longer serialized payload always writes first;
       the shorter one is held back until the longer write has landed.

    That ordering is the exact failure mode of a *shared* temp filename:
    both opens truncate the same inode, the long write fills it, and the
    short write then overwrites only its own (shorter) prefix — leaving the
    long write's tail dangling behind it, unremoved, because O_TRUNC only
    happens at open() and neither writer opens again after the other's
    write. If put_fn's two calls target *different* files instead, this
    same forced ordering is harmless, since there is no shared inode for
    the short write to leave a stale tail in.
    """
    open_barrier = threading.Barrier(2, timeout=5)
    long_written = threading.Event()
    long_len = len(json.dumps(long_payload).encode())

    def ordered_write_text(self, data, *args, **kwargs):
        encoded = data.encode()
        fd = os.open(str(self), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            open_barrier.wait(timeout=5)
            if len(encoded) == long_len:
                os.write(fd, encoded)
                long_written.set()
            else:
                long_written.wait(timeout=5)
                os.write(fd, encoded)
        finally:
            os.close(fd)

    monkeypatch.setattr(Path, "write_text", ordered_write_text)

    threads = [
        threading.Thread(target=put_fn, args=(long_payload,)),
        threading.Thread(target=put_fn, args=(short_payload,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)


def test_shared_tmp_name_from_the_original_brief_can_corrupt_a_race(tmp_path, monkeypatch):
    """Negative control for the test below. Reproduces the plan's original
    put(), whose temp filename was `path.with_suffix(".tmp")` — the same
    name for every writer of a given key, unlike map.cache's per-writer
    unique name. Forcing the exact race _race_writers describes against
    that shared name really does corrupt the entry, proving the mechanism
    the fix closes is real and not hypothetical.
    """
    digest = hashlib.sha256(b"shared-key").hexdigest()
    final = tmp_path / f"{digest}.json"

    def brief_style_put(payload: dict) -> None:
        path = tmp_path / f"{digest}.json"
        tmp = path.with_suffix(".tmp")  # the plan's original scheme
        try:
            tmp.write_text(json.dumps(payload))
            tmp.replace(path)
        except OSError:
            # Matches the brief's own put(): whichever writer's replace()
            # loses the race finds its shared tmp file already renamed away
            # by the other and gives up quietly, exactly as the original
            # code does on any write failure.
            pass

    long_payload = {"writer": "a", "pad": "x" * 5000}
    short_payload = {"writer": "b"}
    _race_writers(monkeypatch, brief_style_put, long_payload, short_payload)

    with pytest.raises(json.JSONDecodeError):
        json.loads(final.read_text())


def test_concurrent_puts_to_the_same_key_do_not_corrupt_each_other(tmp_path, monkeypatch):
    """Positive control: the identical forced race from the test above,
    replayed against the real DiskCache.put(), must not corrupt anything —
    its temp filename is unique per writer (pid + uuid4), so the two
    writers never share the inode the negative control's corruption
    depends on.
    """
    cache = DiskCache(tmp_path)
    long_payload = {"writer": "a", "pad": "x" * 5000}
    short_payload = {"writer": "b"}

    _race_writers(
        monkeypatch,
        lambda payload: cache.put("shared-key", payload),
        long_payload,
        short_payload,
    )

    # Whichever writer's rename lands last wins; what matters is the result
    # is one complete, valid write, never a corrupt or missing entry.
    assert cache.get("shared-key") in (long_payload, short_payload)


def test_put_recreates_a_cache_dir_deleted_mid_run(tmp_path):
    """If the OS cache folder is cleared while the app is running, the next
    put() should re-establish it rather than silently going dark for the
    rest of the process lifetime."""
    cache = DiskCache(tmp_path)
    cache.put("k1", {"a": 1})
    for entry in tmp_path.iterdir():
        entry.unlink()
    tmp_path.rmdir()
    assert not tmp_path.exists()

    cache.put("k2", {"b": 2})

    assert tmp_path.exists()
    assert cache.get("k2") == {"b": 2}


def test_get_on_a_deleted_cache_dir_is_a_miss_not_a_crash(tmp_path):
    cache = DiskCache(tmp_path)
    cache.put("k", {"a": 1})
    for entry in tmp_path.iterdir():
        entry.unlink()
    tmp_path.rmdir()

    assert cache.get("k") is None


def test_total_bytes_survives_a_file_vanishing_mid_scan(tmp_path, monkeypatch):
    """total_bytes() must not raise if a file it globbed is removed by a
    concurrent evictor between glob() listing it and stat() being called on
    it. Deleting the file before the scan starts wouldn't exercise this —
    glob() would simply never list it — so the deletion is injected via
    Path.stat() itself, at the exact moment total_bytes() reaches that file.
    """
    cache = DiskCache(tmp_path)
    cache.put("k1", {"a": 1})
    cache.put("k2", {"b": 2})
    victim = cache._path("k1")
    survivor = cache._path("k2")
    expected_total = survivor.stat().st_size

    real_stat = Path.stat
    triggered = False

    def vanish_on_stat(self, *args, **kwargs):
        nonlocal triggered
        # Path.exists() is itself implemented via stat(), so checking
        # existence here would recurse into this same wrapper; a one-shot
        # flag does the job without that trap.
        if self == victim and not triggered:
            triggered = True
            victim.unlink()  # a concurrent evictor wins the race right here
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", vanish_on_stat)

    # Not just "didn't raise" — the survivor must be counted in full and the
    # vanished victim not counted at all.
    assert cache.total_bytes() == expected_total


def test_budget_smaller_than_a_single_entry_never_exceeds_budget(tmp_path):
    """The budget is a hard cap: an entry too big to fit is evicted right
    back out rather than being allowed to push the cache over budget."""
    payload = {"blob": "x" * 5000}
    size = len(json.dumps(payload).encode())
    cache = DiskCache(tmp_path, budget_bytes=size // 2)

    cache.put("too-big", payload)

    assert cache.total_bytes() <= size // 2
    assert cache.get("too-big") is None


def test_zero_budget_holds_nothing(tmp_path):
    cache = DiskCache(tmp_path, budget_bytes=0)
    cache.put("k", {"a": 1})
    assert cache.total_bytes() == 0


def test_eviction_tolerates_an_entry_disappearing_before_unlink(tmp_path, monkeypatch):
    """One candidate victim vanishing (a concurrent evictor already removed
    it) must not abort the rest of the eviction pass. If it did, entries that
    still need evicting would survive and the cache would stay over budget —
    only reproducible by deleting the *second* victim while _evict() is in
    the middle of removing the *first*, not by deleting it beforehand (which
    would just make _evict() skip it cleanly, never touching the unlink()
    error path at all).
    """
    payload = {"blob": "x" * 2000}
    size = len(json.dumps(payload).encode())
    cache = DiskCache(tmp_path, budget_bytes=size * 10)  # nothing evicted yet
    for name in ("old", "mid", "new", "newest"):
        cache.put(name, payload)
    # Force a deterministic LRU order regardless of filesystem mtime
    # resolution (some filesystems only tick once per second).
    paths = {name: cache._path(name) for name in ("old", "mid", "new", "newest")}
    for i, name in enumerate(("old", "mid", "new", "newest")):
        t = 1_000_000 + i * 1000
        os.utime(paths[name], (t, t))

    real_unlink = Path.unlink

    def sneaky_unlink(self, *args, **kwargs):
        if self == paths["old"] and paths["mid"].exists():
            paths["mid"].unlink()  # a concurrent evictor wins this race
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", sneaky_unlink)

    # Only "newest" should fit; evicting it requires removing old, mid, and
    # new, even though mid disappears out from under the pass.
    cache.budget_bytes = size
    cache._evict()

    assert cache.total_bytes() <= size
    assert not paths["new"].exists()
    assert paths["newest"].exists()


def test_put_overwrites_existing_key_and_refreshes_recency(tmp_path):
    cache = DiskCache(tmp_path)
    cache.put("k", {"v": 1})
    cache.put("k", {"v": 2})
    assert cache.get("k") == {"v": 2}
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_get_touches_mtime_so_lru_order_updates(tmp_path):
    cache = DiskCache(tmp_path)
    cache.put("k", {"v": 1})
    path = next(tmp_path.glob("*.json"))
    old_mtime = path.stat().st_mtime
    os.utime(path, (old_mtime - 1000, old_mtime - 1000))

    cache.get("k")

    assert path.stat().st_mtime > old_mtime - 1000


def test_entry_with_invalid_utf8_bytes_is_treated_as_a_miss(tmp_path):
    """A corrupt entry is not always corrupt *JSON* — it could be corrupt
    *bytes* (a truncated write, a filesystem hiccup). Either way it must
    degrade to a miss, not raise."""
    cache = DiskCache(tmp_path)
    cache.put("k", {"a": 1})
    path = next(tmp_path.glob("*.json"))
    path.write_bytes(b"\xff\xfe\x00garbage")

    assert cache.get("k") is None
