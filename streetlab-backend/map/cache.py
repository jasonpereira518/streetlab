"""Content-addressed disk cache for fetched map data.

Two jobs: make a repeated location load instant and offline, and keep the
footprint bounded. Keys are hashed rather than used as filenames, so an
arbitrary user-entered query can never traverse out of the cache directory or
produce a name the filesystem rejects.

Recency is tracked with the filesystem's own mtime, touched on read. That
keeps the cache a pile of plain JSON files with no index to corrupt or
migrate.

A write goes through a per-writer temp file (name + pid + a random suffix)
that is renamed into place, so a reader never observes a partial write and
two writers racing on the same key cannot interleave into one corrupt file.
Because the temp name does not end in `.json`, it is invisible to the `*.json`
glob that both `total_bytes()` and eviction use — which also means a process
killed between the write and the rename leaves an orphan that would otherwise
sit on disk forever, uncounted and unevictable. `__init__` sweeps any such
orphans left by a previous crash before the cache is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

log = logging.getLogger("streetlab.map")

DEFAULT_BUDGET_BYTES = 99 * 1024 * 1024


def default_cache_dir() -> Path:
    """Platform cache location for map extracts."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "StreetLab" / "osm"
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "StreetLab" / "osm"


class DiskCache:
    def __init__(self, root: Path, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> None:
        self.root = Path(root)
        self.budget_bytes = budget_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self._sweep_orphaned_tmp()

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.root / f"{digest}.json"

    def _sweep_orphaned_tmp(self) -> None:
        """Delete writer temp files left behind by a process that never got
        to rename them into place. A `.tmp` file is only ever a write in
        progress — nothing reads it — so it is always safe to discard."""
        for p in self.root.glob("*.tmp"):
            try:
                p.unlink()
            except OSError:  # pragma: no cover - defensive
                pass

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        # Mark as recently used.
        now = time.time()
        try:
            os.utime(path, (now, now))
        except OSError:  # pragma: no cover - defensive
            pass
        return payload

    def put(self, key: str, payload: dict) -> None:
        path = self._path(key)
        # Unique per call: two writers racing on the same key (or the same
        # writer called twice in quick succession) must never share a temp
        # file, or one could rename the other's half-written content into
        # place.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            # Deliberate extra resilience, not a fix for a promise this
            # module makes elsewhere: nothing requires the cache to survive
            # its own directory being deleted out from under a live process
            # (the OS clearing it, the user emptying it by hand). Recreating
            # it here is cheap and avoids put() going silently dark for the
            # rest of the process's life, so it is worth doing anyway.
            self.root.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload))
            tmp.replace(path)
        except OSError:
            log.warning("could not write cache entry; continuing uncached")
            return
        finally:
            # write_text() may have succeeded while replace() failed (e.g. a
            # racing rmdir of the cache dir); do not leave that half behind.
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover - defensive
                    pass
        self._evict()

    def total_bytes(self) -> int:
        total = 0
        for p in self.root.glob("*.json"):
            try:
                total += p.stat().st_size
            except OSError:
                continue  # vanished under us; another writer's problem now
        return total

    def _evict(self) -> None:
        entries: list[tuple[Path, float, int]] = []
        for p in self.root.glob("*.json"):
            try:
                st = p.stat()
            except OSError:  # pragma: no cover - defensive
                continue  # vanished between glob() and stat()
            entries.append((p, st.st_mtime, st.st_size))
        entries.sort(key=lambda e: e[1])  # oldest (least recently used) first

        total = sum(size for _, _, size in entries)
        while total > self.budget_bytes and entries:
            victim, _, size = entries.pop(0)
            # Account for the space as freed whether or not our own unlink
            # is the one that freed it: a concurrent evictor may have
            # already removed this exact victim, and its bytes are gone
            # from disk either way. Subtracting only on success would leave
            # `total` overcounting and evict more entries than necessary.
            total -= size
            try:
                victim.unlink()
            except OSError:
                continue  # already gone; the space was freed either way
