"""Content-addressed disk cache for detector model weights.

Same shape as `map/cache.py`, applied to a different payload: a weights file
instead of a JSON blob, and a hash the cache *verifies* rather than one it
only uses as a filename. That verification is the reason this module exists
rather than just reusing `DiskCache` directly -- a download that got
truncated or tampered with must never be trusted as if it were a complete,
correct model.

Recency is tracked with the filesystem's own mtime, touched on every
`ensure()` hit, exactly as `map/cache.py` does. That keeps the cache a pile
of plain files under `root` with no separate index to corrupt or migrate.

A write goes through a per-writer temp file (name + pid + a random suffix)
that is hash-verified and then renamed into place, so a reader never
observes a partial or corrupt file and two writers racing on the same spec
cannot interleave into one broken one. A hash mismatch deletes the temp file
and raises -- a half-trusted weight file left on disk would be silently
loaded by a later run and is worse than no file at all.

Downloading is injected as `fetch: (url, dest) -> None`, never imported here.
The real fetcher (using `httpx`, already a project dependency) lives with its
caller; this module only ever calls the seam it is given, which is what
keeps backend tests offline.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

Fetch = Callable[[str, Path], None]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Identifies one detector weights file: where to get it, and the hash
    that proves a downloaded copy is the real thing."""

    name: str
    url: str
    sha256: str
    size_bytes: int


# onnx-community/rtdetr_r18vd, int8. Measured 58.9 ms on CPU here, the fastest
# of the variants tried; see the plan's provider table. RT-DETRv2 was exported and
# measured against this on 2026-08-20 and did NOT replace it: both models scored zero
# vehicle detections on real detector frames, and v1 is faster and 3.7x smaller. See
# docs/measurements/2026-08-20-detector-comparison.md.
DEFAULT_MODEL = ModelSpec(
    name="rtdetr_r18vd_quantized",
    url="https://huggingface.co/onnx-community/rtdetr_r18vd/resolve/main/onnx/model_quantized.onnx",
    sha256="85703b0f56dbaceb89b21122e580fd11e11a879111fd727d0e9abdaf0e3620bf",
    size_bytes=21_713_196,
)


# The same rtdetr_r18vd checkpoint at full precision. NOT the default and not
# shipped: Cycle 5 Phase 2 measures whether post-training int8 quantization is
# what blinds the detector on 9-20 px targets, and that comparison needs the
# unquantized weights of the SAME architecture -- a different model would
# confound the one variable the cell isolates. Cycle 4 measured an fp16
# variant's latency but never its scores, and its fp32 test was RT-DETRv2, a
# different architecture reporting only top-class names.
# Resolved through this same cache -- `ModelCache.ensure(FP32_MODEL,
# fetch_weights)` -- by the Phase 2 measurement script, exactly like
# DEFAULT_MODEL: the hash below is what proves the measured cell ran on
# these exact bytes, not on whatever happened to be at the URL that day.
# Hash and size verified by download on 2026-08-26; see
# docs/measurements/2026-08-26-cycle5-phase2-gates.md for the command.
FP32_MODEL = ModelSpec(
    name="rtdetr_r18vd_fp32",
    url="https://huggingface.co/onnx-community/rtdetr_r18vd/resolve/main/onnx/model.onnx",
    sha256="11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558",
    size_bytes=82_572_357,
)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelCache:
    """Resolves a `ModelSpec` to a local path, downloading at most once."""

    def __init__(self, root: Path, budget_bytes: int) -> None:
        self.root = Path(root)
        self.budget_bytes = budget_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self._sweep_orphaned_tmp()

    def path_for(self, spec: ModelSpec) -> Path:
        # The hash prefix, not the full digest, keeps names short while
        # still making a stale or wrong file impossible to mistake for the
        # one a caller asked for -- a spec change always yields a new path.
        return self.root / f"{spec.name}-{spec.sha256[:16]}.onnx"

    def _sweep_orphaned_tmp(self) -> None:
        """Delete writer temp files left behind by a process that never got
        to rename them into place. A `.tmp` file is only ever a write in
        progress -- nothing reads it -- so it is always safe to discard."""
        for p in self.root.glob("*.tmp"):
            try:
                p.unlink()
            except OSError:  # pragma: no cover - defensive
                pass

    def ensure(self, spec: ModelSpec, fetch: Fetch) -> Path:
        """Return a verified local path for `spec`, fetching only if needed.

        A cache hit is any existing file whose hash still matches -- that
        covers both "already downloaded" and "downloaded but corrupted since
        (disk error, user editing it by hand)", which must be refetched
        rather than trusted.

        Unlike `map/cache.py`'s `put()`, this does not evict on every call.
        A model file is tens of megabytes and a session typically resolves
        one, so folding eviction into every `ensure()` would mean walking
        and stat-ing the whole cache directory on a path that is otherwise
        just a hash check. Budget enforcement is the caller's explicit
        `evict_to_budget()` call instead.
        """
        path = self.path_for(spec)
        if path.exists() and _sha256_of(path) == spec.sha256:
            now = time.time()
            try:
                os.utime(path, (now, now))
            except OSError:  # pragma: no cover - defensive
                pass
            return path

        # Unique per call: two writers racing on the same spec (or the same
        # writer called twice in quick succession) must never share a temp
        # file, or one could rename the other's half-written content into
        # place.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            fetch(spec.url, tmp)
            digest = _sha256_of(tmp)
            if digest != spec.sha256:
                raise ValueError(
                    f"downloaded {spec.name!r} does not match expected "
                    f"sha256 (expected {spec.sha256}, got {digest})"
                )
            tmp.replace(path)
        finally:
            # Either fetch() failed, the hash check failed, or replace()
            # already moved it away -- in every case, no temp file should
            # be left behind.
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover - defensive
                    pass

        return path

    def evict_to_budget(self) -> list[Path]:
        """Delete least-recently-used files until the total is within
        `budget_bytes`. Returns what it removed."""
        entries: list[tuple[Path, float, int]] = []
        for p in self.root.glob("*.onnx"):
            try:
                st = p.stat()
            except OSError:  # pragma: no cover - defensive
                continue  # vanished between glob() and stat()
            entries.append((p, st.st_mtime, st.st_size))
        entries.sort(key=lambda e: e[1])  # oldest (least recently used) first

        removed: list[Path] = []
        total = sum(size for _, _, size in entries)
        while total > self.budget_bytes and entries:
            victim, _, size = entries.pop(0)
            total -= size
            try:
                victim.unlink()
            except OSError:
                continue  # already gone; the space was freed either way
            removed.append(victim)
        return removed
