"""Content-addressed caching at every stage boundary.

Invariant 2 -- ingestion cost is paid once, query cost forever -- only holds if
"once" is really once, including across runs, across config edits that do not
touch the stage, and across machines. That requires the cache key to be a
function of exactly what the stage read.

    key = H(stage, impl, version, params_hash, input_hash, scope_hash)

``input_hash`` is the content hash of the stage's input. ``scope_hash`` covers
anything *else* the implementation read: for a ``DOCUMENT``-scoped enricher, the
parent document's hash. A unit-scoped enricher has an empty scope hash and
therefore survives edits elsewhere in its document -- which is the difference
between reprocessing 10 documents and reprocessing 10 units.

The rule that makes this sound: **a stage must be a pure function of its
declared inputs.** An implementation that reads the clock, a random seed or a
mutable external service without folding it into ``params_hash`` will serve
wrong results from cache. This is stated as a contract because it cannot be
enforced by types.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from indexer.core.accounting import StageFingerprint
from indexer.core.ids import ContentHash, merge_hashes

__all__ = [
    "ArtifactStore",
    "CacheKey",
    "CacheStore",
    "NullCache",
    "cache_key",
    "content_uri",
]

CacheKey = str


def cache_key(
    fingerprint: StageFingerprint,
    input_hash: str,
    *,
    scope_hash: str = "",
) -> CacheKey:
    """Derive the cache key for one stage call.

    Namespaced by stage so two stages with identical inputs never collide, and
    length-prefixed inside ``merge_hashes`` so no concatenation ambiguity exists.
    """
    return merge_hashes(
        fingerprint.stage,
        fingerprint.impl,
        fingerprint.version,
        fingerprint.params_hash,
        input_hash,
        scope_hash,
    )


@runtime_checkable
class CacheStore(Protocol):
    """Byte-oriented content-addressed store.

    Bytes rather than objects on purpose: the cache must survive a process
    restart and a library upgrade, so values are serialised by the caller in a
    format it owns. A pickle-based cache would make every dataclass change a
    silent corpus-wide invalidation, or worse, a silent unpickling error.
    """

    def get(self, key: CacheKey) -> bytes | None: ...

    def put(self, key: CacheKey, value: bytes) -> None: ...

    def has(self, key: CacheKey) -> bool: ...

    def delete(self, key: CacheKey) -> None: ...

    def iter_keys(self, prefix: str = "") -> Iterator[CacheKey]: ...


class NullCache:
    """Caches nothing. The behaviour of ``cache.enabled: false``.

    Exists so that "no cache" is a store like any other rather than a branch in
    every stage. Also the right store for a benchmark that must measure cold
    cost, which invariant 6 asks for regularly.
    """

    __slots__ = ()

    def get(self, key: CacheKey) -> bytes | None:
        return None

    def put(self, key: CacheKey, value: bytes) -> None:
        return None

    def has(self, key: CacheKey) -> bool:
        return False

    def delete(self, key: CacheKey) -> None:
        return None

    def iter_keys(self, prefix: str = "") -> Iterator[CacheKey]:
        return iter(())


@runtime_checkable
class ArtifactStore(Protocol):
    """Where large binaries live: page rasters, figure crops, model artefacts.

    Separate from ``CacheStore`` because the lifetimes differ. A cache entry may
    be evicted at any time and the pipeline recomputes. An artifact is
    *referenced* by a ``MediaRef`` held in an index; evicting it breaks
    provenance. Conflating the two makes a cache clear corrupt the corpus.
    """

    def put(self, content: bytes, media_type: str) -> str: ...

    def get(self, uri: str) -> bytes: ...

    def exists(self, uri: str) -> bool: ...


def content_uri(h: ContentHash, media_type: str) -> str:
    """Canonical artifact URI form: ``indexer://artifact/<hash>?type=<mt>``."""
    return f"indexer://artifact/{h}?type={media_type}"
