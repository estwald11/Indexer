"""Shared plumbing for anything that registers with the frame.

A leaf module, not part of ``indexer.impls``: the evaluation harness registers
judges and bootstrappers too, and ``indexer.eval`` must not depend on the
reference implementations. A harness that imports a specific BM25 index cannot
evaluate a system this library did not build -- which is usually the first
comparison anyone asks for.

``StageImpl`` gives every implementation its fingerprint for free, computed from
declared class attributes plus its validated params. That matters more than it
looks: the fingerprint is half the cache key, and an implementation that forgets
to include a parameter in it will serve stale results for every value of that
parameter. Deriving it from the params dict means forgetting is not possible.

The ``VERSION`` attribute is the other half, and the one that still requires
discipline: bump it whenever behaviour changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from typing import Any, ClassVar

from indexer.core.accounting import StageFingerprint
from indexer.core.ids import hash_obj

__all__ = ["StageImpl", "dataclass_params"]


class StageImpl:
    """Base for every reference implementation."""

    STAGE: ClassVar[str] = ""
    IMPL: ClassVar[str] = ""
    VERSION: ClassVar[str] = "1"

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self._params: dict[str, Any] = dict(params or {})

    def fingerprint(self) -> StageFingerprint:
        return StageFingerprint(
            stage=self.STAGE,
            impl=self.IMPL,
            version=self.VERSION,
            params_hash=hash_obj(self._params),
        )

    def param(self, name: str, default: Any = None) -> Any:
        return self._params.get(name, default)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.STAGE}/{self.IMPL}@{self.VERSION}>"


def dataclass_params(cls: type) -> Any:
    """Build a params validator from a dataclass.

    Unknown keys raise (``TypeError`` from the constructor, wrapped as a
    ``ConfigError`` by the registry), which is the behaviour that makes an open
    params block safe: the schema does not enumerate implementations, but a
    misspelled parameter is still a startup error naming the field.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    known = {f.name for f in fields(cls)}

    def normalize(d: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(d) - known
        if unknown:
            raise TypeError(f"unknown parameter(s) {sorted(unknown)}; accepted: {sorted(known)}")
        return asdict(cls(**d))

    return normalize
