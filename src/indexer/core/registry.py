"""The implementation registry.

"A new project is a config file, not a fork" is a claim about one thing: whether
adding an implementation requires editing frame code. It does not, because the
frame never names implementations. Config names them as strings; the registry
resolves them; nothing between the two has a list to update.

Registration carries a params type, and that is what keeps the config schema
honest. The config file's ``params`` block is an untyped mapping when the YAML
is read -- it has to be, or the schema would enumerate every implementation and
we would be back to forking. It is validated in a second pass against the
params type the implementation registered. So a typo in a param is still a
startup error with a field name and a line, not a ``KeyError`` forty minutes
into a build.

Third-party implementations register through the ``indexer.impls`` entry-point
group, so a package can be installed and used without the frame importing it
eagerly -- which also keeps optional heavy dependencies out of the import path
of a project that does not use them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Generic, TypeVar

from indexer.core.accounting import StageFingerprint
from indexer.core.errors import ConfigError
from indexer.core.ids import hash_obj

__all__ = ["REGISTRY", "Registration", "Registry", "register", "resolve"]

T = TypeVar("T")

#: The stages an implementation may register for. An implementation declares
#: one; the loader refuses to wire a reranker into the parse slot.
STAGES = frozenset(
    {
        "corpus",
        "parse",
        "segment",
        "enrich",
        "index",
        "route",
        "retrieve",
        "fuse",
        "rerank",
        "cache",
        "ledger",
        "judge",
        "bootstrap",
    }
)


@dataclass(frozen=True, slots=True)
class Registration(Generic[T]):
    """One registered implementation.

    ``version`` is the implementation's *behavioural* version, not the package
    version. Bump it whenever output can change -- a reworded prompt, a changed
    default, a fixed bug. It is half of the cache key, and the contract that
    makes caching safe rests on this being done honestly.
    """

    stage: str
    name: str
    version: str
    factory: Callable[..., T]
    #: Validates and normalises the ``params`` mapping from config. Returning a
    #: typed object is preferred; returning a dict is acceptable. Must be
    #: deterministic -- its output is hashed.
    params_model: Callable[[Mapping[str, Any]], Any] | None = None
    #: Free-text: what this implementation assumes, what it costs, when to use
    #: it. Surfaced by ``indexer impls list`` so the choice is documentable
    #: without reading source.
    summary: str = ""
    #: Extras required for this implementation, e.g. ``("parse-pymupdf",)``.
    requires: tuple[str, ...] = field(default_factory=tuple)

    def fingerprint(self, params: Mapping[str, Any]) -> StageFingerprint:
        return StageFingerprint(
            stage=self.stage,
            impl=self.name,
            version=self.version,
            params_hash=hash_obj(self.normalize(params)),
        )

    def normalize(self, params: Mapping[str, Any]) -> Any:
        if self.params_model is None:
            return dict(params)
        try:
            return self.params_model(params)
        except Exception as exc:
            raise ConfigError(f"invalid params for {self.stage}/{self.name}: {exc}") from exc

    def build(self, params: Mapping[str, Any], **extra: Any) -> T:
        """Instantiate. ``extra`` carries frame-supplied collaborators -- the
        cache, the accountant -- which are never configured by the user and so
        never appear in ``params`` or in the fingerprint."""
        return self.factory(self.normalize(params), **extra)


class Registry:
    """A stage-namespaced map of name -> registration."""

    __slots__ = ("_by_stage", "_loaded_entry_points")

    def __init__(self) -> None:
        self._by_stage: dict[str, dict[str, Registration[Any]]] = {}
        self._loaded_entry_points = False

    def add(self, reg: Registration[Any], *, replace: bool = False) -> None:
        if reg.stage not in STAGES:
            raise ConfigError(f"unknown stage {reg.stage!r}; expected one of {sorted(STAGES)}")
        slot = self._by_stage.setdefault(reg.stage, {})
        if reg.name in slot and not replace:
            existing = slot[reg.name]
            raise ConfigError(
                f"{reg.stage}/{reg.name} is already registered by {existing.factory!r}; "
                f"pass replace=True to override deliberately"
            )
        slot[reg.name] = reg

    def get(self, stage: str, name: str) -> Registration[Any]:
        self._load_entry_points()
        slot = self._by_stage.get(stage, {})
        if name not in slot:
            known = ", ".join(sorted(slot)) or "(none registered)"
            raise ConfigError(
                f"no {stage} implementation named {name!r}. Available: {known}. "
                f"If it lives in another package, install it and ensure it declares "
                f"an `indexer.impls` entry point."
            )
        return slot[name]

    def names(self, stage: str) -> tuple[str, ...]:
        self._load_entry_points()
        return tuple(sorted(self._by_stage.get(stage, {})))

    def iter_all(self) -> Iterator[Registration[Any]]:
        self._load_entry_points()
        for slot in self._by_stage.values():
            yield from slot.values()

    def _load_entry_points(self) -> None:
        """Import third-party implementations, once, on first lookup.

        Lazy on purpose: importing every registered implementation at startup
        would load torch for a project that only uses BM25.
        """
        if self._loaded_entry_points:
            return
        self._loaded_entry_points = True
        for ep in entry_points(group="indexer.impls"):
            try:
                ep.load()  # the module registers on import
            except Exception as exc:  # pragma: no cover - environment dependent
                raise ConfigError(
                    f"failed loading implementation package {ep.name!r}: {exc}"
                ) from exc


REGISTRY = Registry()


def register(
    stage: str,
    name: str,
    *,
    version: str,
    params_model: Callable[[Mapping[str, Any]], Any] | None = None,
    summary: str = "",
    requires: tuple[str, ...] = (),
    registry: Registry | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form. The only way implementations enter the frame."""

    def decorate(factory: Callable[..., T]) -> Callable[..., T]:
        (registry or REGISTRY).add(
            Registration(
                stage=stage,
                name=name,
                version=version,
                factory=factory,
                params_model=params_model,
                summary=summary,
                requires=requires,
            )
        )
        return factory

    return decorate


def resolve(stage: str, name: str, registry: Registry | None = None) -> Registration[Any]:
    return (registry or REGISTRY).get(stage, name)
