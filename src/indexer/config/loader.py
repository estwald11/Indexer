"""Loading, overlaying, interpolating and hashing configs.

Four jobs, in order:

1.  **Overlay.** ``extends:`` merges a base file under this one. Ablation arms
    are overlays, which is how two arms are provably identical except in the
    keys that were overridden.
2.  **Interpolate.** ``${env:VAR}`` for secrets and ``${paths.store}`` for
    internal references. A config file must be committable, so no secret is
    ever written in one -- and the *interpolated* values are redacted again
    before the manifest is written.
3.  **Validate, phase 1.** Pydantic: shape, types, cross-stage coherence.
4.  **Hash.** The canonical hash of the resolved config, which identifies the
    index the config builds and appears in every manifest.

Phase 2 validation -- implementation names and their params, against the
registry -- happens in ``resolve_impl``, at pipeline construction. It is
separate because it requires the implementations to be importable, and a config
should be readable and diffable without installing a GPU stack.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from indexer.config.schema import Config, ImplSpec
from indexer.core.errors import ConfigError
from indexer.core.ids import ContentHash, hash_obj
from indexer.core.registry import Registration, Registry, resolve

__all__ = [
    "config_hash",
    "load",
    "load_mapping",
    "redact",
    "resolve_impl",
    "with_overrides",
]

_INTERP = re.compile(r"\$\{([^}]+)\}")

#: Keys whose values never reach a manifest or a log. Matched case-insensitively
#: as substrings, so ``openai_api_key`` and ``auth_token`` are both covered.
_SECRET_HINTS = ("key", "token", "secret", "password", "credential")


def load_mapping(path: str | Path, *, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a config file and its ``extends`` chain into one mapping."""
    p = Path(path).expanduser().resolve()
    if p in _seen:
        chain = " -> ".join(str(s) for s in (*_seen, p))
        raise ConfigError(f"circular extends: {chain}")
    if not p.exists():
        raise ConfigError(f"config not found: {p}")

    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: top level must be a mapping")

    base_ref = raw.pop("extends", None)
    if base_ref:
        base_path = (p.parent / str(base_ref)).resolve()
        base = load_mapping(base_path, _seen=(*_seen, p))
        raw = deep_merge(base, raw)
    return raw


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive merge. Mappings merge; **lists replace**.

    Lists replacing rather than concatenating is the important half. An overlay
    that sets ``enrichers:`` means *these enrichers*, not "these as well as the
    base's" -- appending would make "run without the contextualiser" impossible
    to express, and that is the single most important ablation arm.
    """
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def interpolate(obj: Any, root: Mapping[str, Any]) -> Any:
    """Expand ``${env:VAR}``, ``${env:VAR:default}`` and ``${dotted.path}``."""
    if isinstance(obj, str):
        return _INTERP.sub(lambda m: _expand(m.group(1), root), obj)
    if isinstance(obj, Mapping):
        return {k: interpolate(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [interpolate(v, root) for v in obj]
    return obj


def _expand(ref: str, root: Mapping[str, Any]) -> str:
    if ref.startswith("env:"):
        _, _, rest = ref.partition(":")
        name, _, default = rest.partition(":")
        val = os.environ.get(name)
        if val is None:
            if not default:
                raise ConfigError(
                    f"config references ${{env:{name}}} but it is unset and has no default"
                )
            return default
        return val
    cur: Any = root
    for part in ref.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise ConfigError(f"config references ${{{ref}}}, which does not exist")
        cur = cur[part]
    if isinstance(cur, (Mapping, list)):
        raise ConfigError(f"${{{ref}}} resolves to a {type(cur).__name__}, not a scalar")
    return str(cur)


def with_overrides(raw: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply dotted-path overrides. The ablation runner's only lever.

    ``{"ingestion.enrich.enabled": False}`` sets that key and nothing else --
    which is what makes an ablation arm's label ("enrich off") a true statement
    about the difference between two runs.

    List elements are addressed by index (``indexes.0.enabled``) or, more
    usefully, by name (``indexes[dense].enabled``), because an index's position
    in a list is not a stable thing to write in an ablation spec.

    **Pure**: the input is deep-copied. A shallow copy leaves nested mappings
    and lists shared, so writing through a dotted path mutates the caller's
    config -- and an ablation runner applying arms to one snapshot would leak
    each arm's overrides into the next. The arms would then differ in ways the
    override lists do not state, which is precisely the property the delta table
    depends on.
    """
    out: dict[str, Any] = deepcopy(dict(raw))
    for dotted, value in overrides.items():
        cur: Any = out
        parts = _split_path(dotted)
        for i, part in enumerate(parts[:-1]):
            cur = _descend(cur, part, dotted, create=True, nxt=parts[i + 1])
        last = parts[-1]
        if isinstance(last, int):
            if not isinstance(cur, list) or last >= len(cur):
                raise ConfigError(f"override {dotted!r}: index {last} out of range")
            cur[last] = value
        elif isinstance(cur, list):
            # Replace a named element wholesale: `indexes[visual]` = {...}
            target = _by_name(cur, last, dotted)
            cur[cur.index(target)] = value
        else:
            cur[last] = value
    return out


def _split_path(dotted: str) -> list[str | int]:
    parts: list[str | int] = []
    for seg in dotted.split("."):
        m = re.fullmatch(r"([^\[\]]+)\[([^\[\]]+)\]", seg)
        if m:
            parts.append(m.group(1))
            key = m.group(2)
            parts.append(int(key) if key.isdigit() else key)
        elif seg.isdigit():
            parts.append(int(seg))
        else:
            parts.append(seg)
    return parts


def _descend(cur: Any, part: str | int, dotted: str, *, create: bool, nxt: str | int) -> Any:
    if isinstance(part, int):
        if not isinstance(cur, list) or part >= len(cur):
            raise ConfigError(f"override {dotted!r}: index {part} out of range")
        return cur[part]
    if isinstance(cur, list):
        return _by_name(cur, part, dotted)
    if not isinstance(cur, MutableMapping):
        raise ConfigError(f"override {dotted!r}: {part!r} is not a mapping")
    if part not in cur:
        if not create:
            raise ConfigError(f"override {dotted!r}: {part!r} does not exist")
        cur[part] = [] if isinstance(nxt, int) else {}
    return cur[part]


def _by_name(items: list[Any], name: str | int, dotted: str) -> Any:
    for it in items:
        if isinstance(it, Mapping) and (it.get("name") == name or it.get("impl") == name):
            return it
    raise ConfigError(f"override {dotted!r}: no list element named {name!r}")


def load(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[Config, dict[str, Any]]:
    """Load and validate. Returns the model and the resolved mapping.

    The mapping is returned alongside because the manifest records the resolved
    config verbatim, and round-tripping through the model would silently drop
    the ``params`` blocks that phase 1 keeps open.
    """
    raw = load_mapping(path)
    if overrides:
        raw = with_overrides(raw, overrides)
    resolved = interpolate(raw, raw)
    try:
        cfg = Config.model_validate(resolved)
    except Exception as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return cfg, resolved


def config_hash(resolved: Mapping[str, Any]) -> ContentHash:
    """Identity of a configuration. Appears in every manifest.

    Computed over the redacted form, so rotating an API key does not invalidate
    an index -- a secret is not part of what a config *means*.
    """
    return hash_obj(redact(resolved))


def redact(obj: Any, *, placeholder: str = "<redacted>") -> Any:
    """Strip secret-looking values. Applied before hashing, logging or writing."""
    if isinstance(obj, Mapping):
        return {
            k: placeholder
            if isinstance(k, str) and any(h in k.lower() for h in _SECRET_HINTS)
            else redact(v, placeholder=placeholder)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact(v, placeholder=placeholder) for v in obj]
    return obj


def resolve_impl(
    stage: str, spec: ImplSpec | Mapping[str, Any], *, registry: Registry | None = None
) -> tuple[Registration[Any], Any]:
    """Phase 2: look the implementation up and validate its params.

    Returns the registration and the normalised params, so the caller can build
    the instance and compute its fingerprint from the same normalised value --
    building from one and fingerprinting the other is how a cache ends up keyed
    on something other than what ran.
    """
    impl = spec.impl if isinstance(spec, ImplSpec) else spec["impl"]
    params = spec.params if isinstance(spec, ImplSpec) else spec.get("params", {})
    reg = resolve(stage, impl, registry)
    return reg, reg.normalize(params)
