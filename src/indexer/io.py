"""Filesystem primitives shared by stores and index implementations.

A leaf module for the same reason as ``indexer.textutil``: index
implementations need ``atomic_write``, and importing it from
``indexer.pipeline.stores`` executes ``indexer.pipeline.__init__``, which pulls
in the whole orchestration layer and closes an import cycle.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

__all__ = ["atomic_write", "decode_value", "encode_value"]


def encode_value(v: Any) -> Any:
    """A JSON-safe form of a field or metadata value, dates tagged.

    Shared by the codec and by every index that persists filter fields as JSON.
    Indexes used to write dates as bare ISO strings, so after a reload -- and,
    since the conversion happened on write, before one too -- a date filter
    compared a ``date`` against a ``str``, the evaluator raised, and the filter
    silently excluded every unit.
    """
    if isinstance(v, dict):
        if "__t" in v:  # already encoded
            return v
        return {str(k): encode_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [encode_value(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return [encode_value(x) for x in sorted(v, key=repr)]
    if isinstance(v, datetime):
        return {"__t": "datetime", "v": v.isoformat()}
    if isinstance(v, date):
        return {"__t": "date", "v": v.isoformat()}
    return v


def decode_value(v: Any) -> Any:
    if isinstance(v, dict):
        tag = v.get("__t")
        if tag == "datetime":
            return datetime.fromisoformat(v["v"])
        if tag == "date":
            return date.fromisoformat(v["v"])
        return {k: decode_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [decode_value(x) for x in v]
    return v


def atomic_write(path: Path, data: bytes) -> None:
    """Write via a temp file and rename.

    A build interrupted mid-write must not leave a half-written cache entry that
    later reads as valid. Rename is atomic within a filesystem; the temp file is
    created in the destination directory to keep it so.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
