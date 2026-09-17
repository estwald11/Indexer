"""Filesystem primitives shared by stores and index implementations.

A leaf module for the same reason as ``indexer.textutil``: index
implementations need ``atomic_write``, and importing it from
``indexer.pipeline.stores`` executes ``indexer.pipeline.__init__``, which pulls
in the whole orchestration layer and closes an import cycle.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["atomic_write"]


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
