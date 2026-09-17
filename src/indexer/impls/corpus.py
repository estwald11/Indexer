"""Corpus scanners."""

from __future__ import annotations

import fnmatch
import mimetypes
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indexer.core.document import SourceDocument
from indexer.core.ids import DocumentId, hash_bytes, make_document_id
from indexer.core.registry import register
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["FilesystemScanner"]

_EXTRA_TYPES = {
    ".md": "text/markdown",
    ".rst": "text/x-rst",
    ".txt": "text/plain",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "text/toml",
    ".cfg": "text/plain",
    ".in": "text/plain",
}


class _FileDocument(SourceDocument):
    """A ``SourceDocument`` that knows how to read itself.

    Lazy: a scan of 100k files must not read 100k files, because the plan only
    needs hashes and the majority of documents are usually unchanged.
    """

    __slots__ = ()

    def load(self) -> bytes:
        return Path(self.metadata["path"]).read_bytes()


@dataclass(frozen=True, slots=True)
class FilesystemParams:
    root: str = "."
    include: list[str] = field(default_factory=lambda: ["**/*"])
    exclude: list[str] = field(default_factory=list)
    max_bytes: int = 8_000_000
    follow_symlinks: bool = False
    #: field name -> regex over the path relative to root; group 1 is the value.
    #: Corpora encode real facts in their layout (tenant, matter, package,
    #: year), and those are scanner-supplied metadata -- known, not inferred --
    #: so they belong here rather than in an extraction model that would only
    #: guess at what the directory already states.
    path_metadata: dict[str, str] = field(default_factory=dict)


@register(
    "corpus",
    "filesystem",
    version="1",
    params_model=dataclass_params(FilesystemParams),
    summary="Walk a directory. Document ids derive from the path relative to root.",
)
def _make_filesystem(params: dict[str, Any], **_: Any) -> FilesystemScanner:
    return FilesystemScanner(params)


class FilesystemScanner(StageImpl):
    STAGE, IMPL, VERSION = "corpus", "filesystem", "1"

    def __init__(self, params: dict[str, Any], namespace: str = "") -> None:
        super().__init__(params)
        self.root = Path(params.get("root", ".")).resolve()
        self.include: list[str] = params.get("include", ["**/*"])
        self.exclude: list[str] = params.get("exclude", [])
        self.max_bytes: int = params.get("max_bytes", 8_000_000)
        self.namespace = namespace
        self._path_meta = {k: re.compile(v) for k, v in (params.get("path_metadata") or {}).items()}

    def scan(self) -> Iterable[SourceDocument]:
        if not self.root.exists():
            return
        seen: set[Path] = set()
        for pattern in self.include:
            for path in sorted(self.root.glob(pattern)):
                if not path.is_file() or path in seen:
                    continue
                rel = path.relative_to(self.root).as_posix()
                if any(fnmatch.fnmatch(rel, ex) for ex in self.exclude):
                    continue
                if path.stat().st_size > self.max_bytes:
                    continue
                seen.add(path)
                data = path.read_bytes()
                # The document id derives from the path *relative to root*, not
                # the absolute path. An absolute path makes ids machine-specific,
                # so a rebuild on another machine reprocesses the whole corpus
                # while appearing to work.
                yield _FileDocument(
                    document_id=make_document_id(rel, namespace=self.namespace),
                    source_uri=path.as_uri(),
                    content_hash=hash_bytes(data),
                    media_type=_media_type(path),
                    size_bytes=len(data),
                    metadata={
                        "path": str(path),
                        "relpath": rel,
                        "name": path.name,
                        **self._path_metadata(rel),
                    },
                )

    def _path_metadata(self, rel: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, rx in self._path_meta.items():
            m = rx.search(rel)
            if m:
                out[name] = m.group(1) if m.groups() else m.group(0)
        return out


def _media_type(path: Path) -> str:
    if path.suffix.lower() in _EXTRA_TYPES:
        return _EXTRA_TYPES[path.suffix.lower()]
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def document_id_for(root: str | Path, path: str | Path, namespace: str = "") -> DocumentId:
    """The id a scan would assign. Used by the golden-set bootstrapper."""
    rel = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    return make_document_id(rel, namespace=namespace)
