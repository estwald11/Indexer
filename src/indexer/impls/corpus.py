"""Corpus scanners.

``filesystem`` walks a directory. For a company archive it also has to do what
a folder of files does not make obvious:

*Not read what has not changed.* A scan hashed every file on every run -- the
whole archive read end to end to find the ten documents that changed. File size
and modification time are now remembered per path, and a file whose stat is
unchanged keeps its hash without being opened.

*Take known facts from beside the document.* A sidecar (``contract.pdf`` +
``contract.pdf.meta.json``) supplies the source system's id, the ACL, and any
metadata a DMS export carries. Path rules assign ACLs by folder. These are
facts, not guesses, and they reach every index as filterable fields.

*Open containers.* A signed ``.p7m`` holds the document; an email holds its
attachments; a PEC holds the original message and its attachments; a zip holds
folders of all of these. With ``expand`` the scanner yields what is inside, each
with its own stable id, the container's facts (tenant, ACL, the email's
subject and sender) inherited, and a link to its parent.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indexer.core.document import PARENT_KEY, SourceDocument
from indexer.core.ids import DocumentId, hash_bytes, hash_obj, make_document_id
from indexer.core.registry import register
from indexer.impls.containers import (
    media_type_for,
    parse_email,
    unwrap_p7m,
    zip_members,
)
from indexer.io import atomic_write
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["FilesystemScanner", "document_id_for", "load_located"]

#: Attachments that are part of the envelope rather than documents: the PEC
#: provider's certification data (kept as metadata) and detached signatures.
_ENVELOPE_PARTS = frozenset({"daticert.xml", "smime.p7s"})
_CONTAINERS = ("p7m", "eml", "zip")


class _FileDocument(SourceDocument):
    """A ``SourceDocument`` that knows how to read itself.

    Lazy: a scan of 100k files must not read 100k files, because the plan only
    needs hashes and the majority of documents are usually unchanged. A
    document inside a container is reached by replaying its locator -- unwrap
    the envelope, take attachment 2 -- on the container's bytes.
    """

    __slots__ = ()

    def load(self) -> bytes:
        return load_located(Path(self.metadata["path"]), self.metadata.get("_locator", []))


def load_located(path: Path, locator: Sequence[Sequence[Any]]) -> bytes:
    """The bytes at ``locator`` inside the file at ``path``."""
    data = path.read_bytes()
    for step in locator:
        kind = step[0]
        if kind == "p7m":
            data = unwrap_p7m(data).content
        elif kind == "eml":
            data = parse_email(data).attachments[int(step[1])].payload
        elif kind == "zip":
            data = dict(zip_members(data))[str(step[1])]
        else:  # pragma: no cover - a locator this module did not write
            raise ValueError(f"unknown locator step {step!r}")
    return data


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
    #: Remember each file's size and modification time, and do not re-read a
    #: file whose stat is unchanged. Off means every scan hashes every byte.
    stat_cache: bool = True
    #: A JSON object in ``<document><suffix>`` is merged into the document's
    #: metadata. ``document_id`` in it replaces the path-derived id (the source
    #: system's key survives a move); ``acl`` sets the access list.
    sidecar_suffix: str = ".meta.json"
    #: [{"pattern": "hr/**", "acl": ["group:hr"]}, ...]: glob over the relative
    #: path, first match wins. A sidecar's ``acl`` overrides.
    acl_rules: list[dict[str, Any]] = field(default_factory=list)
    #: The ACL of a document no rule or sidecar covers. ``None`` leaves it
    #: without one; what that means is the query side's ``acl`` setting.
    default_acl: list[str] | None = None
    #: Containers to open: any of p7m, eml, zip.
    expand: list[str] = field(default_factory=list)
    #: How deep containers nest before the scanner stops opening them.
    max_depth: int = 3


@register(
    "corpus",
    "filesystem",
    version="2",
    params_model=dataclass_params(FilesystemParams),
    summary=(
        "Walk a directory. Stable path-derived ids (or a sidecar's source id), "
        "stat-cached hashing, sidecar metadata and ACLs, p7m/eml/PEC/zip expansion."
    ),
)
def _make_filesystem(params: dict[str, Any], **kw: Any) -> FilesystemScanner:
    return FilesystemScanner(
        params, namespace=kw.get("namespace", ""), state_dir=kw.get("state_dir")
    )


class FilesystemScanner(StageImpl):
    STAGE, IMPL, VERSION = "corpus", "filesystem", "2"

    def __init__(
        self,
        params: dict[str, Any],
        namespace: str = "",
        state_dir: str | Path | None = None,
    ) -> None:
        super().__init__(params)
        self.root = Path(params.get("root", ".")).resolve()
        self.include: list[str] = params.get("include", ["**/*"])
        self.exclude: list[str] = params.get("exclude", [])
        self.max_bytes: int = params.get("max_bytes", 8_000_000)
        self.namespace = namespace
        self._path_meta = {k: re.compile(v) for k, v in (params.get("path_metadata") or {}).items()}
        self.sidecar_suffix = str(params.get("sidecar_suffix", ".meta.json") or "")
        self.acl_rules: list[dict[str, Any]] = list(params.get("acl_rules") or [])
        self.default_acl = params.get("default_acl")
        self.expand = [e for e in (params.get("expand") or []) if e in _CONTAINERS]
        self.max_depth = int(params.get("max_depth", 3))
        self.use_stat_cache = bool(params.get("stat_cache", True))
        # One state file per distinct scan configuration: what a file expands
        # into depends on `expand` and `max_depth`, so a changed setting must
        # not reuse descriptors computed under the old one.
        self._state_path = (
            Path(state_dir) / f"{namespace or 'default'}-{hash_obj(self._params)[7:23]}.json"
            if state_dir is not None
            else None
        )
        self._memory_state: dict[str, Any] = {}
        #: Files actually opened by the last scan, for tests and diagnostics.
        self.files_read = 0

    # ------------------------------------------------------------------ scan

    def scan(self) -> Iterable[SourceDocument]:
        if not self.root.exists():
            return
        state = self._load_state()
        fresh: dict[str, Any] = {}
        self.files_read = 0
        seen: set[Path] = set()
        for pattern in self.include:
            for path in sorted(self.root.glob(pattern)):
                if not path.is_file() or path in seen or path == self._state_path:
                    continue
                rel = path.relative_to(self.root).as_posix()
                if any(fnmatch.fnmatch(rel, ex) for ex in self.exclude):
                    continue
                if self.sidecar_suffix and rel.endswith(self.sidecar_suffix):
                    continue  # metadata of another file, not a document
                st = path.stat()
                if st.st_size > self.max_bytes:
                    continue
                seen.add(path)
                entry = state.get(rel)
                if (
                    self.use_stat_cache
                    and entry is not None
                    and entry["size"] == st.st_size
                    and entry["mtime"] == st.st_mtime_ns
                ):
                    descriptors = entry["docs"]
                else:
                    self.files_read += 1
                    descriptors = self._describe(path.name, path.read_bytes(), [], 0)
                fresh[rel] = {"size": st.st_size, "mtime": st.st_mtime_ns, "docs": descriptors}
                yield from self._documents(path, rel, descriptors)
        self._save_state(fresh)

    def _documents(
        self, path: Path, rel: str, descriptors: list[dict[str, Any]]
    ) -> Iterable[SourceDocument]:
        sidecar = self._sidecar(path)
        source_id = sidecar.pop("document_id", None)
        base: dict[str, Any] = {
            "path": str(path),
            "relpath": rel,
            "name": path.name,
            **self._path_metadata(rel),
        }
        acl = sidecar.pop("acl", None)
        if acl is None:
            acl = self._acl_for(rel)
        if acl is not None:
            base["acl"] = list(acl) if isinstance(acl, (list, tuple)) else [str(acl)]
        base.update(sidecar)
        if source_id is not None:
            base["source_id"] = str(source_id)
        # The document id derives from the path *relative to root*, not the
        # absolute path, or from the source system's id when a sidecar gives
        # one. An absolute path makes ids machine-specific, so a rebuild on
        # another machine reprocesses the whole corpus while appearing to work.
        anchor = f"id:{source_id}" if source_id is not None else rel
        ids: dict[str, DocumentId] = {}
        for d in descriptors:
            doc_id = make_document_id(anchor + d["suffix"], namespace=self.namespace)
            ids[d["suffix"]] = doc_id
            meta = {**base, **d["meta"], "_locator": d["locator"]}
            if d["suffix"]:
                meta["relpath"] = rel + d["suffix"]
                meta["container"] = rel
                meta["name"] = d["name"]
                parent = ids.get(d["parent"])
                if parent is not None:
                    meta[PARENT_KEY] = parent
            elif d["name"] != path.name:
                meta["name"] = d["name"]  # an unwrapped envelope: x.pdf.p7m -> x.pdf
            yield _FileDocument(
                document_id=doc_id,
                source_uri=path.as_uri() + (d["suffix"] or ""),
                content_hash=d["hash"],
                media_type=d["media_type"],
                size_bytes=d["size"],
                metadata=meta,
            )

    # ------------------------------------------------------------ containers

    def _describe(
        self, name: str, data: bytes, locator: list[list[Any]], depth: int, *, suffix: str = ""
    ) -> list[dict[str, Any]]:
        """The documents a file is: itself, or -- for a container being
        expanded -- what it holds. Plain data, so it can be remembered."""
        lower = name.lower()
        # By name, not by sniffing: every file would otherwise be probed as a
        # possible CMS structure, and ".p7m" is how Italian archives name them.
        if "p7m" in self.expand and lower.endswith(".p7m") and depth < self.max_depth:
            try:
                signed = unwrap_p7m(data)
            except ValueError:
                signed = None
            if signed is not None:
                inner = re.sub(r"(?i)(\.p7m)+$", "", name) or name
                envelope = {
                    "envelope": "p7m",
                    "signed": True,
                    "signature_verified": False,
                    "signers": list(signed.signers),
                    "signer_ids": list(signed.signer_ids),
                }
                docs = self._describe(
                    inner, signed.content, [*locator, ["p7m"]], depth + 1, suffix=suffix
                )
                for d in docs:
                    if d["suffix"] == suffix:
                        d["meta"] = {**envelope, **d["meta"]}
                return docs
        if "zip" in self.expand and lower.endswith(".zip") and depth < self.max_depth:
            out: list[dict[str, Any]] = []
            try:
                members = list(zip_members(data))
            except Exception:  # a corrupt archive is a document that failed, not a crash
                members = []
            for member, payload in members:
                out.extend(
                    self._describe(
                        member.rsplit("/", 1)[-1],
                        payload,
                        [*locator, ["zip", member]],
                        depth + 1,
                        suffix=f"{suffix}#{member}",
                    )
                )
            for d in out:
                d["meta"].setdefault("archive_path", d["suffix"].split("#")[-1])
            return out
        if "eml" in self.expand and lower.endswith(".eml") and depth < self.max_depth:
            return self._describe_email(name, data, locator, depth, suffix)
        return [self._descriptor(name, data, locator, suffix, {})]

    def _describe_email(
        self, name: str, data: bytes, locator: list[list[Any]], depth: int, suffix: str
    ) -> list[dict[str, Any]]:
        info = parse_email(data)
        facts: dict[str, Any] = {
            "email_subject": info.subject,
            "email_from": info.sender,
            "email_to": list(info.to),
            "email_date": info.date,
        }
        if info.cc:
            facts["email_cc"] = list(info.cc)
        if info.pec is not None:
            facts["pec"] = True
            facts.update({f"pec_{k}": v for k, v in info.pec.items()})
        facts = {k: v for k, v in facts.items() if v not in ("", [], None)}
        docs = [self._descriptor(name, data, locator, suffix, dict(facts))]
        inherited = {k: v for k, v in facts.items() if k.startswith(("email_", "pec"))}
        for att in info.attachments:
            if att.name.lower() in _ENVELOPE_PARTS or att.name.lower().endswith(".p7s"):
                continue
            children = self._describe(
                att.name,
                att.payload,
                [*locator, ["eml", att.index]],
                depth + 1,
                suffix=f"{suffix}#att/{att.index}-{att.name}",
            )
            for c in children:
                c["meta"] = {**inherited, "attachment_name": att.name, **c["meta"]}
                c["parent"] = c.get("parent") or suffix
            docs.extend(children)
        return docs

    @staticmethod
    def _descriptor(
        name: str, data: bytes, locator: list[list[Any]], suffix: str, meta: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "suffix": suffix,
            "parent": suffix.rsplit("#", 1)[0] if "#" in suffix else "",
            "name": name,
            "media_type": media_type_for(name),
            "size": len(data),
            "hash": hash_bytes(data),
            "locator": locator,
            "meta": meta,
        }

    # -------------------------------------------------------------- metadata

    def _path_metadata(self, rel: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, rx in self._path_meta.items():
            m = rx.search(rel)
            if m:
                out[name] = m.group(1) if m.groups() else m.group(0)
        return out

    def _acl_for(self, rel: str) -> list[str] | None:
        for rule in self.acl_rules:
            if fnmatch.fnmatch(rel, str(rule.get("pattern", ""))):
                acl = rule.get("acl", [])
                return [str(a) for a in (acl if isinstance(acl, list) else [acl])]
        return list(self.default_acl) if self.default_acl is not None else None

    def _sidecar(self, path: Path) -> dict[str, Any]:
        if not self.sidecar_suffix:
            return {}
        side = path.with_name(path.name + self.sidecar_suffix)
        if not side.is_file():
            return {}
        try:
            data = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A malformed sidecar must not silently drop the document's ACL and
            # let it through as public: the document is withheld instead, as an
            # empty ACL, which no principal holds.
            return {"acl": [], "sidecar_error": f"unreadable sidecar {side.name}"}
        return dict(data) if isinstance(data, Mapping) else {}

    # ----------------------------------------------------------------- state

    def _load_state(self) -> dict[str, Any]:
        if self._state_path is None:
            return self._memory_state
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return dict(raw.get("files", {})) if raw.get("root") == str(self.root) else {}

    def _save_state(self, state: dict[str, Any]) -> None:
        self._memory_state = state
        if self._state_path is not None:
            atomic_write(
                self._state_path,
                json.dumps({"root": str(self.root), "files": state}).encode("utf-8"),
            )


def _media_type(path: Path) -> str:
    return media_type_for(path.name)


def document_id_for(root: str | Path, path: str | Path, namespace: str = "") -> DocumentId:
    """The id a scan would assign. Used by the golden-set bootstrapper."""
    rel = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    return make_document_id(rel, namespace=namespace)
