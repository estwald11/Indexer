#!/usr/bin/env python3
"""Assemble an evaluation corpus from PyPI source distributions.

Why this corpus. It is real documentation written by many hands: heterogeneous
formats (markdown, reStructuredText, plain text), genuine structure (headings,
code blocks, tables, cross-references), highly variable length, and the uneven
quality that a synthetic corpus never reproduces. It also carries machine-
readable facts -- versions, release dates, Python requirements -- so the
structured path is exercised with real values rather than stubs.

It is reproducible: anyone with PyPI access gets the same files for the same
pinned versions, which matters because an ablation report nobody can re-run is
an anecdote.

Every document keeps ``package`` and ``version`` metadata, which the field
extractor turns into structured fields and the golden-set bootstrapper uses as
the query subject.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

PACKAGES = [
    "requests",
    "urllib3",
    "click",
    "jinja2",
    "flask",
    "sqlalchemy",
    "attrs",
    "pydantic",
    "rich",
    "httpx",
    "pytest",
    "tox",
    "packaging",
    "pluggy",
    "markupsafe",
    "werkzeug",
    "itsdangerous",
    "certifi",
    "idna",
    "charset-normalizer",
    "pyyaml",
    "tomli",
    "platformdirs",
    "filelock",
    "virtualenv",
    "distlib",
    "typing-extensions",
    "sniffio",
    "anyio",
    "h11",
    "httpcore",
    "exceptiongroup",
    "iniconfig",
    "cachetools",
    "chardet",
    "colorama",
    "more-itertools",
    "wrapt",
    "zipp",
    "jsonschema",
    "referencing",
    "rpds-py",
    "fsspec",
    "joblib",
    "python-dateutil",
    "pytz",
    "six",
    "soupsieve",
    "beautifulsoup4",
    "lxml",
]

DOC_SUFFIXES = (".md", ".rst", ".txt")
DOC_DIRS = ("docs/", "doc/", "documentation/")
ROOT_DOCS = (
    "readme",
    "changelog",
    "changes",
    "history",
    "news",
    "contributing",
    "authors",
    "security",
    "upgrading",
    "faq",
)
SKIP_PARTS = ("test", "tests", "_build", "locale", "translations", "node_modules")


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "indexer-eval-corpus/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "indexer-eval-corpus/0.1"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def wanted(name: str) -> bool:
    low = name.lower()
    parts = low.split("/")
    if any(p in SKIP_PARTS for p in parts):
        return False
    if not low.endswith(DOC_SUFFIXES):
        return False
    rest = "/".join(parts[1:])  # strip the sdist's top-level directory
    if any(rest.startswith(d) for d in DOC_DIRS):
        return True
    if "/" not in rest and any(rest.startswith(r) for r in ROOT_DOCS):
        return True
    return False


def extract(
    archive: bytes, url: str, out: Path, pkg: str, version: str, min_bytes: int, max_bytes: int
) -> int:
    written = 0
    members: list[tuple[str, bytes]] = []
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            for info in zf.infolist():
                if not info.is_dir() and wanted(info.filename):
                    members.append((info.filename, zf.read(info)))
    else:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tf:
            for m in tf.getmembers():
                if m.isfile() and wanted(m.name):
                    f = tf.extractfile(m)
                    if f:
                        members.append((m.name, f.read()))

    for name, data in members:
        if not (min_bytes <= len(data) <= max_bytes):
            continue
        rel = "/".join(name.split("/")[1:]) or name.split("/")[-1]
        dest = out / pkg / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written += 1
    if written:
        # Package-level facts, so the structured path has real values to answer
        # from rather than something invented for the demo.
        (out / pkg / "_package.json").write_text(
            json.dumps({"package": pkg, "version": version}, indent=1)
        )
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./data/pypi-docs")
    ap.add_argument("--packages", nargs="*", default=PACKAGES)
    ap.add_argument("--min-bytes", type=int, default=700)
    ap.add_argument("--max-bytes", type=int, default=400_000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    total = 0

    for pkg in args.packages:
        try:
            meta = fetch_json(f"https://pypi.org/pypi/{pkg}/json")
            version = meta["info"]["version"]
            sdist = next(
                (u for u in meta["urls"] if u["packagetype"] == "sdist"),
                None,
            )
            if sdist is None:
                print(f"  {pkg:24} no sdist, skipped", file=sys.stderr)
                continue
            data = fetch_bytes(sdist["url"])
            n = extract(data, sdist["url"], out, pkg, version, args.min_bytes, args.max_bytes)
            total += n
            # Keep the exact version so the corpus is reproducible.
            manifest[pkg] = {"version": version, "files": n, "sdist": sdist["filename"]}
            print(f"  {pkg:24} {version:12} {n:4d} docs")
        except Exception as exc:  # network and archive variety is wide
            print(f"  {pkg:24} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    (out / "_corpus.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"\n{total} documents from {len(manifest)} packages -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
