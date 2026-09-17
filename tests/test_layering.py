"""Tests for the layering policy.

``indexer.core`` carries the contracts and must stay dependency-free: a project
adopting the frame inherits the types and none of the choices. This is easy to
break by accident and impossible to notice, so it is a test.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "src" / "indexer" / "core"

#: Modules in the standard library, plus the package itself.
_ALLOWED_ROOTS = set(sys.stdlib_module_names) | {"indexer"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_core_imports_nothing_third_party() -> None:
    offenders: dict[str, set[str]] = {}
    for py in sorted(CORE.glob("*.py")):
        bad = _imported_roots(py) - _ALLOWED_ROOTS
        if bad:
            offenders[py.name] = bad
    assert not offenders, (
        f"indexer.core must stay dependency-free, but {offenders} were imported. "
        f"Adopting the frame would now inherit those choices."
    )


def test_core_does_not_name_a_concrete_implementation() -> None:
    """The frame never names an implementation; config does, via the registry."""
    banned = ("qdrant", "lancedb", "pymupdf", "docling", "splade", "colqwen", "openai")
    offenders: dict[str, list[str]] = {}
    for py in sorted(CORE.glob("*.py")):
        # Docstrings legitimately discuss reference implementations; code must not.
        code = "\n".join(
            ln for ln in py.read_text().splitlines() if not ln.lstrip().startswith("#")
        )
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                text = ast.unparse(node).lower()
                hits = [b for b in banned if b in text]
                if hits:
                    offenders.setdefault(py.name, []).extend(hits)
    assert not offenders, f"indexer.core imports vendor code: {offenders}"


EVAL = Path(__file__).resolve().parent.parent / "src" / "indexer" / "eval"


def test_eval_does_not_depend_on_any_implementation() -> None:
    """The harness must be able to evaluate a system this library did not build.

    That is usually the first comparison anyone wants -- "is the new pipeline
    better than what we already have?" -- and a harness that imports a specific
    index implementation cannot answer it. It also closed an import cycle:
    pipeline -> eval (for contract checks) -> impls -> pipeline.
    """
    offenders: dict[str, list[str]] = {}
    for py in sorted(EVAL.glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = node.names[0].name
            if mod and mod.startswith("indexer.impls"):
                offenders.setdefault(py.name, []).append(mod)
    assert not offenders, (
        f"indexer.eval imports implementation modules: {offenders}. "
        f"Shared helpers belong in indexer.textutil, indexer.io or indexer.plugin."
    )


def test_no_import_cycles_in_the_package() -> None:
    """Importing any subpackage first must work. A cycle makes order matter."""
    import importlib
    import subprocess
    import sys

    for first in ("indexer.eval", "indexer.impls", "indexer.pipeline", "indexer.config"):
        r = subprocess.run(
            [sys.executable, "-c", f"import {first}"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"importing {first} first fails:\n{r.stderr[-800:]}"
    assert importlib  # keep the import meaningful to linters
