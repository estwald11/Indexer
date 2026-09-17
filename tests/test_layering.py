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
