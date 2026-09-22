"""``indexer.config.check`` must not say OK about a config that cannot run.

It checked the schema only, so ``configs/full.yaml`` -- which named nine
implementations that did not exist -- printed "OK". The check now resolves every
name and validates every params block; missing optional dependencies are
warnings, named.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indexer.config.check import main, resolve_all
from indexer.config.loader import load

BASE = """
schema_version: 1
project: {name: check}
corpus:
  sources:
    - impl: filesystem
      params: {root: ./data}
ingestion:
  parse: {enabled: true, default: {impl: PARSER}}
  segment: {impl: structural}
  index:
    indexes:
      - {name: lexical, kind: lexical, impl: bm25_memory, params: PARAMS}
query:
  route:
    enabled: true
    impl: rules
    paths:
      structured: {targets: []}
      lookup: {targets: [lexical], step_budget: 1}
      iterative: {targets: [lexical], step_budget: 3}
"""


def _write(tmp_path: Path, parser: str = "markdown", params: str = "{}") -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(BASE.replace("PARSER", parser).replace("PARAMS", params))
    return p


def test_a_valid_config_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(_write(tmp_path))]) == 0
    assert "OK" in capsys.readouterr().out


def test_an_unknown_implementation_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(_write(tmp_path, parser="no_such_parser"))]) == 1
    err = capsys.readouterr().err
    assert "no parse implementation named 'no_such_parser'" in err
    assert "ingestion.parse.default" in err


def test_a_misspelled_parameter_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(_write(tmp_path, params="{k_1: 1.5}"))]) == 1
    assert "unknown parameter(s) ['k_1']" in capsys.readouterr().err


def test_schema_only_skips_resolution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--schema-only", str(_write(tmp_path, parser="no_such_parser"))]) == 0
    assert "schema only" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["reference.yaml", "pypi-docs.yaml"])
def test_the_shipped_configs_resolve(name: str) -> None:
    cfg, _ = load(Path("configs") / name)
    assert [f for f in resolve_all(cfg) if f.level == "error"] == []
