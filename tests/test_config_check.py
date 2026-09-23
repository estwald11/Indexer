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


@pytest.mark.parametrize(
    "name", ["reference.yaml", "pypi-docs.yaml", "full.yaml", "it-enterprise.yaml"]
)
def test_the_shipped_configs_resolve(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """full.yaml named five implementations that did not exist until this held."""
    monkeypatch.setenv("LLAMAPARSE_API_KEY", "sk-test")
    cfg, _ = load(Path("configs") / name)
    assert [f for f in resolve_all(cfg) if f.level == "error"] == []


def test_every_arm_of_the_italian_preset_assembles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ablation arm that cannot be built is found by the run that needed it,
    hours in. Every arm is assembled here -- stages constructed, models not
    loaded, nothing called."""
    from indexer.pipeline.build import assemble, declared_fields

    monkeypatch.setenv("ARCHIVE_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("ARCHIVE_ROOT", str(tmp_path / "archive"))
    (tmp_path / "archive").mkdir()
    path = Path("configs") / "it-enterprise.yaml"
    cfg, _ = load(path)
    names, types = declared_fields(cfg)
    # The router's vocabulary covers what the parser, the extractors and the
    # classifier write -- the reason declares_fields exists.
    assert {"importo_totale", "data_documento", "doc_type", "piva", "iban", "anno"} <= set(names)
    assert types["importo_totale"] == "float" and types["data_documento"] == "date"
    for arm in cfg.eval.ablations:
        a = assemble(path, overrides=arm.overrides)
        a.ingestion()
        a.query_engine()
