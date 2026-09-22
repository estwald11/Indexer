"""Tenant isolation: a caller's scope must hold on every path and in every cache.

The frame's multi-tenant story is a caller filter (``Query.filters``) plus
scanner metadata. Both are only as good as the weakest path that ignores them,
and an archive shared by several companies -- or several departments of one --
is exactly where a leak is a incident rather than a quality problem. Each test
here is a reproduction of a leak that existed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indexer.core.predicate import Compare, Op
from indexer.core.query import RoutePath
from indexer.pipeline import assemble

CONFIG = """
schema_version: 1
project: {{name: tenancy}}
paths: {{store: {root}/index, cache: {root}/cache, manifests: {root}/man, artifacts: {root}/art}}
corpus:
  sources:
    - impl: filesystem
      params:
        root: {data}
        include: ["**/*.md"]
        path_metadata: {{tenant: '^([^/]+)/'}}
ingestion:
  parse: {{enabled: true, default: {{impl: markdown}}}}
  segment: {{impl: structural, max_tokens: 200, params: {{merge_below_tokens: 0}}}}
  enrich:
    enabled: true
    enrichers:
      - impl: regex_fields
        scope: unit
        params:
          from_metadata: [tenant]
          fields:
            amount: {{pattern: 'amount ([0-9]+)', type: int}}
  index:
    indexes:
      - {{name: lexical, kind: lexical, impl: bm25_memory}}
      - {{name: fields, kind: structured, impl: sqlite}}
query:
  route:
    enabled: true
    impl: rules
    paths:
      structured: {{targets: [fields]}}
      lookup: {{targets: [lexical], step_budget: 1}}
      iterative: {{targets: [lexical], step_budget: 3}}
"""


def _workspace(tmp_path: Path, files: dict[str, str]) -> Path:
    data = tmp_path / "data"
    for rel, body in files.items():
        p = data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(CONFIG.format(root=tmp_path.as_posix(), data=data.as_posix()))
    return cfg


ACME = Compare("tenant", Op.EQ, "acme")
GLOBEX = Compare("tenant", Op.EQ, "globex")


class TestStructuredPathHonoursCallerScope:
    @pytest.fixture
    def assembly(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        cfg = _workspace(
            tmp_path,
            {
                "acme/invoice.md": "# Invoice\n\nThe invoice amount 1500 is due in March.\n",
                "globex/invoice.md": "# Invoice\n\nThe invoice amount 9000 is due in April.\n",
            },
        )
        a = assemble(cfg)
        assert a.ingestion().build().ok
        return a

    def test_records_from_other_tenants_are_not_returned(self, assembly) -> None:  # type: ignore[no-untyped-def]
        resp = assembly.query_engine().query("invoices with amount greater than 1000", filters=ACME)
        assert str(resp.decision.path) == RoutePath.STRUCTURED
        assert resp.records is not None
        assert [r["amount"] for r in resp.records.as_dicts()] == [1500]

    def test_unscoped_query_still_sees_everything(self, assembly) -> None:  # type: ignore[no-untyped-def]
        resp = assembly.query_engine().query("invoices with amount greater than 1000")
        assert resp.records is not None
        assert sorted(r["amount"] for r in resp.records.as_dicts()) == [1500, 9000]
