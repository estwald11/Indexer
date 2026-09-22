"""Query-side guarantees an agent relies on: access control, result shape, routing
of rerankers and fusion weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from indexer.core.errors import AccessDenied
from indexer.core.query import QueryType, RoutePath
from indexer.pipeline import assemble

CONFIG = """
schema_version: 1
project: {{name: engine}}
paths: {{store: {root}/index, cache: {root}/cache, manifests: {root}/man, artifacts: {root}/art}}
corpus:
  sources:
    - impl: filesystem
      params:
        root: {data}
        include: ["**/*.md"]
        acl_rules:
          - {{pattern: "hr/**", acl: ["group:hr"]}}
          - {{pattern: "sales/**", acl: ["group:sales", "group:hr"]}}
ingestion:
  parse: {{enabled: true, default: {{impl: markdown}}}}
  segment: {{impl: structural, max_tokens: 60, params: {{merge_below_tokens: 0}}}}
  enrich:
    enabled: true
    enrichers:
      - impl: regex_fields
        scope: unit
        params:
          fields: {{bonus: {{pattern: 'bonus ([0-9]+)', type: int}}}}
  index:
    indexes:
      - {{name: lexical, kind: lexical, impl: bm25}}
      - {{name: fields, kind: structured, impl: sqlite}}
query:
  route:
    enabled: true
    impl: rules
    paths:
      structured: {{targets: [fields]}}
      lookup: {{targets: [lexical], step_budget: 1}}
      iterative: {{targets: [lexical], step_budget: 3}}
  access: {{enabled: {access}, missing: {missing}}}
  shape: {shape}
"""

POLICY = "# Policy\n\nThe travel policy covers flights, hotels and meals for employees.\n"
SALARY = "# Salaries\n\nThe bonus 5000 for the travel manager is confidential.\n"
PITCH = "# Pitch\n\nOur travel offer for clients includes flights and hotels.\n"
MANUAL = "# Manual\n\n" + "\n\n".join(
    f"## Chapter {i}\n\nTravel expenses rule {i}: flights must be booked early." for i in range(6)
)


def _setup(
    tmp_path: Path,
    *,
    access: bool = False,
    missing: str = "deny",
    shape: str = "{enabled: false}",
    files: dict[str, str] | None = None,
):  # type: ignore[no-untyped-def]
    data = tmp_path / "data"
    for rel, body in (
        files or {"hr/salary.md": SALARY, "sales/pitch.md": PITCH, "policy.md": POLICY}
    ).items():
        p = data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        CONFIG.format(
            root=tmp_path.as_posix(),
            data=data.as_posix(),
            access=str(access).lower(),
            missing=missing,
            shape=shape,
        )
    )
    a = assemble(cfg)
    assert a.ingestion().build().ok
    return a


def _names(resp) -> set[str]:  # type: ignore[no-untyped-def]
    return {h.unit.unit.metadata["relpath"] for h in resp.hits if h.unit}


class TestAccessControl:
    def test_a_query_sees_only_what_its_principals_may(self, tmp_path: Path) -> None:
        a = _setup(tmp_path, access=True)
        engine = a.query_engine()
        assert _names(engine.query("travel flights hotels", principals=("group:sales",))) == {
            "sales/pitch.md"
        }
        assert _names(engine.query("travel", principals=("group:hr",))) == {
            "hr/salary.md",
            "sales/pitch.md",
        }

    def test_documents_without_an_acl_follow_the_missing_setting(self, tmp_path: Path) -> None:
        denied = _setup(tmp_path / "deny", access=True, missing="deny")
        allowed = _setup(tmp_path / "allow", access=True, missing="allow")
        q = "travel policy employees"
        assert "policy.md" not in _names(denied.query_engine().query(q, principals=("u",)))
        assert "policy.md" in _names(allowed.query_engine().query(q, principals=("u",)))

    def test_the_structured_path_is_scoped_too(self, tmp_path: Path) -> None:
        a = _setup(tmp_path, access=True)
        resp = a.query_engine().query(
            "entries with bonus greater than 100", principals=("group:sales",)
        )
        assert str(resp.decision.path) == RoutePath.STRUCTURED
        assert resp.records is not None and resp.records.is_empty()
        resp = a.query_engine().query(
            "entries with bonus greater than 100", principals=("group:hr",)
        )
        assert resp.records is not None and [r["bonus"] for r in resp.records.as_dicts()] == [5000]

    def test_a_query_without_principals_is_refused(self, tmp_path: Path) -> None:
        a = _setup(tmp_path, access=True)
        with pytest.raises(AccessDenied):
            a.query_engine().query("travel")

    def test_off_by_default(self, tmp_path: Path) -> None:
        a = _setup(tmp_path)
        assert len(_names(a.query_engine().query("travel flights"))) == 3


class TestShape:
    def test_at_most_n_units_per_document(self, tmp_path: Path) -> None:
        files = {"manual.md": MANUAL, "policy.md": POLICY}
        loose = _setup(tmp_path / "a", files=files)
        capped = _setup(tmp_path / "b", files=files, shape="{enabled: true, max_per_document: 2}")
        q = "travel expenses flights"
        docs = [h.document_id for h in capped.query_engine().query(q, top_k=10).hits]
        assert max(docs.count(d) for d in set(docs)) <= 2
        assert len(set(docs)) == 2  # the cap made room for the other document
        assert len(loose.query_engine().query(q, top_k=10).hits) > len(docs)

    def test_the_same_passage_in_two_files_is_shown_once(self, tmp_path: Path) -> None:
        a = _setup(
            tmp_path,
            files={"2024/policy.md": POLICY, "copia/policy.md": POLICY},
            shape="{enabled: true}",
        )
        hits = a.query_engine().query("travel policy flights hotels").hits
        assert len(hits) == 1
        assert hits[0].explain["duplicates"][0]["document_id"] != hits[0].document_id

    def test_neighbouring_units_come_with_the_hit(self, tmp_path: Path) -> None:
        a = _setup(
            tmp_path, files={"manual.md": MANUAL}, shape="{enabled: true, expand_neighbors: 1}"
        )
        hit = a.query_engine().query("rule 3").hits[0]
        assert "rule 2" in hit.explain["before"][0] and "rule 4" in hit.explain["after"][0]


class TestFusionByType:
    def test_weights_follow_the_query_type(self) -> None:
        from indexer.core.accounting import InMemoryAccountant
        from indexer.core.cache import NullCache
        from indexer.core.ids import DocumentId, UnitId
        from indexer.core.provenance import Provenance, Span
        from indexer.core.results import Hit, RankedList
        from indexer.core.stages import StageContext
        from indexer.impls.fuse import RRFFuser

        def lst(source: str, ids: list[str]) -> RankedList:
            return RankedList(
                hits=tuple(
                    Hit(
                        unit_id=UnitId(u),
                        document_id=DocumentId(u),
                        rank=i,
                        score=1.0,
                        index=source,
                        provenance=Provenance(document_id=DocumentId(u), span=Span(0, 1)),
                    )
                    for i, u in enumerate(ids, start=1)
                ),
                source=source,
            )

        lists = [lst("lexical", ["a"]), lst("dense", ["b"])]
        fuser = RRFFuser(
            {"weights": {"dense": 2.0}, "weights_by_type": {"factual": {"lexical": 3.0}}}
        )

        def top(qtype: str) -> str:
            ctx = StageContext(
                cache=NullCache(), accountant=InMemoryAccountant(), attrs={"query_type": qtype}
            )
            return str(fuser.fuse(lists, ctx).hits[0].unit_id)

        assert top(str(QueryType.COMPARATIVE)) == "b"  # the default weights favour dense
        assert top(str(QueryType.FACTUAL)) == "a"  # factual questions weight lexical up


def test_decision_log_records_the_rewrite(tmp_path: Path) -> None:
    a = _setup(tmp_path)
    engine = a.query_engine()
    engine.query("travel")
    log = Path(a.paths.store) / "route-decisions.jsonl"
    record = json.loads(log.read_text().splitlines()[-1])
    assert "rewritten_query" in record


def test_a_path_can_name_its_own_reranker(tmp_path: Path) -> None:
    """``route.paths.<p>.rerank`` was accepted by the schema and never read."""
    a = _setup(tmp_path)
    cfg = tmp_path / "c.yaml"
    text = cfg.read_text().replace(
        "lookup: {targets: [lexical], step_budget: 1}",
        "lookup: {targets: [lexical], step_budget: 1, rerank: noop}",
    )
    cfg.write_text(text + "  rerank: {enabled: true, impl: lexical_overlap}\n")
    engine = assemble(cfg).query_engine()
    lookup = engine.query("travel flights")
    assert str(lookup.decision.path) == RoutePath.LOOKUP
    assert lookup.reranked is not None and lookup.reranked.source == "fuse:rrf"  # noop
    cfg.write_text(
        text.replace(", rerank: noop", "") + "  rerank: {enabled: true, impl: lexical_overlap}\n"
    )
    default = assemble(cfg).query_engine().query("travel flights")
    assert default.reranked is not None and default.reranked.source == "rerank:lexical_overlap"
    assert a is not None
