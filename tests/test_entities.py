"""Identifiers with check digits, and the join to the company's own registers.

The test values are real in form: ``00743110157`` is a valid partita IVA,
``RSSMRA85T10A562S`` a valid codice fiscale, ``IT60X0542811101000000123456``
a valid IBAN. Each has a sibling one digit off, which must be rejected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indexer.core.accounting import InMemoryAccountant
from indexer.core.predicate import Compare, In, Op, StructuredQuery
from indexer.core.stages import StageContext
from indexer.impls.enrich_entities import extract_entities
from indexer.pipeline import assemble
from indexer.validators import valid_codice_fiscale, valid_iban, valid_piva


class TestValidators:
    @pytest.mark.parametrize("value", ["00743110157", "IT00743110157", "IT 00743110157"])
    def test_valid_partite_iva(self, value: str) -> None:
        assert valid_piva(value)

    @pytest.mark.parametrize("value", ["00743110158", "0074311015", "00000000000", "abc"])
    def test_invalid_partite_iva(self, value: str) -> None:
        assert not valid_piva(value)

    def test_codice_fiscale_persons_companies_and_omocodia(self) -> None:
        assert valid_codice_fiscale("RSSMRA85T10A562S")
        assert valid_codice_fiscale("rssmra85t10a562s")
        assert not valid_codice_fiscale("RSSMRA85T10A562T")
        assert valid_codice_fiscale("00743110157")  # a company's fiscal code
        # Omocodia replaces digits with letters, and the control letter follows.
        assert not valid_codice_fiscale("RSSMRA85T10A56NS")

    def test_iban(self) -> None:
        assert valid_iban("IT60X0542811101000000123456")
        assert valid_iban("IT60 X054 2811 1010 0000 0123 456")
        assert not valid_iban("IT60X0542811101000000123457")
        assert not valid_iban("IT60X054281110100000012345")  # an Italian IBAN is 27 long


class TestExtraction:
    def test_only_checked_identifiers_are_kept(self) -> None:
        text = (
            "Fornitore Rossi Srl, P.IVA 00743110157, C.F. RSSMRA85T10A562S, "
            "IBAN IT60 X054 2811 1010 0000 0123 456. Tel. 02 12345678901. "
            "Scrivere a Ufficio@Rossi.it. Un'altra P.IVA 12345678901 è errata."
        )
        found, rejected = extract_entities(text)
        assert found["piva"] == ["IT00743110157"]
        assert found["codice_fiscale"] == ["RSSMRA85T10A562S"]
        assert found["iban"] == ["IT60X0542811101000000123456"]
        assert found["email"] == ["ufficio@rossi.it"]
        # A phone number is not a VAT number, and a wrong one is kept for review.
        assert rejected == {"piva": ["IT12345678901"]}


CONFIG = """
schema_version: 1
project: {{name: anagrafiche}}
paths: {{store: {root}/index, cache: {root}/cache, manifests: {root}/man, artifacts: {root}/art}}
corpus:
  sources:
    - impl: filesystem
      params: {{root: {data}, include: ["**/*.md"]}}
ingestion:
  parse: {{enabled: true, default: {{impl: markdown}}}}
  segment: {{impl: structural}}
  enrich:
    enabled: true
    enrichers:
      - {{impl: entities, scope: unit}}
      - impl: master_data
        scope: unit
        params:
          sources:
            - path: {register}
              match: {{fields: [piva, codice_fiscale], column: partita_iva}}
              emit: {{codice_cliente: cliente_codice, ragione_sociale: cliente_nome}}
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
"""

REGISTER = (
    "codice_cliente;ragione_sociale;partita_iva\n"
    "C001;Rossi Forniture Srl;00743110157\n"
    "C002;Verdi Snc;12345678903\n"
)


class TestMasterData:
    def _setup(self, tmp_path: Path, register: str = REGISTER):  # type: ignore[no-untyped-def]
        data = tmp_path / "data"
        data.mkdir(exist_ok=True)
        (data / "ordine.md").write_text(
            "# Ordine 77\n\nOrdine emesso a Rossi Forniture Srl, P.IVA IT00743110157.\n"
        )
        (data / "reclamo.md").write_text(
            "# Reclamo\n\nIl cliente con partita IVA 12345678903 segnala un ritardo.\n"
        )
        reg = tmp_path / "clienti.csv"
        reg.write_text(register, encoding="utf-8")
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            CONFIG.format(root=tmp_path.as_posix(), data=data.as_posix(), register=reg.as_posix())
        )
        return cfg

    def test_documents_carry_the_customer_they_mention(self, tmp_path: Path) -> None:
        a = assemble(self._setup(tmp_path))
        assert a.ingestion().build().ok
        idx = a.indexes["fields"]
        ctx = StageContext(cache=a.cache, accountant=InMemoryAccountant())
        rs = idx.structured_query(  # type: ignore[attr-defined]
            StructuredQuery(
                where=Compare("cliente_codice", Op.EQ, "C001"),
                select=("document_id", "cliente_nome"),
                level="document",
            ),
            ctx,
        )
        assert [r[1] for r in rs.rows] == ["Rossi Forniture Srl"]
        # And entity lookup by identifier works across documents.
        rs = idx.structured_query(  # type: ignore[attr-defined]
            StructuredQuery(where=In("piva", ("IT12345678903",)), level="document"), ctx
        )
        assert len(rs.rows) == 1

    def test_updating_the_register_relinks(self, tmp_path: Path) -> None:
        cfg = self._setup(tmp_path)
        assert assemble(cfg).ingestion().build().ok
        (tmp_path / "clienti.csv").write_text(
            REGISTER.replace("C002;Verdi Snc", "C009;Verdi & Figli Snc"), encoding="utf-8"
        )
        a = assemble(cfg)
        res = a.ingestion().build()
        assert {c.kind.value for c in res.plan} == {"restaged"}
        stored = [a.unit_store.get(u) for u in a.unit_store.all_ids()]
        codes = {c for u in stored if u for c in u.fields().get("cliente_codice", ())}
        assert codes == {"C001", "C009"}
