"""Entities and master data: which companies, people and accounts a unit is about.

An agent acting on an archive asks about *things* -- a customer, a supplier, a
bank account -- more often than about words. Two enrichers make those things
queryable:

``entities``     Finds VAT numbers, fiscal codes, IBANs and email addresses in a
                 unit's text and keeps only those whose check digit holds. A
                 candidate that fails is kept out of the index and recorded for
                 review, not guessed at: eleven digits near "IVA" are a phone
                 number as often as a VAT number.
``master_data``  Joins those identifiers -- and any a parser already knows, like
                 FatturaPA's parties -- to the company's own registers (a CSV
                 export of customers or suppliers), so a unit that mentions a
                 VAT number carries the customer code it belongs to. The
                 register's bytes are part of the fingerprint: updating it
                 re-links every unit, and nothing else.

Both write multi-valued fields (a unit can mention several companies), which
every index filters on existentially.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indexer.core.document import ParsedDocument
from indexer.core.ids import hash_bytes, hash_obj
from indexer.core.registry import register
from indexer.core.stages import EnrichContext
from indexer.core.unit import ContextScope, Enrichment, Unit
from indexer.plugin import StageImpl, dataclass_params
from indexer.validators import (
    normalize_iban,
    normalize_piva,
    valid_codice_fiscale,
    valid_iban,
    valid_piva,
)

__all__ = ["EntityExtractor", "MasterDataLinker"]

#: A VAT number needs its prefix or a label within reach: bare eleven-digit
#: runs are phone numbers and order codes as often as they are VAT numbers.
_PIVA = re.compile(
    r"(?:\bIT\s?(\d{11})\b)"
    r"|(?:(?:p\.?\s?iva|partita\s+iva|p\.?\s?i\.?|vat(?:\s+(?:no|number|n\.?))?|"
    r"c\.?\s?f\.?\s?/\s?p\.?\s?iva)[\s:.\-n°]{0,6}(?:IT\s?)?(\d{11})\b)",
    re.I,
)
_CF = re.compile(
    r"\b([A-Z]{6}[0-9LMNPQRSTUV]{2}[ABCDEHLMPRST][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z])\b",
    re.I,
)
_IBAN = re.compile(r"\b([A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30})\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

_KINDS = ("piva", "codice_fiscale", "iban", "email")


@dataclass(frozen=True, slots=True)
class EntityParams:
    kinds: list[str] = field(default_factory=lambda: list(_KINDS))
    #: Keep only candidates whose check digit holds. Off only to measure what
    #: validation removes.
    validate: bool = True


@register(
    "enrich",
    "entities",
    version="1",
    params_model=dataclass_params(EntityParams),
    summary=(
        "VAT numbers, fiscal codes, IBANs and emails, kept only when their check digit "
        "holds. No model. Multi-valued fields for filtering and entity lookup."
    ),
)
def _make_entities(params: dict[str, Any], **_: Any) -> EntityExtractor:
    return EntityExtractor(params)


class EntityExtractor(StageImpl):
    STAGE, IMPL, VERSION = "enrich", "entities", "1"
    name = "entities"
    scope = ContextScope.UNIT
    reads_prior = False

    def input_hash(self, unit: Unit, document: ParsedDocument, prior: Any) -> str:
        return hash_obj({"text": unit.text})

    def enrich(self, units: Sequence[Unit], ctx: EnrichContext) -> Sequence[Enrichment]:
        kinds = set(self.param("kinds", _KINDS))
        check = bool(self.param("validate", True))
        fp = self.fingerprint().key()
        out = []
        for u in units:
            found, rejected = extract_entities(u.text, kinds, validate=check)
            out.append(
                Enrichment(
                    enricher=self.name,
                    fingerprint=fp,
                    fields={k: tuple(v) for k, v in found.items() if v},
                    extra={"rejected": rejected} if rejected else {},
                    scope=self.scope,
                )
            )
        return out


def extract_entities(
    text: str, kinds: set[str] | None = None, *, validate: bool = True
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(found, rejected) identifiers in ``text``, normalised, in order."""
    kinds = set(_KINDS) if kinds is None else kinds
    found: dict[str, list[str]] = {k: [] for k in kinds}
    rejected: dict[str, list[str]] = {}

    def keep(kind: str, value: str, ok: bool) -> None:
        if ok or not validate:
            if value not in found[kind]:
                found[kind].append(value)
        else:
            rejected.setdefault(kind, [])
            if value not in rejected[kind]:
                rejected[kind].append(value)

    if "piva" in kinds:
        for m in _PIVA.finditer(text):
            digits = m.group(1) or m.group(2)
            keep("piva", normalize_piva(digits), valid_piva(digits))
    if "codice_fiscale" in kinds:
        for m in _CF.finditer(text):
            cf = m.group(1).upper()
            keep("codice_fiscale", cf, valid_codice_fiscale(cf))
    if "iban" in kinds:
        for m in _IBAN.finditer(text):
            iban = normalize_iban(m.group(1))
            if iban[:2].isalpha() and len(iban) >= 15:
                keep("iban", iban, valid_iban(iban))
    if "email" in kinds:
        for m in _EMAIL.finditer(text):
            keep("email", m.group(0).lower(), True)
    return found, rejected


# --------------------------------------------------------------------------- #
# master data                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MasterDataParams:
    #: [{path, match: {fields: [...], column}, emit: {column: field}, delimiter}]
    #: ``match.fields`` are the unit's fields or metadata to look up (the
    #: ``entities`` output, a parser's ``cedente_piva``...); ``column`` is the
    #: register's key column; ``emit`` copies register columns to fields.
    sources: list[dict[str, Any]] = field(default_factory=list)


@register(
    "enrich",
    "master_data",
    version="1",
    params_model=dataclass_params(MasterDataParams),
    summary=(
        "Joins identifiers found in a unit (VAT numbers, fiscal codes, codes) to the "
        "company's registers (CSV), emitting customer/supplier codes and names."
    ),
)
def _make_master_data(params: dict[str, Any], **_: Any) -> MasterDataLinker:
    return MasterDataLinker(params)


class MasterDataLinker(StageImpl):
    """A deterministic join from what a unit mentions to who it is.

    Reads earlier enrichers' fields (it runs after ``entities``) and scanner or
    parser metadata, so it keeps the frame's default, widest cache key.
    """

    STAGE, IMPL, VERSION = "enrich", "master_data", "1"
    name = "master_data"
    scope = ContextScope.UNIT
    reads_prior = True

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._tables: list[tuple[dict[str, Any], dict[str, list[dict[str, str]]], str]] = []
        for spec in params.get("sources", []):
            raw = Path(spec["path"]).read_bytes()
            rows = _read_csv(raw, spec.get("delimiter"))
            column = spec["match"]["column"]
            index: dict[str, list[dict[str, str]]] = {}
            for row in rows:
                key = _key(row.get(column, ""))
                if key:
                    index.setdefault(key, []).append(row)
            self._tables.append((spec, index, str(hash_bytes(raw))))

    def fingerprint(self) -> Any:
        from indexer.core.accounting import StageFingerprint

        # The registers' contents, not only their paths: a customer added to
        # the CSV must re-link the units that mention them.
        return StageFingerprint(
            stage=self.STAGE,
            impl=self.IMPL,
            version=self.VERSION,
            params_hash=hash_obj(
                {"params": self._params, "registers": [h for _, _, h in self._tables]}
            ),
        )

    def enrich(self, units: Sequence[Unit], ctx: EnrichContext) -> Sequence[Enrichment]:
        fp = self.fingerprint().key()
        out = []
        for u in units:
            known: dict[str, Any] = dict(u.metadata)
            for e in ctx.prior.get(u.unit_id, {}).values():
                known.update(e.fields)
            fields: dict[str, tuple[Any, ...]] = {}
            for spec, index, _ in self._tables:
                match = spec["match"]
                names = match.get("fields") or [match.get("field")]
                emitted: dict[str, list[str]] = {}
                for name in names:
                    for value in _values(known.get(name)):
                        for row in index.get(_key(value), []):
                            for column, target in (spec.get("emit") or {}).items():
                                v = row.get(column, "")
                                if v and v not in emitted.setdefault(target, []):
                                    emitted[target].append(v)
                for target, values in emitted.items():
                    fields[target] = tuple(values)
            out.append(
                Enrichment(enricher=self.name, fingerprint=fp, fields=fields, scope=self.scope)
            )
        return out


def _values(v: Any) -> list[Any]:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _key(value: Any) -> str:
    """Join key: identifiers compared without spaces, case or the IT prefix."""
    s = re.sub(r"[\s.\-]+", "", str(value)).upper()
    if re.fullmatch(r"IT\d{11}", s):
        s = s[2:]
    return s


def _read_csv(raw: bytes, delimiter: str | None) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - cp1252 decodes everything but five bytes
        text = raw.decode("utf-8", errors="replace")
    if delimiter is None:
        # Italian spreadsheet exports use ";" -- the comma is the decimal mark.
        head = text.splitlines()[0] if text else ""
        delimiter = ";" if head.count(";") > head.count(",") else ","
    return [
        {k.strip(): (v or "").strip() for k, v in row.items() if k}
        for row in csv.DictReader(io.StringIO(text), delimiter=delimiter)
    ]
