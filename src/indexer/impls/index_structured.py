"""Structured index: SQLite over extracted fields.

This is where invariant 5's questions go instead of vector search. It is also
the one index with a second capability (``StructuredCapable``), which is why the
frame asks for capabilities by ``isinstance`` rather than putting
``structured_query`` on every ``Index``.

The predicate compiler is the interesting part. ``indexer.core.predicate``
defines an AST with a reference in-memory evaluator; this compiles the same AST
to parameterised SQL. The conformance requirement is that the two agree, and
there is a test that checks it over generated predicates -- otherwise the same
question returns different answers depending on which store answered it.

Schema note: fields are stored in a key/value table rather than as columns,
because the field set comes from config and changes per project. A column per
field would mean a migration every time a corpus gains an extraction rule, which
is precisely the fork this library exists to avoid.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from indexer.core.ids import DocumentId, UnitId
from indexer.core.predicate import (
    Aggregation,
    And,
    Compare,
    Exists,
    In,
    Not,
    Op,
    Or,
    Predicate,
    StructuredQuery,
    TextMatch,
)
from indexer.core.provenance import Provenance, Span
from indexer.core.registry import register
from indexer.core.results import Hit, RankedList, RecordSet
from indexer.core.stages import IndexQuery, IndexStatsView, IndexWriteReceipt, StageContext
from indexer.core.unit import EnrichedUnit
from indexer.plugin import StageImpl, dataclass_params

__all__ = ["SqliteStructuredIndex", "compile_predicate"]

_SQL_OPS = {Op.EQ: "=", Op.NE: "!=", Op.LT: "<", Op.LTE: "<=", Op.GT: ">", Op.GTE: ">="}


@dataclass(frozen=True, slots=True)
class SqliteParams:
    path: str = "./var/index/fields.db"
    #: Fields to index. Empty means index whatever enrich produced, which is the
    #: right default: the extraction schema already lives in the enricher config
    #: and repeating it here would let the two drift apart.
    fields: list[str] | None = None


@register(
    "index",
    "sqlite",
    version="2",
    params_model=dataclass_params(SqliteParams),
    summary="SQLite over extracted fields. Answers the STRUCTURED route without vector search.",
)
def _make_sqlite(params: dict[str, Any], **kw: Any) -> SqliteStructuredIndex:
    return SqliteStructuredIndex(params, name=kw.get("name", "fields"))


class SqliteStructuredIndex(StageImpl):
    STAGE, IMPL, VERSION = "index", "sqlite", "2"
    kind = "structured"

    def __init__(self, params: dict[str, Any], name: str = "fields") -> None:
        super().__init__(params)
        self.name = name
        self.path = Path(params.get("path", "./var/index/fields.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._check_identity()

    def _check_identity(self) -> None:
        """Start empty if the database was built as something else.

        Same rule as the other indexes: the ledger restages every document when
        this index's fingerprint changes, and rows left from the old
        configuration -- fields a narrowed ``fields:`` list no longer keeps --
        would otherwise survive as "unchanged".
        """
        want = json.dumps(self.store_identity(), sort_keys=True)
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'identity'").fetchone()
        if row is not None and row["value"] != want:
            self._conn.executescript("DELETE FROM fields; DELETE FROM units;")
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(units)")}
        if "record_hash" not in cols:
            # A database from version 1. Its rows are kept (the ledger restages
            # them all for the version change) and rewritten as they arrive.
            self._conn.execute("ALTER TABLE units ADD COLUMN record_hash TEXT")
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('identity', ?)", (want,)
        )
        self._conn.commit()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS units (
                unit_id     TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                source_uri  TEXT,
                span_start  INTEGER,
                span_end    INTEGER,
                section     TEXT,
                text        TEXT,
                surface_hash TEXT,
                record_hash TEXT
            );
            CREATE TABLE IF NOT EXISTS fields (
                unit_id TEXT NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
                name    TEXT NOT NULL,
                -- One typed column per storage class, exactly one non-null.
                -- Storing everything as TEXT would make every numeric and
                -- temporal comparison a string comparison, which is the bug
                -- invariant 5 exists to prevent.
                v_text  TEXT,
                v_num   REAL,
                v_int   INTEGER,
                v_date  TEXT,
                PRIMARY KEY (unit_id, name)
            );
            CREATE INDEX IF NOT EXISTS ix_fields_name ON fields(name);
            CREATE INDEX IF NOT EXISTS ix_units_doc   ON units(document_id);
            PRAGMA foreign_keys = ON;
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ write

    def upsert(self, units: Sequence[EnrichedUnit], ctx: StageContext) -> IndexWriteReceipt:
        written = skipped = 0
        cur = self._conn.cursor()
        for eu in units:
            record = str(eu.record_hash)
            row = cur.execute(
                "SELECT record_hash FROM units WHERE unit_id = ?", (eu.unit_id,)
            ).fetchone()
            # Skipped only when the whole record matches. Comparing the surface
            # hash skipped every unit whose fields changed without its text --
            # which is every unit, when the change is a corrected extraction
            # rule -- and the structured path kept answering with old values.
            if row and row["record_hash"] == record:
                skipped += 1
                continue
            cur.execute(
                "INSERT OR REPLACE INTO units "
                "(unit_id, document_id, source_uri, span_start, span_end, section, text, "
                "surface_hash, record_hash)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    eu.unit_id,
                    eu.document_id,
                    eu.unit.provenance.source_uri,
                    eu.unit.provenance.span.start,
                    eu.unit.provenance.span.end,
                    " > ".join(eu.unit.section_path),
                    eu.indexing_text(),
                    str(eu.indexing_hash),
                    record,
                ),
            )
            cur.execute("DELETE FROM fields WHERE unit_id = ?", (eu.unit_id,))
            wanted = self.param("fields")
            for k, v in eu.fields().items():
                if wanted and k not in wanted:
                    continue
                cur.execute(
                    "INSERT OR REPLACE INTO fields (unit_id, name, v_text, v_num, v_int, v_date)"
                    " VALUES (?,?,?,?,?,?)",
                    (eu.unit_id, k, *_typed_columns(v)),
                )
            written += 1
        self._conn.commit()
        return IndexWriteReceipt(written=written, skipped=skipped)

    def flush(self) -> None:
        """SQLite commits per batch already; this makes the capability uniform."""
        self._conn.commit()

    def delete(self, unit_ids: Sequence[UnitId], ctx: StageContext) -> int:
        cur = self._conn.cursor()
        cur.executemany("DELETE FROM fields WHERE unit_id = ?", [(u,) for u in unit_ids])
        cur.executemany("DELETE FROM units WHERE unit_id = ?", [(u,) for u in unit_ids])
        self._conn.commit()
        return cur.rowcount if cur.rowcount > 0 else 0

    def delete_document(self, document_id: DocumentId, ctx: StageContext) -> int:
        cur = self._conn.cursor()
        ids = [
            r["unit_id"]
            for r in cur.execute("SELECT unit_id FROM units WHERE document_id = ?", (document_id,))
        ]
        return self.delete([UnitId(i) for i in ids], ctx)

    # ------------------------------------------------------------------- read

    def search(self, query: IndexQuery, ctx: StageContext) -> RankedList:
        """Filter-only retrieval. **Empty when there is nothing to filter on.**

        A structured index has no text ranking: it cannot say which of two rows
        better answers a sentence. With a filter it can still say which rows
        match, and those come back in a stable, documented order (unit id)
        rather than a fabricated score order.

        With *no* filter there is neither a constraint nor a ranking, and the
        honest answer is nothing. Returning "the first k rows" instead looks
        like a result and is not one: the rows are the same for every query, so
        fusion receives a constant list of arbitrary units carrying the same
        rank-1 weight as a real top hit, and it displaces genuine matches in
        exactly the same way for every query in the set. An empty list costs a
        fuser nothing; a confident wrong list costs it the top of the ranking.
        """
        if query.filters is None and query.unit_ids is None:
            return RankedList(
                hits=(),
                source=self.name,
                query_text=query.text,
                fingerprint=self.fingerprint().key(),
            )
        where: str = ""
        params: list[Any] = []
        if query.filters is not None:
            where, params = compile_predicate(query.filters)
        sql = "SELECT * FROM units"
        if where:
            sql += f" WHERE unit_id IN (SELECT unit_id FROM units WHERE {where})"
        if query.unit_ids is not None:
            ids = list(query.unit_ids)
            if not ids:
                return RankedList(
                    hits=(),
                    source=self.name,
                    query_text=query.text,
                    fingerprint=self.fingerprint().key(),
                )
            placeholders = ", ".join("?" for _ in ids)
            sql += (" AND " if where else " WHERE ") + f"unit_id IN ({placeholders})"
            params = [*params, *ids]
        sql += " ORDER BY unit_id LIMIT ?"
        rows = self._conn.execute(sql, [*params, query.top_k]).fetchall()
        return RankedList(
            hits=tuple(
                Hit(
                    unit_id=UnitId(r["unit_id"]),
                    document_id=DocumentId(r["document_id"]),
                    rank=i,
                    score=1.0,
                    index=self.name,
                    provenance=Provenance(
                        document_id=DocumentId(r["document_id"]),
                        span=Span(r["span_start"], r["span_end"]),
                        source_uri=r["source_uri"] or "",
                    ),
                    matched_text=r["text"] or "",
                )
                for i, r in enumerate(rows, start=1)
            ),
            source=self.name,
            query_text=query.text,
            fingerprint=self.fingerprint().key(),
        )

    def structured_query(self, query: StructuredQuery, ctx: StageContext) -> RecordSet:
        """The STRUCTURED route's destination. Rows, not a ranking."""
        where, params = compile_predicate(query.where) if query.where else ("", [])

        # Every select expression is aliased to its logical name. Without the
        # alias sqlite names the column after the expression text, and the row
        # lookup by field name fails -- silently returning nothing on some
        # drivers, which would look like "no rows matched".
        select_parts: list[str] = []
        columns: list[str] = []
        for f in query.select:
            select_parts.append(f'{_field_expr(f)} AS "{f}"')
            columns.append(f)
        for agg in query.aggregations:
            name = _agg_name(agg)
            select_parts.append(f'{_agg_expr(agg)} AS "{name}"')
            columns.append(name)
        if not select_parts:
            select_parts = ['u.unit_id AS "unit_id"', 'u.document_id AS "document_id"']
            columns = ["unit_id", "document_id"]

        sql = f"SELECT {', '.join(select_parts)}, GROUP_CONCAT(u.unit_id) AS _src FROM units u"
        if where:
            sql += f" WHERE u.unit_id IN (SELECT unit_id FROM units WHERE {where})"
        if query.group_by:
            sql += " GROUP BY " + ", ".join(_field_expr(g) for g in query.group_by)
        elif query.aggregations:
            pass  # a bare aggregate over the whole set
        else:
            sql += " GROUP BY u.unit_id"
        if query.order_by:
            sql += " ORDER BY " + ", ".join(
                f"{_field_expr(f)} {'DESC' if desc else 'ASC'}" for f, desc in query.order_by
            )
        limit = int(query.limit) if query.limit else 1000

        # One row past the limit says whether the answer was cut; only then is
        # the full count worth a second query.
        rows = self._conn.execute(f"{sql} LIMIT {limit + 1}", params).fetchall()
        truncated = len(rows) > limit
        rows = rows[:limit]
        total = (
            int(self._conn.execute(f"SELECT COUNT(*) AS n FROM ({sql})", params).fetchone()["n"])
            if truncated
            else len(rows)
        )
        return RecordSet(
            columns=tuple(columns),
            rows=tuple(tuple(r[c] for c in columns) for r in rows),
            # Every row carries the units it came from, so a number in an answer
            # is as citable as a passage.
            sources=tuple(
                tuple(UnitId(x) for x in (r["_src"] or "").split(",") if x) for r in rows
            ),
            fingerprint=self.fingerprint().key(),
            total=total,
            truncated=truncated,
        )

    def all_unit_ids(self) -> list[UnitId]:
        """Every unit id, ascending. Used by tests and by ablation tooling."""
        return [
            UnitId(r["unit_id"])
            for r in self._conn.execute("SELECT unit_id FROM units ORDER BY unit_id")
        ]

    def stats(self) -> IndexStatsView:
        n = self._conn.execute("SELECT COUNT(*) c FROM units").fetchone()["c"]
        names = [
            r["name"] for r in self._conn.execute("SELECT DISTINCT name FROM fields ORDER BY name")
        ]
        return IndexStatsView(
            unit_count=n,
            size_bytes=self.path.stat().st_size if self.path.exists() else None,
            detail={"fields": names},
        )


# --------------------------------------------------------------------------- #
# predicate -> SQL                                                             #
# --------------------------------------------------------------------------- #


def _typed_columns(v: Any) -> tuple[Any, Any, Any, Any]:
    """(v_text, v_num, v_int, v_date) -- exactly one non-null."""
    if isinstance(v, bool):
        return (None, None, int(v), None)
    if isinstance(v, (datetime, date)):
        return (None, None, None, v.isoformat())
    if isinstance(v, int):
        return (None, None, v, None)
    if isinstance(v, float):
        return (None, v, None, None)
    return (None if v is None else str(v), None, None, None)


def _field_expr(name: str) -> str:
    if name in ("unit_id", "document_id", "source_uri", "section", "text"):
        return f"u.{name}"
    return (
        f"(SELECT COALESCE(f.v_text, f.v_num, f.v_int, f.v_date) FROM fields f "
        f"WHERE f.unit_id = u.unit_id AND f.name = {_quote(name)})"
    )


def _quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _agg_expr(agg: Aggregation) -> str:
    if agg.op == "count":
        return "COUNT(*)"
    return f"{agg.op.upper()}({_field_expr(agg.field or 'unit_id')})"


def _agg_name(agg: Aggregation) -> str:
    return f"{agg.op}_{agg.field}" if agg.field else str(agg.op)


def compile_predicate(pred: Predicate) -> tuple[str, list[Any]]:
    """Compile the AST to parameterised SQL over the ``units`` alias.

    Parameterised throughout: field *names* come from config and are quoted,
    field *values* come from queries and are always bound. A structured route
    receiving a user's text is an injection surface otherwise.

    Must agree with ``indexer.core.predicate.evaluate``. ``tests/test_sql.py``
    checks the two against each other over generated predicates.
    """
    params: list[Any] = []

    def walk(p: Predicate) -> str:
        match p:
            case Compare(field=f, op=op, value=v):
                col, val = _column_and_value(f, v)
                if v is None:
                    return (
                        f"{_exists_sql(f, col)} IS NULL"
                        if op is Op.EQ
                        else f"{_exists_sql(f, col)} IS NOT NULL"
                    )
                params.append(val)
                return f"{_exists_sql(f, col)} {_SQL_OPS[op]} ?"
            case In(field=f, values=vs):
                if not vs:
                    return "0"
                col, _ = _column_and_value(f, vs[0])
                params.extend(_bind(v) for v in vs)
                placeholders = ", ".join("?" for _ in vs)
                return f"{_exists_sql(f, col)} IN ({placeholders})"
            case Exists(field=f, present=present):
                clause = (
                    f"EXISTS (SELECT 1 FROM fields f WHERE f.unit_id = units.unit_id "
                    f"AND f.name = {_quote(f)} AND "
                    "COALESCE(f.v_text,f.v_num,f.v_int,f.v_date) IS NOT NULL)"
                )
                return clause if present else f"NOT {clause}"
            case TextMatch(field=f, value=v, mode=mode):
                col, _ = _column_and_value(f, v)
                pattern = {"contains": f"%{v}%", "prefix": f"{v}%", "exact": v}.get(mode)
                if pattern is None:
                    raise ValueError(f"unknown TextMatch mode {mode!r}")
                params.append(pattern)
                return f"{_exists_sql(f, col)} LIKE ? ESCAPE '\\'"
            case And(clauses=cs):
                return "(" + " AND ".join(walk(c) for c in cs) + ")" if cs else "1"
            case Or(clauses=cs):
                return "(" + " OR ".join(walk(c) for c in cs) + ")" if cs else "0"
            case Not(clause=c):
                return f"NOT ({walk(c)})"
        raise TypeError(f"cannot compile {type(p).__name__}")

    return walk(pred), params


def _column_and_value(field_name: str, sample: Any) -> tuple[str, Any]:
    if isinstance(sample, bool):
        return "v_int", int(sample)
    if isinstance(sample, (datetime, date)):
        return "v_date", sample.isoformat()
    if isinstance(sample, int):
        return "v_int", sample
    if isinstance(sample, float):
        return "v_num", sample
    return "v_text", sample


def _bind(v: Any) -> Any:
    return _column_and_value("", v)[1]


def _exists_sql(field_name: str, column: str) -> str:
    return (
        f"(SELECT f.{column} FROM fields f WHERE f.unit_id = units.unit_id "
        f"AND f.name = {_quote(field_name)})"
    )
