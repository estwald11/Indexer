"""Structured index: SQLite over extracted fields and scanner metadata.

This is where invariant 5's questions go instead of vector search. It is also
the one index with extra capabilities (``StructuredCapable``, and describing its
schema and its documents to an agent), which is why the frame asks for
capabilities by ``isinstance`` rather than putting them on every ``Index``.

The predicate compiler is the interesting part. ``indexer.core.predicate``
defines an AST with a reference in-memory evaluator; this compiles the same AST
to parameterised SQL. The conformance requirement is that the two agree, and
there is a test that checks it over generated predicates -- otherwise the same
question returns different answers depending on which store answered it. Three
places where they used to disagree, and now do not:

* A number compared across storage classes. ``amount > 1000`` bound the integer
  and compared the integer column, so an amount extracted as the float 1500.0
  never matched. Numbers now compare against both numeric columns.
* Case-insensitive text matching outside ASCII. SQLite's LIKE folds ASCII only,
  the evaluator folds everything; "CITTÀ" and "città" matched in memory and not
  here. Matching now uses the same ``str.casefold``, registered as a function.
* Negation over an absent field. A scalar subquery yields NULL, and NOT NULL is
  NULL, so ``Not(Compare(...))`` dropped units the evaluator kept. Comparisons
  are now EXISTS clauses, which are never NULL.

Schema
------
Fields are stored key/value rather than as columns, because the field set comes
from config and changes per project; a column per field would mean a migration
every time a corpus gains an extraction rule. A field may hold several values
(the groups an ACL grants, every VAT number a document mentions), one row each,
and every predicate over such a field is existential -- as in the evaluator.

A second pair of tables holds the same facts per *document*: the union of its
units' values in reading order. ``StructuredQuery(level="document")`` queries
those, so a document matches when its supplier is named in one unit and its
total in another, and a count counts documents.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
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
#: Columns of ``units`` addressable by name in select, group and order clauses.
_UNIT_COLUMNS = ("unit_id", "document_id", "source_uri", "section", "text")
_DOC_COLUMNS = ("document_id", "source_uri", "unit_count")
_SCHEMA_VERSION = "3"


@dataclass(frozen=True, slots=True)
class SqliteParams:
    path: str = "./var/index/fields.db"
    #: Fields to index. Empty means index whatever enrich produced plus the
    #: scanner's metadata, which is the right default: the extraction schema
    #: already lives in the enricher config and repeating it here would let the
    #: two drift apart.
    fields: list[str] | None = None


@register(
    "index",
    "sqlite",
    version="3",
    params_model=dataclass_params(SqliteParams),
    summary="SQLite over extracted fields. Answers the STRUCTURED route without vector search.",
)
def _make_sqlite(params: dict[str, Any], **kw: Any) -> SqliteStructuredIndex:
    return SqliteStructuredIndex(params, name=kw.get("name", "fields"))


class SqliteStructuredIndex(StageImpl):
    STAGE, IMPL, VERSION = "index", "sqlite", "3"
    kind = "structured"

    def __init__(self, params: dict[str, Any], name: str = "fields") -> None:
        super().__init__(params)
        self.name = name
        self.path = Path(params.get("path", "./var/index/fields.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # The evaluator's case folding, so TextMatch agrees outside ASCII.
        self._conn.create_function("py_fold", 1, _fold, deterministic=True)
        self._open()
        self._touched: set[str] = set()

    # ------------------------------------------------------------------ schema

    def _open(self) -> None:
        """Create the schema, or start over if the database was built as
        something else.

        Same rule as the other indexes: the ledger restages every document when
        this index's fingerprint changes, and rows left from the old
        configuration would otherwise survive as "unchanged".
        """
        self._conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        want = json.dumps({**self.store_identity(), "schema": _SCHEMA_VERSION}, sort_keys=True)
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'identity'").fetchone()
        if row is not None and row["value"] != want:
            self._conn.executescript(
                """
                DROP TABLE IF EXISTS fields; DROP TABLE IF EXISTS units;
                DROP TABLE IF EXISTS doc_fields; DROP TABLE IF EXISTS documents;
                """
            )
        elif row is None:
            # A database from before the identity was recorded has the old
            # single-valued schema; it cannot be altered in place.
            has_units = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='units'"
            ).fetchone()
            if has_units:
                self._conn.executescript("DROP TABLE IF EXISTS fields; DROP TABLE units;")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS units (
                unit_id      TEXT PRIMARY KEY,
                document_id  TEXT NOT NULL,
                source_uri   TEXT,
                span_start   INTEGER,
                span_end     INTEGER,
                section      TEXT,
                text         TEXT,
                ordinal      INTEGER,
                surface_hash TEXT,
                record_hash  TEXT
            );
            CREATE TABLE IF NOT EXISTS fields (
                unit_id TEXT NOT NULL,
                name    TEXT NOT NULL,
                ord     INTEGER NOT NULL DEFAULT 0,
                -- One typed column per storage class, exactly one non-null.
                -- Storing everything as TEXT would make every numeric and
                -- temporal comparison a string comparison, which is the bug
                -- invariant 5 exists to prevent.
                v_text  TEXT,
                v_num   REAL,
                v_int   INTEGER,
                v_date  TEXT,
                PRIMARY KEY (unit_id, name, ord)
            );
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                source_uri  TEXT,
                unit_count  INTEGER,
                first_unit  TEXT
            );
            CREATE TABLE IF NOT EXISTS doc_fields (
                document_id TEXT NOT NULL,
                name        TEXT NOT NULL,
                ord         INTEGER NOT NULL,
                v_text  TEXT,
                v_num   REAL,
                v_int   INTEGER,
                v_date  TEXT,
                PRIMARY KEY (document_id, name, ord)
            );
            CREATE INDEX IF NOT EXISTS ix_fields_name ON fields(name, v_text);
            CREATE INDEX IF NOT EXISTS ix_fields_num  ON fields(name, v_num, v_int);
            CREATE INDEX IF NOT EXISTS ix_units_doc   ON units(document_id);
            CREATE INDEX IF NOT EXISTS ix_docf_name   ON doc_fields(name, v_text);
            """
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('identity', ?)", (want,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------ write

    def upsert(self, units: Sequence[EnrichedUnit], ctx: StageContext) -> IndexWriteReceipt:
        written = skipped = 0
        cur = self._conn.cursor()
        wanted = self.param("fields")
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
                "INSERT OR REPLACE INTO units (unit_id, document_id, source_uri, span_start, "
                "span_end, section, text, ordinal, surface_hash, record_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    eu.unit_id,
                    eu.document_id,
                    eu.unit.provenance.source_uri,
                    eu.unit.provenance.span.start,
                    eu.unit.provenance.span.end,
                    " > ".join(eu.unit.section_path),
                    eu.indexing_text(),
                    eu.unit.ordinal,
                    str(eu.indexing_hash),
                    record,
                ),
            )
            cur.execute("DELETE FROM fields WHERE unit_id = ?", (eu.unit_id,))
            for k, v in eu.filter_fields().items():
                if wanted and k not in wanted:
                    continue
                cur.executemany(
                    "INSERT INTO fields (unit_id, name, ord, v_text, v_num, v_int, v_date) "
                    "VALUES (?,?,?,?,?,?,?)",
                    [(eu.unit_id, k, i, *_typed_columns(x)) for i, x in enumerate(_values(v))],
                )
            self._touched.add(eu.document_id)
            written += 1
        self._refresh_documents()
        self._conn.commit()
        return IndexWriteReceipt(written=written, skipped=skipped)

    def flush(self) -> None:
        """SQLite commits per batch already; this makes the capability uniform."""
        self._refresh_documents()
        self._conn.commit()

    def delete(self, unit_ids: Sequence[UnitId], ctx: StageContext) -> int:
        cur = self._conn.cursor()
        ids = list(unit_ids)
        for chunk in _chunks(ids, 500):
            marks = ",".join("?" for _ in chunk)
            self._touched.update(
                r["document_id"]
                for r in cur.execute(
                    f"SELECT DISTINCT document_id FROM units WHERE unit_id IN ({marks})", chunk
                )
            )
        cur.executemany("DELETE FROM fields WHERE unit_id = ?", [(u,) for u in ids])
        cur.executemany("DELETE FROM units WHERE unit_id = ?", [(u,) for u in ids])
        n = cur.rowcount if cur.rowcount > 0 else 0
        self._refresh_documents()
        self._conn.commit()
        return n

    def delete_document(self, document_id: DocumentId, ctx: StageContext) -> int:
        cur = self._conn.cursor()
        ids = [
            r["unit_id"]
            for r in cur.execute("SELECT unit_id FROM units WHERE document_id = ?", (document_id,))
        ]
        return self.delete([UnitId(i) for i in ids], ctx)

    def _refresh_documents(self) -> None:
        """Rebuild the per-document rows of every document touched since the last
        refresh: the union of its units' values, in reading order."""
        if not self._touched:
            return
        cur = self._conn.cursor()
        for doc_id in sorted(self._touched):
            cur.execute("DELETE FROM doc_fields WHERE document_id = ?", (doc_id,))
            units = cur.execute(
                "SELECT unit_id, source_uri FROM units WHERE document_id = ? "
                "ORDER BY ordinal, unit_id",
                (doc_id,),
            ).fetchall()
            if not units:
                cur.execute("DELETE FROM documents WHERE document_id = ?", (doc_id,))
                continue
            cur.execute(
                "INSERT OR REPLACE INTO documents (document_id, source_uri, unit_count, first_unit)"
                " VALUES (?,?,?,?)",
                (doc_id, units[0]["source_uri"], len(units), units[0]["unit_id"]),
            )
            seen: dict[str, list[tuple[Any, ...]]] = {}
            rows = cur.execute(
                "SELECT f.name, f.v_text, f.v_num, f.v_int, f.v_date FROM fields f "
                "JOIN units u ON u.unit_id = f.unit_id WHERE u.document_id = ? "
                "ORDER BY u.ordinal, u.unit_id, f.name, f.ord",
                (doc_id,),
            ).fetchall()
            for r in rows:
                values = seen.setdefault(r["name"], [])
                cols = (r["v_text"], r["v_num"], r["v_int"], r["v_date"])
                if cols not in values:
                    values.append(cols)
            cur.executemany(
                "INSERT INTO doc_fields (document_id, name, ord, v_text, v_num, v_int, v_date) "
                "VALUES (?,?,?,?,?,?,?)",
                [
                    (doc_id, name, i, *cols)
                    for name, values in seen.items()
                    for i, cols in enumerate(values)
                ],
            )
        self._touched.clear()

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
        empty = RankedList(
            hits=(), source=self.name, query_text=query.text, fingerprint=self.fingerprint().key()
        )
        if query.filters is None and query.unit_ids is None:
            return empty
        where: list[str] = []
        params: list[Any] = []
        if query.filters is not None:
            clause, params = compile_predicate(query.filters)
            where.append(clause)
        if query.unit_ids is not None:
            ids = list(query.unit_ids)
            if not ids:
                return empty
            where.append(f"units.unit_id IN ({', '.join('?' for _ in ids)})")
            params = [*params, *ids]
        sql = f"SELECT * FROM units WHERE {' AND '.join(where)} ORDER BY unit_id LIMIT ?"
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
        if query.level not in ("unit", "document"):
            raise ValueError(f"level must be unit or document, not {query.level!r}")
        self._refresh_documents()
        doc_level = query.level == "document"
        table, key, fields_table = (
            ("documents", "document_id", "doc_fields")
            if doc_level
            else ("units", "unit_id", "fields")
        )
        columns_of = _DOC_COLUMNS if doc_level else _UNIT_COLUMNS
        where, params = (
            compile_predicate(query.where, table=table, key=key, fields_table=fields_table)
            if query.where
            else ("", [])
        )

        def expr(name: str) -> str:
            return _field_expr(
                name, table=table, key=key, fields_table=fields_table, columns=columns_of
            )

        # Every select expression is aliased to its logical name. Without the
        # alias sqlite names the column after the expression text, and the row
        # lookup by field name fails -- silently returning nothing on some
        # drivers, which would look like "no rows matched".
        select_parts: list[str] = []
        columns: list[str] = []
        for f in query.select:
            select_parts.append(f'{expr(f)} AS "{f}"')
            columns.append(f)
        for agg in query.aggregations:
            name = _agg_name(agg)
            select_parts.append(f'{_agg_expr(agg, expr, key)} AS "{name}"')
            columns.append(name)
        if not select_parts:
            select_parts = [f'{table}.{key} AS "{key}"']
            columns = [key]
            if not doc_level:
                select_parts.append('units.document_id AS "document_id"')
                columns.append("document_id")

        src = f"GROUP_CONCAT({table}.{'first_unit' if doc_level else 'unit_id'})"
        sql = f"SELECT {', '.join(select_parts)}, {src} AS _src FROM {table}"
        if where:
            sql += f" WHERE {where}"
        if query.group_by:
            sql += " GROUP BY " + ", ".join(expr(g) for g in query.group_by)
        elif query.aggregations:
            pass  # a bare aggregate over the whole set
        else:
            sql += f" GROUP BY {table}.{key}"
        if query.order_by:
            sql += " ORDER BY " + ", ".join(
                f"{expr(f)} {'DESC' if desc else 'ASC'}" for f, desc in query.order_by
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

    # ---------------------------------------------------------- introspection

    def describe_schema(self, *, sample_values: int = 5) -> list[dict[str, Any]]:
        """Every field this index holds: its types, how many units and documents
        carry it, its range, and a few example values.

        What an agent needs before it can write a predicate: the field names,
        and whether ``importo`` is a number and ``data_fattura`` a date, without
        which it guesses -- and a guessed type queries a column the value was
        never written to.
        """
        self._refresh_documents()
        out: list[dict[str, Any]] = []
        names = [r["name"] for r in self._conn.execute("SELECT DISTINCT name FROM fields")]
        for name in sorted(names):
            stats = self._conn.execute(
                "SELECT COUNT(DISTINCT unit_id) AS units, "
                "SUM(v_text IS NOT NULL) AS t, SUM(v_num IS NOT NULL) AS n, "
                "SUM(v_int IS NOT NULL) AS i, SUM(v_date IS NOT NULL) AS d, "
                "MIN(COALESCE(v_num, v_int)) AS lo, MAX(COALESCE(v_num, v_int)) AS hi, "
                "MIN(v_date) AS dlo, MAX(v_date) AS dhi FROM fields WHERE name = ?",
                (name,),
            ).fetchone()
            docs = self._conn.execute(
                "SELECT COUNT(DISTINCT document_id) AS n FROM doc_fields WHERE name = ?", (name,)
            ).fetchone()["n"]
            counts = (
                ("str", stats["t"]),
                ("float", stats["n"]),
                ("int", stats["i"]),
                ("date", stats["d"]),
            )
            types = [t for t, c in counts if c]
            info: dict[str, Any] = {
                "name": name,
                "types": types,
                "units": stats["units"],
                "documents": docs,
            }
            if stats["lo"] is not None:
                info["min"], info["max"] = stats["lo"], stats["hi"]
            if stats["dlo"] is not None:
                info["min_date"], info["max_date"] = stats["dlo"], stats["dhi"]
            if stats["t"]:
                info["examples"] = [
                    r["v_text"]
                    for r in self._conn.execute(
                        "SELECT v_text, COUNT(*) AS c FROM fields WHERE name = ? AND "
                        "v_text IS NOT NULL GROUP BY v_text ORDER BY c DESC, v_text LIMIT ?",
                        (name, sample_values),
                    )
                ]
            out.append(info)
        return out

    def document_fields(self, document_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Per-document field values, for a catalog card or result shaping.

        Single values come back as scalars, repeated ones as tuples in reading
        order.
        """
        self._refresh_documents()
        out: dict[str, dict[str, Any]] = {}
        ids = list(dict.fromkeys(document_ids))
        for chunk in _chunks(ids, 500):
            marks = ",".join("?" for _ in chunk)
            for r in self._conn.execute(
                f"SELECT document_id, name, ord, v_text, v_num, v_int, v_date FROM doc_fields "
                f"WHERE document_id IN ({marks}) ORDER BY document_id, name, ord",
                chunk,
            ):
                value = _row_value(r)
                slot = out.setdefault(r["document_id"], {})
                if r["name"] in slot:
                    prev = slot[r["name"]]
                    slot[r["name"]] = (*prev, value) if isinstance(prev, tuple) else (prev, value)
                else:
                    slot[r["name"]] = value
        return out

    def unit_ids_for(self, document_ids: Iterable[str]) -> list[UnitId]:
        """The units of these documents, for scoping retrieval to them."""
        out: list[UnitId] = []
        ids = list(dict.fromkeys(document_ids))
        for chunk in _chunks(ids, 500):
            marks = ",".join("?" for _ in chunk)
            out.extend(
                UnitId(r["unit_id"])
                for r in self._conn.execute(
                    f"SELECT unit_id FROM units WHERE document_id IN ({marks}) "
                    f"ORDER BY document_id, ordinal, unit_id",
                    chunk,
                )
            )
        return out

    def all_unit_ids(self) -> list[UnitId]:
        """Every unit id, ascending. Used by tests and by ablation tooling."""
        return [
            UnitId(r["unit_id"])
            for r in self._conn.execute("SELECT unit_id FROM units ORDER BY unit_id")
        ]

    def stats(self) -> IndexStatsView:
        n = self._conn.execute("SELECT COUNT(*) c FROM units").fetchone()["c"]
        docs = self._conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
        names = [
            r["name"] for r in self._conn.execute("SELECT DISTINCT name FROM fields ORDER BY name")
        ]
        return IndexStatsView(
            unit_count=n,
            size_bytes=self.path.stat().st_size if self.path.exists() else None,
            detail={"fields": names, "documents": docs},
        )


# --------------------------------------------------------------------------- #
# values                                                                       #
# --------------------------------------------------------------------------- #


def _fold(s: Any) -> Any:
    return s.casefold() if isinstance(s, str) else s


def _values(v: Any) -> list[Any]:
    """A field's values: a list for a multi-valued field, one for a scalar."""
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _typed_columns(v: Any) -> tuple[Any, Any, Any, Any]:
    """(v_text, v_num, v_int, v_date) -- exactly one non-null."""
    if isinstance(v, bool):
        return (None, None, int(v), None)
    if isinstance(v, datetime):
        return (None, None, None, v.isoformat())
    if isinstance(v, date):
        return (None, None, None, v.isoformat())
    if isinstance(v, int):
        return (None, None, v, None)
    if isinstance(v, float):
        return (None, v, None, None)
    return (None if v is None else str(v), None, None, None)


def _row_value(r: sqlite3.Row) -> Any:
    """A typed value from whichever storage column holds it."""
    if r["v_text"] is not None:
        return r["v_text"]
    if r["v_num"] is not None:
        return float(r["v_num"])
    if r["v_int"] is not None:
        return int(r["v_int"])
    if r["v_date"] is not None:
        d = str(r["v_date"])
        return datetime.fromisoformat(d) if "T" in d else date.fromisoformat(d)
    return None


def _chunks(items: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


# --------------------------------------------------------------------------- #
# predicate -> SQL                                                             #
# --------------------------------------------------------------------------- #


def _quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _value_expr(sample: Any) -> tuple[str, Any]:
    """The column expression a value compares against, and the value to bind.

    Numbers compare against both numeric columns: a field written as 1500.0 in
    one unit and 1500 in another is one field, and ``amount > 1000`` must see
    both. Booleans live in ``v_int`` as 0/1, as they compare in Python.
    """
    if isinstance(sample, bool):
        return "f.v_int", int(sample)
    if isinstance(sample, datetime):
        return "f.v_date", sample.isoformat()
    if isinstance(sample, date):
        return "f.v_date", sample.isoformat()
    if isinstance(sample, (int, float)):
        return "COALESCE(f.v_num, f.v_int)", sample
    return "f.v_text", sample


def _any_value(fields_table: str, table: str, key: str, name: str, condition: str) -> str:
    return (
        f"EXISTS (SELECT 1 FROM {fields_table} f WHERE f.{key} = {table}.{key} "
        f"AND f.name = {_quote(name)} AND {condition})"
    )


def compile_predicate(
    pred: Predicate,
    *,
    table: str = "units",
    key: str = "unit_id",
    fields_table: str = "fields",
) -> tuple[str, list[Any]]:
    """Compile the AST to parameterised SQL over ``table``.

    Parameterised throughout: field *names* come from config and are quoted,
    field *values* come from queries and are always bound. A structured route
    receiving a user's text is an injection surface otherwise.

    Must agree with ``indexer.core.predicate.evaluate``. ``tests/test_sql.py``
    checks the two against each other over generated predicates.
    """
    params: list[Any] = []
    present = "COALESCE(f.v_text, f.v_num, f.v_int, f.v_date) IS NOT NULL"

    def walk(p: Predicate) -> str:
        match p:
            case Compare(field=f, op=op, value=v):
                if v is None:
                    # EQ None means "absent"; every other operator against None
                    # is false in the evaluator, NE included. ``Exists`` is the
                    # way to ask for presence.
                    if op is Op.EQ:
                        return f"NOT {_any_value(fields_table, table, key, f, present)}"
                    return "0"
                col, val = _value_expr(v)
                params.append(val)
                return _any_value(fields_table, table, key, f, f"{col} {_SQL_OPS[op]} ?")
            case In(field=f, values=vs):
                if not vs:
                    return "0"
                col, _ = _value_expr(vs[0])
                params.extend(_value_expr(v)[1] for v in vs)
                marks = ", ".join("?" for _ in vs)
                return _any_value(fields_table, table, key, f, f"{col} IN ({marks})")
            case Exists(field=f, present=is_present):
                clause = _any_value(fields_table, table, key, f, present)
                return clause if is_present else f"NOT {clause}"
            case TextMatch(field=f, value=v, mode=mode):
                folded = _fold(v)
                if mode == "contains":
                    params.append(folded)
                    cond = "instr(py_fold(f.v_text), ?) > 0"
                elif mode == "prefix":
                    params.extend([folded, folded])
                    cond = "substr(py_fold(f.v_text), 1, length(?)) = ?"
                elif mode == "exact":
                    params.append(folded)
                    cond = "py_fold(f.v_text) = ?"
                else:
                    raise ValueError(f"unknown TextMatch mode {mode!r}")
                return _any_value(fields_table, table, key, f, cond)
            case And(clauses=cs):
                return "(" + " AND ".join(walk(c) for c in cs) + ")" if cs else "1"
            case Or(clauses=cs):
                return "(" + " OR ".join(walk(c) for c in cs) + ")" if cs else "0"
            case Not(clause=c):
                return f"NOT ({walk(c)})"
        raise TypeError(f"cannot compile {type(p).__name__}")

    return walk(pred), params


def _field_expr(
    name: str,
    *,
    table: str = "units",
    key: str = "unit_id",
    fields_table: str = "fields",
    columns: Sequence[str] = _UNIT_COLUMNS,
) -> str:
    """A field's value for select, group and order: the first in reading order."""
    if name in columns:
        return f"{table}.{name}"
    return (
        f"(SELECT COALESCE(f.v_text, f.v_num, f.v_int, f.v_date) FROM {fields_table} f "
        f"WHERE f.{key} = {table}.{key} AND f.name = {_quote(name)} ORDER BY f.ord LIMIT 1)"
    )


def _agg_expr(agg: Aggregation, expr: Any, key: str) -> str:
    if agg.op == "count":
        if agg.distinct and agg.field:
            return f"COUNT(DISTINCT {expr(agg.field)})"
        return "COUNT(*)"
    if agg.field is None:
        raise ValueError(f"{agg.op} needs a field")
    inner = expr(agg.field)
    if agg.op in ("sum", "avg"):
        # Numbers stored as text by a lenient extractor would otherwise be
        # summed as 0 by SQLite's affinity rules, without a word.
        return f"{agg.op.upper()}(CAST({inner} AS REAL))"
    return f"{agg.op.upper()}({inner})"


def _agg_name(agg: Aggregation) -> str:
    if agg.distinct and agg.field:
        return f"{agg.op}_distinct_{agg.field}"
    return f"{agg.op}_{agg.field}" if agg.field else str(agg.op)
