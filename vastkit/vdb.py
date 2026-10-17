"""VastDB helpers: SDK scans/writes, Query Engine SQL builder + dialect lint, local DuckDB stand-in.

Rules (from the kit and blueprint source):
- Never write, drop or alter vss-collection / vss-prompts-events. Write to your own schema.
  ``table()`` hands those back read-only; ``qe_query`` runs one SELECT/WITH statement only.
- SDK selects: name columns explicitly and leave out vectors/vectors_visual; predicate
  selects on tables with vector columns need ``patch_vector_predicates()`` (as the blueprint's
  source-code/ingest/vastdb-writer/common/vastdb_patch.py does).
- QE SQL (ADBC, Linux/macOS only): table ref is "<bucket>/<schema>"."<table>"; IN is rejected,
  so use OR chains; GROUP BY/JOIN/DISTINCT/window/ILIKE are unverified -> lint them out and
  aggregate in Python.
- DuckDB runs the same SELECT shape locally (array_cosine_distance over FLOAT[256]).
"""
from __future__ import annotations

import base64
import datetime as dt
import functools
import json
import os
import re
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

import pyarrow as pa

from .config import Settings, settings as _settings

DIM = 256
PROTECTED_TABLES = {"vss-collection", "vss-prompts-events"}
VECTOR_COLUMNS = {"vectors", "vectors_visual"}
VECTOR_TYPE = pa.list_(pa.field("item", pa.float32(), nullable=False), DIM)

# Text columns that are always safe to scan from vss-collection
SAFE_COLUMNS = [
    "pk", "source", "filename", "segment_number", "segment_start_sec", "segment_end_sec", "reasoning_content",
    "object_classes", "object_counts", "max_detection_conf", "perception_ok", "detection_sidecar_uri",
    "cosmos_model", "tokens_used", "processing_time", "is_public", "upload_timestamp", "duration",
    "total_segments", "original_video", "camera_id", "capture_type", "location", "extra_metadata",
]


def _endpoint(url: str) -> str:
    return url if url.startswith(("http://", "https://")) else f"http://{url}"


# ---------------------------------------------------------------- SDK
def patch_vector_predicates() -> None:
    """Drop list<float>/fixed_size_list fields from SDK predicate requests.

    Same type check as the blueprint's source-code/ingest/vastdb-writer/common/vastdb_patch.py (the backend's
    source-code/retrieval/video-backend/src/services/vastdb_service.py drops fixed_size_list only).
    """
    import vastdb._internal as _internal
    if getattr(_internal.build_query_data_request, "_vastkit_patched", False):
        return
    original = _internal.build_query_data_request

    def patched(schema, predicate=None, field_names=None):
        skip = {f.name for f in schema
                if "fixed_size_list" in str(f.type) or ("list<" in str(f.type) and "float" in str(f.type))}
        keep = pa.schema([f for f in schema if f.name not in skip])
        names = [n for n in (field_names or []) if n not in skip] if field_names is not None else None
        return original(keep, predicate, names)

    patched._vastkit_patched = True  # type: ignore[attr-defined]
    _internal.build_query_data_request = patched


def connect(cfg: Settings | None = None, *, endpoint: str | None = None):
    """vastdb session on the data VIP (S3_ENDPOINT). Run on the lab VM (private network)."""
    import vastdb
    cfg = cfg or _settings()
    ep = endpoint or cfg.s3_endpoint or cfg.vdb_endpoint
    if not (ep and cfg.access_key and cfg.secret_key):
        raise RuntimeError("S3_ENDPOINT / ACCESS_KEY / SECRET_KEY unset (VastDB is reachable from the lab VM)")
    return vastdb.connect(endpoint=_endpoint(ep), access=cfg.access_key, secret=cfg.secret_key, ssl_verify=False)


# vastdb.Table attributes that only read and return plain data. Everything else (tx, schema, projection(s),
# imports_table, blob_expansion, handle, ...) is refused: those hand back raw objects that can write.
READ_ONLY_ATTRS = frozenset({"select", "select_splits", "columns", "arrow_schema", "name", "path", "ref", "stats",
                             "get_stats", "sorted_columns", "retrieve_column_names"})


def _read_only(name: str) -> None:
    if name not in READ_ONLY_ATTRS:
        raise PermissionError(f"{name} refused: the shared VSS tables are read-only (allowed: "
                              f"{', '.join(sorted(READ_ONLY_ATTRS))}; write to your own schema)")


def _sdk_table(proxy: "ReadOnlyTable") -> Any:
    return object.__getattribute__(proxy, "_ReadOnlyTable__inner")


def _call_read_only(proxy: "ReadOnlyTable", name: str, *a: Any, **kw: Any) -> Any:
    _read_only(name)
    return getattr(_sdk_table(proxy), name)(*a, **kw)


class ReadOnlyTable:
    """Guard against accidental writes to the shared VSS tables; not a security boundary.

    ``READ_ONLY_ATTRS`` pass through and every other name raises PermissionError, which blocks the write
    methods and the navigators that return writable SDK objects. Methods come back as ``functools.partial``
    over this proxy, so no attribute, bound method or closure hands back the SDK table, and underscore names
    raise AttributeError. A determined caller can still reach it (``object.__getattribute__``, or the SDK itself).
    """

    __slots__ = ("__inner",)

    def __init__(self, inner: Any):
        self.__inner = inner

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)  # the held table and private SDK state (_tx, _metadata) are not proxied
        _read_only(name)
        v = getattr(_sdk_table(self), name)
        return functools.partial(_call_read_only, self, name) if callable(v) else v


def _shared(table_name: str, schema_name: str, cfg: Settings) -> bool:
    """Any vss-* table or the vss-schema schema (whatever cfg says), plus the configured shared names.

    Case-insensitive throughout, in case the backend resolves names that way.
    """
    t, s = table_name.lower(), schema_name.lower()
    shared_tables = {(n or "").lower() for n in (cfg.vdb_collection, cfg.vdb_prompts_collection)}
    return (t.startswith("vss-") or t in PROTECTED_TABLES or t in shared_tables
            or s == "vss-schema" or s == (cfg.vdb_schema or "").lower())


@contextmanager
def table(name: str | None = None, *, schema: str | None = None, session: Any = None,
          cfg: Settings | None = None) -> Iterator[Any]:
    """SDK table in a transaction; the shared VSS tables/schema come back as a ``ReadOnlyTable``."""
    cfg = cfg or _settings()
    s = session or connect(cfg)
    tname, sname = name or cfg.vdb_collection, schema or cfg.vdb_schema
    with s.transaction() as tx:
        t = tx.bucket(cfg.vastdb_bucket).schema(sname).table(tname)
        yield ReadOnlyTable(t) if _shared(tname, sname, cfg) else t


def scan(columns: Sequence[str] = tuple(SAFE_COLUMNS), *, predicate: Any = None, limit: int | None = None,
         table_name: str | None = None, schema: str | None = None, allow_vectors: bool = False,
         cfg: Settings | None = None) -> pa.Table:
    """Column-scoped select. Vector columns need allow_vectors=True (SDK reads often fail on them)."""
    cols = list(columns)
    if not cols:
        raise ValueError("name the columns: select() with no columns on vss-collection is unsafe")
    if VECTOR_COLUMNS & set(cols) and not allow_vectors:
        raise ValueError("vector columns excluded by default; pass allow_vectors=True to try")
    if predicate is not None:
        patch_vector_predicates()
    with table(table_name, schema=schema, cfg=cfg) as t:
        kw: dict[str, Any] = {"columns": cols, "internal_row_id": False}
        if predicate is not None:
            kw["predicate"] = predicate
        if limit:
            kw["limit_rows"] = limit
        try:
            reader = t.select(**kw)
        except TypeError:  # older SDKs: no limit_rows / internal_row_id
            kw.pop("limit_rows", None)
            kw.pop("internal_row_id", None)
            reader = t.select(**kw)
        tbl = reader.read_all()
    return tbl.slice(0, limit) if limit else tbl


def ensure_table(schema_name: str, table_name: str, arrow_schema: pa.Schema, *, cfg: Settings | None = None,
                 session: Any = None) -> None:
    """Create <bucket>/<schema_name>/<table_name> if missing (409/'already exists' = success)."""
    cfg = cfg or _settings()
    if _shared(table_name, schema_name, cfg):
        raise PermissionError(f"{schema_name}/{table_name} refused: write to your own schema, never the VSS tables")
    s = session or connect(cfg)
    try:
        with s.transaction() as tx:
            b = tx.bucket(cfg.vastdb_bucket)
            sch = b.schema(schema_name, fail_if_missing=False) or b.create_schema(schema_name, fail_if_exists=False)
            if not sch.table(table_name, fail_if_missing=False):
                sch.create_table(table_name, arrow_schema, fail_if_exists=False)
    except Exception as exc:  # concurrent create
        if "already exists" not in str(exc).lower() and "409" not in str(exc) and "conflict" not in str(exc).lower():
            raise


def insert(schema_name: str, table_name: str, rows: list[dict[str, Any]], arrow_schema: pa.Schema, *,
           cfg: Settings | None = None, session: Any = None) -> int:
    cfg = cfg or _settings()
    if _shared(table_name, schema_name, cfg):
        raise PermissionError(f"{schema_name}/{table_name} is protected: write to your own schema")
    if not rows:
        return 0
    s = session or connect(cfg)
    ensure_table(schema_name, table_name, arrow_schema, cfg=cfg, session=s)
    with s.transaction() as tx:
        t = tx.bucket(cfg.vastdb_bucket).schema(schema_name).table(table_name)
        t.insert(pa.Table.from_pylist(rows, schema=arrow_schema))
    return len(rows)


def arrow_schema(**cols: Any) -> pa.Schema:
    """arrow_schema(id=pa.utf8(), t=pa.float64(), emb='vector') -> pa.Schema ('vector' = 256-d float32)."""
    return pa.schema([(k, VECTOR_TYPE if v == "vector" else v) for k, v in cols.items()])


_TAG = "$vastkit"
_TIME_TYPES = {"datetime": dt.datetime, "date": dt.date, "time": dt.time}


def _outbox_default(v: Any) -> Any:
    """JSON fallback for Outbox rows: tag datetimes/bytes so replay restores them; numpy -> Python values."""
    for kind, cls in _TIME_TYPES.items():  # datetime first: it subclasses date
        if isinstance(v, cls):
            return {_TAG: kind, "v": v.isoformat()}
    if isinstance(v, (bytes, bytearray)):
        return {_TAG: "bytes", "v": base64.b64encode(v).decode("ascii")}
    if hasattr(v, "tolist"):  # numpy arrays and scalars
        return v.tolist()
    return str(v)


def _outbox_hook(d: dict[str, Any]) -> Any:
    kind = d.get(_TAG)
    if kind is None or len(d) != 2:
        return d
    if kind == "bytes":
        return base64.b64decode(d["v"])
    return _TIME_TYPES[kind].fromisoformat(d["v"]) if kind in _TIME_TYPES else d


class Outbox:
    """Append-only local JSONL of rows that failed to write; ``replay`` retries them later.

    Datetimes/bytes are type-tagged and numpy values stored as Python values, so replay inserts what
    ``add`` was given. Each group that replays is followed by a done marker, so it is never re-inserted.
    """

    def __init__(self, path: str = "outbox.jsonl"):
        self.path = path

    def _append(self, records: Iterable[dict[str, Any]]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, default=_outbox_default) + "\n")

    def add(self, schema_name: str, table_name: str, rows: list[dict[str, Any]]) -> None:
        self._append({"schema": schema_name, "table": table_name, "row": r} for r in rows)

    def _pending(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Rows not yet replayed, by (schema, table): a done marker drops that group's earlier rows."""
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        if not os.path.exists(self.path):
            return groups
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line, object_hook=_outbox_hook)
                key = (d["schema"], d["table"])
                if d.get("done"):
                    groups.pop(key, None)
                else:
                    groups.setdefault(key, []).append(d["row"])
        return groups

    def replay(self, arrow_schemas: dict[str, pa.Schema], **kw: Any) -> int:
        n = 0
        for (sch, tbl), rows in self._pending().items():
            n += insert(sch, tbl, rows, arrow_schemas[tbl], **kw)
            self._append([{"schema": sch, "table": tbl, "done": True}])
        if os.path.exists(self.path):
            os.replace(self.path, self.path + ".done")
        return n


# ---------------------------------------------------------------- SQL (Query Engine via ADBC)
_BANNED = re.compile(r"\b(IN\s*\(|ILIKE\b|OVER\s*\(|GROUP\s+BY\b|DISTINCT\b|JOIN\b|HAVING\b)", re.I)
_WRITES = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|TRUNCATE|GRANT|INTO)\b", re.I)
# string literals, "quoted identifiers" and comments, matched left to right so none hides inside another;
# a -- comment ends at CR or LF, as in DuckDB and Trino, so a lone CR cannot hide the rest of the line
_OPAQUE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\r\n]*|/\*.*?\*/", re.S)


def lint_sql(sql: str) -> str:
    """Accept one read-only SELECT/WITH statement free of constructs the event QE rejects (IN) or hasn't verified.

    Raises ValueError otherwise. String literals, quoted identifiers and comments are not linted, so anything
    that could end one where this scan does not is refused: a backslash inside quotes, a quote right after a
    letter (E'', U&'' escape strings) and nested /* comments.
    """
    for m in _OPAQUE.finditer(sql):
        tok, before = m.group(0), sql[m.start() - 1] if m.start() else " "
        if tok[0] in "'\"" and ("\\" in tok or before.isalnum() or before in "_&"):
            raise ValueError("escape-string literals refused: no backslash inside quotes and no letter right "
                             "before a quote (E'', U&''); put a space before the quote")
        if tok.startswith("/*") and "/*" in tok[2:]:
            raise ValueError("nested /* comments refused")
    stripped = _OPAQUE.sub(lambda m: m.group(0)[0] * 2 if m.group(0)[0] in "'\"" else " ", sql)
    if not re.match(r"[\s(]*(SELECT|WITH)\b", stripped, re.I):
        raise ValueError("qe_query runs a single SELECT / WITH statement only")
    if re.search(r"[;$]", stripped):  # '$': dollar-quoting could hide a ';' from the literal scan
        raise ValueError("one statement only: ';' or '$' outside string literals is refused")
    m = _WRITES.search(stripped)
    if m:
        raise ValueError(f"write/DDL keyword {m.group(0)!r} refused: qe_query is read-only")
    m = _BANNED.search(stripped)
    if m:
        raise ValueError(f"QE-unsafe SQL construct {m.group(0)!r}: use OR chains / aggregate in Python")
    return sql


def q(value: str) -> str:
    """SQL string literal with '' escaping (the QE path has no bind parameters)."""
    return "'" + str(value).replace("'", "''") + "'"


def or_chain(col: str, values: Iterable[Any]) -> str:
    vals = list(values)
    if not vals:
        return "FALSE"
    return "(" + " OR ".join(f"{col} = {q(v) if isinstance(v, str) else v}" for v in vals) + ")"


def table_ref(bucket: str, schema: str, table_name: str) -> str:
    return f'"{bucket}/{schema}"."{table_name}"'


def cosine_distance_sql(qvec: Sequence[float], col: str = "vectors") -> str:
    """``array_cosine_distance(col::FLOAT[n], ARRAY[...]::FLOAT[n])``: the QE/DuckDB expression only (1 - cosine).

    No ORDER BY or LIMIT: how results are filtered and ordered is the project's to write.
    """
    n = len(qvec)
    vec = "[" + ",".join(f"{float(x):.7g}" for x in qvec) + "]"
    return f"array_cosine_distance({col}::FLOAT[{n}], ARRAY{vec}::FLOAT[{n}])"


ADBC_CANDIDATES = ("/opt/adbc-driver/libadbc_driver_vastdb.so", "/tmp/libadbc_driver_vastdb.so")


def find_adbc_driver() -> str | None:
    """$VAST_ADBC_DRIVER, /opt/adbc-driver, /tmp, then the pip package.

    /opt/adbc-driver then /tmp is the blueprint backend's order (VastDBService._setup_adbc in
    source-code/retrieval/video-backend/src/services/vastdb_service.py); the env var and the pip package are
    vastkit's (the backend downloads the driver instead).
    """
    import os
    explicit = os.environ.get("VAST_ADBC_DRIVER")
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for cand in ADBC_CANDIDATES:
        if os.path.isfile(cand):
            return cand
    try:
        import adbc_driver_vastdb  # type: ignore
        return adbc_driver_vastdb.get_driver_path()
    except ImportError:
        return None


def qe_query(sql: str, *, endpoint: str | None = None, cfg: Settings | None = None,
             driver_path: str | None = None) -> pa.Table:
    """Run one read-only SELECT/WITH (checked by ``lint_sql``) on the Query Engine VIP through ADBC.

    Linux/macOS driver only: run it on the VM.
    """
    lint_sql(sql)
    import sys
    if sys.platform == "win32" and not driver_path:
        raise RuntimeError("the VastDB ADBC driver has no Windows build: run SQL on the lab VM "
                           "(or use vdb.duckdb_standin offline)")
    import adbc_driver_manager.dbapi as dbapi
    cfg = cfg or _settings()
    driver_path = driver_path or find_adbc_driver()
    if not driver_path:
        raise RuntimeError("ADBC VastDB driver not found (set VAST_ADBC_DRIVER or pip install adbc-driver-vastdb)")
    ep = endpoint or cfg.vdb_endpoint
    with dbapi.connect(driver=driver_path, db_kwargs={"vast.db.endpoint": _endpoint(ep),
                                                      "vast.db.access_key": cfg.access_key,
                                                      "vast.db.secret_key": cfg.secret_key}) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetch_arrow_table()


# ---------------------------------------------------------------- local stand-in
def _duck_type(values: Iterable[Any]) -> str:
    """DuckDB type from all non-None values of a column (ints widen to DOUBLE when a float appears)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "VARCHAR"
    if all(isinstance(v, bool) for v in vals):
        return "BOOLEAN"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return "DOUBLE" if any(isinstance(v, float) for v in vals) else "BIGINT"
    if all(isinstance(v, (list, tuple)) for v in vals):
        return _duck_type(x for v in vals for x in v) + "[]"
    return "VARCHAR"


def duckdb_standin(rows: list[dict[str, Any]], *, bucket: str = "team-mock-vss-db", schema: str = "vss-schema",
                   table_name: str = "vss-collection", path: str = ":memory:"):
    """DuckDB with the same "<bucket>/<schema>"."<table>" ref and FLOAT[256] columns as the QE.

    rows: dicts with any of SAFE_COLUMNS plus optional 'vectors'/'vectors_visual' lists. Columns are the
    union of the rows' keys; each type comes from all of that column's non-None values, not just rows[0].
    """
    if not rows:
        raise ValueError("no rows")
    import duckdb
    con = duckdb.connect(path)
    sch = f"{bucket}/{schema}"
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{sch}"')
    cols = list(dict.fromkeys(c for r in rows for c in r if not c.startswith("_")))
    defs = [f'"{c}" FLOAT[{DIM}]' if c in VECTOR_COLUMNS else f'"{c}" {_duck_type(r.get(c) for r in rows)}'
            for c in cols]
    con.execute(f'CREATE OR REPLACE TABLE "{sch}"."{table_name}" ({", ".join(defs)})')
    placeholders = ", ".join("?" for _ in cols)
    con.executemany(f'INSERT INTO "{sch}"."{table_name}" VALUES ({placeholders})',
                    [[r.get(c) for c in cols] for r in rows])
    return con
