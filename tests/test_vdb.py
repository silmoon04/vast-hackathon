import datetime as dt
import math
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from vastkit.config import Settings
from vastkit.mocks.corpus import default_corpus
from vastkit.mocks.gpu_mock import _vec
from vastkit import vdb


def test_lint():
    vdb.lint_sql("SELECT a FROM t WHERE x = 'IN (1)' AND (y = 1 OR y = 2)")
    for bad in ("SELECT a FROM t WHERE x IN (1,2)", "SELECT a, COUNT(*) FROM t GROUP BY a",
                "SELECT DISTINCT a FROM t", "SELECT * FROM a JOIN b ON a.x=b.x", "SELECT a FROM t WHERE a ILIKE 'x'"):
        with pytest.raises(ValueError):
            vdb.lint_sql(bad)


SHARED = vdb.table_ref("team-x-vss-db", "vss-schema", "vss-collection")


@pytest.mark.parametrize("sql", [
    f"DELETE FROM {SHARED} WHERE 1=1",
    'DROP TABLE "team-x-vss-db/vss-schema"."vss-prompts-events"',
    f"INSERT INTO {SHARED} (pk) VALUES ('x')",
    f"UPDATE {SHARED} SET is_public = TRUE",
    f"SELECT pk FROM {SHARED}; DELETE FROM {SHARED}",
    "SELECT 1;",
    f"WITH d AS (DELETE FROM {SHARED} RETURNING pk) SELECT pk FROM d",
    f"/* SELECT */ DELETE FROM {SHARED}",
    f"SELECT pk INTO copy_tbl FROM {SHARED}",
    f"SELECT $$'$$; DROP TABLE {SHARED}; SELECT $$'$$",
    "TRUNCATE t", "ALTER TABLE t ADD COLUMN a INT", "CREATE TABLE t (a INT)", "MERGE INTO t USING s ON 1=1",
    "GRANT SELECT ON t TO x",
    # escape strings: DuckDB ends E'\'' at the second quote, the literal scan does not
    "SELECT E'\\'' ; DROP TABLE t; --'",
    "SELECT 1 FROM t WHERE a = E'\\'' OR 1=1; DELETE FROM t; --'",
    "SELECT 1 FROM t WHERE a = e'x'", "SELECT U&'d\\0061t'", "SELECT 1 FROM t WHERE a = 'x\\'",
    # nested comments: DuckDB ends the comment at the second */, the scan at the first
    "SELECT 1 /* /* */ ' */ ; DROP TABLE t; -- '",
    # DuckDB and Trino end a -- comment at a lone CR too
    'SELECT 1 --x\r; DROP TABLE "b/vss-schema"."vss-collection"',
    f"WITH a AS (SELECT 1) --x\rINSERT INTO {SHARED} SELECT * FROM a",
])
def test_lint_refuses_writes_and_chains(sql):
    with pytest.raises(ValueError):
        vdb.lint_sql(sql)
    with pytest.raises(ValueError):  # qe_query lints before it needs a driver or endpoint
        vdb.qe_query(sql, cfg=Settings())


def test_lint_ignores_literals_identifiers_and_comments():
    for ok in (f"SELECT pk FROM {SHARED} WHERE reasoning_content = 'a; DROP TABLE t; DELETE'",
               'SELECT "update-log", "join" FROM "b/s"."created" -- why; not\n WHERE x = 1',
               "  (SELECT a FROM t)", "WITH x AS (SELECT a FROM t) SELECT a FROM x",
               "SELECT updated_at, created_by FROM t /* ; */", "SELECT a FROM t WHERE b = 'it''s' -- C:\\x",
               "SELECT a FROM t WHERE (b='x')", "SELECT a -- note\r\nFROM t"):
        assert vdb.lint_sql(ok) == ok


def test_lint_payloads_really_write_on_duckdb():
    """The escape-string and nested-comment payloads refused above do drop a table when run directly."""
    import duckdb
    for sql in ("SELECT E'\\'' ; DROP TABLE t; --'", "SELECT 1 /* /* */ ' */ ; DROP TABLE t; -- '",
                "SELECT 1 --x\r; DROP TABLE t"):
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t (a INT)")
        con.execute(sql)
        assert con.execute("SHOW TABLES").fetchall() == []
        with pytest.raises(ValueError):
            vdb.lint_sql(sql)


def test_or_chain_and_quote():
    assert vdb.or_chain("camera_id", ["a", "o'b"]) == "(camera_id = 'a' OR camera_id = 'o''b')"
    assert vdb.or_chain("n", []) == "FALSE"


def test_scan_guards():
    with pytest.raises(ValueError):
        vdb.scan([])
    with pytest.raises(ValueError):
        vdb.scan(["source", "vectors"])


def test_protected_tables():
    s = pa.schema([("a", pa.int64())])
    with pytest.raises(PermissionError):
        vdb.insert("hackathon", "vss-collection", [{"a": 1}], s)
    raw = _FakeTable()
    cfg = Settings(vastdb_bucket="b", vdb_schema="team-x-view", vdb_collection="coll")  # VDB_SCHEMA overridden
    for schema, name in (("vss-schema", "results"), ("team-x", "vss-anything"), ("team-x", "VSS-Prompts"),
                         ("team-x", "coll"), ("team-x-view", "t"),
                         # names compare case-insensitively: schema, configured schema, configured table
                         ("VSS-Schema", "results"), ("Team-X-View", "t"), ("team-x", "COLL")):
        with pytest.raises(PermissionError):
            vdb.ensure_table(schema, name, s, cfg=cfg, session=_fake_session(raw))
        for rows in ([{"a": 1}], []):
            with pytest.raises(PermissionError):
                vdb.insert(schema, name, rows, s, cfg=cfg, session=_fake_session(raw))
    assert raw.inserted == []
    assert vdb.insert("team-x", "results", [{"a": 1}], s, cfg=cfg, session=_fake_session(raw)) == 1
    assert len(raw.inserted) == 1


def _rows():
    rows = default_corpus()
    for r in rows:
        r["vectors"] = _vec(r["reasoning_content"].lower())
        r["vectors_visual"] = _vec("v" + r["source"])
    rows[0]["vectors_visual"] = [0.0] * 256  # failed visual embed
    return rows


def test_duckdb_cosine_distance_sql():
    rows = _rows()
    con = vdb.duckdb_standin(rows)
    expr = vdb.cosine_distance_sql(_vec(rows[3]["reasoning_content"].lower()))
    assert "ORDER BY" not in expr and "LIMIT" not in expr
    ref = vdb.table_ref("team-mock-vss-db", "vss-schema", "vss-collection")
    srcs = [rows[3]["source"], rows[4]["source"]]
    got = dict(con.execute(vdb.lint_sql(f"SELECT source, {expr} AS distance FROM {ref} "
                                        f"WHERE {vdb.or_chain('source', srcs)}")).fetchall())
    assert math.isclose(got[srcs[0]], 0.0, abs_tol=1e-5) and got[srcs[1]] > 1e-3
    assert not hasattr(vdb, "local_knn") and not hasattr(vdb, "knn_sql")  # removed: ranking is built at the event


def test_duckdb_standin_types_from_all_rows():
    rows = [{"pk": "a", "score": 1, "conf": None, "tags": None, "n": 2, "flag": None},
            {"pk": "b", "score": 0.73, "conf": 0.5, "tags": ["x", "y"], "n": 3, "flag": True},
            {"pk": "c", "score": 2.6, "conf": 0.25, "tags": [], "extra": "late", "n": None, "flag": False}]
    con = vdb.duckdb_standin(rows, table_name="t")
    ref = '"team-mock-vss-db/vss-schema"."t"'
    types = dict(con.execute(f"SELECT column_name, data_type FROM information_schema.columns "
                             f"WHERE table_name = 't'").fetchall())
    assert types == {"pk": "VARCHAR", "score": "DOUBLE", "conf": "DOUBLE", "tags": "VARCHAR[]", "n": "BIGINT",
                     "flag": "BOOLEAN", "extra": "VARCHAR"}
    got = con.execute(f"SELECT score, conf, tags, extra FROM {ref} ORDER BY pk").fetchall()
    assert got == [(1.0, None, None, None), (0.73, 0.5, ["x", "y"], None), (2.6, 0.25, [], "late")]
    with pytest.raises(ValueError):
        vdb.duckdb_standin([])


class _FakeTable:
    name = "t"

    def __init__(self):
        self.inserted: list[pa.Table] = []

    def select(self, **kw):
        return "reader"

    def insert(self, tbl):
        self.inserted.append(tbl)


def _fake_session(tbl: _FakeTable):
    sch = SimpleNamespace(table=lambda n, **kw: tbl)

    @contextmanager
    def transaction():
        yield SimpleNamespace(bucket=lambda b: SimpleNamespace(schema=lambda s, **kw: sch))
    return SimpleNamespace(transaction=transaction)


def test_table_shared_is_read_only():
    cfg = Settings(vastdb_bucket="b")
    raw = _FakeTable()
    for name, schema in ((None, None), ("vss-prompts-events", "team-x"), ("anything", "vss-schema"),
                         ("vss-other", "team-x"), ("anything", "VSS-SCHEMA"), ("VSS-Collection", "team-x")):
        with vdb.table(name, schema=schema, session=_fake_session(raw), cfg=cfg) as t:
            assert isinstance(t, vdb.ReadOnlyTable) and t.select(columns=["pk"]) == "reader" and t.name == "t"
            for write in ("insert", "update", "delete", "drop", "drop_column", "add_column", "rename",
                          "create_projection", "import_files", "insert_in_column_batches",
                          # navigators that hand back raw, writable SDK objects
                          "tx", "schema", "handle", "projection", "projections", "blob_expansion", "imports_table",
                          "partitions", "sorted_table", "vector_search"):
                with pytest.raises(PermissionError):
                    getattr(t, write)
            with pytest.raises(AttributeError):
                t._tx  # private SDK state is never proxied
            for leak in ("_inner", "_ReadOnlyTable__inner", "_get", "__dict__", "__getstate__", "__reduce_ex__"):
                with pytest.raises(AttributeError):  # no plain attribute hands back the raw table
                    getattr(t, leak)
            sel = t.select
            # wrapped, not bound to the raw table, and no closure holds it (the round-3 introspection probe)
            assert getattr(sel, "__self__", None) is None and getattr(sel, "__closure__", None) is None
            assert all(a is not raw for a in sel.args)
            assert not any(raw is c.cell_contents for c in sel.func.__closure__ or ())
            with pytest.raises(PermissionError):  # the wrapper re-checks the name
                sel.func(t, "insert", "rows")
    assert raw.inserted == []
    with vdb.table("results", schema="team-x", session=_fake_session(raw), cfg=cfg) as t:
        t.insert("rows")
    assert raw.inserted == ["rows"]


def _outbox_rows():
    return [{"ts": dt.datetime(2026, 10, 17, 10, 0, 0, 123456, tzinfo=dt.timezone.utc), "day": dt.date(2026, 10, 17),
             "emb": np.linspace(0, 1, vdb.DIM, dtype=np.float32), "score": np.float32(0.5), "n": np.int64(7),
             "ok": np.bool_(True), "blob": b"\x00\xff", "note": None},
            {"ts": dt.datetime(2026, 10, 17, 11), "day": dt.date(2026, 10, 18), "emb": [0.0] * vdb.DIM,
             "score": 1.25, "n": 3, "ok": False, "blob": b"", "note": "x"}]


OUTBOX_SCHEMA = vdb.arrow_schema(ts=pa.timestamp("us", tz="UTC"), day=pa.date32(), emb="vector",
                                 score=pa.float32(), n=pa.int64(), ok=pa.bool_(), blob=pa.binary(), note=pa.utf8())


def test_outbox_round_trips_typed_values(tmp_path, monkeypatch):
    got: list[pa.Table] = []
    monkeypatch.setattr(vdb, "insert", lambda sch, tbl, rows, schema, **kw: got.append(
        pa.Table.from_pylist(rows, schema=schema)) or len(rows))
    ob = vdb.Outbox(str(tmp_path / "outbox.jsonl"))
    ob.add("team-x", "events", _outbox_rows())
    assert ob.replay({"events": OUTBOX_SCHEMA}) == 2
    assert got[0].equals(pa.Table.from_pylist(_outbox_rows(), schema=OUTBOX_SCHEMA))
    assert (tmp_path / "outbox.jsonl.done").exists() and ob.replay({"events": OUTBOX_SCHEMA}) == 0


def test_outbox_does_not_replay_completed_groups(tmp_path, monkeypatch):
    calls: list[tuple[str, int]] = []
    fail = {"b"}

    def fake_insert(sch, tbl, rows, schema, **kw):
        if tbl in fail:
            raise ConnectionError("VastDB down")
        calls.append((tbl, len(rows)))
        return len(rows)

    monkeypatch.setattr(vdb, "insert", fake_insert)
    s = pa.schema([("i", pa.int64())])
    ob = vdb.Outbox(str(tmp_path / "outbox.jsonl"))
    ob.add("team-x", "a", [{"i": 1}, {"i": 2}])
    ob.add("team-x", "b", [{"i": 3}])
    with pytest.raises(ConnectionError):
        ob.replay({"a": s, "b": s})
    assert calls == [("a", 2)]
    ob.add("team-x", "a", [{"i": 4}])  # new rows after a's done marker still replay
    fail.clear()
    assert ob.replay({"a": s, "b": s}) == 2
    assert calls == [("a", 2), ("b", 1), ("a", 1)]


def test_arrow_schema_vector():
    s = vdb.arrow_schema(id=pa.utf8(), emb="vector")
    assert s.field("emb").type == vdb.VECTOR_TYPE
