"""What the DataEngine segment table alone (5 s counts in VastDB, here duckdb) would answer for the count/total
questions, using only segments that were COMPLETE at question time. Shows what the per-frame tracker adds.
SQL avoids IN / ILIKE / GROUP BY / JOIN / window (VastDB subset)."""
from __future__ import annotations

# make the module importable on its own (protokit lives in prototypes/)
import sys as _sys  # noqa: E402
from pathlib import Path as _P  # noqa: E402
_HERE_PATHS = [str(_P(__file__).resolve().parents[1]), str(_P(__file__).resolve().parent)]
_sys.path[:0] = [x for x in _HERE_PATHS if x not in _sys.path]

from typing import Any

import protokit as pk

VEH = ("car", "truck", "bus", "motorcycle")


def _cols(df, group: str) -> list[str]:
    names = ("person",) if group == "person" else VEH
    return [f"n_{n}" for n in names if f"n_{n}" in df.columns]


def compute(name: str, script: dict[str, Any]) -> dict[str, Any]:
    seg = pk.segments(name)
    con = pk.vastdb({"seg": seg})
    alarm = script["alarm"]
    out = []
    for q in script["questions"]:
        for c in q.get("checks", []):
            if c["type"] not in ("count_now", "total"):
                continue
            cols = _cols(seg, c["group"])
            if not cols:
                got, sql = 0, "-- no such class in the table"
            elif c["type"] == "count_now":
                pk_cols = [x.replace("n_", "peak_") for x in cols]
                expr = " + ".join(pk_cols)
                sql = (f"SELECT {expr} AS n FROM seg WHERE \"end\" <= {q['t']:.2f} ORDER BY \"end\" DESC LIMIT 1")
                row = con.execute(sql).fetchone()
                got = int(row[0]) if row else 0
            else:
                expr = " + ".join(cols)
                sql = (f"SELECT SUM({expr}) AS n FROM seg WHERE \"end\" <= {q['t']:.2f} AND \"end\" > {alarm:.2f}")
                row = con.execute(sql).fetchone()
                got = int(row[0] or 0)
            lo, hi = c["range"]
            out.append({"id": q["id"], "type": c["type"], "group": c["group"], "sql": sql, "got": got,
                        "truth_range": c["range"], "right": lo <= got <= hi})
    return {"answers": out, "right": sum(a["right"] for a in out), "n": len(out)}
