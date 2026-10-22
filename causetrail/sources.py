"""Data gateways: one interface, two back-ends.

The agent never talks to a database directly. It writes SQL against *logical*
table names in braces (``{orders}``, ``{support_tickets}``...) and a gateway
resolves them to physical, federated names and executes the query.

* :class:`LocalGateway`   - offline. Loads the four fragmented demo files into an
  in-memory DuckDB, so cross-source SQL works with no infrastructure.
* :class:`ZetarisGateway` - Zetaris Lightning (federated/virtual SQL). Same SQL,
  the table map points each logical table at a Zetaris datasource.

A tiny macro, ``DAYS_BETWEEN(a, b)``, papers over the one dialect difference the
agent needs (DuckDB ``date_diff`` vs Spark-style ``datediff``).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Logical tables -> which physical source they come from (for docs / schema).
LOGICAL_TABLES: dict[str, str] = {
    "orders": "orders.db (SQLite)",
    "order_items": "orders.db (SQLite)",
    "payments": "orders.db (SQLite)",
    "refunds": "orders.db (SQLite)",
    "ad_spend": "ad_spend.csv (CSV)",
    "support_tickets": "support_tickets.json (JSON export)",
    "skus": "inventory_supplier.xlsx#skus (Excel)",
    "warehouses": "inventory_supplier.xlsx#warehouses (Excel)",
    "suppliers": "inventory_supplier.xlsx#suppliers (Excel)",
    "lead_times": "inventory_supplier.xlsx#lead_times (Excel)",
}

_TABLE_RE = re.compile(r"\{([a-z_]+)\}")
_DAYS_RE = re.compile(r"DAYS_BETWEEN\s*\(", re.I)


def _expand_days_between(sql: str, render) -> str:
    """Rewrite DAYS_BETWEEN(a, b) -> dialect SQL; paren-aware so args may nest."""
    out, pos = [], 0
    while (m := _DAYS_RE.search(sql, pos)) is not None:
        out.append(sql[pos : m.start()])
        depth, i, args, start = 1, m.end(), [], m.end()
        while i < len(sql) and depth:
            ch = sql[i]
            if ch == "'":
                i = sql.index("'", i + 1)
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    args.append(sql[start:i])
            elif ch == "," and depth == 1:
                args.append(sql[start:i])
                start = i + 1
            i += 1
        if depth or len(args) != 2:
            raise GatewayError("malformed DAYS_BETWEEN(a, b)")
        out.append(render(_expand_days_between(args[0].strip(), render), _expand_days_between(args[1].strip(), render)))
        pos = i
    out.append(sql[pos:])
    return "".join(out)


class GatewayError(RuntimeError):
    pass


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    sql: str  # the physical SQL actually executed
    elapsed_ms: float = 0.0

    def dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, r)) for r in self.rows]


@dataclass
class DataGateway(ABC):
    """Common interface. ``run_sql`` takes logical SQL, returns rows."""

    name: str = "abstract"
    table_map: dict[str, str] = field(default_factory=dict)

    def resolve(self, sql: str) -> str:
        def table(m: re.Match) -> str:
            key = m.group(1)
            if key not in self.table_map:
                raise GatewayError(f"unknown logical table {{{key}}}")
            return self.table_map[key]

        sql = _TABLE_RE.sub(table, sql)
        return _expand_days_between(sql, self.days_between)

    def days_between(self, a: str, b: str) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    @abstractmethod
    def run_sql(self, sql: str) -> QueryResult: ...

    @abstractmethod
    def list_schema(self) -> dict[str, list[tuple[str, str]]]:
        """logical table -> [(column, type)]"""

    def describe(self) -> str:
        """Schema as compact text, for prompts and the CLI."""
        lines = []
        for t, cols in self.list_schema().items():
            src = LOGICAL_TABLES.get(t, "?")
            lines.append(f"{{{t}}}  -- from {src}\n    " + ", ".join(f"{c} {ty}" for c, ty in cols))
        return "\n".join(lines)


class LocalGateway(DataGateway):
    """DuckDB over the local demo files. Emulates federation offline."""

    def __init__(self, data_dir: str | Path):
        import duckdb

        super().__init__(name="local-duckdb", table_map={t: t for t in LOGICAL_TABLES})
        self.data_dir = Path(data_dir)
        if not (self.data_dir / "orders.db").exists():
            raise GatewayError(f"no demo data in {self.data_dir} - run `python -m causetrail gen-data`")
        self.con = duckdb.connect(":memory:")
        self._load()

    def _load(self) -> None:
        import pandas as pd

        d = self.data_dir
        # SQLite: read via the stdlib, keep native column types.
        sq = sqlite3.connect(d / "orders.db")
        for t in ("orders", "order_items", "payments", "refunds"):
            df = pd.read_sql_query(f"SELECT * FROM {t}", sq)
            self._register(t, df)
        sq.close()
        self.con.execute("ALTER TABLE orders ADD COLUMN order_date DATE")
        self.con.execute("UPDATE orders SET order_date = CAST(order_ts AS TIMESTAMP)::DATE")
        for t, c in (("orders", "shipped_date"), ("orders", "promised_ship_date"), ("refunds", "refunded_at")):
            self.con.execute(f"ALTER TABLE {t} ALTER {c} TYPE DATE USING CAST({c} AS DATE)")
        self.con.execute("ALTER TABLE orders ALTER order_ts TYPE TIMESTAMP USING CAST(order_ts AS TIMESTAMP)")
        self.con.execute(
            "ALTER TABLE payments ALTER attempted_at TYPE TIMESTAMP USING CAST(attempted_at AS TIMESTAMP)"
        )
        # CSV
        self.con.execute(
            f"CREATE TABLE ad_spend AS SELECT * FROM read_csv_auto('{(d / 'ad_spend.csv').as_posix()}')"
        )
        # JSON helpdesk export: flatten, keep the awkward "#123" order reference.
        raw = json.loads((d / "support_tickets.json").read_text())["tickets"]
        df = pd.json_normalize(raw)
        df["tags"] = df["tags"].apply(lambda x: ",".join(x))
        df = df.rename(columns={"requester.customer_id": "requester_customer_id", "requester.lang": "requester_lang"})
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)
        self._register("support_tickets", df)
        # Excel workbook, one table per sheet.
        for sheet in ("skus", "warehouses", "suppliers", "lead_times"):
            df = pd.read_excel(d / "inventory_supplier.xlsx", sheet_name=sheet, keep_default_na=False)
            if "week_start" in df:
                df["week_start"] = pd.to_datetime(df["week_start"]).dt.date
            self._register(sheet, df)

    def _register(self, name: str, df) -> None:
        self.con.register("_tmp_df", df)
        self.con.execute(f"CREATE TABLE {name} AS SELECT * FROM _tmp_df")
        self.con.unregister("_tmp_df")

    def days_between(self, a: str, b: str) -> str:
        return f"date_diff('day', CAST({a} AS DATE), CAST({b} AS DATE))"

    def run_sql(self, sql: str) -> QueryResult:
        phys = self.resolve(sql)
        t0 = time.perf_counter()
        try:
            cur = self.con.execute(phys)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
        except Exception as e:  # duckdb raises many types
            raise GatewayError(f"{type(e).__name__}: {e}") from e
        return QueryResult(cols, rows, phys, (time.perf_counter() - t0) * 1000)

    def list_schema(self) -> dict[str, list[tuple[str, str]]]:
        out: dict[str, list[tuple[str, str]]] = {}
        for t in LOGICAL_TABLES:
            cols = self.con.execute(f"DESCRIBE {t}").fetchall()
            out[t] = [(c[0], str(c[1])) for c in cols]
        return out


class ZetarisGateway(DataGateway):
    """Zetaris Lightning over its REST SQL endpoint.

    SEAM - not yet exercised against a live instance. Config (env vars):
      ZETARIS_REST_URL   default http://localhost:8888/api/v1.0
      ZETARIS_API_KEY    bearer token (required)
      ZETARIS_ORG_ID     default 1
      ZETARIS_TABLE_MAP  JSON {"orders": "shop_sqlite.main.orders", ...}
                         or a path to a JSON file with that mapping.
    Call shape mirrors zetaris-getting-started/scripts/zetaris_sql.py
    (POST /sql-editor/sqls/run with {queryId, sql, source, limit}).
    """

    def __init__(self, table_map: dict[str, str] | None = None, limit: int = 10000):
        mapping = table_map or self._env_table_map()
        super().__init__(name="zetaris", table_map=mapping)
        self.base = os.environ.get("ZETARIS_REST_URL", "http://localhost:8888/api/v1.0").rstrip("/")
        self.org = os.environ.get("ZETARIS_ORG_ID", "1")
        self._key = os.environ.get("ZETARIS_API_KEY", "")
        self.limit = limit
        if not self._key:
            raise GatewayError("ZETARIS_API_KEY is not set")

    @staticmethod
    def _env_table_map() -> dict[str, str]:
        raw = os.environ.get("ZETARIS_TABLE_MAP", "")
        if not raw:
            raise GatewayError("ZETARIS_TABLE_MAP is not set (logical table -> Zetaris table)")
        if raw.strip().startswith("{"):
            return json.loads(raw)
        return json.loads(Path(raw).read_text())

    def days_between(self, a: str, b: str) -> str:
        # Zetaris Lightning speaks Spark SQL.
        return f"datediff(CAST({b} AS DATE), CAST({a} AS DATE))"

    def _call(self, path: str, body: dict | None = None) -> Any:
        req = Request(
            self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            method="POST" if body is not None else "GET",
            headers={
                "Authorization": f"Bearer {self._key}",
                "X-Org-ID": self.org,
                "X-Request-ID": str(uuid.uuid4()),
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode() or "null")
        except HTTPError as e:
            raise GatewayError(f"Zetaris HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e
        except (URLError, OSError) as e:
            raise GatewayError(f"cannot reach Zetaris at {self.base}: {getattr(e, 'reason', e)}") from e

    def run_sql(self, sql: str) -> QueryResult:
        phys = self.resolve(sql)
        t0 = time.perf_counter()
        out = self._call(
            "/sql-editor/sqls/run",
            {"queryId": str(uuid.uuid4()), "sql": phys, "source": "SqlEditor", "limit": self.limit},
        )
        cols, rows = _parse_zetaris_result(out)
        return QueryResult(cols, rows, phys, (time.perf_counter() - t0) * 1000)

    def list_schema(self) -> dict[str, list[tuple[str, str]]]:
        out = {}
        for logical, phys in self.table_map.items():
            res = self._call(
                "/sql-editor/sqls/run",
                {"queryId": str(uuid.uuid4()), "sql": f"DESCRIBE {phys}", "source": "SqlEditor", "limit": 500},
            )
            _, rows = _parse_zetaris_result(res)
            out[logical] = [(str(r[0]), str(r[1]) if len(r) > 1 else "?") for r in rows]
        return out


def _parse_zetaris_result(out: Any) -> tuple[list[str], list[tuple]]:
    """Accept the shapes we have seen documented: {data: [[...]]} with optional
    column metadata, or {data: [{col: val}]}. Zetaris returns values as strings,
    so numeric-looking strings are coerced. TODO: confirm on a live instance."""
    if not isinstance(out, dict):
        return [], []
    data = out.get("data") or []
    meta = out.get("columns") or out.get("schema") or out.get("header") or []
    cols = [c.get("name") if isinstance(c, dict) else str(c) for c in meta]
    if data and isinstance(data[0], dict):
        cols = cols or list(data[0])
        return cols, [tuple(_coerce(r.get(c)) for c in cols) for r in data]
    rows = [tuple(_coerce(v) for v in r) for r in data]
    if not cols and rows:
        cols = [f"c{i}" for i in range(len(rows[0]))]
    return cols, rows


_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$")


def _coerce(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s.lower() in ("null", "none", ""):
        return None
    if _INT_RE.match(s) and not (len(s) > 1 and s.startswith("0")):
        return int(s)
    if _FLOAT_RE.match(s):
        return float(s)
    return v


def make_gateway(kind: str, data_dir: str | Path = "data") -> DataGateway:
    if kind == "local":
        return LocalGateway(data_dir)
    if kind == "zetaris":
        return ZetarisGateway()
    raise ValueError(f"unknown gateway {kind!r}")
