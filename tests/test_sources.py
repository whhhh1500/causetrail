import hashlib

import pytest

from causetrail.datagen import generate
from causetrail.sources import LOGICAL_TABLES, GatewayError, LocalGateway, ZetarisGateway


def test_generation_is_reproducible(tmp_path, data_dir):
    generate(tmp_path, seed=7)
    for f in ("ad_spend.csv", "support_tickets.json"):
        a = hashlib.sha256((tmp_path / f).read_bytes()).hexdigest()
        b = hashlib.sha256((data_dir / f).read_bytes()).hexdigest()
        assert a == b, f


def test_all_logical_tables_queryable(gw):
    schema = gw.list_schema()
    assert set(schema) == set(LOGICAL_TABLES)
    for t in LOGICAL_TABLES:
        assert gw.run_sql(f"SELECT COUNT(*) FROM {{{t}}}").rows[0][0] > 0, t


def test_dataset_size(gw):
    n = gw.run_sql("SELECT COUNT(*) FROM {orders}").rows[0][0]
    assert 5000 < n < 20000
    weeks = gw.run_sql("SELECT COUNT(DISTINCT date_trunc('week', order_date)) FROM {orders}").rows[0][0]
    assert weeks == 12


def test_cross_source_join_sqlite_json_excel(gw):
    # helpdesk JSON references orders as "#<id>": joining needs a key transform
    sql = """
    SELECT s.family, COUNT(*) AS tickets
    FROM {support_tickets} t
    JOIN {orders} o ON o.order_id = CAST(REPLACE(t.order_ref, '#', '') AS BIGINT)
    JOIN {order_items} i ON i.order_id = o.order_id
    JOIN {skus} s ON s.sku = i.sku
    WHERE o.region = 'APAC' GROUP BY s.family ORDER BY tickets DESC
    """
    rows = gw.run_sql(sql).rows
    assert rows[0][0] == "Wearables"


def test_signal_is_not_visible_in_refunds_table_alone(gw):
    # The planted cause is upstream: refund reasons are mostly a generic code.
    rows = gw.run_sql("""
        SELECT reason_code, COUNT(*) FROM {refunds} r JOIN {orders} o ON o.order_id = r.order_id
        WHERE o.region='APAC' AND r.refunded_at >= DATE '2026-09-28' GROUP BY 1 ORDER BY 2 DESC""").rows
    assert rows[0][0] == "customer_request"
    assert "late" not in {r[0] for r in rows}


def test_days_between_macro_nests(gw):
    sql = "SELECT DAYS_BETWEEN(DATE '2026-01-01', COALESCE(NULL, DATE '2026-01-11')) AS d"
    assert gw.run_sql(sql).rows[0][0] == 10


def test_unknown_table_raises(gw):
    with pytest.raises(GatewayError):
        gw.run_sql("SELECT * FROM {nope}")


def test_bad_sql_raises_gateway_error(gw):
    with pytest.raises(GatewayError):
        gw.run_sql("SELECT no_such_col FROM {orders}")


def test_zetaris_gateway_resolves_without_network(monkeypatch):
    monkeypatch.setenv("ZETARIS_API_KEY", "test-key")
    z = ZetarisGateway(table_map={"orders": "shop_sqlite.main.orders"})
    sql = z.resolve("SELECT DAYS_BETWEEN(o.order_date, o.shipped_date) FROM {orders} o")
    assert "shop_sqlite.main.orders" in sql and "datediff(" in sql


def test_zetaris_requires_key(monkeypatch):
    monkeypatch.delenv("ZETARIS_API_KEY", raising=False)
    with pytest.raises(GatewayError):
        ZetarisGateway(table_map={"orders": "x"})


def test_missing_data_dir(tmp_path):
    with pytest.raises(GatewayError):
        LocalGateway(tmp_path / "empty")


def test_zetaris_result_parsing_coerces_strings():
    from causetrail.sources import _parse_zetaris_result

    cols, rows = _parse_zetaris_result({"columns": [{"name": "region"}, {"name": "n"}, {"name": "rate"}],
                                        "data": [["APAC", "327", "8.26"], ["NA", "0", "null"]]})
    assert cols == ["region", "n", "rate"]
    assert rows == [("APAC", 327, 8.26), ("NA", 0, None)]
