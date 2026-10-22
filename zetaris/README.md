# Zetaris wiring (work in progress)

CauseTrail's agent writes SQL against **logical** table names (`{orders}`, `{support_tickets}`, ...).
`ZetarisGateway` maps each logical name to a table in Zetaris Lightning and sends the query to the
REST SQL endpoint (`POST {ZETARIS_REST_URL}/sql-editor/sqls/run`), so the four fragmented sources are
joined *inside Zetaris* instead of being copied anywhere.

Plan:
1. Run `python -m causetrail gen-data` to produce the four sources in `./data`.
2. Register them as Zetaris datasources. Zetaris cannot read a file on the laptop: files must sit in
   storage it can reach (e.g. S3) or be loaded into a database it can reach. Plan: `orders.db` tables ->
   a Postgres datasource (`CREATE DATASOURCE ...`); CSV/JSON/Excel -> S3 (or a second Postgres schema),
   exposed under a Lightning database. Record the physical `<source>.<schema>.<table>` names.
3. Fill in `table_map.example.json` with those names and point `ZETARIS_TABLE_MAP` at it.
4. `python -m causetrail schema --gateway zetaris` then `python -m causetrail ask "..." --gateway zetaris`.

The `DAYS_BETWEEN(a, b)` macro is rendered as Spark-style `datediff(b, a)` for Zetaris.

**Status:** not yet run against a live instance (no instance provisioned yet). Response parsing in
`_parse_zetaris_result` accepts the shapes in the official getting-started scripts and must be confirmed.
