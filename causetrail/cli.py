"""Command line: ``python -m causetrail <command>``.

  gen-data   write the seeded demo sources to ./data
  schema     show the logical tables the agent can query (and where they live)
  sql        run one logical SQL statement through a gateway
  ask        investigate a question and write report.md + evidence.json
  llm-check  verify the configured model endpoint answers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _gateway(args):
    from .sources import make_gateway

    if args.gateway == "local" and not (Path(args.data) / "orders.db").exists():
        from .datagen import generate

        print(f"(no demo data in {args.data}; generating it with seed 7)", file=sys.stderr)
        generate(args.data, seed=7)
    return make_gateway(args.gateway, args.data)


def cmd_gen(args) -> int:
    from .datagen import generate

    p = generate(args.data, seed=args.seed)
    for f in sorted(p.root.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size:>9,d} bytes")
    return 0


def cmd_schema(args) -> int:
    print(_gateway(args).describe())
    return 0


def cmd_sql(args) -> int:
    res = _gateway(args).run_sql(args.query)
    print(" | ".join(res.columns))
    for r in res.rows[: args.limit]:
        print(" | ".join("" if v is None else str(v) for v in r))
    if len(res.rows) > args.limit:
        print(f"... {len(res.rows) - args.limit} more rows")
    return 0


def cmd_ask(args) -> int:
    from .agent import Agent
    from .llm import make_llm
    from .report import render
    from .scope import UnsupportedQuestion

    gw = _gateway(args)
    llm = make_llm(args.llm)
    print(f"CauseTrail {__version__} · gateway={gw.name} · model={llm.label()}", file=sys.stderr)
    try:
        inv = Agent(gw, llm, max_steps=args.max_steps, verbose=not args.quiet).investigate(args.question)
    except UnsupportedQuestion as e:
        print(f"can't investigate that yet: {e}", file=sys.stderr)
        return 2
    md = render(inv, include_sql=not args.no_sql)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(md)
    (out / "evidence.json").write_text(inv.ledger.to_json())
    (out / "trace.txt").write_text("\n".join(inv.trace) + "\n")
    if args.full:
        print(md)
    else:
        print(md.split("## Evidence ledger")[0])
    print(f"wrote {out/'report.md'}, {out/'evidence.json'}, {out/'trace.txt'}", file=sys.stderr)
    return 0


def cmd_llm_check(args) -> int:
    from .llm import MeterlessClient, make_llm

    llm = make_llm(args.llm)
    if isinstance(llm, MeterlessClient):
        print("models:", llm.list_models())
        print("using:", llm.resolve_model())
    print(llm.label(), "->", llm.chat([{"role": "user", "content": "Reply with the single word: ready"}])[:200])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="causetrail", description="Evidence-first root-cause agent for business metrics")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, data=True):
        p.add_argument("--gateway", choices=["local", "zetaris"], default="local")
        p.add_argument("--data", default="data", help="demo data dir (local gateway)")

    p = sub.add_parser("gen-data", help="generate the seeded demo sources")
    p.add_argument("--data", default="data")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_gen)

    p = sub.add_parser("schema", help="list logical tables")
    common(p)
    p.set_defaults(fn=cmd_schema)

    p = sub.add_parser("sql", help="run one logical SQL query")
    common(p)
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(fn=cmd_sql)

    p = sub.add_parser("ask", help="investigate a metric question")
    common(p)
    p.add_argument("question")
    p.add_argument("--llm", choices=["auto", "meterless", "nvidia", "cc1500", "none"], default="auto")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--out", default="out")
    p.add_argument("--full", action="store_true", help="print the full report incl. evidence ledger")
    p.add_argument("--no-sql", action="store_true", help="omit SQL from the report")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("llm-check", help="ping the configured model")
    p.add_argument("--llm", choices=["auto", "meterless", "nvidia", "cc1500"], default="auto")
    p.set_defaults(fn=cmd_llm_check)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
