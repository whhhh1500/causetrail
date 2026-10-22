"""Markdown report. Every printed claim passes the ledger's citation guard."""
from __future__ import annotations

from .agent import REFUTED, SUPPORTED, Investigation
from .evidence import Claim, UncitedClaimError


def _cite(inv: Investigation, c: Claim) -> str:
    inv.ledger.check_claim(c)  # raises UncitedClaimError: an unsourced sentence never prints
    return f"{c.text} " + " ".join(f"[{e}]" for e in c.evidence_ids)


def render(inv: Investigation, *, include_sql: bool = True) -> str:
    s = inv.scope
    L: list[str] = []
    L.append("# CauseTrail investigation")
    L.append(f"> **Question:** {inv.question}")
    if s:
        L.append(f"> **Scope:** {s.metric} · {s.region} · {s.label} ({s.t0} → {s.t1}) vs baseline {s.baseline_label} "
                 f"({s.b0} → {s.b1}) · data as of {s.as_of}")
    L.append(f"> **Gateway:** `{inv.gateway}` · **Model:** `{inv.llm}` · **Steps:** {inv.steps_used}/{inv.max_steps} "
             f"· **Evidence entries:** {len(inv.ledger)}")
    L.append("")
    L.append("## Answer")
    for c in inv.answer:
        L.append(f"- {_cite(inv, c)}")
    L.append("")
    L.append(f"**Confidence:** {inv.confidence_label} ({inv.confidence:.2f})")
    for w in inv.confidence_reasons:
        L.append(f"  - {w}")
    if inv.summary:
        L.append("")
        L.append("## Summary (model-written, every sentence checked against the ledger)")
        L.append(" ".join(_cite(inv, c) for c in inv.summary))
    L.append("")
    L.append("## Hypotheses")
    L.append("| id | hypothesis | status | evidence |")
    L.append("|---|---|---|---|")
    icon = {SUPPORTED: "✅ supported", REFUTED: "❌ ruled out", "inconclusive": "❔ inconclusive",
            "not_tested": "⏭ not tested"}
    for r in inv.results:
        src = " *(LLM-proposed)*" if r.source == "llm" else ""
        L.append(f"| {r.id} | {r.title}{src} | {icon.get(r.status, r.status)} | "
                 f"{', '.join(r.evidence_ids) or '-'} |")
    ruled = [r for r in inv.results if r.status == REFUTED]
    if ruled:
        L.append("")
        L.append("## Ruled out (and the evidence that killed them)")
        for r in ruled:
            killer = [e for e in r.evidence_ids if inv.ledger[e].verdict == "refutes"] or r.evidence_ids
            tag = " *(LLM-judged, advisory)*" if r.source == "llm" else ""
            L.append(f"- **{r.id} – {r.title}.**{tag} " + " ".join(
                _cite(inv, Claim(inv.ledger[e].finding, [e], "refutation")) for e in killer))
    L.append("")
    L.append("## What I could not check")
    for x in inv.limits:
        L.append(f"- {x}")
    L.append("")
    L.append("## Evidence ledger")
    for e in inv.ledger.all():
        L.append(f"### {e.id} · {e.hypothesis_id} · {e.verdict}")
        L.append(f"*{e.purpose}* — `{e.gateway}`, {e.row_count} rows, {e.elapsed_ms} ms, sha `{e.sql_sha}`")
        L.append("")
        L.append(f"**Finding:** {e.finding}")
        if include_sql:
            L.append("```sql")
            L.append(e.sql)
            L.append("```")
        if e.columns:
            L.append("| " + " | ".join(e.columns) + " |")
            L.append("|" + "---|" * len(e.columns))
            for row in e.row_sample:
                L.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
            if e.row_count > len(e.row_sample):
                L.append(f"*… {e.row_count - len(e.row_sample)} more rows*")
        L.append("")
    if inv.rejected:
        L.append("## Appendix: sentences the citation guard refused")
        for x in inv.rejected:
            L.append(f"- {x}")
    return "\n".join(L) + "\n"


__all__ = ["render", "UncitedClaimError"]
