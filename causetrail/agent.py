"""The investigation loop.

question -> scope -> confirm the metric moved -> plan hypotheses ->
for each hypothesis: render SQL -> run through the gateway -> record evidence ->
judge (supports / refutes / inconclusive) -> unlock follow-ups -> ... (bounded)
-> pick the deepest supported causal chain -> cited claims -> report.

The LLM is used where language helps (reading the question, prioritising and
proposing extra hypotheses, judging ad-hoc queries, writing the summary). Every
LLM output is validated against the data; whatever fails validation is
dropped and logged, never printed as fact.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from .evidence import Claim, Evidence, EvidenceLedger, UncitedClaimError, _NUM_RE, _close
from .hypotheses import (CONTEXT, INCONCLUSIVE, LIBRARY, METRIC_MOVED, REFUTES, ROOTS, SUPPORTS,
                         Hypothesis, Outcome, Probe)
from .llm import LLMClient, LLMUnavailable, NullLLM
from .scope import Scope, UnsupportedQuestion, build_scope, parse_question_rules
from .sources import DataGateway, GatewayError

SUPPORTED, REFUTED = "supported", "refuted"


@dataclass
class HypothesisResult:
    id: str
    title: str
    mechanism: str
    status: str  # supported | refuted | inconclusive | not_tested
    evidence_ids: list[str] = field(default_factory=list)
    bindings: dict = field(default_factory=dict)
    layer: int = 0
    source: str = "library"  # library | llm
    note: str = ""


@dataclass
class Investigation:
    question: str
    scope: Scope | None
    ledger: EvidenceLedger
    results: list[HypothesisResult]
    answer: list[Claim]
    summary: list[Claim]
    confidence: float
    confidence_label: str
    confidence_reasons: list[str]
    limits: list[str]
    rejected: list[str]  # LLM outputs the guards refused, with the reason
    trace: list[str]
    gateway: str
    llm: str
    steps_used: int
    max_steps: int

    def by_status(self, status: str) -> list[HypothesisResult]:
        return [r for r in self.results if r.status == status]

    def root(self) -> HypothesisResult | None:
        sup = [r for r in self.results if r.status == SUPPORTED and r.layer > 0]
        return max(sup, key=lambda r: r.layer) if sup else None


SELECT_ONLY = re.compile(r"^\s*(select|with)\b", re.I)
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|copy|pragma|install|load)\b", re.I)


class Agent:
    def __init__(self, gateway: DataGateway, llm: LLMClient | None = None, *, max_steps: int = 20,
                 max_llm_hypotheses: int = 2, verbose: bool = False):
        self.gw = gateway
        self.llm = llm or NullLLM()
        self.max_steps = max_steps
        self.max_llm_hypotheses = max_llm_hypotheses
        self.verbose = verbose

    # ------------------------------------------------------------------ helpers
    def _log(self, trace: list[str], msg: str) -> None:
        trace.append(msg)
        if self.verbose:
            print(f"  · {msg}", flush=True)

    def _data_bounds(self) -> tuple[date, date]:
        r = self.gw.run_sql("SELECT MIN(order_date) AS lo, MAX(order_date) AS hi FROM {orders}").rows[0]
        lo, hi = (v if isinstance(v, date) else date.fromisoformat(str(v)[:10]) for v in r)
        return lo, hi

    def _llm_json(self, task: str, system: str, user: str, trace: list[str]):
        try:
            return self.llm.ask_json(task, system, user)
        except (LLMUnavailable, ValueError) as e:
            self._log(trace, f"LLM {task} unavailable -> deterministic fallback ({str(e)[:80]})")
            return None

    # ------------------------------------------------------------------ scope
    def scope_for(self, question: str, trace: list[str]) -> Scope:
        lo, hi = self._data_bounds()
        rules = None
        try:
            rules = parse_question_rules(question, hi)
        except UnsupportedQuestion as e:
            rules_err = e
        got = self._llm_json(
            "parse_scope",
            "Extract the metric investigation scope. metric must be 'refund_rate'; region one of APAC, NA, EU, "
            "LATAM; year and ISO week as integers. Keys: metric, region, year, week.",
            f"Question: {question}\nData covers {lo} to {hi}.", trace)
        parsed = rules
        if isinstance(got, dict):
            try:
                cand = {"metric": str(got["metric"]), "region": str(got["region"]).upper(),
                        "year": int(got["year"]), "week": int(got["week"])}
                if rules and cand != rules:
                    # The rule parser read explicit tokens; trust it over the model on conflict.
                    self._log(trace, f"LLM scope {cand} disagrees with rule parse {rules}; using rule parse")
                else:
                    parsed = cand
            except (KeyError, TypeError, ValueError):
                self._log(trace, "LLM scope reply malformed; using rule parse")
        if parsed is None:
            raise rules_err  # noqa: F821 - set when rules is None
        sc = build_scope(question, parsed["metric"], parsed["region"], parsed["year"], parsed["week"], lo, hi)
        self._log(trace, f"scope: {sc.metric} in {sc.region} {sc.label} vs baseline {sc.baseline_label}")
        return sc

    # ------------------------------------------------------------------ probes
    def _run_probe(self, h: Hypothesis, p: Probe, scope: Scope, bindings: dict, ledger: EvidenceLedger,
                   trace: list[str]) -> tuple[Evidence, Outcome]:
        sql = p.render(scope, bindings)
        purpose = p.purpose.format(region=scope.region, **bindings)
        try:
            res = self.gw.run_sql(sql)
            out = p.judge(res.dicts(), scope, bindings)
            ev = ledger.record(hypothesis_id=h.id, purpose=purpose, sql=sql, physical_sql=res.sql,
                               gateway=self.gw.name, columns=res.columns, rows=res.rows, finding=out.finding,
                               metrics=out.metrics, verdict=out.verdict, elapsed_ms=res.elapsed_ms)
        except GatewayError as e:
            out = Outcome("error", f"Query failed: {e}")
            ev = ledger.record(hypothesis_id=h.id, purpose=purpose, sql=sql, physical_sql=sql, gateway=self.gw.name,
                               columns=[], rows=[], finding=out.finding, verdict="error")
        self._log(trace, f"{ev.id} [{h.id}] {out.verdict}: {out.finding[:110]}")
        return ev, out

    def _test(self, h: Hypothesis, scope, bindings, ledger, trace, budget: list[int]) -> HypothesisResult:
        res = HypothesisResult(h.id, h.title_for(scope, bindings), h.mechanism, INCONCLUSIVE,
                               bindings=dict(bindings), layer=h.layer)
        verdicts = []
        for p in h.probes:
            if budget[0] <= 0:
                res.note = "step budget exhausted mid-hypothesis"
                break
            budget[0] -= 1
            ev, out = self._run_probe(h, p, scope, res.bindings, ledger, trace)
            res.evidence_ids.append(ev.id)
            res.bindings.update(out.bindings)
            verdicts.append(out.verdict)
            if out.verdict == REFUTES and h.id != "H0":
                break  # one clean refutation is enough; don't spend budget
        decisive = [v for v in verdicts if v != CONTEXT]
        need = h.min_support if h.min_support is not None else len([p for p in h.probes]) - verdicts.count(CONTEXT)
        if REFUTES in decisive and h.id != "H0":
            res.status = REFUTED
        elif decisive.count(SUPPORTS) >= max(need, 1) and "error" not in decisive:
            res.status = SUPPORTED
        else:
            res.status = INCONCLUSIVE
        return res

    # ------------------------------------------------------------------ LLM extras
    def _plan(self, scope: Scope, h0: Evidence, trace) -> tuple[list[str], list[dict]]:
        catalog = "\n".join(f"{hid}: {LIBRARY[hid].title} ({LIBRARY[hid].mechanism})" for hid in ROOTS)
        covered = "\n".join(f"- {h.title} ({h.mechanism})" for h in LIBRARY.values() if h.id not in ROOTS)
        got = self._llm_json(
            "plan",
            "You are planning a root-cause investigation for a business metric. Order the candidate hypotheses "
            "from most to least likely and optionally propose up to 2 extra hypotheses not in the list, each "
            "with ONE read-only SQL query over the logical tables (write table names in braces, e.g. {orders}) "
            "and a falsification criterion. Keys: order (list of ids), extra (list of {title, mechanism, sql, "
            "refute_if}).",
            f"Question: {scope.question}\nScope: {scope.region} {scope.label}, baseline {scope.baseline_label} "
            f"({scope.b0}..{scope.b1}), target {scope.t0}..{scope.t1}.\nFinding so far [{h0.id}]: {h0.finding}\n"
            f"Candidates:\n{catalog}\nAlready covered as automatic drill-downs (do NOT propose these again):\n"
            f"{covered}\nSchema:\n{self.gw.describe()}", trace)
        order = list(ROOTS)
        extras: list[dict] = []
        if isinstance(got, dict):
            ids = [i for i in got.get("order", []) if i in ROOTS]
            order = list(dict.fromkeys(ids + list(ROOTS)))  # never drop a candidate: we falsify all of them
            for x in (got.get("extra") or [])[: self.max_llm_hypotheses]:
                if isinstance(x, dict) and x.get("sql") and x.get("title"):
                    extras.append(x)
            self._log(trace, f"plan from LLM: order={order}, extra={len(extras)}")
        return order, extras

    def _test_llm_hypothesis(self, n: int, x: dict, scope, ledger, trace, budget) -> HypothesisResult:
        hid = f"X{n}"
        res = HypothesisResult(hid, str(x["title"])[:160], str(x.get("mechanism", ""))[:200], INCONCLUSIVE,
                               source="llm")
        sql = str(x["sql"]).strip().rstrip(";")
        if not SELECT_ONLY.match(sql) or FORBIDDEN.search(sql) or ";" in sql:
            res.note = "rejected: not a single read-only SELECT"
            self._log(trace, f"{hid} {res.note}")
            return res
        budget[0] -= 1
        try:
            qr = self.gw.run_sql(sql)
        except GatewayError as e:
            ev = ledger.record(hypothesis_id=hid, purpose=res.title, sql=sql, physical_sql=sql, gateway=self.gw.name,
                               columns=[], rows=[], finding=f"Query failed: {str(e)[:200]}", verdict="error")
            res.evidence_ids.append(ev.id)
            res.note = "LLM SQL failed"
            self._log(trace, f"{ev.id} [{hid}] error: {str(e)[:100]}")
            return res
        rows_txt = json.dumps([list(map(str, r)) for r in qr.rows[:30]])
        got = self._llm_json(
            "judge",
            "Judge a hypothesis from query results. verdict is one of supports, refutes, inconclusive. finding is "
            "one sentence and may only use numbers that appear in the rows. Keys: verdict, finding.",
            f"Hypothesis: {res.title}\nRefute if: {x.get('refute_if', '')}\nColumns: {qr.columns}\nRows: {rows_txt}",
            trace)
        verdict, finding = INCONCLUSIVE, "No judgement available."
        if isinstance(got, dict) and got.get("verdict") in (SUPPORTS, REFUTES, INCONCLUSIVE):
            verdict, finding = got["verdict"], str(got.get("finding", ""))[:400]
            row_nums = [float(v) for r in qr.rows for v in r if isinstance(v, (int, float)) and not isinstance(v, bool)]
            bad = [m for m in _NUM_RE.findall(finding) if not any(_close(float(m.replace(",", "")), a) for a in row_nums)]
            if bad:
                verdict, finding = INCONCLUSIVE, f"LLM judgement discarded (numbers {bad} not in the result rows)."
        ev = ledger.record(hypothesis_id=hid, purpose=res.title, sql=sql, physical_sql=qr.sql, gateway=self.gw.name,
                           columns=qr.columns, rows=qr.rows, finding=finding, verdict=verdict,
                           elapsed_ms=qr.elapsed_ms)
        res.evidence_ids.append(ev.id)
        res.status = {SUPPORTS: SUPPORTED, REFUTES: REFUTED}.get(verdict, INCONCLUSIVE)
        self._log(trace, f"{ev.id} [{hid}] {verdict}: {finding[:100]}")
        return res

    def _summarise(self, ledger: EvidenceLedger, results, trace, rejected) -> list[Claim]:
        if isinstance(self.llm, NullLLM):
            return []
        ev_txt = "\n".join(f"[{e.id}] ({e.hypothesis_id}, {e.verdict}) {e.finding}" for e in ledger.all())
        res_txt = "\n".join(f"{r.id} {r.status}: {r.title}" for r in results)
        got = self._llm_json(
            "narrate",
            "Write a 3-5 sentence executive summary for an operations analyst. Every sentence must cite the "
            "evidence ids it relies on and may only use numbers that appear in those evidence findings. "
            "Keys: summary (list of {text, evidence_ids}).",
            f"Hypothesis results:\n{res_txt}\nEvidence ledger:\n{ev_txt}", trace)
        out: list[Claim] = []
        if not isinstance(got, dict):
            return out
        for item in (got.get("summary") or [])[:6]:
            if not isinstance(item, dict):
                continue
            c = Claim(str(item.get("text", "")).strip(), [str(i) for i in item.get("evidence_ids") or []], "summary")
            try:
                out.append(ledger.check_claim(c))
            except UncitedClaimError as e:
                rejected.append(str(e))
                self._log(trace, f"guard refused an LLM sentence: {str(e)[:100]}")
        return out

    # ------------------------------------------------------------------ main loop
    def investigate(self, question: str) -> Investigation:
        trace: list[str] = []
        ledger = EvidenceLedger()
        rejected: list[str] = []
        scope = self.scope_for(question, trace)
        budget = [self.max_steps]
        results: list[HypothesisResult] = []

        h0 = self._test(METRIC_MOVED, scope, {}, ledger, trace, budget)
        results.append(h0)
        trend_ev = ledger[h0.evidence_ids[0]]
        if trend_ev.verdict != SUPPORTS:
            self._log(trace, "metric did not move beyond normal variation -> stop")
            return self._finish(question, scope, ledger, results, [], rejected, trace, budget)

        order, extras = self._plan(scope, trend_ev, trace)
        queue: list[tuple[str, dict]] = [(hid, {}) for hid in order]
        tested: set[str] = set()
        while queue and budget[0] > 0:
            hid, binds = queue.pop(0)
            if hid in tested:
                continue
            h = LIBRARY[hid]
            if any(k not in binds for k in h.requires):
                continue
            tested.add(hid)
            r = self._test(h, scope, binds, ledger, trace, budget)
            results.append(r)
            if r.status == SUPPORTED:
                for child in h.unlocks:
                    queue.insert(0, (child, dict(r.bindings)))  # drill down immediately
                    self._log(trace, f"{hid} supported -> drill into {child} with {r.bindings}")
        for n, x in enumerate(extras, 1):
            if budget[0] <= 0:
                break
            results.append(self._test_llm_hypothesis(n, x, scope, ledger, trace, budget))
        for hid, _ in queue:
            if hid not in tested and hid in ROOTS:
                results.append(HypothesisResult(hid, LIBRARY[hid].title, LIBRARY[hid].mechanism, "not_tested",
                                                note="step budget exhausted"))
        return self._finish(question, scope, ledger, results, [], rejected, trace, budget)

    # ------------------------------------------------------------------ conclusions
    def _finish(self, question, scope, ledger, results, _unused, rejected, trace, budget) -> Investigation:
        inv = Investigation(question, scope, ledger, results, [], [], 0.0, "low", [], [], rejected, trace,
                            self.gw.name, self.llm.label(), self.max_steps - budget[0], self.max_steps)
        root = inv.root()
        h0 = results[0]
        trend = ledger[h0.evidence_ids[0]]
        if trend.verdict != SUPPORTS:
            inv.answer = [Claim(f"No investigation needed: {trend.finding}", [trend.id], "answer")]
            inv.confidence, inv.confidence_label = 0.8, "high"
            inv.confidence_reasons = ["the metric change is within the 25% threshold used for 'moved'"]
        elif root is None:
            inv.answer = [Claim("No tested mechanism was supported by the evidence; see ruled-out and "
                                "inconclusive hypotheses below.", [trend.id], "answer")]
            inv.confidence, inv.confidence_label = 0.2, "low"
            inv.confidence_reasons = ["no hypothesis survived its falsification tests"]
        else:
            chain = sorted([r for r in results if r.status == SUPPORTED and r.layer > 0], key=lambda r: -r.layer)
            inv.answer.append(Claim(f"Most likely root cause: {root.title}.", root.evidence_ids, "answer"))
            for r in chain:
                for eid in r.evidence_ids:
                    if ledger[eid].verdict == SUPPORTS:
                        inv.answer.append(Claim(ledger[eid].finding, [eid], "finding"))
            inv.answer.append(Claim(trend.finding, [trend.id], "finding"))
            inv.confidence, inv.confidence_label, inv.confidence_reasons = self._confidence(inv, chain)
        for c in inv.answer:
            ledger.check_claim(c)  # our own sentences go through the same guard
        inv.summary = self._summarise(ledger, results, trace, rejected)
        inv.limits = self._limits(inv)
        return inv

    def _confidence(self, inv: Investigation, chain) -> tuple[float, str, list[str]]:
        score, why = 0.35, []
        layers = {r.layer for r in chain}
        for lyr, label in ((1, "the excess is concentrated in one segment"),
                           (2, "a mechanism links that segment to refunds"),
                           (3, "an upstream cause precedes the mechanism and passes a control check")):
            if lyr in layers:
                score += 0.15
                why.append(f"+0.15 {label}")
        # Only deterministic (library) refutations count; LLM-judged ones are advisory.
        refuted = [r for r in inv.results if r.status == REFUTED and r.source == "library"]
        bonus = min(0.15, 0.05 * len(refuted))
        if bonus:
            score += bonus
            why.append(f"+{bonus:.2f} {len(refuted)} competing hypotheses ruled out")
        errs = [e for e in inv.ledger.all() if e.verdict == "error"]
        if errs:
            score -= 0.1
            why.append(f"-0.10 {len(errs)} queries failed")
        if any(r.status == "not_tested" for r in inv.results):
            score -= 0.1
            why.append("-0.10 some candidates were not tested (step budget)")
        why.append("-0.05 observational data only; no experiment")
        score = round(max(0.05, min(0.95, score - 0.05)), 2)
        label = "high" if score >= 0.8 else "medium" if score >= 0.55 else "low"
        return score, label, why

    def _limits(self, inv: Investigation) -> list[str]:
        out = [
            "No carrier/tracking data is connected, so warehouse delay and carrier delay cannot be separated.",
            "Refund reason codes are coarse; most extra refunds are a generic 'customer_request', so motive is "
            "inferred from support tickets, not stated by customers in the refund record.",
            "Observational data only: the causal claim rests on timing (supplier slip precedes delays), "
            "dose-response (late orders refund more) and a control (other suppliers of the same products). "
            "No experiment was run.",
        ]
        if inv.scope:
            out.append(f"Data ends {inv.scope.as_of}; refunds for orders placed late in {inv.scope.label} are still "
                       "accruing, so the latest week is a lower bound.")
        for r in inv.results:
            if r.status == INCONCLUSIVE and r.id != "H0":
                out.append(f"{r.id} '{r.title}' stayed inconclusive" + (f" ({r.note})" if r.note else "") + ".")
            if r.status == "not_tested":
                out.append(f"{r.id} '{r.title}' was not tested: {r.note}.")
        if inv.rejected:
            out.append(f"{len(inv.rejected)} model-written sentence(s) were refused by the citation guard (see appendix).")
        return out
