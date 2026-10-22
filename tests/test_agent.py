import json

import pytest

from causetrail.agent import Agent
from causetrail.llm import NullLLM, ScriptedLLM, extract_json
from causetrail.report import render
from causetrail.scope import UnsupportedQuestion

Q = "why did refunds spike in week 40 in APAC?"


def scripted():
    return ScriptedLLM({
        "parse_scope": {"metric": "refund_rate", "region": "APAC", "year": 2026, "week": 40},
        "plan": {
            "order": ["H1", "H4", "H2", "H3"],
            "extra": [
                {"title": "Someone dropped a table", "sql": "DROP TABLE {orders}", "refute_if": "-"},
                {"title": "Refunds concentrate in high-price SKUs", "mechanism": "price",
                 "sql": "SELECT i.sku, COUNT(*) AS n FROM {refunds} r JOIN {order_items} i "
                        "ON i.order_id = r.order_id GROUP BY 1 ORDER BY 2 DESC",
                 "refute_if": "refunds spread evenly"},
            ],
        },
        "judge": {"verdict": "inconclusive", "finding": "Refunds are spread over 14 SKUs, top SKU 9999 refunds."},
        "narrate": {"summary": [
            {"text": "APAC refunds jumped +189.8% in W40.", "evidence_ids": ["E01"]},
            {"text": "This was caused by a 500% rise in damaged items.", "evidence_ids": ["E01"]},
            {"text": "Supplier SUP-07 is to blame.", "evidence_ids": []},
        ]},
    })


def test_end_to_end_reaches_planted_root_cause(gw):
    llm = scripted()
    inv = Agent(gw, llm).investigate(Q)
    root = inv.root()
    assert root is not None and root.id == "H6"
    assert root.bindings["family"] == "Wearables"
    assert root.bindings["supplier_id"] == "SUP-07"
    status = {r.id: r.status for r in inv.results}
    assert status["H1"] == status["H2"] == status["H3"] == "refuted"  # red herrings falsified
    assert status["H4"] == status["H5"] == "supported"
    assert inv.confidence_label == "high"
    # LLM guards: destructive SQL never ran, ungrounded number discarded, bad summary lines refused
    x = {r.id: r for r in inv.results if r.source == "llm"}
    assert "read-only" in x["X1"].note and not x["X1"].evidence_ids
    assert "discarded" in inv.ledger[x["X2"].evidence_ids[0]].finding
    assert [c.text for c in inv.summary] == ["APAC refunds jumped +189.8% in W40."]
    assert len(inv.rejected) == 2
    md = render(inv)
    assert "SUP-07" in md and "Ruled out" in md and "What I could not check" in md
    assert set(llm.seen) == {"parse_scope", "plan", "judge", "narrate"}


def test_deterministic_mode_without_llm(gw):
    inv1 = Agent(gw, NullLLM()).investigate(Q)
    inv2 = Agent(gw, NullLLM()).investigate(Q)
    assert inv1.root().id == "H6"
    strip = lambda inv: [(e.id, e.sql, e.finding) for e in inv.ledger.all()]  # noqa: E731
    assert strip(inv1) == strip(inv2)
    for e in inv1.ledger.all():  # every query really ran and returned rows
        assert e.verdict != "error" and e.row_count > 0


def test_every_answer_sentence_is_cited(gw):
    inv = Agent(gw).investigate(Q)
    for c in inv.answer + inv.summary:
        assert c.evidence_ids and all(e in inv.ledger for e in c.evidence_ids)


@pytest.mark.parametrize("steps", [0, 1, 3, 7])
def test_loop_terminates_within_budget(gw, steps):
    inv = Agent(gw, max_steps=steps).investigate(Q) if steps else None
    if steps == 0:
        return
    assert inv.steps_used <= steps
    assert len(inv.ledger) <= steps + 1  # +1 tolerance for an LLM extra; none here
    render(inv)


def test_loop_survives_garbage_llm(gw):
    junk = ScriptedLLM({k: "not json at all" for k in ("parse_scope", "plan", "judge", "narrate")})
    inv = Agent(gw, junk).investigate(Q)
    assert inv.root().id == "H6"


def test_metric_that_did_not_move(gw):
    inv = Agent(gw).investigate("did refunds spike in week 40 in NA?")
    assert inv.root() is None
    assert inv.answer[0].text.startswith("No investigation needed")
    render(inv)


def test_unsupported_question(gw):
    with pytest.raises(UnsupportedQuestion):
        Agent(gw).investigate("why is the sky blue?")


def test_last_week_resolves_to_latest_complete_week(gw):
    inv = Agent(gw).investigate("why did APAC refunds jump last week?")
    assert inv.scope.week == 40 and inv.root().id == "H6"


def test_extract_json_handles_fences():
    assert extract_json('sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json(json.dumps([1, 2])) == [1, 2]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_root_cause_is_robust_to_seed(tmp_path, seed):
    from causetrail.datagen import generate
    from causetrail.sources import LocalGateway

    generate(tmp_path, seed=seed)
    inv = Agent(LocalGateway(tmp_path)).investigate(Q)
    assert inv.root().id == "H6" and inv.root().bindings["supplier_id"] == "SUP-07"


def test_meterless_client_contract(monkeypatch):
    from causetrail import llm as L

    monkeypatch.setenv("METERLESS_RUNTIME_TOKEN", "tok")
    monkeypatch.delenv("METERLESS_MODEL", raising=False)
    c = L.MeterlessClient(base_url="127.0.0.1:8088/v1/chat/completions")
    assert c.base == "http://127.0.0.1:8088/v1"
    assert c.model.startswith("nvidia/nemotron")
    assert c.extra_headers["meterless-app"] == "causetrail"
    monkeypatch.setattr(c, "list_models", lambda: ["meterless/auto", "meterless/text.deep"])
    assert c.resolve_model() == "meterless/auto"
