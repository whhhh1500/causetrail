import pytest

from causetrail.evidence import Claim, EvidenceLedger, UncitedClaimError, UngroundedNumberError


def _ledger():
    led = EvidenceLedger()
    led.record(hypothesis_id="H0", purpose="p", sql="SELECT 1", physical_sql="SELECT 1", gateway="test",
               columns=["x"], rows=[(1,)], finding="APAC refund rate was 8.3% vs 2.8% (+189.8%).",
               metrics={"ratio": 0.67, "late_orders": 147})
    return led


def test_uncited_claim_is_refused():
    with pytest.raises(UncitedClaimError):
        _ledger().check_claim(Claim("Refunds rose because of the supplier.", []))


def test_claim_citing_unknown_evidence_is_refused():
    with pytest.raises(UncitedClaimError):
        _ledger().check_claim(Claim("Refunds rose.", ["E99"]))


def test_invented_number_is_refused():
    with pytest.raises(UngroundedNumberError):
        _ledger().check_claim(Claim("Refunds rose 63%.", ["E01"]))


def test_grounded_claims_pass_including_ratio_as_percent():
    led = _ledger()
    led.check_claim(Claim("Refund rate went from 2.8% to 8.3%, +189.8%.", ["E01"]))
    led.check_claim(Claim("Campaign orders refunded at 67% of the others' rate across 147 late orders.", ["E01"]))


def test_identifiers_are_not_numbers():
    # W40, SUP-07 and dates are identifiers, not quantities to be grounded
    _ledger().check_claim(Claim("In W40 supplier SUP-07 slipped.", ["E01"]))


def test_evidence_entry_fields():
    e = _ledger()["E01"]
    assert e.id == "E01" and e.sql and e.gateway == "test" and e.row_sample == [[1]]
    assert e.created_at and e.sql_sha and e.finding


def test_renderer_refuses_uncited_claim(gw):
    from causetrail.agent import Agent
    from causetrail.report import render

    inv = Agent(gw).investigate("why did refunds spike in week 40 in APAC?")
    inv.answer.append(Claim("An unsourced sentence sneaks in.", []))
    with pytest.raises(UncitedClaimError):
        render(inv)
