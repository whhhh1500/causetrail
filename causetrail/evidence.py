"""The evidence ledger.

Every query the agent runs becomes an :class:`Evidence` entry: id, the logical
SQL, the physical SQL the gateway executed, which gateway, a sample of the rows
that came back, a timestamp, and the finding derived from those rows.

Every sentence the agent wants to say becomes a :class:`Claim` that must cite
one or more evidence ids. :meth:`EvidenceLedger.check_claim` enforces two rules:

1. **No uncited claims** - a claim with no evidence id, or with an id that is
   not in the ledger, raises :class:`UncitedClaimError`.
2. **No invented numbers** - every number in the claim text must appear in the
   finding/metrics of the evidence it cites (small rounding tolerance). This is
   what stops an LLM from writing a plausible but unsupported "refunds up 63%".

The report renderer calls ``check_claim`` on every claim before printing it,
so an unsourced sentence cannot reach the output.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any

SAMPLE_ROWS = 8
_NUM_RE = re.compile(r"(?<![A-Za-z0-9_-])-?\d+(?:,\d{3})*(?:\.\d+)?")


class UncitedClaimError(ValueError):
    """Raised when a claim has no (valid) evidence behind it."""


class UngroundedNumberError(UncitedClaimError):
    """Raised when a claim states a number its cited evidence does not contain."""


def _jsonable(v: Any) -> Any:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, float):
        return round(v, 6)
    if hasattr(v, "item"):  # numpy / decimal-ish scalars
        try:
            return v.item()
        except Exception:
            pass
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return str(v)


@dataclass
class Evidence:
    id: str
    hypothesis_id: str
    purpose: str
    sql: str
    physical_sql: str
    gateway: str
    columns: list[str]
    row_count: int
    row_sample: list[list[Any]]
    finding: str
    metrics: dict[str, Any] = field(default_factory=dict)
    verdict: str = "context"  # supports | refutes | inconclusive | context | error
    created_at: str = ""
    elapsed_ms: float = 0.0
    sql_sha: str = ""

    def numbers(self) -> list[float]:
        nums = [float(n.replace(",", "")) for n in _NUM_RE.findall(self.finding)]
        for v in self.metrics.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                nums.append(float(v))
        return nums


@dataclass
class Claim:
    text: str
    evidence_ids: list[str]
    kind: str = "finding"  # finding | answer | refutation | limit


class EvidenceLedger:
    def __init__(self) -> None:
        self._entries: dict[str, Evidence] = {}

    # -- recording -------------------------------------------------------------
    def record(
        self,
        *,
        hypothesis_id: str,
        purpose: str,
        sql: str,
        physical_sql: str,
        gateway: str,
        columns: list[str],
        rows: list[tuple],
        finding: str,
        metrics: dict[str, Any] | None = None,
        verdict: str = "context",
        elapsed_ms: float = 0.0,
    ) -> Evidence:
        eid = f"E{len(self._entries) + 1:02d}"
        ev = Evidence(
            id=eid,
            hypothesis_id=hypothesis_id,
            purpose=purpose,
            sql=sql.strip(),
            physical_sql=physical_sql.strip(),
            gateway=gateway,
            columns=list(columns),
            row_count=len(rows),
            row_sample=[[_jsonable(v) for v in r] for r in rows[:SAMPLE_ROWS]],
            finding=finding,
            metrics={k: _jsonable(v) for k, v in (metrics or {}).items()},
            verdict=verdict,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            elapsed_ms=round(elapsed_ms, 1),
            sql_sha=hashlib.sha256(physical_sql.encode()).hexdigest()[:12],
        )
        self._entries[eid] = ev
        return ev

    # -- access ----------------------------------------------------------------
    def __getitem__(self, eid: str) -> Evidence:
        return self._entries[eid]

    def __contains__(self, eid: str) -> bool:
        return eid in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def all(self) -> list[Evidence]:
        return list(self._entries.values())

    def for_hypothesis(self, hid: str) -> list[Evidence]:
        return [e for e in self._entries.values() if e.hypothesis_id == hid]

    # -- the rule --------------------------------------------------------------
    def check_claim(self, claim: Claim, *, check_numbers: bool = True) -> Claim:
        if not claim.evidence_ids:
            raise UncitedClaimError(f"uncited claim refused: {claim.text!r}")
        missing = [e for e in claim.evidence_ids if e not in self._entries]
        if missing:
            raise UncitedClaimError(f"claim cites unknown evidence {missing}: {claim.text!r}")
        if check_numbers:
            allowed: list[float] = []
            for eid in claim.evidence_ids:
                allowed.extend(self._entries[eid].numbers())
            for raw in _NUM_RE.findall(claim.text):
                n = float(raw.replace(",", ""))
                if not any(_close(n, a) for a in allowed):
                    raise UngroundedNumberError(
                        f"number {raw} in claim is not in cited evidence {claim.evidence_ids}: {claim.text!r}"
                    )
        return claim

    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in self._entries.values()], indent=2)


def _close(n: float, a: float) -> bool:
    if abs(n - a) <= 0.051:
        return True
    if a != 0 and abs(n - a) / abs(a) <= 0.01:
        return True
    # allow a ratio stated as a percentage and vice versa (0.083 <-> 8.3)
    return abs(n - a * 100) <= 0.051 or abs(n * 100 - a) <= 0.051
