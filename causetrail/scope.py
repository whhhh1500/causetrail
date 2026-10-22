"""Turn a plain-language question into a concrete investigation scope."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta

REGIONS = ("APAC", "NA", "EU", "LATAM")
REGION_ALIASES = {
    "apac": "APAC", "asia": "APAC", "asia-pacific": "APAC", "asia pacific": "APAC",
    "na": "NA", "north america": "NA", "us": "NA", "usa": "NA",
    "eu": "EU", "europe": "EU", "emea": "EU",
    "latam": "LATAM", "latin america": "LATAM", "south america": "LATAM",
}
METRICS = {"refund_rate": ("refund", "refunds", "refund rate", "refunded")}


class UnsupportedQuestion(ValueError):
    pass


@dataclass
class Scope:
    question: str
    metric: str
    region: str
    year: int
    week: int
    t0: date  # target week start (Mon)
    t1: date  # target week end (exclusive)
    b0: date  # baseline start
    b1: date  # baseline end (exclusive)
    c0: date  # incident cohort start (2 weeks before target, catches lagging refunds)
    as_of: date
    nb: int  # baseline weeks

    @property
    def label(self) -> str:
        return f"{self.year}-W{self.week:02d}"

    @property
    def baseline_label(self) -> str:
        return f"W{self.b0.isocalendar()[1]:02d}–W{(self.b1 - timedelta(days=1)).isocalendar()[1]:02d}"

    def params(self) -> dict[str, str]:
        d = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(self).items()}
        return {k: str(v) for k, v in d.items()}


def build_scope(question: str, metric: str, region: str, year: int, week: int,
                data_start: date, as_of: date, baseline_weeks: int = 7, gap_weeks: int = 3) -> Scope:
    if metric not in METRICS:
        raise UnsupportedQuestion(f"metric {metric!r} not supported yet (supported: {list(METRICS)})")
    if region not in REGIONS:
        raise UnsupportedQuestion(f"region {region!r} unknown (known: {REGIONS})")
    t0 = date.fromisocalendar(year, week, 1)
    t1 = t0 + timedelta(days=7)
    if t0 > as_of:
        raise UnsupportedQuestion(f"{year}-W{week} is after the data ends ({as_of})")
    b1 = t0 - timedelta(days=7 * gap_weeks)
    # skip the first data week: refunds lag orders, so it under-counts.
    first_full = date.fromisocalendar(*data_start.isocalendar()[:2], 1) + timedelta(days=7)
    b0 = max(first_full, b1 - timedelta(days=7 * baseline_weeks))
    nb = (b1 - b0).days // 7
    if nb < 2:
        raise UnsupportedQuestion("not enough history before the target week for a baseline")
    return Scope(question, metric, region, year, week, t0, t1, b0, b1, t0 - timedelta(days=14), as_of, nb)


def parse_question_rules(question: str, as_of: date) -> dict:
    """Deterministic parser: metric, region, iso week. Used when no LLM, and to
    validate what an LLM returns."""
    q = question.lower()
    metric = next((m for m, kws in METRICS.items() if any(k in q for k in kws)), None)
    if metric is None:
        raise UnsupportedQuestion("could not find a supported metric in the question (try 'refunds')")
    region = None
    for alias in sorted(REGION_ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", q):
            region = REGION_ALIASES[alias]
            break
    if region is None:
        raise UnsupportedQuestion("which region? (APAC, NA, EU, LATAM)")
    m = re.search(r"\b(?:week|wk|w)\s*(\d{1,2})\b", q)
    year = as_of.isocalendar()[0]
    if m:
        week = int(m.group(1))
    elif "last week" in q or "this week" in q or "recent" in q:
        # latest complete ISO week in the data
        last_sun = as_of if as_of.weekday() == 6 else as_of - timedelta(days=as_of.weekday() + 1)
        year, week = last_sun.isocalendar()[:2]
    else:
        raise UnsupportedQuestion("which week? (e.g. 'week 40' or 'last week')")
    return {"metric": metric, "region": region, "year": year, "week": week}
