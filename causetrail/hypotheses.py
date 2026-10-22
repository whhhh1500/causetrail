"""Hypothesis library for the refund-rate playbook.

Each :class:`Hypothesis` names a mechanism and carries one or more probes. A
probe is SQL (logical tables in ``{braces}``, scope parameters as ``$name``)
plus a *judge* - a deterministic function that reads the rows and returns a
verdict, a one-line finding, numeric metrics and optional bindings (e.g. which
product family is to blame) that unlock follow-up hypotheses.

Probes are written as falsification tests: each states what would have to be
true if the mechanism were real, and the judge refutes the hypothesis when the
data says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from string import Template
from typing import Callable

from .scope import Scope

SUPPORTS, REFUTES, INCONCLUSIVE, CONTEXT = "supports", "refutes", "inconclusive", "context"


@dataclass
class Outcome:
    verdict: str
    finding: str
    metrics: dict = field(default_factory=dict)
    bindings: dict = field(default_factory=dict)


Judge = Callable[[list[dict], Scope, dict], Outcome]


@dataclass
class Probe:
    id: str
    purpose: str
    sql: str
    judge: Judge

    def render(self, scope: Scope, bindings: dict) -> str:
        return Template(self.sql).substitute({**scope.params(), **{k: str(v) for k, v in bindings.items()}})


@dataclass
class Hypothesis:
    id: str
    title: str  # may use {family}/{region}/... placeholders
    mechanism: str
    probes: list[Probe]
    requires: tuple[str, ...] = ()  # bindings needed before it can run
    unlocks: tuple[str, ...] = ()  # hypotheses queued when this one is supported
    min_support: int | None = None  # default: every non-context probe must support
    layer: int = 0  # depth in the causal chain (higher = closer to root cause)

    def title_for(self, scope: Scope, bindings: dict) -> str:
        return self.title.format(region=scope.region, **bindings)


def pct(x: float) -> float:
    return round(x * 100, 1)


def _share(part: float, total: float) -> float:
    return part / total if total > 0 else 0.0


# --- H0: did the metric actually move, and where? -----------------------------

TREND_SQL = """
WITH ord AS (
  SELECT CASE WHEN o.order_date >= DATE '$t0' THEN 'target' ELSE 'baseline' END AS period,
         COUNT(*) AS orders
  FROM {orders} o
  WHERE o.region = '$region'
    AND ((o.order_date >= DATE '$b0' AND o.order_date < DATE '$b1')
      OR (o.order_date >= DATE '$t0' AND o.order_date < DATE '$t1'))
  GROUP BY 1),
ref AS (
  SELECT CASE WHEN r.refunded_at >= DATE '$t0' THEN 'target' ELSE 'baseline' END AS period,
         COUNT(*) AS refunds
  FROM {refunds} r JOIN {orders} o ON o.order_id = r.order_id
  WHERE o.region = '$region'
    AND ((r.refunded_at >= DATE '$b0' AND r.refunded_at < DATE '$b1')
      OR (r.refunded_at >= DATE '$t0' AND r.refunded_at < DATE '$t1'))
  GROUP BY 1)
SELECT ord.period, ord.orders, COALESCE(ref.refunds, 0) AS refunds,
       ROUND(COALESCE(ref.refunds, 0) * 100.0 / ord.orders, 2) AS refund_rate_pct
FROM ord LEFT JOIN ref ON ref.period = ord.period
ORDER BY ord.period
"""


def judge_trend(rows, s: Scope, b) -> Outcome:
    by = {r["period"]: r for r in rows}
    if "target" not in by or "baseline" not in by:
        return Outcome(INCONCLUSIVE, "Not enough data in the target or baseline window.")
    t, bl = by["target"], by["baseline"]
    tr, br = t["refunds"] / t["orders"], bl["refunds"] / bl["orders"]
    lift = pct(tr / br - 1) if br else 0.0
    m = {"target_rate_pct": pct(tr), "baseline_rate_pct": pct(br), "lift_pct": lift,
         "target_refunds": t["refunds"], "target_orders": t["orders"]}
    f = (f"{s.region} refund rate was {pct(tr)}% in {s.label} ({t['refunds']} refunds / {t['orders']} orders) "
         f"vs {pct(br)}% over the {s.baseline_label} baseline, a change of {lift:+}%.")
    moved = br > 0 and tr / br >= 1.25 and t["refunds"] >= 5
    return Outcome(SUPPORTS if moved else REFUTES, f, m)


REGIONS_SQL = """
WITH ord AS (
  SELECT o.region,
         COUNT(CASE WHEN o.order_date >= DATE '$t0' AND o.order_date < DATE '$t1' THEN 1 END) AS target_orders,
         COUNT(CASE WHEN o.order_date >= DATE '$b0' AND o.order_date < DATE '$b1' THEN 1 END) AS baseline_orders
  FROM {orders} o GROUP BY o.region),
ref AS (
  SELECT o.region,
         COUNT(CASE WHEN r.refunded_at >= DATE '$t0' AND r.refunded_at < DATE '$t1' THEN 1 END) AS target_refunds,
         COUNT(CASE WHEN r.refunded_at >= DATE '$b0' AND r.refunded_at < DATE '$b1' THEN 1 END) AS baseline_refunds
  FROM {refunds} r JOIN {orders} o ON o.order_id = r.order_id GROUP BY o.region)
SELECT ord.region,
       ROUND(ref.target_refunds * 100.0 / ord.target_orders, 2) AS target_rate_pct,
       ROUND(ref.baseline_refunds * 100.0 / ord.baseline_orders, 2) AS baseline_rate_pct
FROM ord JOIN ref ON ref.region = ord.region
ORDER BY ord.region
"""


def judge_regions(rows, s: Scope, b) -> Outcome:
    lifts = {r["region"]: (r["target_rate_pct"] / r["baseline_rate_pct"] - 1) * 100
             for r in rows if r["baseline_rate_pct"]}
    others = {k: round(v, 1) for k, v in lifts.items() if k != s.region}
    mine = round(lifts.get(s.region, 0.0), 1)
    concentrated = all(v < 25 for v in others.values()) and mine >= 25
    other_txt = ", ".join(f"{k} {v:+}%" for k, v in sorted(others.items()))
    f = (f"Change vs baseline by region: {s.region} {mine:+}%; other regions {other_txt}. "
         + ("The move is specific to " + s.region + "." if concentrated else "The move is not specific to one region."))
    m = {"region_lift_pct": mine, **{f"lift_{k}_pct": v for k, v in others.items()}}
    return Outcome(SUPPORTS if concentrated else INCONCLUSIVE, f, m)


METRIC_MOVED = Hypothesis(
    "H0", "The {region} refund rate really moved in the target week", "metric check",
    [Probe("H0.trend", "Confirm the refund-rate change (target week vs baseline)", TREND_SQL, judge_trend),
     Probe("H0.regions", "Is the change specific to the region?", REGIONS_SQL, judge_regions)],
    min_support=1,
)

# --- H1: marketing mix -----------------------------------------------------------

SPEND_SQL = """
SELECT a.channel, a.campaign,
       ROUND(SUM(CASE WHEN a.date >= DATE '$t0' AND a.date < DATE '$t1' THEN a.spend_usd ELSE 0 END), 0) AS target_spend,
       ROUND(SUM(CASE WHEN a.date >= DATE '$b0' AND a.date < DATE '$b1' THEN a.spend_usd ELSE 0 END) / $nb, 0) AS baseline_spend_per_week
FROM {ad_spend} a
WHERE a.market = '$region'
GROUP BY a.channel, a.campaign
ORDER BY target_spend DESC, a.campaign
"""


def judge_spend(rows, s: Scope, b) -> Outcome:
    tgt = sum(r["target_spend"] for r in rows)
    base = sum(r["baseline_spend_per_week"] for r in rows)
    new = [r["campaign"] for r in rows if r["baseline_spend_per_week"] == 0 and r["target_spend"] > 0]
    lift = pct(tgt / base - 1) if base else 0.0
    f = (f"{s.region} ad spend was ${tgt:,.0f} in {s.label} vs ${base:,.0f}/week in the baseline ({lift:+}%)"
         + (f"; new campaign(s) running: {', '.join(new)}." if new else "."))
    return Outcome(CONTEXT, f, {"target_spend": tgt, "baseline_spend_per_week": base, "spend_lift_pct": lift},
                   {"new_campaigns": ",".join(new)})


CHANNEL_SQL = """
SELECT o.channel, COUNT(*) AS orders, COUNT(r.order_id) AS refunded_orders,
       ROUND(COUNT(r.order_id) * 100.0 / COUNT(*), 2) AS refund_rate_pct
FROM {orders} o
LEFT JOIN (SELECT DISTINCT order_id FROM {refunds}) r ON r.order_id = o.order_id
WHERE o.region = '$region' AND o.order_date >= DATE '$c0' AND o.order_date < DATE '$t1'
GROUP BY o.channel
ORDER BY refund_rate_pct DESC, o.channel
"""


def judge_channel(rows, s: Scope, b) -> Outcome:
    ps = next((r for r in rows if r["channel"] == "paid_social"), None)
    rest = [r for r in rows if r["channel"] != "paid_social"]
    if not ps or not rest:
        return Outcome(INCONCLUSIVE, "No paid_social orders in the incident window.")
    ps_rate = ps["refunded_orders"] / ps["orders"]
    rest_rate = sum(r["refunded_orders"] for r in rest) / sum(r["orders"] for r in rest)
    ratio = round(ps_rate / rest_rate, 2) if rest_rate else 0.0
    f = (f"Orders placed {s.c0}..{s.label}: paid_social refund rate {pct(ps_rate)}% "
         f"vs {pct(rest_rate)}% for all other channels (ratio {ratio}). "
         + ("Campaign customers do refund more." if ratio >= 1.5 else
            "Campaign customers refund no more than everyone else, so the campaign does not explain the spike."))
    v = SUPPORTS if ratio >= 1.5 else REFUTES if ratio < 1.3 else INCONCLUSIVE
    return Outcome(v, f, {"paid_social_rate_pct": pct(ps_rate), "other_rate_pct": pct(rest_rate), "ratio": ratio})


MARKETING = Hypothesis(
    "H1", "A paid-social campaign brought in refund-prone customers",
    "more ad spend -> new, lower-intent customers -> more refunds",
    [Probe("H1.spend", "Did ad spend / campaigns change? (precondition)", SPEND_SQL, judge_spend),
     Probe("H1.channel", "Falsify: do campaign-channel orders refund more than other orders?", CHANNEL_SQL,
           judge_channel)],
)

# --- H2: payments -----------------------------------------------------------------

PAYMENT_SQL = """
SELECT CASE WHEN o.order_date >= DATE '$t0' THEN 'target' ELSE 'baseline' END AS period,
       COUNT(DISTINCT o.order_id) AS orders,
       COUNT(CASE WHEN p.status = 'failed' THEN 1 END) AS failed_attempts,
       ROUND(COUNT(CASE WHEN p.status = 'failed' THEN 1 END) * 100.0 / COUNT(DISTINCT o.order_id), 2) AS failed_per_100_orders
FROM {orders} o JOIN {payments} p ON p.order_id = o.order_id
WHERE o.region = '$region'
  AND ((o.order_date >= DATE '$b0' AND o.order_date < DATE '$b1')
    OR (o.order_date >= DATE '$t0' AND o.order_date < DATE '$t1'))
GROUP BY 1
ORDER BY 1
"""


def judge_payment(rows, s: Scope, b) -> Outcome:
    by = {r["period"]: r for r in rows}
    if len(by) < 2:
        return Outcome(INCONCLUSIVE, "Missing payment data for one of the windows.")
    t, bl = by["target"]["failed_per_100_orders"], by["baseline"]["failed_per_100_orders"]
    lift = round(t / bl, 2) if bl else 0.0
    f = (f"Payment failures were {t} per 100 orders in {s.label} vs {bl} in the baseline (ratio {lift}). "
         + ("Payment failures rose." if lift >= 1.25 else "Payment failures are flat."))
    return Outcome(SUPPORTS if lift >= 1.25 else REFUTES, f,
                   {"target_failed_per_100": t, "baseline_failed_per_100": bl, "ratio": lift})


CHARGEBACK_SQL = """
SELECT r.reason_code,
       COUNT(CASE WHEN r.refunded_at >= DATE '$t0' AND r.refunded_at < DATE '$t1' THEN 1 END) AS target_refunds,
       ROUND(COUNT(CASE WHEN r.refunded_at >= DATE '$b0' AND r.refunded_at < DATE '$b1' THEN 1 END) * 1.0 / $nb, 2) AS baseline_refunds_per_week
FROM {refunds} r JOIN {orders} o ON o.order_id = r.order_id
WHERE o.region = '$region'
GROUP BY r.reason_code
ORDER BY target_refunds DESC, r.reason_code
"""


def _excess(rows, key):
    ex = {r[key]: r["target_refunds"] - r["baseline_refunds_per_week"] for r in rows}
    total = sum(v for v in ex.values() if v > 0)
    return ex, total


def judge_chargeback(rows, s: Scope, b) -> Outcome:
    ex, total = _excess(rows, "reason_code")
    share = _share(max(ex.get("chargeback", 0), 0), total)
    cr = _share(max(ex.get("customer_request", 0), 0), total)
    f = (f"Chargebacks account for {pct(share)}% of the {total:.1f} excess refunds in {s.label}; "
         f"'customer_request' accounts for {pct(cr)}%.")
    v = SUPPORTS if share >= 0.5 else REFUTES if share < 0.25 else INCONCLUSIVE
    return Outcome(v, f, {"chargeback_share_pct": pct(share), "customer_request_share_pct": pct(cr),
                          "excess_refunds": round(total, 1)})


PAYMENTS = Hypothesis(
    "H2", "Payment problems (failures, chargebacks) drove the refunds",
    "payment failures / disputes -> forced refunds",
    [Probe("H2.failures", "Falsify: did payment failure rates rise?", PAYMENT_SQL, judge_payment),
     Probe("H2.chargebacks", "Falsify: are the extra refunds chargebacks?", CHARGEBACK_SQL, judge_chargeback)],
    min_support=1,
)

# --- H3: product quality -------------------------------------------------------------


def judge_quality(rows, s: Scope, b) -> Outcome:
    ex, total = _excess(rows, "reason_code")
    q = sum(max(ex.get(k, 0), 0) for k in ("damaged", "not_as_described"))
    share = _share(q, total)
    cr = _share(max(ex.get("customer_request", 0), 0), total)
    f = (f"Quality reasons (damaged, not_as_described) explain {pct(share)}% of the excess refunds; "
         f"{pct(cr)}% are a generic 'customer_request' - the refund table alone cannot say why.")
    v = SUPPORTS if share >= 0.5 else REFUTES if share < 0.25 else INCONCLUSIVE
    return Outcome(v, f, {"quality_share_pct": pct(share), "customer_request_share_pct": pct(cr)})


QUALITY = Hypothesis(
    "H3", "A product-quality problem (damaged / not as described) drove refunds",
    "defects -> returns",
    [Probe("H3.reasons", "Falsify: are the extra refunds quality-coded?",
           CHARGEBACK_SQL.replace("ORDER BY target_refunds DESC, r.reason_code", "ORDER BY r.reason_code"), judge_quality)],
)

# --- H4: product mix -------------------------------------------------------------------

FAMILY_SQL = """
SELECT s.family,
       COUNT(DISTINCT CASE WHEN r.refunded_at >= DATE '$t0' AND r.refunded_at < DATE '$t1' THEN r.refund_id END) AS target_refunds,
       ROUND(COUNT(DISTINCT CASE WHEN r.refunded_at >= DATE '$b0' AND r.refunded_at < DATE '$b1' THEN r.refund_id END) * 1.0 / $nb, 2) AS baseline_refunds_per_week
FROM {refunds} r
JOIN {orders} o ON o.order_id = r.order_id
JOIN {order_items} i ON i.order_id = o.order_id
JOIN {skus} s ON s.sku = i.sku
WHERE o.region = '$region'
GROUP BY s.family
ORDER BY target_refunds DESC, s.family
"""


def judge_family(rows, s: Scope, b) -> Outcome:
    ex, total = _excess(rows, "family")
    if total <= 0:
        return Outcome(REFUTES, "No family has more refunds than its baseline.")
    top = max(ex, key=ex.get)
    share = _share(ex[top], total)
    r = next(x for x in rows if x["family"] == top)
    f = (f"{top} accounts for {pct(share)}% of the excess refunds in {s.region} {s.label} "
         f"({r['target_refunds']} refunds vs {r['baseline_refunds_per_week']}/week baseline).")
    v = SUPPORTS if share >= 0.5 else INCONCLUSIVE
    return Outcome(v, f, {"top_family_share_pct": pct(share), "family_target_refunds": r["target_refunds"],
                          "family_baseline_per_week": r["baseline_refunds_per_week"]}, {"family": top})


PRODUCT_MIX = Hypothesis(
    "H4", "The excess refunds are concentrated in one product family",
    "something specific to one family", [Probe("H4.family", "Which product family carries the excess?",
                                               FAMILY_SQL, judge_family)],
    unlocks=("H5",), layer=1,
)

# --- H5: fulfilment ----------------------------------------------------------------------

LATE_SQL = """
SELECT CASE WHEN o.order_date >= DATE '$c0' THEN 'incident' ELSE 'baseline' END AS period,
       CASE WHEN s.family = '$family' THEN '$family' ELSE 'other families' END AS segment,
       COUNT(*) AS orders,
       ROUND(AVG(DAYS_BETWEEN(o.order_date, COALESCE(o.shipped_date, DATE '$as_of'))), 1) AS avg_days_to_ship,
       ROUND(SUM(CASE WHEN COALESCE(o.shipped_date, DATE '$as_of') > o.promised_ship_date THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS late_pct
FROM {orders} o
JOIN {order_items} i ON i.order_id = o.order_id
JOIN {skus} s ON s.sku = i.sku
WHERE o.region = '$region'
  AND ((o.order_date >= DATE '$b0' AND o.order_date < DATE '$b1')
    OR (o.order_date >= DATE '$c0' AND o.order_date < DATE '$t1'))
GROUP BY 1, 2
ORDER BY 2, 1
"""


def judge_late(rows, s: Scope, b) -> Outcome:
    fam = b["family"]
    g = {(r["period"], r["segment"]): r for r in rows}
    try:
        fi, fb = g[("incident", fam)], g[("baseline", fam)]
        oi, ob = g[("incident", "other families")], g[("baseline", "other families")]
    except KeyError:
        return Outcome(INCONCLUSIVE, "Missing a period/segment for the late-shipment comparison.")
    jump = fi["late_pct"] - fb["late_pct"]
    other_jump = oi["late_pct"] - ob["late_pct"]
    f = (f"{fam} orders in {s.region} shipped late {fi['late_pct']}% of the time in the incident window "
         f"(avg {fi['avg_days_to_ship']} days to ship) vs {fb['late_pct']}% in the baseline "
         f"({fb['avg_days_to_ship']} days); other families went from {ob['late_pct']}% to {oi['late_pct']}%.")
    ok = jump >= 10 and fi["late_pct"] >= 2 * max(fb["late_pct"], 1) and other_jump < 5
    v = SUPPORTS if ok else REFUTES if jump < 3 else INCONCLUSIVE
    return Outcome(v, f, {"family_late_pct": fi["late_pct"], "family_baseline_late_pct": fb["late_pct"],
                          "family_days_to_ship": fi["avg_days_to_ship"],
                          "family_baseline_days_to_ship": fb["avg_days_to_ship"],
                          "other_late_pct": oi["late_pct"], "other_baseline_late_pct": ob["late_pct"]})


LATE_REFUND_SQL = """
SELECT CASE WHEN COALESCE(o.shipped_date, DATE '$as_of') > o.promised_ship_date THEN 'late' ELSE 'on_time' END AS shipping,
       COUNT(*) AS orders, COUNT(r.order_id) AS refunded_orders,
       ROUND(COUNT(r.order_id) * 100.0 / COUNT(*), 2) AS refund_rate_pct
FROM {orders} o
LEFT JOIN (SELECT DISTINCT order_id FROM {refunds}) r ON r.order_id = o.order_id
WHERE o.region = '$region' AND o.order_date >= DATE '$b0' AND o.order_date < DATE '$t1'
GROUP BY 1
ORDER BY 1
"""


def judge_late_refund(rows, s: Scope, b) -> Outcome:
    g = {r["shipping"]: r for r in rows}
    if "late" not in g or "on_time" not in g:
        return Outcome(INCONCLUSIVE, "No late orders to compare.")
    lr, orr = g["late"]["refund_rate_pct"], g["on_time"]["refund_rate_pct"]
    ratio = round(lr / orr, 1) if orr else 0.0
    f = (f"In {s.region}, late orders were refunded {lr}% of the time vs {orr}% for on-time orders "
         f"({ratio}x, {g['late']['orders']} late orders).")
    v = SUPPORTS if ratio >= 3 and g["late"]["orders"] >= 20 else REFUTES if ratio < 1.5 else INCONCLUSIVE
    return Outcome(v, f, {"late_refund_rate_pct": lr, "on_time_refund_rate_pct": orr, "ratio": ratio,
                          "late_orders": g["late"]["orders"]})


TICKET_SQL = """
WITH delay_tickets AS (
  SELECT CAST(REPLACE(t.order_ref, '#', '') AS BIGINT) AS order_id, MIN(t.created_at) AS first_ticket_at
  FROM {support_tickets} t
  WHERE LOWER(t.subject) LIKE '%where is my order%' OR LOWER(t.subject) LIKE '%not shipped%'
     OR LOWER(t.subject) LIKE '%delivery%' OR LOWER(t.subject) LIKE '%delayed%'
     OR LOWER(t.tags) LIKE '%shipping%' OR LOWER(t.tags) LIKE '%delivery%'
  GROUP BY 1),
x AS (
  SELECT r.order_id, r.refunded_at,
         CASE WHEN r.refunded_at >= DATE '$t0' THEN 'target' ELSE 'baseline' END AS period
  FROM {refunds} r JOIN {orders} o ON o.order_id = r.order_id
  WHERE o.region = '$region'
    AND ((r.refunded_at >= DATE '$b0' AND r.refunded_at < DATE '$b1')
      OR (r.refunded_at >= DATE '$t0' AND r.refunded_at < DATE '$t1')))
SELECT x.period, COUNT(*) AS refunds, COUNT(d.order_id) AS preceded_by_delay_ticket,
       ROUND(COUNT(d.order_id) * 100.0 / COUNT(*), 1) AS pct_with_delay_ticket
FROM x LEFT JOIN delay_tickets d
  ON d.order_id = x.order_id AND CAST(d.first_ticket_at AS DATE) <= x.refunded_at
GROUP BY x.period
ORDER BY x.period
"""


def judge_tickets(rows, s: Scope, b) -> Outcome:
    g = {r["period"]: r for r in rows}
    if "target" not in g:
        return Outcome(INCONCLUSIVE, "No refunds in the target week.")
    t = g["target"]["pct_with_delay_ticket"]
    bl = g.get("baseline", {}).get("pct_with_delay_ticket", 0.0)
    f = (f"{t}% of {s.region} refunds in {s.label} were preceded by a shipping-delay support ticket "
         f"(joined from the helpdesk JSON export) vs {bl}% in the baseline.")
    v = SUPPORTS if t >= 30 and t >= 3 * max(bl, 1) else REFUTES if t < 10 else INCONCLUSIVE
    return Outcome(v, f, {"target_ticket_pct": t, "baseline_ticket_pct": bl})


FULFILMENT = Hypothesis(
    "H5", "Late shipments of {family} in {region} drove the refunds",
    "slow fulfilment -> 'where is my order?' -> cancellation refunds",
    [Probe("H5.late", "Did {family} start shipping late (and only {family})?", LATE_SQL, judge_late),
     Probe("H5.late_refund", "Falsify: are late orders actually refunded more?", LATE_REFUND_SQL,
           judge_late_refund),
     Probe("H5.tickets", "Falsify: did customers complain about delays before refunding?", TICKET_SQL,
           judge_tickets)],
    requires=("family",), unlocks=("H6",), min_support=2, layer=2,
)

# --- H6: supplier ---------------------------------------------------------------------------

LEAD_SQL = """
SELECT lt.week_start, lt.warehouse, lt.supplier_id, sp.supplier_name, lt.promised_lead_days, lt.actual_lead_days,
       lt.actual_lead_days - lt.promised_lead_days AS slip_days, lt.on_hand_units
FROM {lead_times} lt
JOIN {warehouses} w ON w.warehouse = lt.warehouse
JOIN {suppliers} sp ON sp.supplier_id = lt.supplier_id
WHERE w.serves_region = '$region' AND lt.family = '$family'
  AND lt.week_start >= DATE '$b0' AND lt.week_start < DATE '$t1'
ORDER BY lt.week_start
"""


def judge_lead(rows, s: Scope, b) -> Outcome:
    if not rows:
        return Outcome(INCONCLUSIVE, "No lead-time rows for this family/region.")
    import datetime as _dt

    def wk(r):
        v = r["week_start"]
        return v if isinstance(v, _dt.date) else _dt.date.fromisoformat(str(v)[:10])

    base = [r for r in rows if wk(r) < s.b1]
    recent = [r for r in rows if wk(r) >= s.b1]
    if not base or not recent:
        return Outcome(INCONCLUSIVE, "Lead-time history does not cover baseline and incident windows.")
    b_slip = round(sum(r["slip_days"] for r in base) / len(base), 1)
    peak = max(recent, key=lambda r: r["slip_days"])
    first = next((r for r in rows if r["slip_days"] >= 3), None)
    b_hand = sum(r["on_hand_units"] for r in base) / len(base)
    low_hand = min(r["on_hand_units"] for r in recent)
    drop = pct(1 - low_hand / b_hand) if b_hand else 0.0
    sup, name = peak["supplier_id"], peak["supplier_name"]
    start_wk = f"W{wk(first).isocalendar()[1]:02d}" if first else "n/a"
    f = (f"Supplier {sup} ({name}) for {b['family']} at {peak['warehouse']} slipped to {peak['actual_lead_days']} days "
         f"lead time vs {peak['promised_lead_days']} promised (baseline slip {b_slip} days); the slip started in "
         f"{start_wk}, and on-hand stock fell {drop}% below its baseline average.")
    ok = peak["slip_days"] >= 5 and b_slip <= 2 and drop >= 50
    v = SUPPORTS if ok else REFUTES if peak["slip_days"] < 2 else INCONCLUSIVE
    return Outcome(v, f, {"peak_lead_days": peak["actual_lead_days"], "promised_lead_days": peak["promised_lead_days"],
                          "baseline_slip_days": b_slip, "stock_drop_pct": drop},
                   {"supplier_id": sup, "supplier_name": name, "warehouse": peak["warehouse"], "slip_start": start_wk})


CONTROL_SQL = """
WITH slip AS (
  SELECT w.serves_region AS region, MAX(lt.supplier_id) AS supplier_id,
         MAX(lt.actual_lead_days - lt.promised_lead_days) AS max_slip_days
  FROM {lead_times} lt JOIN {warehouses} w ON w.warehouse = lt.warehouse
  WHERE lt.family = '$family' AND lt.week_start >= DATE '$b0' AND lt.week_start < DATE '$t1'
  GROUP BY w.serves_region),
ref AS (
  SELECT o.region,
         COUNT(CASE WHEN r.refunded_at >= DATE '$t0' AND r.refunded_at < DATE '$t1' THEN 1 END) AS target_refunds,
         ROUND(COUNT(CASE WHEN r.refunded_at >= DATE '$b0' AND r.refunded_at < DATE '$b1' THEN 1 END) * 1.0 / $nb, 2) AS baseline_refunds_per_week
  FROM {refunds} r
  JOIN {orders} o ON o.order_id = r.order_id
  JOIN {order_items} i ON i.order_id = o.order_id
  JOIN {skus} s ON s.sku = i.sku
  WHERE s.family = '$family'
  GROUP BY o.region)
SELECT slip.region, slip.supplier_id, slip.max_slip_days,
       COALESCE(ref.target_refunds, 0) AS target_refunds, COALESCE(ref.baseline_refunds_per_week, 0) AS baseline_refunds_per_week
FROM slip LEFT JOIN ref ON ref.region = slip.region
ORDER BY slip.max_slip_days DESC, slip.region
"""


def judge_control(rows, s: Scope, b) -> Outcome:
    mine = next((r for r in rows if r["region"] == s.region), None)
    others = [r for r in rows if r["region"] != s.region]
    if not mine or not others:
        return Outcome(INCONCLUSIVE, "No control regions for this family.")
    clean = [r for r in others if r["max_slip_days"] <= 2
             and r["target_refunds"] <= max(2 * r["baseline_refunds_per_week"], r["baseline_refunds_per_week"] + 4)]
    other_txt = "; ".join(f"{r['region']} via {r['supplier_id']}: max slip {r['max_slip_days']} days, "
                          f"{r['target_refunds']} refunds vs {r['baseline_refunds_per_week']}/week" for r in others)
    f = (f"Control check for {b['family']}: {s.region} via {mine['supplier_id']} slipped {mine['max_slip_days']} days "
         f"with {mine['target_refunds']} refunds vs {mine['baseline_refunds_per_week']}/week. Other regions - "
         f"{other_txt}. " + ("Same products from other suppliers did not see a refund spike."
                              if len(clean) == len(others) else "Some control regions also moved."))
    v = SUPPORTS if len(clean) == len(others) and mine["max_slip_days"] >= 5 else INCONCLUSIVE
    return Outcome(v, f, {"control_regions_clean": len(clean), "control_regions": len(others),
                          "own_max_slip_days": mine["max_slip_days"]})


SUPPLIER = Hypothesis(
    "H6", "A supplier lead-time slip for {family} at the {region} warehouse caused the late shipments",
    "supplier slip -> stock-out -> backorders -> late shipments",
    [Probe("H6.lead", "Did the supplier's lead time slip before the delays?", LEAD_SQL, judge_lead),
     Probe("H6.control", "Falsify with a control: did the same products from other suppliers stay fine?",
           CONTROL_SQL, judge_control)],
    requires=("family",), layer=3,
)

LIBRARY: dict[str, Hypothesis] = {h.id: h for h in (MARKETING, PAYMENTS, QUALITY, PRODUCT_MIX, FULFILMENT, SUPPLIER)}
ROOTS = ("H1", "H2", "H3", "H4")  # what the planner may choose from at the start
