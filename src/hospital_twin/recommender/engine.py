"""AI decision support: detect risks, propose actions, verify them in the twin.

Pipeline:
  1. Risk detection on the ML forecasts and the baseline twin projection.
  2. Candidate interventions are generated from a playbook of hospital
     operational levers (surge beds, discharge drive, extra physicians,
     ambulance diversion, elective re-planning, nurse redeployment).
  3. Every simulable intervention is applied to the twin and re-simulated
     with common random numbers; the paired KPI deltas (with 90% CI) become
     the quantified, explainable "expected impact".
  4. Interventions that do not improve the weighted outcome are rejected and
     shown as such, so administrators see what was considered and why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..simulation.runner import MCResult, compare
from ..simulation.scenario import Scenario

IMPACT_METRICS = [
    "door_to_doc_mean_min", "boarding_mean_h", "boarding_p90_h", "lwbs", "ICU_full_hours",
    "MED_full_hours", "SURG_full_hours", "electives_done", "electives_cancelled", "electives_postponed",
    "outliers", "icu_exit_block_h", "diverted",
]
METRIC_LABELS = {
    "door_to_doc_mean_min": "Door-to-doctor (min)",
    "boarding_mean_h": "ED boarding mean (h)",
    "boarding_p90_h": "ED boarding P90 (h)",
    "lwbs": "Left without being seen",
    "ICU_full_hours": "Hours ICU full",
    "MED_full_hours": "Hours medical ward full",
    "SURG_full_hours": "Hours surgical ward full",
    "electives_done": "Electives performed",
    "electives_cancelled": "Electives cancelled",
    "electives_postponed": "Electives postponed",
    "outliers": "Outlier placements",
    "icu_exit_block_h": "ICU exit-block (bed-h)",
    "diverted": "Ambulances diverted",
}
# Value of a one-unit reduction of each KPI (higher = more important).
WEIGHTS = {
    "door_to_doc_mean_min": 0.6, "boarding_mean_h": 15.0, "boarding_p90_h": 4.0, "lwbs": 1.5,
    "ICU_full_hours": 3.0, "MED_full_hours": 1.0, "SURG_full_hours": 1.0, "electives_done": -3.0,
    "electives_cancelled": 4.0,
    "electives_postponed": 2.0, "outliers": 0.4, "icu_exit_block_h": 0.15, "diverted": 0.8,
}
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "info": 3}


@dataclass
class Recommendation:
    rid: str
    severity: str
    category: str
    title: str
    rationale: str
    actions: list[str]
    owner: str
    icon: str
    scenario_changes: dict | None = None
    action_cost: float = 0.0
    impact: pd.DataFrame | None = None
    score: float = 0.0
    accepted: bool = True
    evidence: dict = field(default_factory=dict)

    def headline_impact(self, n: int = 3) -> list[str]:
        if self.impact is None:
            return []
        imp = self.impact.copy()
        imp["value"] = -imp["delta"] * imp["metric"].map(WEIGHTS)
        imp = imp[imp["delta"].abs() > 1e-6].sort_values("value", ascending=False).head(n)
        out = []
        for r in imp.itertuples():
            pct = (r.delta / r.baseline * 100) if abs(r.baseline) > 1e-9 else float("nan")
            arrow = "▼" if r.delta < 0 else "▲"
            tail = f" ({pct:+.0f}%)" if math.isfinite(pct) else ""
            out.append(f"{arrow} {METRIC_LABELS[r.metric]}: {r.baseline:.1f} → {r.scenario:.1f}{tail}")
        return out


def merge_scenario(base: Scenario, changes: dict, name: str) -> Scenario:
    d = {}
    for k, v in changes.items():
        if k in ("bed_delta", "doctor_delta"):
            merged = dict(getattr(base, k))
            for kk, vv in v.items():
                merged[kk] = merged.get(kk, 0) + vv
            d[k] = merged
        elif k in ("lounge_delta",):
            d[k] = getattr(base, k) + v
        elif k == "elective_postpone_frac":
            d[k] = min(0.9, base.elective_postpone_frac + v)
        else:
            d[k] = v
    return base.with_(name=name, **d)


def _risk(ctx, base: MCResult) -> dict:
    k = base.kpis
    beds = base.beds
    arr = ctx.champion("ed_arrivals").iloc[:7]
    recent = ctx.hist["daily"]["ed_arrivals"].iloc[-28:].mean()
    icu_fc = ctx.champion("icu_census").iloc[:7]
    med_fc = ctx.champion("med_census").iloc[:7]
    surg_fc = ctx.champion("surg_census").iloc[:7]
    return {
        "arrivals_next7": float(arr["yhat"].sum()),
        "arrivals_change_pct": float((arr["yhat"].mean() / recent - 1) * 100),
        "arrivals_peak_day": arr["hi"].idxmax(),
        "arrivals_peak_hi": float(arr["hi"].max()),
        "icu_fc_peak": float(icu_fc["yhat"].max()),
        "icu_fc_peak_hi": float(icu_fc["hi"].max()),
        "icu_fc_peak_day": icu_fc["hi"].idxmax(),
        "med_fc_peak_hi": float(med_fc["hi"].max()),
        "surg_fc_peak_hi": float(surg_fc["hi"].max()),
        "p_icu_full": float((k["ICU_full_hours"] > 0).mean()),
        "icu_full_h": float(k["ICU_full_hours"].mean()),
        "icu_full_h_p90": float(k["ICU_full_hours"].quantile(0.9)),
        "ward_full_h": float((k["MED_full_hours"] + k["SURG_full_hours"]).mean()),
        "med_occ": float(k["MED_occ_mean"].mean()),
        "surg_occ": float(k["SURG_occ_mean"].mean()),
        "icu_occ": float(k["ICU_occ_mean"].mean()),
        "d2d_p90": float(k["door_to_doc_p90_min"].mean()),
        "d2d_mean": float(k["door_to_doc_mean_min"].mean()),
        "lwbs_rate": float(k["lwbs_rate"].mean()),
        "boarding_p90": float(k["boarding_p90_h"].mean()),
        "boarding_mean": float(k["boarding_mean_h"].mean()),
        "electives_cancelled": float(k["electives_cancelled"].mean()),
        "beds": beds,
    }


def _candidates(r: dict, elective_plan=None, start: pd.Timestamp | None = None,
                horizon_days: int = 8) -> list[Recommendation]:
    beds = r["beds"]
    recs: list[Recommendation] = []
    icu_trigger = r["p_icu_full"] > 0.25 or r["icu_fc_peak_hi"] > 0.92 * beds["ICU"]
    if icu_trigger:
        n = int(np.clip(math.ceil(r["icu_fc_peak_hi"] - 0.85 * beds["ICU"]), 2, 6))
        sev = "critical" if r["p_icu_full"] > 0.6 and r["icu_full_h"] > 12 else "high"
        why = (f"ICU census forecast peaks at {r['icu_fc_peak']:.1f} beds (P90 {r['icu_fc_peak_hi']:.1f}) "
               f"on {r['icu_fc_peak_day']:%a %d %b} against {beds['ICU']} staffed beds. The twin shows the ICU "
               f"full for {r['icu_full_h']:.0f} h on average over the next week "
               f"(P90 {r['icu_full_h_p90']:.0f} h, {r['p_icu_full'] * 100:.0f}% of simulated futures).")
        recs.append(Recommendation(
            "icu_surge", sev, "Capacity", f"Open {n} temporary ICU beds (convert PACU / HDU bays)", why,
            [f"Convert {n} PACU/HDU bays to level-3 capable beds for 7 days",
             f"Assign {math.ceil(n / 2)} additional ICU-qualified nurses per shift",
             "Daily 08:00 ICU bed huddle to step patients down early"],
            "Critical Care Director", "🫁", {"bed_delta": {"ICU": n}}, action_cost=8.0 * n,
            evidence={"icu_peak_hi": r["icu_fc_peak_hi"]},
        ))
        recs.append(Recommendation(
            "defer_electives", "high", "Surgery", "Defer 30% of elective list, starting with ICU-dependent cases",
            why + " Elective post-op ICU demand competes directly with emergencies.",
            ["Postpone non-urgent (P3/P4) cases needing post-op ICU",
             "Protect all P1 (urgent) cases", "Re-book deferred cases into the optimiser's next free slots"],
            "Surgical Services Lead", "🗓️", {"elective_postpone_frac": 0.3}, action_cost=4.0,
        ))
    ward_trigger = r["ward_full_h"] > 2 or max(r["med_occ"], r["surg_occ"]) > 90
    if ward_trigger:
        recs.append(Recommendation(
            "discharge_drive", "high" if r["ward_full_h"] > 12 else "medium", "Patient flow",
            "Launch discharge-before-noon drive and extend the discharge lounge",
            f"Medical ward averages {r['med_occ']:.0f}% occupancy and surgical {r['surg_occ']:.0f}%; wards are "
            f"projected to be full for {r['ward_full_h']:.0f} h this week, which blocks ED admissions "
            f"(mean boarding {r['boarding_mean']:.1f} h).",
            ["Consultant ward rounds by 09:00, discharge summaries prepared the evening before",
             "Open 8 extra discharge-lounge chairs (08:00–20:00)", "Pharmacy fast-track for TTO medications",
             "Transport desk pre-booking for confirmed discharges"],
            "Bed Manager", "🏃", {"early_discharge": True, "lounge_delta": 8}, action_cost=3.0,
        ))
    ed_trigger = r["d2d_p90"] > 40 or r["lwbs_rate"] > 1.5 or r["arrivals_change_pct"] > 6
    if ed_trigger:
        recs.append(Recommendation(
            "ed_physician", "high" if r["lwbs_rate"] > 3 else "medium", "Staffing",
            "Add one ED physician to the day and evening shifts",
            f"ED arrivals are forecast at {r['arrivals_next7']:.0f} over 7 days "
            f"({r['arrivals_change_pct']:+.1f}% vs the last 4 weeks; peak P90 {r['arrivals_peak_hi']:.0f} on "
            f"{r['arrivals_peak_day']:%a %d %b}). Door-to-doctor P90 is {r['d2d_p90']:.0f} min and "
            f"{r['lwbs_rate']:.1f}% of patients leave without being seen.",
            ["Call in 1 locum emergency physician 07:00–15:00 and 15:00–23:00",
             "Open the fast-track (ESI 4–5) stream during 10:00–22:00 peak"],
            "ED Clinical Lead", "🩺", {"doctor_delta": {"Day": 1, "Evening": 1}}, action_cost=6.0,
        ))
    if r["boarding_p90"] > 4:
        recs.append(Recommendation(
            "diversion", "critical" if r["boarding_p90"] > 8 else "high", "Network",
            "Request partial ambulance diversion (25% of ESI 2–3) to partner hospitals",
            f"Admitted ED patients wait {r['boarding_mean']:.1f} h on average for a bed (P90 "
            f"{r['boarding_p90']:.1f} h). Diverting a share of stable ambulance traffic protects ED capacity "
            "for critical arrivals.",
            ["Notify regional ambulance control", "Share live bed availability with partner hospitals",
             "Review diversion status every 4 h"],
            "Duty Hospital Manager", "🚑", {"divert_frac": 0.25}, action_cost=5.0,
        ))
    if elective_plan is not None and r["electives_cancelled"] > 3:
        recs.append(Recommendation(
            "elective_plan", "medium", "Surgery", "Adopt the bed-aware optimised elective schedule",
            f"On the current booking pattern ~{r['electives_cancelled']:.0f} electives are cancelled on the day "
            "this week because no post-op bed is free. The optimiser books cases around the forecast emergency load.",
            ["Publish optimised theatre list", "Call patients 48 h ahead to confirm",
             "Keep one theatre open for emergencies"],
            "Theatre Manager", "🔪", {"elective_plan": elective_plan.plan_cases(start, horizon_days)},
            action_cost=1.0,
        ))
    return recs


def _bundle(accepted: list[Recommendation]) -> dict:
    out: dict = {}
    for r in accepted:
        for k, v in (r.scenario_changes or {}).items():
            if isinstance(v, dict):
                cur = out.setdefault(k, {})
                for kk, vv in v.items():
                    cur[kk] = cur.get(kk, 0) + vv
            elif k == "elective_postpone_frac":
                out[k] = max(out.get(k, 0), v)
            else:
                out[k] = v
    return out


def generate_recommendations(ctx, base_scenario: Scenario | None = None, base: MCResult | None = None,
                             roster_summary: dict | None = None, elective_plan=None, reps: int = 16,
                             horizon_h: int = 168, seed: int = 0) -> tuple[list[Recommendation], MCResult, dict]:
    base_scenario = base_scenario or Scenario()
    if base is None:
        base = ctx.simulate(base_scenario, reps=reps, horizon_h=horizon_h, seed=seed)
    risk = _risk(ctx, base)
    recs = _candidates(risk, elective_plan, ctx.start, horizon_h // 24 + 1)
    for r in recs:
        sc = merge_scenario(base_scenario, r.scenario_changes or {}, f"{base_scenario.name} + {r.title}")
        alt = ctx.simulate(sc, reps=len(base.kpis), horizon_h=horizon_h, seed=seed)
        r.impact = compare(base, alt, IMPACT_METRICS)
        r.score = float((-r.impact["delta"] * r.impact["metric"].map(WEIGHTS)).sum() - r.action_cost)
        r.accepted = r.score > 0
    accepted = [r for r in recs if r.accepted]
    if len(accepted) >= 2:
        changes = _bundle(accepted)
        sc = merge_scenario(base_scenario, changes, f"{base_scenario.name} + integrated plan")
        alt = ctx.simulate(sc, reps=len(base.kpis), horizon_h=horizon_h, seed=seed)
        imp = compare(base, alt, IMPACT_METRICS)
        cost = sum(r.action_cost for r in accepted)
        top = min(accepted, key=lambda r: SEVERITY_RANK[r.severity]).severity
        recs.insert(0, Recommendation(
            "bundle", top, "Integrated plan", f"Integrated response plan ({len(accepted)} coordinated actions)",
            "Combining the individually beneficial actions below, re-simulated together to capture interactions "
            "(e.g. faster discharges free the beds that extra ICU capacity needs downstream).",
            [r.title for r in accepted], "Chief Operating Officer", "🧭", changes, cost, imp,
            float((-imp["delta"] * imp["metric"].map(WEIGHTS)).sum() - cost), True,
        ))
    if roster_summary and (roster_summary.get("agency_shifts", 0) or roster_summary.get("short_shifts", 0)):
        recs.append(Recommendation(
            "nurse_roster", "high" if roster_summary.get("short_shifts", 0) else "medium", "Staffing",
            f"Approve optimised nurse roster: {roster_summary['float_shifts']} float-pool placements, "
            f"{roster_summary['agency_shifts']} agency shifts",
            "Forecast P90 census converted to nurse-to-patient ratios exceeds the fixed template roster. "
            f"The CP-SAT optimiser covers {roster_summary['scheduled_shifts'] + roster_summary['agency_shifts']} "
            f"of {roster_summary['required_shifts']} required shifts.",
            ["Publish the optimised roster", "Book agency shifts 72 h ahead for best rates",
             "Redeploy cross-trained float nurses as scheduled"],
            "Director of Nursing", "👩‍⚕️", evidence=roster_summary, score=roster_summary.get("agency_shifts", 0) * 2.0,
        ))
    if risk["arrivals_change_pct"] > 4:
        recs.append(Recommendation(
            "demand_alert", "info", "Forecast", f"ED demand rising {risk['arrivals_change_pct']:+.0f}% next week",
            f"Champion model forecasts {risk['arrivals_next7']:.0f} ED arrivals over the next 7 days with the "
            f"busiest day {risk['arrivals_peak_day']:%A %d %b} (P90 {risk['arrivals_peak_hi']:.0f}).",
            ["Brief department heads at the daily safety huddle"], "Operations Analyst", "📈",
        ))
    recs.sort(key=lambda r: (not r.accepted, SEVERITY_RANK[r.severity], -r.score))
    return recs, base, risk
