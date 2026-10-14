"""Bed-aware elective surgery scheduling (OR-Tools CP-SAT).

Chooses which waiting-list patients to operate on which day over the next two
weeks so that theatre time is respected and post-operative ICU / ward demand
fits *beside* the forecast emergency load. Clinical priority and waiting time
drive the objective; urgent (P1) cases are strongly protected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from ..config import DEFAULT_CONFIG, HospitalConfig
from ..data.generator import india_holidays

PROCEDURES = [
    # name, specialty, theatre hours, ICU days, ward days, relative frequency
    ("Total knee replacement", "Orthopaedics", 2.5, 0, 3, 10),
    ("Total hip replacement", "Orthopaedics", 2.5, 0, 3, 7),
    ("Laparoscopic cholecystectomy", "General Surgery", 1.5, 0, 1, 12),
    ("Inguinal hernia repair", "General Surgery", 1.2, 0, 1, 10),
    ("Thyroidectomy", "ENT / Endocrine", 2.0, 0, 1, 5),
    ("Hysterectomy", "Gynaecology", 2.0, 0, 2, 7),
    ("Spinal fusion", "Neurosurgery", 4.0, 1, 4, 4),
    ("Colectomy", "General Surgery", 3.5, 1, 5, 4),
    ("Radical prostatectomy", "Urology", 3.0, 0, 2, 4),
    ("CABG (bypass)", "Cardiothoracic", 4.5, 2, 5, 4),
    ("Valve replacement", "Cardiothoracic", 5.0, 2, 5, 2),
    ("Craniotomy", "Neurosurgery", 4.0, 2, 4, 2),
    ("Mastectomy", "Surgical Oncology", 2.5, 0, 2, 5),
    ("Cataract (day case)", "Ophthalmology", 0.6, 0, 0, 10),
]
PRIORITY = {"P1": (0.15, 7, 100), "P2": (0.35, 30, 40), "P3": (0.35, 90, 15), "P4": (0.15, 365, 5)}


def generate_waitlist(today: pd.Timestamp, n: int = 240, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = np.array([p[5] for p in PROCEDURES], dtype=float)
    pick = rng.choice(len(PROCEDURES), size=n, p=w / w.sum())
    pr = rng.choice(list(PRIORITY), size=n, p=[v[0] for v in PRIORITY.values()])
    rows = []
    for i, (k, p) in enumerate(zip(pick, pr)):
        name, spec, hrs, icu, ward, _ = PROCEDURES[k]
        target = PRIORITY[p][1]
        waited = int(rng.integers(0, int(target * 1.2) + 1))
        rows.append({
            "case_id": f"E{i + 1:04d}", "procedure": name, "specialty": spec, "priority": p,
            "hours": round(hrs * rng.uniform(0.85, 1.2), 1), "icu_days": icu, "ward_days": ward,
            "listed": (today - pd.Timedelta(days=waited)).normalize(), "waited_days": waited,
            "due": (today + pd.Timedelta(days=target - waited)).normalize(),
        })
    return pd.DataFrame(rows)


@dataclass
class ElectivePlan:
    schedule: pd.DataFrame  # waitlist rows + day / date (NaT = postponed)
    daily: pd.DataFrame  # per day: OT hours used/capacity, ICU / SURG beds used/available
    status: str
    objective: float

    def summary(self) -> dict:
        s = self.schedule
        done = s["date"].notna()
        p1 = s["priority"] == "P1"
        late = done & (s["date"] > s["due"])
        return {
            "scheduled": int(done.sum()),
            "postponed": int((~done).sum()),
            "p1_scheduled_pct": float(done[p1].mean() * 100) if p1.any() else 100.0,
            "late_cases": int(late.sum()),
            "theatre_util_pct": float(self.daily["ot_used"].sum() / max(1e-9, self.daily["ot_cap"].sum()) * 100),
            "bed_overflow_days": int(((self.daily["icu_used"] > self.daily["icu_avail"])
                                      | (self.daily["surg_used"] > self.daily["surg_avail"])).sum()),
            "status": self.status,
        }

    def plan_cases(self, start: pd.Timestamp, days: int) -> list[list[list[float]]]:
        """Booked cases per simulation day index as [icu_days, ward_days, hours] (feeds the twin)."""
        s = self.schedule.dropna(subset=["date"])
        day = (s["date"] - start.normalize()).dt.days
        return [[[float(r.icu_days), float(r.ward_days), float(r.hours)] for r in s[day == d].itertuples()]
                for d in range(days)]


def capacity_for_electives(
    census_fc: pd.DataFrame, elective_dow: pd.DataFrame, snapshot_electives: dict,
    cfg: HospitalConfig = DEFAULT_CONFIG, target_occ: float = 0.95,
) -> pd.DataFrame:
    """Beds available for *new* electives per day = target capacity − emergency forecast − legacy electives."""
    out = []
    for i, d in enumerate(census_fc.index):
        row = {"date": d}
        for u, col in (("ICU", "ICU"), ("SURG", "SURG")):
            emergency = census_fc.loc[d, u] - elective_dow.loc[d.dayofweek, u]
            legacy = snapshot_electives.get(u, 0) * np.exp(-(i + 0.5) / (2.0 if u == "ICU" else 3.0))
            row[f"{col.lower()}_avail"] = max(0.0, cfg.beds[u] * target_occ - emergency - legacy)
        out.append(row)
    return pd.DataFrame(out).set_index("date")


def optimize_electives(
    waitlist: pd.DataFrame, capacity: pd.DataFrame, cfg: HospitalConfig = DEFAULT_CONFIG,
    ot_hours_per_day: float | None = None, time_limit: float = 15.0,
) -> ElectivePlan:
    dates = list(capacity.index)
    hol = india_holidays({d.year for d in dates})
    ot_cap = ot_hours_per_day or (cfg.theatres - 1) * 10.0  # one theatre kept for emergencies
    open_days = [i for i, d in enumerate(dates) if d.dayofweek < 5 and d.date() not in hol]
    m = cp_model.CpModel()
    y = {(r.case_id, d): m.NewBoolVar(f"y_{r.case_id}_{d}") for r in waitlist.itertuples() for d in open_days}
    rows = {r.case_id: r for r in waitlist.itertuples()}
    for cid in rows:
        m.AddAtMostOne(y[cid, d] for d in open_days)
    for d in open_days:
        m.Add(sum(int(rows[c].hours * 10) * y[c, d] for c in rows) <= int(ot_cap * 10))

    slack = {}
    for t in range(len(dates)):
        for unit, col in (("ICU", "icu_avail"), ("SURG", "surg_avail")):
            terms = []
            for c, r in rows.items():
                icu, ward = int(r.icu_days), int(r.ward_days)
                for d in open_days:
                    if d > t:
                        continue
                    if unit == "ICU" and t < d + icu:
                        terms.append(y[c, d])
                    if unit == "SURG" and d + icu <= t < d + icu + ward:
                        terms.append(y[c, d])
            if terms:
                slack[t, unit] = m.NewIntVar(0, 50, f"sl_{t}_{unit}")
                m.Add(sum(terms) <= int(np.floor(capacity.iloc[t][col])) + slack[t, unit])

    obj = []
    for (c, d), v in y.items():
        r = rows[c]
        w = PRIORITY[r.priority][2] + r.waited_days / 10
        date = dates[d]
        late = max(0, (date - r.due).days)
        obj.append(int(w * 10) * v - int((late * 8 + d * 1) * 10) * v)
    obj += [-4000 * s for s in slack.values()]
    m.Maximize(sum(obj))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    st = solver.Solve(m)
    status = solver.StatusName(st)
    assigned = {c: dates[d] for (c, d), v in y.items() if solver.Value(v)}
    sched = waitlist.copy()
    sched["date"] = sched["case_id"].map(assigned)
    sched["date"] = pd.to_datetime(sched["date"])
    return ElectivePlan(sched, _daily_usage(sched, capacity, ot_cap, open_days), status,
                        float(solver.ObjectiveValue()))


def fifo_plan(waitlist: pd.DataFrame, capacity: pd.DataFrame, cfg: HospitalConfig = DEFAULT_CONFIG) -> ElectivePlan:
    """Traditional approach: book strictly by listing date until theatre time runs out (bed-blind)."""
    dates = list(capacity.index)
    hol = india_holidays({d.year for d in dates})
    ot_cap = (cfg.theatres - 1) * 10.0
    open_days = [i for i, d in enumerate(dates) if d.dayofweek < 5 and d.date() not in hol]
    used = {d: 0.0 for d in open_days}
    assigned = {}
    for r in waitlist.sort_values("listed").itertuples():
        for d in open_days:
            if used[d] + r.hours <= ot_cap:
                used[d] += r.hours
                assigned[r.case_id] = dates[d]
                break
    sched = waitlist.copy()
    sched["date"] = pd.to_datetime(sched["case_id"].map(assigned))
    return ElectivePlan(sched, _daily_usage(sched, capacity, ot_cap, open_days), "FIFO", 0.0)


def _daily_usage(sched: pd.DataFrame, capacity: pd.DataFrame, ot_cap: float, open_days) -> pd.DataFrame:
    dates = list(capacity.index)
    s = sched.dropna(subset=["date"])
    rows = []
    for t, date in enumerate(dates):
        start = (s["date"] - dates[0]).dt.days
        icu = ((start <= t) & (t < start + s["icu_days"])).sum()
        surg = ((start + s["icu_days"] <= t) & (t < start + s["icu_days"] + s["ward_days"])).sum()
        rows.append({
            "date": date, "ot_used": float(s.loc[s["date"] == date, "hours"].sum()),
            "ot_cap": ot_cap if t in open_days else 0.0, "cases": int((s["date"] == date).sum()),
            "icu_used": int(icu), "icu_avail": float(capacity.iloc[t]["icu_avail"]),
            "surg_used": int(surg), "surg_avail": float(capacity.iloc[t]["surg_avail"]),
        })
    return pd.DataFrame(rows)
