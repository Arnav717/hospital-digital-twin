"""Nurse rostering with OR-Tools CP-SAT.

Demand per unit and shift comes from the census forecasts and nurse-to-patient
ratios. Each nurse works at most one shift per day, no Night→Day turnarounds,
respects contracted min/max shifts, a cap on nights, and requested days off.
Float-pool nurses can be redeployed across the units they are skilled for;
agency nurses close residual gaps at a premium.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from ..config import ALL_UNITS, DEFAULT_CONFIG, SHIFTS, HospitalConfig

FIRST = ["Priya", "Anitha", "Kavya", "Divya", "Lakshmi", "Meena", "Revathi", "Sangeetha", "Deepa",
         "Nandhini", "Arun", "Karthik", "Suresh", "Vignesh", "Rahul", "Asha", "Gayathri", "Janani",
         "Keerthana", "Swathi", "Ramya", "Pooja", "Harini", "Bala", "Dinesh", "Sneha", "Vidya",
         "Shalini", "Joseph", "Mary", "Fatima", "Ayesha", "Rekha", "Uma", "Bhavani", "Manoj"]
LAST = ["R", "S", "K", "M", "P", "V", "N", "A", "J", "T", "G", "B"]

POOL_SIZE = {"ED": 26, "ICU": 48, "MED": 104, "SURG": 64, "FLOAT": 18}
COST = {"regular": 100, "night": 130, "redeploy": 15, "agency": 380, "short": 2500, "over": 25}
AGENCY_CAP = 4


@dataclass
class Nurse:
    nid: str
    name: str
    home: str  # unit code or FLOAT
    skills: tuple[str, ...]
    min_shifts: int = 4
    max_shifts: int = 5
    days_off: set[int] = field(default_factory=set)


def build_nurse_pool(seed: int = 3, days: int = 7) -> list[Nurse]:
    rng = np.random.default_rng(seed)
    nurses = []
    i = 0
    for home, n in POOL_SIZE.items():
        for _ in range(n):
            i += 1
            if home == "FLOAT":
                skills = ("ICU", "MED", "SURG") if rng.random() < 0.25 else ("ED", "MED", "SURG")
                mn, mx = 2, 5
            else:
                extra = {"MED": ("SURG",), "SURG": ("MED",), "ED": ("MED",), "ICU": ()}[home]
                skills = (home,) + (extra if rng.random() < 0.3 else ())
                mn, mx = 4, 5
            off = set(rng.choice(days, size=int(rng.random() < 0.35), replace=False).tolist())
            name = f"{FIRST[rng.integers(len(FIRST))]} {LAST[rng.integers(len(LAST))]}."
            nurses.append(Nurse(f"N{i:03d}", name, home, skills, mn, mx, off))
    return nurses


def staffing_demand(census_fc: pd.DataFrame, factors: pd.DataFrame,
                    cfg: HospitalConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """census_fc: index=date, columns=units (expected daily mean census at chosen risk level)."""
    rows = []
    for d in census_fc.index:
        for s in SHIFTS:
            for u in ALL_UNITS:
                c = float(census_fc.loc[d, u]) * float(factors.loc[u, s])
                rows.append({"date": d, "shift": s, "unit": u, "census": c,
                             "required": cfg.required_nurses(u, c, s)})
    return pd.DataFrame(rows)


@dataclass
class RosterResult:
    assignments: pd.DataFrame  # nurse, date, shift, unit
    coverage: pd.DataFrame  # date, shift, unit, required, scheduled, agency, short
    cost: float
    status: str
    solve_seconds: float

    def summary(self) -> dict:
        c = self.coverage
        return {
            "required_shifts": int(c["required"].sum()),
            "scheduled_shifts": int(c["scheduled"].sum()),
            "agency_shifts": int(c["agency"].sum()),
            "short_shifts": int(c["short"].sum()),
            "over_shifts": int((c["scheduled"] + c["agency"] - c["required"]).clip(lower=0).sum()),
            # shifts worked by unit-based nurses outside their home unit (float-pool placements excluded)
            "redeployments": int(((self.assignments["unit"] != self.assignments["home"])
                                  & (self.assignments["home"] != "FLOAT")).sum()) if len(self.assignments) else 0,
            "float_shifts": int((self.assignments["home"] == "FLOAT").sum()) if len(self.assignments) else 0,
            "cost": self.cost,
            "status": self.status,
        }


def optimize_roster(nurses: list[Nurse], demand: pd.DataFrame, time_limit: float = 20.0,
                    allow_agency: bool = True) -> RosterResult:
    dates = sorted(demand["date"].unique())
    D, S = len(dates), list(SHIFTS)
    req = {(dates.index(r.date), r.shift, r.unit): int(r.required) for r in demand.itertuples()}
    m = cp_model.CpModel()
    x = {}
    for n in nurses:
        for d in range(D):
            for s in S:
                for u in n.skills:
                    x[n.nid, d, s, u] = m.NewBoolVar(f"x_{n.nid}_{d}_{s}_{u}")
    for n in nurses:
        for d in range(D):
            day = [x[n.nid, d, s, u] for s in S for u in n.skills]
            if d in n.days_off:
                m.Add(sum(day) == 0)
            else:
                m.AddAtMostOne(day)
            if d + 1 < D:
                m.Add(sum(x[n.nid, d, "Night", u] for u in n.skills)
                      + sum(x[n.nid, d + 1, "Day", u] for u in n.skills) <= 1)
        total = sum(x[n.nid, d, s, u] for d in range(D) for s in S for u in n.skills)
        avail = D - len(n.days_off)
        m.Add(total <= min(n.max_shifts, avail))
        m.Add(total >= min(n.min_shifts, avail))
        m.Add(sum(x[n.nid, d, "Night", u] for d in range(D) for u in n.skills) <= 2)

    by_unit = {u: [n for n in nurses if u in n.skills] for u in ALL_UNITS}
    agency, short, over = {}, {}, {}
    for (d, s, u), r in req.items():
        staff = sum(x[n.nid, d, s, u] for n in by_unit[u])
        agency[d, s, u] = m.NewIntVar(0, AGENCY_CAP if allow_agency else 0, f"a_{d}_{s}_{u}")
        short[d, s, u] = m.NewIntVar(0, r, f"sh_{d}_{s}_{u}")
        over[d, s, u] = m.NewIntVar(0, 60, f"ov_{d}_{s}_{u}")
        m.Add(staff + agency[d, s, u] + short[d, s, u] - over[d, s, u] == r)

    home = _home_map(nurses)
    cost_terms = []
    for (nid, d, s, u), v in x.items():
        c = COST["night"] if s == "Night" else COST["regular"]
        if home[nid] != u:
            c += COST["redeploy"]
        cost_terms.append(c * v)
    cost_terms += [COST["agency"] * v for v in agency.values()]
    cost_terms += [COST["short"] * v for v in short.values()]
    cost_terms += [COST["over"] * v for v in over.values()]
    m.Minimize(sum(cost_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    st = solver.Solve(m)
    status = solver.StatusName(st)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"roster infeasible: {status}")

    names = {n.nid: n.name for n in nurses}
    rows = [{"nurse": nid, "name": names[nid], "home": home[nid], "date": dates[d], "shift": s, "unit": u}
            for (nid, d, s, u), v in x.items() if solver.Value(v)]
    asg = pd.DataFrame(rows, columns=["nurse", "name", "home", "date", "shift", "unit"])
    cov = demand.copy()
    sched = asg.groupby(["date", "shift", "unit"]).size()
    cov["scheduled"] = [int(sched.get((r.date, r.shift, r.unit), 0)) for r in cov.itertuples()]
    cov["agency"] = [solver.Value(agency[dates.index(r.date), r.shift, r.unit]) for r in cov.itertuples()]
    cov["short"] = [solver.Value(short[dates.index(r.date), r.shift, r.unit]) for r in cov.itertuples()]
    return RosterResult(asg, cov, float(solver.ObjectiveValue()), status, solver.WallTime())


def _home_map(nurses: list[Nurse]) -> dict[str, str]:
    return {n.nid: n.home for n in nurses}


def template_roster(nurses: list[Nurse], demand: pd.DataFrame) -> RosterResult:
    """Traditional fixed roster: each nurse works a fixed rotation on their home unit
    (float pool spread evenly), with no forecast-driven redeployment and no agency booking."""
    dates = sorted(demand["date"].unique())
    rows = []
    for i, n in enumerate(nurses):
        unit = n.home if n.home != "FLOAT" else n.skills[i % len(n.skills)]
        shift = SHIFTS[i % 3]
        worked = 0
        for d_i, d in enumerate(dates):
            if (d_i + i) % 7 in (5, 6) or d_i in n.days_off or worked >= n.max_shifts:
                continue
            rows.append({"nurse": n.nid, "name": n.name, "home": n.home, "date": d, "shift": shift, "unit": unit})
            worked += 1
    asg = pd.DataFrame(rows)
    sched = asg.groupby(["date", "shift", "unit"]).size()
    cov = demand.copy()
    cov["scheduled"] = [int(sched.get((r.date, r.shift, r.unit), 0)) for r in cov.itertuples()]
    cov["agency"] = 0
    cov["short"] = (cov["required"] - cov["scheduled"]).clip(lower=0)
    cost = (len(asg) * COST["regular"] + cov["short"].sum() * COST["short"]
            + (cov["scheduled"] - cov["required"]).clip(lower=0).sum() * COST["over"])
    return RosterResult(asg, cov, float(cost), "TEMPLATE", 0.0)


def required_from_forecast(fc: pd.DataFrame, level: str = "hi") -> pd.DataFrame:
    """Pivot champion census forecasts into date × unit at a risk level (yhat / hi)."""
    m = {"ed_census": "ED", "icu_census": "ICU", "med_census": "MED", "surg_census": "SURG"}
    f = fc[fc["champion"] & fc["target"].isin(m)]
    out = f.pivot(index="date", columns="target", values=level).rename(columns=m)
    return out[list(ALL_UNITS)].iloc[:7].apply(lambda s: s.map(lambda v: max(v, 0.0)))


def nurse_hours_short(coverage: pd.DataFrame) -> float:
    return float(coverage["short"].sum() * 8)

