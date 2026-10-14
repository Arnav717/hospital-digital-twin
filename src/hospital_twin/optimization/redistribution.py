"""Patient redistribution as a min-cost transport LP (SciPy / HiGHS).

Excess demand on a unit (projected peak above staffed beds, plus current
boarders) is moved to clinically acceptable internal outlier beds or partner
hospitals, minimising clinical/transfer cost; unplaced patients are penalised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from ..config import PARTNER_HOSPITALS

# Internal outlier compatibility (source unit -> acceptable host unit, cost).
INTERNAL = {("MED", "SURG"): 1.0, ("SURG", "MED"): 1.5}
UNPLACED_PENALTY = 60.0


def plan_redistribution(excess: dict[str, float], spare: dict[str, float],
                        partners: list[dict] = PARTNER_HOSPITALS) -> pd.DataFrame:
    """excess: unit -> patients to place; spare: internal unit -> free beds."""
    srcs = [u for u, v in excess.items() if v > 0.5]
    if not srcs:
        return pd.DataFrame(columns=["from", "to", "patients", "cost_per_patient", "km", "lat", "lon"])
    arcs = []  # (src, dest_label, capacity_key, cost, km, lat, lon)
    for s in srcs:
        for (a, b), c in INTERNAL.items():
            if a == s and spare.get(b, 0) > 0:
                arcs.append((s, f"Outlier bed — {b}", ("int", b), c, 0.0, None, None))
        for p in partners:
            if p["free"].get(s, 0) > 0:
                cost = 4.0 + p["km"] / 4 + (6.0 if s == "ICU" else 0.0)
                arcs.append((s, p["name"], ("ext", p["name"], s), cost, p["km"], p["lat"], p["lon"]))
        arcs.append((s, "Unplaced (escalate)", ("none", s), UNPLACED_PENALTY, 0.0, None, None))
    n = len(arcs)
    c = np.array([a[3] for a in arcs])
    A_eq = np.zeros((len(srcs), n))
    b_eq = np.array([float(excess[s]) for s in srcs])
    for j, a in enumerate(arcs):
        A_eq[srcs.index(a[0]), j] = 1
    caps = {}
    for j, a in enumerate(arcs):
        key = a[2]
        if key[0] == "int":
            caps.setdefault(key, (float(spare[key[1]]), []))[1].append(j)
        elif key[0] == "ext":
            p = next(p for p in partners if p["name"] == key[1])
            caps.setdefault(key, (float(p["free"][key[2]]), []))[1].append(j)
    A_ub = np.zeros((len(caps), n))
    b_ub = np.zeros(len(caps))
    for i, (cap, js) in enumerate(caps.values()):
        A_ub[i, js] = 1
        b_ub[i] = cap
    res = linprog(c, A_ub=A_ub if len(caps) else None, b_ub=b_ub if len(caps) else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n, method="highs")
    rows = []
    for j, a in enumerate(arcs):
        v = res.x[j] if res.success else 0.0
        if v > 1e-6:
            rows.append({"from": a[0], "to": a[1], "patients": round(float(v), 1), "cost_per_patient": a[3],
                         "km": a[4], "lat": a[5], "lon": a[6]})
    return pd.DataFrame(rows)


def excess_from_projection(ts_p90: pd.DataFrame, beds: dict, hours: int = 48) -> tuple[dict, dict]:
    """Excess demand and spare beds over the next `hours` from a twin projection."""
    w = ts_p90.iloc[:hours]
    excess, spare = {}, {}
    for u in ("ICU", "MED", "SURG"):
        demand_peak = (w[u] + w[f"{u}_queue"]).max()
        excess[u] = max(0.0, demand_peak - beds[u] * 0.97)
        spare[u] = max(0.0, beds[u] * 0.95 - w[u].max())
    return excess, spare
