"""The live digital-twin state: current hospital snapshot + near-term outlook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DEFAULT_CONFIG, UNIT_NAMES, HospitalConfig


@dataclass
class UnitState:
    code: str
    name: str
    beds: int
    occupied: int
    waiting: int = 0
    electives: int = 0

    @property
    def occupancy(self) -> float:
        return self.occupied / self.beds * 100 if self.beds else 0.0

    @property
    def free(self) -> int:
        return max(0, self.beds - self.occupied)


@dataclass
class HospitalState:
    time: pd.Timestamp
    units: dict[str, UnitState]
    ed_waiting: int
    ed_treatment: int
    ed_boarding: int
    lounge: int
    ot_busy: int
    doctors: int
    arrivals_24h: int
    arrivals_prev_24h: int
    esi_counts: dict[int, int]

    @property
    def status(self) -> str:
        icu = self.units["ICU"].occupancy
        wards = max(self.units["MED"].occupancy, self.units["SURG"].occupancy)
        if icu >= 97 or (wards >= 98 and self.ed_boarding >= 4) or self.ed_boarding >= 10:
            return "CRITICAL"
        if max(icu, wards) >= 90 or self.ed_boarding >= 4:
            return "STRAINED"
        return "STABLE"


def current_state(snapshot: dict, hourly: pd.DataFrame, cfg: HospitalConfig = DEFAULT_CONFIG) -> HospitalState:
    t = pd.Timestamp(snapshot["time"])
    inp = pd.DataFrame(snapshot["inpatients"])
    ed = pd.DataFrame(snapshot["ed"]) if snapshot["ed"] else pd.DataFrame(columns=["stage", "esi"])
    units = {}
    for u in ("ICU", "MED", "SURG"):
        sub = inp[inp["unit"] == u] if len(inp) else inp
        units[u] = UnitState(u, UNIT_NAMES[u], cfg.beds[u], len(sub),
                             electives=int(sub["elective"].sum()) if len(sub) else 0)
    stage = ed["stage"].value_counts().to_dict() if len(ed) else {}
    in_bed = stage.get("treatment", 0) + stage.get("boarding", 0)
    units = {"ED": UnitState("ED", UNIT_NAMES["ED"], cfg.beds["ED"], in_bed, stage.get("waiting", 0)), **units}
    h = hourly[hourly.index < t]
    a24 = int(h["arrivals"].iloc[-24:].sum())
    a48 = int(h["arrivals"].iloc[-48:-24].sum())
    esi = ed["esi"].value_counts().to_dict() if len(ed) else {}
    return HospitalState(
        time=t, units=units, ed_waiting=stage.get("waiting", 0), ed_treatment=stage.get("treatment", 0),
        ed_boarding=stage.get("boarding", 0), lounge=snapshot.get("lounge", 0),
        ot_busy=snapshot.get("ot_busy", 0), doctors=snapshot.get("doctors", 0),
        arrivals_24h=a24, arrivals_prev_24h=a48, esi_counts={int(k): int(v) for k, v in esi.items()},
    )


def shift_factors(hourly: pd.DataFrame) -> pd.DataFrame:
    """Mean census in each shift relative to the daily mean (unit × shift)."""
    from ..config import shift_of_hour

    cols = {"ED": "ed_occupied", "ICU": "ICU", "MED": "MED", "SURG": "SURG"}
    h = hourly.copy()
    h["shift"] = [shift_of_hour(x) for x in h.index.hour]
    h["day"] = h.index.normalize()
    out = {}
    for u, c in cols.items():
        dm = h.groupby("day")[c].transform("mean").replace(0, np.nan)
        out[u] = (h[c] / dm).groupby(h["shift"]).mean()
    return pd.DataFrame(out).T


def elective_census_by_dow(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily[["icu_elective", "surg_elective"]].copy()
    d["dow"] = d.index.dayofweek
    return d.groupby("dow").mean().rename(columns={"icu_elective": "ICU", "surg_elective": "SURG"})
