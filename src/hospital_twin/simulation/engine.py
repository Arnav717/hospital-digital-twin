"""Discrete-event simulation of patient flow through the hospital (SimPy).

Flow modelled:

    arrival ─► triage ─► ED bed ─► doctor ─► work-up ─► disposition
                                                         ├─ home
                                                         ├─ board ─► MED / SURG / ICU (outlier beds after a wait)
                                                         └─ emergency OT ─► ICU / SURG
    elective list ─► bed check ─► OT ─► SURG / ICU
    ICU ─► step-down ward (holds the ICU bed while waiting = exit block) ─► discharge
    ward ─► discharge window ─► discharge lounge / transport ─► home

The same engine generates the synthetic "historical" hospital record and runs
the forward-looking what-if simulations, so the twin is calibrated by design.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import simpy
from scipy.special import ndtr, ndtri

from ..config import DEFAULT_CONFIG, HospitalConfig, LogNormal, shift_of_hour
from .scenario import Scenario

DISPOSITIONS = ("HOME", "MED", "SURG", "ICU", "OT")
WARDS = ("MED", "SURG")

TS_COLUMNS = [
    "t", "ed_occupied", "ed_waiting", "ed_boarding", "ICU", "MED", "SURG",
    "ICU_queue", "MED_queue", "SURG_queue", "ot_busy", "lounge", "doctors",
    "arrivals", "ICU_elective", "SURG_elective", "icu_blocked",
]


class Patient:
    __slots__ = (
        "pid", "esi", "ambulance", "elective", "surgical", "arrival", "triage_end",
        "ed_bed", "doc_start", "dispo_time", "ed_depart", "dispo", "unit", "unit_in",
        "icu_in", "icu_out", "discharge", "outcome", "loc", "kind", "since", "procedure", "plan",
    )

    def __init__(self, pid: int, arrival: float, esi: int = 0, ambulance: bool = False,
                 elective: bool = False):
        self.pid = pid
        self.esi = esi
        self.ambulance = ambulance
        self.elective = elective
        self.surgical = elective
        self.arrival = arrival
        self.triage_end = self.ed_bed = self.doc_start = self.dispo_time = None
        self.ed_depart = self.unit_in = self.icu_in = self.icu_out = self.discharge = None
        self.dispo = self.unit = self.outcome = self.procedure = self.plan = None
        self.loc = "ED_WAIT"
        self.kind = None
        self.since = arrival

    def record(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__ if k not in ("loc", "since", "plan")}


class Sampler:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def ln(self, d: LogNormal, mult: float = 1.0) -> float:
        return float(self.rng.lognormal(d.mu, d.sigma)) * mult

    def ln_residual(self, d: LogNormal, elapsed: float, mult: float = 1.0) -> float:
        """Remaining time given `elapsed` time already spent (truncated log-normal)."""
        if elapsed <= 0:
            return self.ln(d, mult)
        mu = d.mu + math.log(mult)
        lo = float(ndtr((math.log(elapsed) - mu) / d.sigma))
        u = min(lo + (1.0 - lo) * self.rng.random(), 1 - 1e-9)
        total = math.exp(mu + d.sigma * float(ndtri(u)))
        return max(total - elapsed, 0.25)


@dataclass
class SimResult:
    timeseries: pd.DataFrame
    kpis: dict
    flows: Counter
    patients: list[dict] = field(default_factory=list)
    snapshot: dict | None = None


class HospitalSim:
    def __init__(
        self,
        *,
        hourly_rates: np.ndarray,
        start: pd.Timestamp,
        scenario: Scenario | None = None,
        cfg: HospitalConfig = DEFAULT_CONFIG,
        seed: int = 0,
        warm: dict | None = None,
        holidays: set | frozenset = frozenset(),
        record: bool = False,
        surge_hours: list[tuple[float, int, float]] | None = None,
        elective_rate: np.ndarray | None = None,
    ):
        self.cfg = cfg
        self.sc = scenario or Scenario()
        self.start = pd.Timestamp(start).floor("h")
        self.start_hour = self.start.hour
        self.horizon = float(len(hourly_rates))
        self.rng = np.random.default_rng(seed)
        self.s = Sampler(self.rng)
        self.record = record
        self.holidays = holidays
        self.elective_rate = elective_rate
        self.env = simpy.Environment()
        env = self.env

        beds = {u: max(1, cfg.beds[u] + self.sc.bed_delta.get(u, 0)) for u in cfg.beds}
        self.beds = beds
        self.ed = simpy.PriorityResource(env, beds["ED"])
        self.triage = simpy.Resource(env, cfg.triage_nurses)
        self.units = {u: simpy.Resource(env, beds[u]) for u in ("ICU", "MED", "SURG")}
        self.ot = simpy.PriorityResource(env, cfg.theatres)
        self.lounge = simpy.Resource(env, max(0, cfg.lounge_beds + self.sc.lounge_delta) or 1)
        self.lounge_cap = max(0, cfg.lounge_beds + self.sc.lounge_delta)
        self.doc_schedule = {
            sh: max(1, n + self.sc.doctor_delta.get(sh, 0)) for sh, n in cfg.ed_doctors.items()
        }
        self.doctors = simpy.PriorityResource(env, max(self.doc_schedule.values()))
        self._doc_holds: list = []
        self.docs_on_duty = 0

        self.disp_cum = {k: np.cumsum(v) for k, v in cfg.disposition.items()}
        self.window = (8, 18) if self.sc.early_discharge else cfg.discharge_window
        self.transport_mult = 0.6 if self.sc.early_discharge else 1.0
        self.los_mult = self.sc.los_multiplier

        # State / statistics
        self._pid = 0
        self.active: dict[int, Patient] = {}
        self.patients: list[Patient] = []
        self.flows: Counter = Counter()
        self.boarding = 0
        self.icu_blocked = 0
        self.arr_hour = 0
        self.elective_census = {"ICU": 0, "SURG": 0, "MED": 0}
        self.door_to_doc: list[float] = []
        self.ed_los: list[float] = []
        self.boarding_times: list[float] = []
        self.counts: Counter = Counter()
        self.icu_block_hours = 0.0
        self.ts: list[tuple] = []

        self._build_arrivals(hourly_rates, surge_hours)
        if warm:
            self._warm_start(warm)
        env.process(self._arrival_process())
        env.process(self._doctor_controller())
        env.process(self._elective_process())
        env.process(self._monitor())

    # ------------------------------------------------------------------ helpers
    def _new_patient(self, **kw) -> Patient:
        self._pid += 1
        p = Patient(self._pid, self.env.now, **kw)
        self.active[p.pid] = p
        if self.record:
            self.patients.append(p)
        return p

    def _finish(self, p: Patient, outcome: str):
        p.outcome = outcome
        p.discharge = self.env.now
        self.active.pop(p.pid, None)

    def _local_hour(self, t: float | None = None) -> float:
        t = self.env.now if t is None else t
        return (self.start_hour + t) % 24

    def _day_index(self, t: float | None = None) -> int:
        t = self.env.now if t is None else t
        return int((self.start_hour + t) // 24)

    def _date_of_day(self, d: int) -> pd.Timestamp:
        return self.start.normalize() + pd.Timedelta(days=d)

    def _window_delay(self) -> float:
        lo, hi = self.window
        h = self._local_hour()
        if lo <= h < hi:
            return 0.0
        wait = (lo - h) % 24
        return wait + self.rng.random() * (hi - lo) * 0.6

    def _drop(self, res: simpy.Resource, req):
        if req.triggered:
            res.release(req)
        else:
            req.cancel()

    def _move(self, p: Patient, loc: str, kind: str | None = None):
        p.loc = loc
        p.since = self.env.now
        if kind:
            p.kind = kind

    # ------------------------------------------------------------------ arrivals
    def _build_arrivals(self, rates: np.ndarray, surge_hours):
        cfg, rng, sc = self.cfg, self.rng, self.sc
        rates = np.asarray(rates, dtype=float) * sc.arrival_multiplier
        counts = rng.poisson(np.clip(rates, 0, None))
        hours = np.repeat(np.arange(len(rates)), counts)
        times = hours + rng.random(hours.size)
        esi = rng.choice(5, size=hours.size, p=cfg.esi_mix) + 1
        surges = list(surge_hours or []) + [(s.hour, s.patients, s.duration) for s in sc.surges]
        surge_flag = np.zeros(hours.size, dtype=bool)
        for hour, n, dur in surges:
            st = hour + rng.random(n) * dur
            times = np.concatenate([times, st])
            esi = np.concatenate([esi, rng.choice(5, size=n, p=cfg.mass_casualty_mix) + 1])
            surge_flag = np.concatenate([surge_flag, np.ones(n, dtype=bool)])
        order = np.argsort(times, kind="stable")
        self.arr_times = times[order]
        self.arr_esi = esi[order]
        amb_p = np.asarray(cfg.ambulance_prob)[self.arr_esi - 1]
        self.arr_amb = (rng.random(self.arr_times.size) < amb_p) | surge_flag[order]

    def _arrival_process(self):
        env, rng = self.env, self.rng
        divert = self.sc.divert_frac
        for t, esi, amb in zip(self.arr_times, self.arr_esi, self.arr_amb):
            dt = t - env.now
            if dt > 0:
                yield env.timeout(dt)
            esi = int(esi)
            self.arr_hour += 1
            self.counts["arrivals"] += 1
            src = "Ambulance" if amb else "Walk-in"
            if amb and divert > 0 and esi in (2, 3) and rng.random() < divert:
                self.counts["diverted"] += 1
                self.flows[(src, "Diverted to partner")] += 1
                continue
            self.flows[(src, f"ESI {esi}")] += 1
            env.process(self._ed_patient(esi, bool(amb)))

    # ------------------------------------------------------------------ ED
    def _ed_patient(self, esi: int, amb: bool, skip_triage: bool = False, arrival: float | None = None):
        env, cfg, s = self.env, self.cfg, self.s
        p = self._new_patient(esi=esi, ambulance=amb)
        if arrival is not None:
            p.arrival = arrival
        if esi > 1 and not skip_triage:
            with self.triage.request() as r:
                yield r
                yield env.timeout(s.ln(cfg.triage_time))
        p.triage_end = env.now
        bed = self.ed.request(priority=esi)
        patience = cfg.lwbs_patience[esi - 1]
        if patience < 1e8:
            res = yield bed | env.timeout(self.rng.exponential(patience))
            if bed not in res:
                bed.cancel()
                self.counts["lwbs"] += 1
                self.flows[(f"ESI {esi}", "Left without being seen")] += 1
                self._finish(p, "lwbs")
                return
        else:
            yield bed
        p.ed_bed = env.now
        self._move(p, "ED_TREAT")
        doc = self.doctors.request(priority=esi)
        yield doc
        p.doc_start = env.now
        self.door_to_doc.append(env.now - p.arrival)
        yield env.timeout(s.ln(cfg.doctor_time[esi - 1]))
        self.doctors.release(doc)
        yield env.timeout(s.ln(cfg.ed_workup[esi - 1]))
        dispo = DISPOSITIONS[int(np.searchsorted(self.disp_cum[esi], self.rng.random(), side="right"))]
        yield from self._ed_dispose(p, bed, dispo)

    def _ed_dispose(self, p: Patient, bed, dispo: str):
        env, cfg, s = self.env, self.cfg, self.s
        p.dispo = dispo
        p.dispo_time = env.now
        label = {"HOME": "Discharged from ED", "MED": "Medical Ward", "SURG": "Surgical Ward",
                 "ICU": "ICU", "OT": "Emergency Surgery"}[dispo]
        self.flows[(f"ESI {p.esi}", label)] += 1
        if dispo == "HOME":
            self.ed.release(bed)
            p.ed_depart = env.now
            self.ed_los.append(env.now - p.arrival)
            self._finish(p, "home")
            return
        self.counts[f"admit_{dispo}"] += 1
        self._move(p, "ED_BOARD")
        self.boarding += 1
        t0 = env.now
        if dispo == "OT":
            p.surgical = True
            ot = self.ot.request(priority=0)
            yield ot
            self._board_done(p, bed, t0)
            yield env.timeout(s.ln(cfg.emergency_surgery_time))
            self.ot.release(ot)
            icu = self.rng.random() < cfg.emergency_postop_icu_prob
            dest, kind = ("ICU", "ICU_POSTOP") if icu else ("SURG", "SURG")
            self.flows[("Emergency Surgery", "ICU" if icu else "Surgical Ward")] += 1
            unit, req = yield from self._acquire(dest, overflow=not icu)
            yield from self._inpatient(p, unit, req, kind)
            return
        if dispo == "SURG":
            p.surgical = True
        unit, req = yield from self._acquire(dispo, overflow=dispo != "ICU")
        self._board_done(p, bed, t0)
        yield from self._inpatient(p, unit, req, dispo)

    def _board_done(self, p: Patient, bed, t0: float):
        self.boarding -= 1
        self.boarding_times.append(self.env.now - t0)
        self.ed.release(bed)
        p.ed_depart = self.env.now
        self.ed_los.append(self.env.now - p.arrival)

    # ------------------------------------------------------------------ inpatient
    def _acquire(self, unit: str, overflow: bool):
        res_unit = self.units[unit]
        req = res_unit.request()
        if not overflow or unit == "ICU":
            yield req
            return unit, req
        res = yield req | self.env.timeout(self.cfg.overflow_after)
        if req in res:
            return unit, req
        alt = "SURG" if unit == "MED" else "MED"
        req2 = self.units[alt].request()
        res = yield req | req2
        if req in res:
            self._drop(self.units[alt], req2)
            return unit, req
        self._drop(res_unit, req)
        self.counts["outliers"] += 1
        return alt, req2

    def _inpatient(self, p: Patient, unit: str, req, kind: str, los: float | None = None):
        env, cfg, s = self.env, self.cfg, self.s
        p.unit = p.unit or unit
        p.unit_in = p.unit_in if p.unit_in is not None else env.now
        self._move(p, unit, kind)
        if p.elective:
            self.elective_census[unit] += 1
        if unit == "ICU":
            p.icu_in = env.now
        if los is None:
            if p.plan is not None and kind in ("ICU_POSTOP", "SURG_ELECTIVE", "STEPDOWN"):
                days = p.plan[0] if kind == "ICU_POSTOP" else p.plan[1]
                los = s.ln(LogNormal(max(days, 0.3) * 24, 0.5), self.los_mult)
            else:
                los = s.ln(cfg.los_hours[kind], self.los_mult)
        yield env.timeout(los)
        if unit == "ICU":
            p.icu_out = env.now
            stepdown = (p.plan[1] > 0) if p.plan is not None else self.rng.random() < cfg.icu_stepdown_prob
            if stepdown:
                ward = "SURG" if p.surgical else "MED"
                self.icu_blocked += 1
                t0 = env.now
                p.loc = "ICU_BLOCK"
                w, wreq = yield from self._acquire(ward, overflow=True)
                self.icu_blocked -= 1
                self.icu_block_hours += env.now - t0
                self._leave(p, "ICU")
                self.units["ICU"].release(req)
                self.flows[("ICU", "Medical Ward" if w == "MED" else "Surgical Ward")] += 1
                yield from self._inpatient(p, w, wreq, "STEPDOWN")
                return
            self.flows[("ICU", "Discharged home")] += 1
            self._leave(p, "ICU")
            self.units["ICU"].release(req)
            self._finish(p, "home")
            return
        yield env.timeout(self._window_delay())
        yield from self._discharge(p, unit, req)

    def _leave(self, p: Patient, unit: str):
        if p.elective:
            self.elective_census[unit] -= 1

    def _discharge(self, p: Patient, unit: str, req):
        env = self.env
        tt = self.s.ln(self.cfg.transport_time, self.transport_mult)
        label = "Medical Ward" if unit == "MED" else "Surgical Ward"
        if self.lounge_cap and self.lounge.count < self.lounge_cap:
            lr = self.lounge.request()
            yield lr
            self._leave(p, unit)
            self.units[unit].release(req)
            self.flows[(label, "Discharge Lounge")] += 1
            self.flows[("Discharge Lounge", "Discharged home")] += 1
            self._move(p, "LOUNGE")
            yield env.timeout(tt)
            self.lounge.release(lr)
        else:
            yield env.timeout(tt)
            self._leave(p, unit)
            self.units[unit].release(req)
            self.flows[(label, "Discharged home")] += 1
        self._finish(p, "home")

    # ------------------------------------------------------------------ electives
    def _elective_cases(self, d: int) -> list:
        """Cases for day d: [icu_days, ward_days, theatre_hours] from a plan, or None (random case-mix)."""
        if self.sc.elective_plan is not None:
            return list(self.sc.elective_plan[d]) if d < len(self.sc.elective_plan) else []
        return [None] * self._elective_count(d)

    def _elective_count(self, d: int) -> int:
        date = self._date_of_day(d)
        if date.dayofweek >= 5 or date.date() in self.holidays:
            return 0
        mean = self.cfg.elective_daily_mean
        if self.elective_rate is not None and d < len(self.elective_rate):
            mean = float(self.elective_rate[d])
        return int(self.rng.poisson(mean))

    def _elective_process(self):
        env = self.env
        yield env.timeout((7 - self.start_hour) % 24)
        while True:
            cases = self._elective_cases(self._day_index())
            n = len(cases)
            keep = self.rng.random(n) >= self.sc.elective_postpone_frac
            postponed = int(n - keep.sum())
            self.counts["electives_listed"] += n
            self.counts["electives_postponed"] += postponed
            if postponed:
                self.flows[("Elective list", "Postponed")] += postponed
            for case, k in zip(cases, keep):
                if k:
                    env.process(self._elective(case))
            yield env.timeout(24)

    def _elective(self, case=None):
        env, cfg, s = self.env, self.cfg, self.s
        if case is None:
            icu = self.rng.random() < cfg.elective_postop_icu_prob
            day_case = False
        else:
            icu = case[0] > 0
            day_case = case[0] <= 0 and case[1] <= 0
        unit = "ICU" if icu else "SURG"
        r = self.units[unit]
        if not day_case and r.capacity - r.count - len(r.queue) <= cfg.elective_bed_reserve:
            self.counts["electives_cancelled"] += 1
            self.flows[("Elective list", "Cancelled — no bed")] += 1
            return
        p = self._new_patient(elective=True)
        p.plan = case
        p.loc = "OT_WAIT"
        day = self._day_index()
        ot = self.ot.request(priority=1)
        yield ot
        if self._local_hour() >= 18 or self._day_index() != day:
            self.ot.release(ot)
            self.counts["electives_cancelled"] += 1
            self.flows[("Elective list", "Cancelled — theatre overrun")] += 1
            self._finish(p, "cancelled")
            return
        self.flows[("Elective list", "Elective Surgery")] += 1
        self._move(p, "OT")
        dur = s.ln(LogNormal(case[2], 0.25)) if case is not None else s.ln(cfg.surgery_time)
        yield env.timeout(dur)
        self.ot.release(ot)
        self.counts["electives_done"] += 1
        if day_case:
            self.flows[("Elective Surgery", "Day case — home")] += 1
            self._finish(p, "home")
            return
        self.flows[("Elective Surgery", "ICU" if icu else "Surgical Ward")] += 1
        u, req = yield from self._acquire(unit, overflow=not icu)
        yield from self._inpatient(p, u, req, "ICU_POSTOP" if icu else "SURG_ELECTIVE")

    # ------------------------------------------------------------------ staffing
    def _doctor_controller(self):
        env = self.env
        maxd = self.doctors.capacity
        while True:
            want = self.doc_schedule[shift_of_hour(int(self._local_hour()))]
            hold = maxd - want
            while len(self._doc_holds) < hold:
                self._doc_holds.append(self.doctors.request(priority=-1))
            while len(self._doc_holds) > hold:
                self._drop(self.doctors, self._doc_holds.pop())
            self.docs_on_duty = want
            yield env.timeout(1.0)

    # ------------------------------------------------------------------ warm start
    def _warm_start(self, warm: dict):
        for w in warm.get("inpatients", []):
            self.env.process(self._warm_inpatient(w))
        for w in warm.get("ed", []):
            self.env.process(self._warm_ed(w))

    def _warm_inpatient(self, w: dict):
        p = self._new_patient(elective=bool(w.get("elective")))
        p.surgical = bool(w.get("surgical"))
        p.arrival = -float(w.get("elapsed", 0))
        unit, kind = w["unit"], w["kind"]
        req = self.units[unit].request()
        yield req
        los = self.s.ln_residual(self.cfg.los_hours[kind], float(w.get("elapsed", 0)), self.los_mult)
        yield from self._inpatient(p, unit, req, kind, los=los)

    def _warm_ed(self, w: dict):
        stage, esi = w["stage"], int(w["esi"])
        arrival = -float(w.get("elapsed", 0))
        if stage == "waiting":
            yield from self._ed_patient(esi, False, skip_triage=True, arrival=arrival)
            return
        p = self._new_patient(esi=esi)
        p.arrival = arrival
        bed = self.ed.request(priority=esi)
        yield bed
        if stage == "treatment":
            yield self.env.timeout(self.s.ln(self.cfg.ed_workup[esi - 1]) * 0.5)
            dispo = DISPOSITIONS[int(np.searchsorted(self.disp_cum[esi], self.rng.random(), side="right"))]
        else:
            dispo = w.get("dest") or "MED"
        yield from self._ed_dispose(p, bed, dispo)

    # ------------------------------------------------------------------ monitor
    def _monitor(self):
        env = self.env
        yield env.timeout(1e-6)
        while True:
            u = self.units
            self.ts.append((
                env.now, self.ed.count,
                len(self.ed.queue) + len(self.triage.queue) + self.triage.count,
                self.boarding, u["ICU"].count, u["MED"].count, u["SURG"].count,
                len(u["ICU"].queue), len(u["MED"].queue), len(u["SURG"].queue),
                self.ot.count, self.lounge.count if self.lounge_cap else 0, self.docs_on_duty,
                self.arr_hour, self.elective_census["ICU"], self.elective_census["SURG"],
                self.icu_blocked,
            ))
            self.arr_hour = 0
            yield env.timeout(1.0)

    # ------------------------------------------------------------------ run
    def run(self, snapshot: bool = False) -> SimResult:
        self.env.run(until=self.horizon)
        ts = pd.DataFrame(self.ts, columns=TS_COLUMNS)
        ts["t"] = ts["t"].round().astype(int)
        # arrivals were counted over the hour *preceding* each sample; re-label to the hour they occurred in
        ts["arrivals"] = ts["arrivals"].shift(-1, fill_value=0)
        ts["time"] = self.start + pd.to_timedelta(ts["t"], unit="h")
        return SimResult(
            timeseries=ts,
            kpis=self._kpis(ts),
            flows=self.flows,
            patients=[p.record() for p in self.patients] if self.record else [],
            snapshot=self._snapshot() if snapshot else None,
        )

    def _kpis(self, ts: pd.DataFrame) -> dict:
        def stat(xs, fn, scale=1.0):
            return float(fn(np.asarray(xs)) * scale) if len(xs) else 0.0

        c, beds = self.counts, self.beds
        seen = max(1, c["arrivals"] - c["diverted"])
        k = {
            "arrivals": c["arrivals"],
            "diverted": c["diverted"],
            "lwbs": c["lwbs"],
            "lwbs_rate": c["lwbs"] / seen * 100,
            "door_to_doc_mean_min": stat(self.door_to_doc, np.mean, 60),
            "door_to_doc_p90_min": stat(self.door_to_doc, lambda a: np.percentile(a, 90), 60),
            "ed_los_mean_h": stat(self.ed_los, np.mean),
            "ed_los_p90_h": stat(self.ed_los, lambda a: np.percentile(a, 90)),
            "boarding_mean_h": stat(self.boarding_times, np.mean),
            "boarding_p90_h": stat(self.boarding_times, lambda a: np.percentile(a, 90)),
            "ed_waiting_peak": float(ts["ed_waiting"].max()) if len(ts) else 0.0,
            "outliers": c["outliers"],
            "icu_exit_block_h": self.icu_block_hours,
            "electives_listed": c["electives_listed"],
            "electives_done": c["electives_done"],
            "electives_cancelled": c["electives_cancelled"],
            "electives_postponed": c["electives_postponed"],
        }
        for u in ("ICU", "MED", "SURG"):
            occ = ts[u] / beds[u] * 100 if len(ts) else pd.Series([0.0])
            k[f"{u}_occ_mean"] = float(occ.mean())
            k[f"{u}_occ_peak"] = float(occ.max())
            k[f"{u}_full_hours"] = float(((ts[u] >= beds[u]) & (ts[f"{u}_queue"] > 0)).sum()) if len(ts) else 0.0
            k[f"admit_{u}"] = c[f"admit_{u}"]
        k["ed_occ_mean"] = float((ts["ed_occupied"] / beds["ED"] * 100).mean()) if len(ts) else 0.0
        return k

    def _snapshot(self) -> dict:
        now = self.env.now
        inpatients, ed = [], []
        for p in self.active.values():
            if p.loc in ("ICU", "MED", "SURG", "ICU_BLOCK"):
                unit = "ICU" if p.loc == "ICU_BLOCK" else p.loc
                inpatients.append({
                    "unit": unit, "kind": p.kind or unit, "elapsed": round(now - p.since, 2),
                    "elective": p.elective, "surgical": p.surgical, "esi": p.esi,
                })
            elif p.loc in ("ED_WAIT", "ED_TREAT", "ED_BOARD"):
                stage = {"ED_WAIT": "waiting", "ED_TREAT": "treatment", "ED_BOARD": "boarding"}[p.loc]
                ed.append({"esi": p.esi, "stage": stage, "dest": p.dispo if p.dispo in ("MED", "SURG", "ICU", "OT") else None,
                           "elapsed": round(now - p.arrival, 2)})
        return {
            "time": (self.start + pd.Timedelta(hours=now)).isoformat(),
            "inpatients": inpatients,
            "ed": ed,
            "lounge": self.lounge.count if self.lounge_cap else 0,
            "ot_busy": self.ot.count,
            "doctors": self.docs_on_duty,
        }
