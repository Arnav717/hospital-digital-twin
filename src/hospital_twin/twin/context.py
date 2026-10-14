"""Loads the built twin (history, forecasts, snapshot) and runs projections on it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from ..config import ARTIFACT_DIR, DATA_DIR, DEFAULT_CONFIG, HospitalConfig
from ..data.generator import india_holidays, load_history
from ..forecasting.pipeline import champion_forecast
from ..simulation.runner import MCResult, run_monte_carlo
from ..simulation.scenario import Scenario
from .state import HospitalState, current_state, elective_census_by_dow, shift_factors


@dataclass
class TwinContext:
    hist: dict
    forecast: pd.DataFrame
    forecast_info: dict
    cfg: HospitalConfig = DEFAULT_CONFIG

    @property
    def snapshot(self) -> dict:
        return self.hist["snapshot"]

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.snapshot["time"])

    @cached_property
    def state(self) -> HospitalState:
        return current_state(self.snapshot, self.hist["hourly"], self.cfg)

    @cached_property
    def holidays(self) -> set:
        return set(india_holidays([self.start.year, self.start.year + 1]))

    @cached_property
    def shift_factors(self) -> pd.DataFrame:
        return shift_factors(self.hist["hourly"])

    @cached_property
    def elective_dow(self) -> pd.DataFrame:
        return elective_census_by_dow(self.hist["daily"])

    def champion(self, target: str) -> pd.DataFrame:
        return champion_forecast(self.forecast, target)

    def census_forecast(self, level: str = "yhat") -> pd.DataFrame:
        m = {"ed_census": "ED", "icu_census": "ICU", "med_census": "MED", "surg_census": "SURG"}
        f = self.forecast[self.forecast["champion"] & self.forecast["target"].isin(m)]
        return f.pivot(index="date", columns="target", values=level).rename(columns=m)[list(m.values())]

    def simulate(self, scenario: Scenario | None = None, reps: int = 24, horizon_h: int = 168,
                 seed: int = 0, n_jobs: int | None = None) -> MCResult:
        return run_monte_carlo(
            scenario or Scenario(), forecast=self.champion("ed_arrivals"), profile=self.hist["profile"],
            start=self.start, warm=self.snapshot, horizon_h=horizon_h, reps=reps, seed=seed,
            cfg=self.cfg, holidays=self.holidays, n_jobs=n_jobs,
        )


def load_context() -> TwinContext:
    hist = load_history(DATA_DIR)
    fc = pd.read_parquet(ARTIFACT_DIR / "forecast.parquet")
    info = json.loads((ARTIFACT_DIR / "forecast_metrics.json").read_text())
    return TwinContext(hist, fc, info)
