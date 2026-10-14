"""Monte-Carlo execution of forward-looking twin simulations."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config import DEFAULT_CONFIG, HospitalConfig
from .engine import HospitalSim
from .scenario import Scenario

Z90 = 1.2816


@dataclass
class MCResult:
    scenario: Scenario
    kpis: pd.DataFrame  # one row per replication
    ts_mean: pd.DataFrame
    ts_p10: pd.DataFrame
    ts_p90: pd.DataFrame
    sample_ts: pd.DataFrame  # one representative replication (for animation)
    flows: dict
    beds: dict

    def kpi_summary(self) -> pd.DataFrame:
        k = self.kpis
        return pd.DataFrame({
            "mean": k.mean(), "p10": k.quantile(0.1), "p90": k.quantile(0.9),
        })


def forecast_hourly_rates(
    fc: pd.DataFrame, profile: np.ndarray, start: pd.Timestamp, horizon_h: int, z: float = 0.0
) -> np.ndarray:
    """Hourly arrival rates from a daily forecast with columns yhat, lo, hi.

    `z` draws a coherent demand path from the forecast's 80% interval (z=0 → point forecast).
    """
    start = pd.Timestamp(start).floor("h")
    idx = pd.date_range(start, periods=horizon_h, freq="h")
    day = idx.normalize()
    f = fc.reindex(day.unique()).ffill().bfill()
    if z >= 0:
        daily = f["yhat"] + z * (f["hi"] - f["yhat"]) / Z90
    else:
        daily = f["yhat"] + z * (f["yhat"] - f["lo"]) / Z90
    daily = daily.clip(lower=1).reindex(day).to_numpy()
    return daily * profile[idx.dayofweek, idx.hour]


def _one(scenario, fc, profile, start, horizon_h, warm, seed, cfg, holidays):
    rng = np.random.default_rng(10_000 + seed)
    z = float(np.clip(rng.normal(), -2.5, 2.5))
    rates = forecast_hourly_rates(fc, profile, start, horizon_h, z)
    sim = HospitalSim(hourly_rates=rates, start=start, scenario=scenario, cfg=cfg,
                      seed=seed, warm=warm, holidays=holidays)
    r = sim.run()
    return r.kpis, r.timeseries, dict(r.flows), sim.beds


def run_monte_carlo(
    scenario: Scenario,
    *,
    forecast: pd.DataFrame,
    profile: np.ndarray,
    start: pd.Timestamp,
    warm: dict | None,
    horizon_h: int = 168,
    reps: int = 24,
    seed: int = 0,
    cfg: HospitalConfig = DEFAULT_CONFIG,
    holidays: set | frozenset = frozenset(),
    n_jobs: int | None = None,
) -> MCResult:
    """Run `reps` replications. Seeds are shared across scenarios (common random numbers)."""
    n_jobs = n_jobs or min(reps, os.cpu_count() or 1)
    args = (scenario, forecast, profile, pd.Timestamp(start), horizon_h, warm)
    if n_jobs == 1:
        out = [_one(*args, seed + i, cfg, holidays) for i in range(reps)]
    else:
        out = Parallel(n_jobs=n_jobs)(
            delayed(_one)(*args, seed + i, cfg, holidays) for i in range(reps)
        )
    kpis = pd.DataFrame([o[0] for o in out])
    stack = pd.concat([o[1].assign(rep=i) for i, o in enumerate(out)])
    num = stack.drop(columns=["time"]).groupby("t")
    times = out[0][1].set_index("t")["time"]
    ts_mean = num.mean().drop(columns="rep").assign(time=times)
    ts_p10 = num.quantile(0.1).drop(columns="rep").assign(time=times)
    ts_p90 = num.quantile(0.9).drop(columns="rep").assign(time=times)
    flows: Counter = Counter()
    for o in out:
        flows.update(o[2])
    flows = {k: v / reps for k, v in flows.items()}
    # representative replication = the one whose ICU+ward load is closest to the median
    load = [o[1][["ICU", "MED", "SURG"]].sum().sum() for o in out]
    rep_i = int(np.argsort(load)[len(load) // 2])
    return MCResult(scenario, kpis, ts_mean, ts_p10, ts_p90, out[rep_i][1], flows, out[0][3])


def compare(base: MCResult, alt: MCResult, metrics: list[str]) -> pd.DataFrame:
    """Paired comparison (common random numbers) of KPI means with a 90% interval on the delta."""
    rows = []
    for m in metrics:
        d = alt.kpis[m].to_numpy() - base.kpis[m].to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        rows.append({
            "metric": m, "baseline": base.kpis[m].mean(), "scenario": alt.kpis[m].mean(),
            "delta": d.mean(), "ci_lo": d.mean() - 1.645 * se, "ci_hi": d.mean() + 1.645 * se,
        })
    return pd.DataFrame(rows)
