"""Quantitative evaluation of every layer of the twin → reports/evaluation.md."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import ARTIFACT_DIR, REPORT_DIR
from .recommender.engine import METRIC_LABELS, generate_recommendations
from .simulation.engine import HospitalSim
from .simulation.scenario import preset_scenarios
from .twin.context import load_context

VALIDATION_DAYS = 56
BURN_IN = 21


def _replay(rates: np.ndarray, start: pd.Timestamp, seed: int, holidays: set) -> pd.DataFrame:
    sim = HospitalSim(hourly_rates=rates, start=start, seed=seed, holidays=holidays)
    ts = sim.run().timeseries
    ts = ts[ts["time"] >= start + pd.Timedelta(days=BURN_IN)]
    return ts.set_index("time")[["ed_occupied", "ICU", "MED", "SURG", "arrivals"]]


def validate_twin(ctx, reps: int = 10) -> pd.DataFrame:
    """Replay the last weeks of observed demand through the twin and compare daily census."""
    hourly = ctx.hist["hourly"]
    end = hourly.index[-1].normalize()
    start = end - pd.Timedelta(days=VALIDATION_DAYS + BURN_IN)
    rates = hourly.loc[start: end - pd.Timedelta(hours=1), "rate"].to_numpy()
    runs = Parallel(n_jobs=-1)(delayed(_replay)(rates, start, 100 + i, ctx.holidays) for i in range(reps))
    sim = sum(r.groupby(r.index.normalize()).mean() for r in runs) / reps
    obs = hourly.loc[start + pd.Timedelta(days=BURN_IN): end - pd.Timedelta(hours=1),
                     ["ed_occupied", "ICU", "MED", "SURG", "arrivals"]]
    obs = obs.groupby(obs.index.normalize()).mean()
    rows = []
    for c, name in [("arrivals", "ED arrivals / h"), ("ed_occupied", "ED census"), ("ICU", "ICU census"),
                    ("MED", "Medical ward census"), ("SURG", "Surgical ward census")]:
        o = obs[c]
        s = sim[c].reindex(o.index)
        rows.append({"series": name, "observed_mean": o.mean(), "twin_mean": s.mean(),
                     "MAPE_%": float((np.abs(s - o) / o.clip(lower=1)).mean() * 100),
                     "correlation": float(np.corrcoef(o, s)[0, 1])})
    return pd.DataFrame(rows)


def ablation(ctx, reps: int, ops: dict) -> pd.DataFrame:
    rows = []
    presets = preset_scenarios()
    for name in ("Baseline", "Monsoon Fever Surge", "Mass Casualty Incident", "ICU Bed Closure"):
        sc = presets[name]
        recs, base, _ = generate_recommendations(ctx, sc, reps=reps, elective_plan=ops["elective_plan"])
        best = next((r for r in recs if r.accepted and r.impact is not None), None)
        row = {"scenario": name, "plan": best.title if best else "—"}
        for m in ("door_to_doc_mean_min", "boarding_mean_h", "lwbs", "ICU_full_hours", "electives_cancelled"):
            row[f"{METRIC_LABELS[m]} (reactive)"] = base.kpis[m].mean()
            row[f"{METRIC_LABELS[m]} (AI plan)"] = (
                float(best.impact.set_index("metric").loc[m, "scenario"]) if best else base.kpis[m].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def evaluate(reps: int = 24) -> dict:
    ctx = load_context()
    info = ctx.forecast_info
    ops = joblib.load(ARTIFACT_DIR / "operations.joblib")
    print("Validating twin against observed history…")
    val = validate_twin(ctx)
    m = pd.DataFrame(info["metrics"])
    champ = m[m.apply(lambda r: info["champions"][r["target"]] == r["model"], axis=1)]
    fc_tab = champ[["target", "model", "MAE", "RMSE", "MAPE", "Skill vs naive %"]].copy()
    fc_tab["80% interval coverage"] = fc_tab["target"].map(info["coverage80"])
    ro, rt = ops["roster"].summary(), ops["template"].summary()
    eo, ef = ops["elective_plan"].summary(), ops["elective_fifo"].summary()
    roster_tab = pd.DataFrame([{"roster": "Fixed template", **rt}, {"roster": "CP-SAT optimised", **ro}])
    el_tab = pd.DataFrame([{"plan": "FIFO by listing date", **ef}, {"plan": "CP-SAT bed-aware", **eo}])
    print("Running ablation across stress scenarios…")
    abl = ablation(ctx, reps, ops)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md = [
        "# Hospital Digital Twin — Evaluation Report", "",
        f"Snapshot: {ctx.start:%d %b %Y %H:%M} · history: {len(ctx.hist['daily'])} days · "
        f"{len(ctx.hist['encounters']):,} encounters", "",
        "## 1. Forecasting (rolling-origin back-test, 10 origins × 14-day horizon)", "",
        _md(fc_tab), "",
        "Skill = MAE reduction vs seasonal-naive. Coverage = share of back-test actuals inside the calibrated 80% band.", "",
        "## 2. Twin validation (replay of the last 8 weeks of observed demand, 10 replications)", "",
        _md(val), "",
        "## 3. Nurse rostering (next 7 days, P90 demand)", "",
        _md(roster_tab.drop(columns=["status"])), "",
        "## 4. Elective scheduling (next 14 days)", "",
        _md(el_tab.drop(columns=["status"])), "",
        "## 5. Ablation — reactive operations vs simulation-verified AI plan", "",
        _md(abl, ".1f"), "",
    ]
    (REPORT_DIR / "evaluation.md").write_text("\n".join(md))
    out = {"forecast": fc_tab.to_dict(orient="records"), "validation": val.to_dict(orient="records"),
           "roster": roster_tab.to_dict(orient="records"), "electives": el_tab.to_dict(orient="records"),
           "ablation": abl.to_dict(orient="records")}
    (REPORT_DIR / "evaluation.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"Report written to {REPORT_DIR / 'evaluation.md'}")
    return out
