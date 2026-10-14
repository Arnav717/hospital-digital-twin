"""Back-testing, model selection, calibrated intervals and SHAP explanations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed

from ..config import ARTIFACT_DIR
from .models import FEATURE_LABELS, MODELS, TARGETS, LightGBMRecursive, make_features

HORIZON = 14
N_ORIGINS = 10
ORIGIN_STEP = 14


def _predict_all(y: pd.Series, horizon: int) -> dict[str, pd.Series]:
    preds = {name: cls().fit(y).predict(horizon) for name, cls in MODELS.items()}
    preds["Ensemble"] = (preds["SARIMA"] + preds["LightGBM"]) / 2
    return preds


def backtest_target(y: pd.Series, target: str) -> pd.DataFrame:
    rows = []
    last = len(y) - HORIZON
    for k in range(N_ORIGINS):
        cut = last - k * ORIGIN_STEP
        train, test = y.iloc[:cut], y.iloc[cut: cut + HORIZON]
        for name, pred in _predict_all(train, HORIZON).items():
            rows.append(pd.DataFrame({
                "target": target, "model": name, "origin": train.index[-1], "date": test.index,
                "h": np.arange(1, len(test) + 1), "y": test.to_numpy(), "yhat": pred.to_numpy()[: len(test)],
            }))
    return pd.concat(rows, ignore_index=True)


def metrics(bt: pd.DataFrame) -> pd.DataFrame:
    e = bt["yhat"] - bt["y"]
    bt = bt.assign(ae=e.abs(), se=e**2, ape=(e.abs() / bt["y"].clip(lower=1)) * 100)
    m = bt.groupby(["target", "model"]).agg(MAE=("ae", "mean"), RMSE=("se", "mean"), MAPE=("ape", "mean"))
    m["RMSE"] = np.sqrt(m["RMSE"])
    naive = m.xs("Seasonal naive", level="model")["MAE"]
    m["Skill vs naive %"] = (1 - m["MAE"] / naive.reindex(m.index.get_level_values(0)).to_numpy()) * 100
    return m.reset_index()


def residual_quantiles(bt: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Per-horizon 10th/90th residual quantiles, pooled over a ±2 day window."""
    r = bt.assign(res=bt["y"] - bt["yhat"])
    out = []
    for h in range(1, horizon + 1):
        w = r[(r["h"] >= h - 2) & (r["h"] <= h + 2)]["res"]
        out.append({"h": h, "q10": w.quantile(0.1), "q90": w.quantile(0.9)})
    return pd.DataFrame(out)


def _final_for_target(y: pd.Series, target: str, bt: pd.DataFrame, champion: str):
    preds = _predict_all(y, HORIZON)
    rows = []
    for name, pred in preds.items():
        q = residual_quantiles(bt[bt["model"] == name])
        rows.append(pd.DataFrame({
            "target": target, "model": name, "date": pred.index, "h": np.arange(1, HORIZON + 1),
            "yhat": pred.to_numpy(), "lo": pred.to_numpy() + q["q10"].to_numpy(),
            "hi": pred.to_numpy() + q["q90"].to_numpy(), "champion": name == champion,
        }))
    fc = pd.concat(rows, ignore_index=True)
    for c in ("yhat", "lo", "hi"):
        fc[c] = fc[c].clip(lower=0)

    # SHAP explanations of the LightGBM model.
    lgbm = LightGBMRecursive().fit(y)
    _, Xf = lgbm.predict(HORIZON, return_features=True)
    Xh = make_features(y)[lgbm.features].iloc[-180:]
    expl = shap.TreeExplainer(lgbm.model)
    sv_hist = expl.shap_values(Xh)
    sv_fut = expl.shap_values(Xf)
    glob = pd.DataFrame({"target": target, "feature": lgbm.features,
                         "importance": np.abs(sv_hist).mean(axis=0)})
    loc = []
    for i, d in enumerate(Xf.index):
        for j, f in enumerate(lgbm.features):
            loc.append({"target": target, "date": d, "feature": f, "shap": float(sv_fut[i, j]),
                        "value": float(Xf.iloc[i, j]) if pd.notna(Xf.iloc[i, j]) else np.nan})
    base = float(np.ravel(expl.expected_value)[0])
    return fc, glob, pd.DataFrame(loc).assign(base=base)


def run_forecasting(daily: pd.DataFrame, out_dir: Path = ARTIFACT_DIR, n_jobs: int = -1,
                    verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Back-testing {len(MODELS) + 1} models × {len(TARGETS)} targets "
              f"({N_ORIGINS} rolling origins, {HORIZON}-day horizon)…")
    series = {t: daily[t].astype(float).asfreq("D").ffill() for t in TARGETS}
    bts = Parallel(n_jobs=n_jobs)(delayed(backtest_target)(series[t], t) for t in TARGETS)
    bt = pd.concat(bts, ignore_index=True)
    m = metrics(bt)
    champions = (m[m["model"] != "Seasonal naive"].sort_values("MAE")
                 .groupby("target").head(1).set_index("target")["model"].to_dict())
    if verbose:
        print("  champions:", champions)
        print("Fitting final models + SHAP…")
    finals = Parallel(n_jobs=n_jobs)(
        delayed(_final_for_target)(series[t], t, bt[bt["target"] == t], champions[t]) for t in TARGETS
    )
    fc = pd.concat([f[0] for f in finals], ignore_index=True)
    shap_global = pd.concat([f[1] for f in finals], ignore_index=True)
    shap_local = pd.concat([f[2] for f in finals], ignore_index=True)

    # Interval coverage of champion intervals on the back-test (calibration check).
    cov = {}
    for t, champ in champions.items():
        b = bt[(bt["target"] == t) & (bt["model"] == champ)]
        q = residual_quantiles(b).set_index("h")
        lo = b["yhat"] + q.loc[b["h"], "q10"].to_numpy()
        hi = b["yhat"] + q.loc[b["h"], "q90"].to_numpy()
        cov[t] = float(((b["y"] >= lo) & (b["y"] <= hi)).mean() * 100)

    bt.to_parquet(out_dir / "backtest.parquet")
    fc.to_parquet(out_dir / "forecast.parquet")
    shap_global.to_parquet(out_dir / "shap_global.parquet")
    shap_local.to_parquet(out_dir / "shap_local.parquet")
    info = {"champions": champions, "coverage80": cov, "metrics": m.to_dict(orient="records"),
            "feature_labels": FEATURE_LABELS, "targets": TARGETS}
    (out_dir / "forecast_metrics.json").write_text(json.dumps(info, indent=1, default=str))
    return {"forecast": fc, "backtest": bt, "metrics": m, "champions": champions, "coverage": cov}


def champion_forecast(fc: pd.DataFrame, target: str) -> pd.DataFrame:
    f = fc[(fc["target"] == target) & (fc["champion"])]
    return f.set_index("date")[["yhat", "lo", "hi"]]
