import numpy as np
import pandas as pd
import pytest

from hospital_twin.config import ARTIFACT_DIR, DEFAULT_CONFIG
from hospital_twin.data.generator import (
    daily_arrival_rate,
    hourly_profile_matrix,
    hourly_rates,
)
from hospital_twin.forecasting.models import (
    LightGBMRecursive,
    Sarima,
    SeasonalNaive,
    make_features,
)
from hospital_twin.forecasting.pipeline import residual_quantiles
from hospital_twin.optimization.electives import (
    capacity_for_electives,
    generate_waitlist,
    optimize_electives,
)
from hospital_twin.optimization.redistribution import plan_redistribution
from hospital_twin.optimization.roster import Nurse, optimize_roster, template_roster
from hospital_twin.simulation.engine import HospitalSim
from hospital_twin.simulation.runner import (
    compare,
    forecast_hourly_rates,
    run_monte_carlo,
)
from hospital_twin.simulation.scenario import Scenario, Surge, preset_scenarios

START = pd.Timestamp("2026-03-02 00:00")  # a Monday


def rates(days=10, per_day=200.0):
    daily = pd.Series(per_day, index=pd.date_range(START, periods=days, freq="D"))
    return hourly_rates(daily, hourly_profile_matrix())


# ---------------------------------------------------------------- simulation
def test_simulation_respects_capacity_and_counts_arrivals():
    r = rates(14)
    res = HospitalSim(hourly_rates=r, start=START, seed=1).run(snapshot=True)
    ts = res.timeseries
    for u in ("ICU", "MED", "SURG"):
        assert ts[u].max() <= DEFAULT_CONFIG.beds[u]
    assert ts["ed_occupied"].max() <= DEFAULT_CONFIG.beds["ED"]
    assert abs(res.kpis["arrivals"] - r.sum()) < 5 * np.sqrt(r.sum())
    assert res.kpis["electives_listed"] > 0
    assert res.snapshot["inpatients"], "steady-state hospital should hold inpatients"


def test_simulation_is_reproducible():
    a = HospitalSim(hourly_rates=rates(5), start=START, seed=7).run().kpis
    b = HospitalSim(hourly_rates=rates(5), start=START, seed=7).run().kpis
    assert a == b


def test_warm_start_from_snapshot():
    snap = HospitalSim(hourly_rates=rates(14), start=START, seed=2).run(snapshot=True).snapshot
    t = pd.Timestamp(snap["time"])
    ts = HospitalSim(hourly_rates=rates(2), start=t, seed=3, warm=snap).run().timeseries
    n_icu = sum(p["unit"] == "ICU" for p in snap["inpatients"])
    assert abs(ts["ICU"].iloc[0] - n_icu) <= 2


def test_extra_icu_beds_reduce_icu_saturation():
    stress = Scenario(arrival_multiplier=1.3)
    more = stress.with_(bed_delta={"ICU": 8})
    kw = dict(hourly_rates=rates(14), start=START)
    a = np.mean([HospitalSim(scenario=stress, seed=s, **kw).run().kpis["ICU_full_hours"] for s in range(3)])
    b = np.mean([HospitalSim(scenario=more, seed=s, **kw).run().kpis["ICU_full_hours"] for s in range(3)])
    assert b < a


def test_surge_and_diversion():
    base = HospitalSim(hourly_rates=rates(3), start=START, seed=4).run().kpis
    sc = Scenario(surges=[Surge(hour=10, patients=40)], divert_frac=0.5)
    alt = HospitalSim(hourly_rates=rates(3), start=START, seed=4, scenario=sc).run().kpis
    assert alt["arrivals"] >= base["arrivals"] + 30
    assert alt["diverted"] > 0


def test_planned_day_cases_need_no_bed():
    plan = [[[0, 0, 0.5]] * 10 for _ in range(4)]
    res = HospitalSim(hourly_rates=rates(4), start=START, seed=5, scenario=Scenario(elective_plan=plan)).run()
    assert res.kpis["electives_cancelled"] == 0
    assert res.flows[("Elective Surgery", "Day case — home")] > 0


def test_scenario_roundtrip():
    for sc in preset_scenarios().values():
        assert Scenario.from_dict(sc.to_dict()).key() == sc.key()


def test_monte_carlo_and_paired_compare():
    fc = pd.DataFrame({"yhat": 200.0, "lo": 180.0, "hi": 220.0}, index=pd.date_range(START, periods=4, freq="D"))
    prof = hourly_profile_matrix()
    r = forecast_hourly_rates(fc, prof, START, 48)
    assert r.shape == (48,) and abs(r[:24].sum() - 200) < 1e-6
    a = run_monte_carlo(Scenario(), forecast=fc, profile=prof, start=START, warm=None, horizon_h=48, reps=3, n_jobs=1)
    b = run_monte_carlo(Scenario(), forecast=fc, profile=prof, start=START, warm=None, horizon_h=48, reps=3, n_jobs=1)
    c = compare(a, b, ["door_to_doc_mean_min"])
    assert c["delta"].abs().max() < 1e-9  # common random numbers → identical runs


# ---------------------------------------------------------------- forecasting
@pytest.fixture(scope="module")
def series():
    idx = pd.date_range("2024-01-01", periods=500, freq="D")
    return daily_arrival_rate(idx)["rate"]


def test_features_have_no_leakage(series):
    X = make_features(series)
    assert (X["lag1"].iloc[1:].to_numpy() == series.iloc[:-1].to_numpy()).all()


@pytest.mark.parametrize("cls", [SeasonalNaive, Sarima, LightGBMRecursive])
def test_models_forecast_horizon(series, cls):
    f = cls().fit(series).predict(14)
    assert len(f) == 14 and f.index[0] == series.index[-1] + pd.Timedelta(days=1)
    assert np.isfinite(f).all() and (f > 0).all()


def test_residual_quantiles_order():
    rng = np.random.default_rng(0)
    bt = pd.DataFrame({"h": np.tile(np.arange(1, 15), 20), "y": rng.normal(0, 1, 280), "yhat": 0.0})
    q = residual_quantiles(bt)
    assert (q["q10"] < q["q90"]).all()


# ---------------------------------------------------------------- optimisation
def test_roster_covers_demand_and_rules():
    nurses = [Nurse(f"N{i}", f"n{i}", "MED", ("MED",), 0, 5) for i in range(12)]
    dates = pd.date_range(START, periods=7, freq="D")
    demand = pd.DataFrame([{"date": d, "shift": s, "unit": "MED", "census": 0, "required": 2}
                           for d in dates for s in ("Day", "Evening", "Night")])
    res = optimize_roster(nurses, demand, time_limit=5, allow_agency=False)
    assert res.coverage["short"].sum() == 0
    a = res.assignments
    assert a.groupby(["nurse", "date"]).size().max() == 1
    assert a.groupby("nurse").size().max() <= 5
    nights = a[a["shift"] == "Night"]
    for r in nights.itertuples():
        nxt = a[(a["nurse"] == r.nurse) & (a["date"] == r.date + pd.Timedelta(days=1)) & (a["shift"] == "Day")]
        assert nxt.empty
    assert template_roster(nurses, demand).coverage["short"].sum() >= 0


def test_elective_plan_respects_theatre_and_beds():
    days = pd.date_range(START, periods=14, freq="D")
    census = pd.DataFrame({"ED": 20, "ICU": 18.0, "MED": 150, "SURG": 80.0}, index=days)
    dow = pd.DataFrame({"ICU": [2.0] * 7, "SURG": [30.0] * 7})
    cap = capacity_for_electives(census, dow, {})
    plan = optimize_electives(generate_waitlist(START, n=80), cap, time_limit=10)
    d = plan.daily
    assert (d["ot_used"] <= d["ot_cap"] + 1e-6).all()
    assert plan.summary()["bed_overflow_days"] == 0
    assert plan.schedule.dropna(subset=["date"])["date"].dt.dayofweek.max() < 5


def test_redistribution_balances_excess():
    plan = plan_redistribution({"ICU": 3, "MED": 10, "SURG": 0}, {"ICU": 0, "MED": 0, "SURG": 4})
    placed = plan.groupby("from")["patients"].sum()
    assert placed["ICU"] == pytest.approx(3) and placed["MED"] == pytest.approx(10)
    assert plan[(plan["from"] == "MED") & plan["to"].str.contains("Outlier")]["patients"].sum() == pytest.approx(4)


# ---------------------------------------------------------------- API
@pytest.mark.skipif(not (ARTIFACT_DIR / "operations.joblib").exists(), reason="run `hospital-twin build` first")
def test_api_endpoints_are_json_safe():
    from fastapi.testclient import TestClient

    from hospital_twin.api.main import app

    c = TestClient(app)
    assert c.get("/state").json()["status"] in ("STABLE", "STRAINED", "CRITICAL")
    assert len(c.get("/forecast/ed_arrivals").json()["points"]) == 14
    assert c.get("/forecast/nope").status_code == 404
    assert c.get("/roster").status_code == 200
    assert c.get("/electives").status_code == 200
