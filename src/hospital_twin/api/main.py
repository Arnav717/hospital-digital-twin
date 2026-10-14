"""REST API exposing the twin to other hospital systems."""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import ARTIFACT_DIR
from ..forecasting.models import TARGETS
from ..recommender.engine import generate_recommendations
from ..simulation.scenario import Scenario, Surge
from ..twin.context import TwinContext, load_context

app = FastAPI(title="Hospital Digital Twin API", version="1.0.0",
              description="Forecast, simulate and get decision support from the hospital digital twin.")


@lru_cache(maxsize=1)
def ctx() -> TwinContext:
    return load_context()


class SurgeIn(BaseModel):
    hour: float = Field(ge=0)
    patients: int = Field(ge=0, le=500)
    duration: float = 2.0


class ScenarioIn(BaseModel):
    name: str = "API scenario"
    arrival_multiplier: float = Field(1.0, ge=0.1, le=5)
    surges: list[SurgeIn] = []
    bed_delta: dict[str, int] = {}
    doctor_delta: dict[str, int] = {}
    elective_postpone_frac: float = Field(0.0, ge=0, le=1)
    divert_frac: float = Field(0.0, ge=0, le=1)
    early_discharge: bool = False
    lounge_delta: int = 0
    los_multiplier: float = Field(1.0, ge=0.5, le=2)
    reps: int = Field(16, ge=2, le=64)
    horizon_hours: int = Field(168, ge=24, le=336)

    def to_scenario(self) -> Scenario:
        d = self.model_dump(exclude={"reps", "horizon_hours", "surges"})
        return Scenario(**d, surges=[Surge(**s.model_dump()) for s in self.surges])


@app.get("/health")
def health():
    return {"status": "ok", "built": (ARTIFACT_DIR / "recommendations.joblib").exists()}


@app.get("/state")
def state():
    s = ctx().state
    return {
        "time": s.time.isoformat(), "status": s.status,
        "units": {k: {"beds": u.beds, "occupied": u.occupied, "occupancy_pct": round(u.occupancy, 1)}
                  for k, u in s.units.items()},
        "ed": {"waiting": s.ed_waiting, "treatment": s.ed_treatment, "boarding": s.ed_boarding},
        "arrivals_24h": s.arrivals_24h,
    }


@app.get("/forecast/{target}")
def forecast(target: str, model: str | None = None):
    if target not in TARGETS:
        raise HTTPException(404, f"unknown target; choose from {list(TARGETS)}")
    fc = ctx().forecast
    f = fc[fc["target"] == target]
    f = f[f["model"] == model] if model else f[f["champion"]]
    if f.empty:
        raise HTTPException(404, "unknown model")
    return {"target": target, "model": f["model"].iloc[0],
            "points": [{"date": r.date.date().isoformat(), "yhat": round(r.yhat, 2), "lo80": round(r.lo, 2),
                        "hi80": round(r.hi, 2)} for r in f.itertuples()]}


@app.post("/simulate")
def simulate(sc: ScenarioIn):
    mc = ctx().simulate(sc.to_scenario(), reps=sc.reps, horizon_h=sc.horizon_hours, n_jobs=1)
    summ = mc.kpi_summary().round(3)
    return {"scenario": sc.name, "kpis": summ.to_dict(orient="index"),
            "hourly_mean": mc.ts_mean.drop(columns=["time"]).round(2).to_dict(orient="list")}


@app.post("/recommendations")
def recommendations(sc: ScenarioIn):
    ops = joblib.load(ARTIFACT_DIR / "operations.joblib")
    recs, _, risk = generate_recommendations(ctx(), sc.to_scenario(), reps=sc.reps, horizon_h=sc.horizon_hours,
                                             roster_summary=ops["roster"].summary(),
                                             elective_plan=ops["elective_plan"])
    return {"risk": {k: (v if isinstance(v, (int, float, dict)) else str(v)) for k, v in risk.items()},
            "recommendations": [{
                "id": r.rid, "severity": r.severity, "title": r.title, "owner": r.owner, "accepted": r.accepted,
                "rationale": r.rationale, "actions": r.actions, "score": round(r.score, 2),
                "impact": r.impact.round(3).to_dict(orient="records") if r.impact is not None else None,
            } for r in recs]}


@app.get("/roster")
def roster():
    ops = joblib.load(ARTIFACT_DIR / "operations.joblib")
    r = ops["roster"]
    return {"summary": r.summary(), "coverage": _records(r.coverage)}


@app.get("/electives")
def electives():
    ops = joblib.load(ARTIFACT_DIR / "operations.joblib")
    p = ops["elective_plan"]
    return {"summary": p.summary(), "schedule": _records(p.schedule)}


def _records(df) -> list[dict]:
    """JSON-safe records (NaN/NaT → null, timestamps → ISO dates)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))
