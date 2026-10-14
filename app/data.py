"""Cached access to the built twin for the Streamlit pages."""

from __future__ import annotations

import json

import joblib
import streamlit as st

from hospital_twin.config import ARTIFACT_DIR, REPORT_DIR
from hospital_twin.simulation.scenario import Scenario
from hospital_twin.twin.context import TwinContext, load_context


def built() -> bool:
    return (ARTIFACT_DIR / "recommendations.joblib").exists()


def require_build() -> None:
    if not built():
        st.error("The twin has not been built yet. Run `uv run hospital-twin build` in the project folder, then reload.")
        st.stop()


@st.cache_resource(show_spinner="Loading digital twin…")
def ctx() -> TwinContext:
    return load_context()


@st.cache_resource(show_spinner=False)
def baseline():
    return joblib.load(ARTIFACT_DIR / "baseline_mc.joblib")


@st.cache_resource(show_spinner=False)
def operations() -> dict:
    return joblib.load(ARTIFACT_DIR / "operations.joblib")


@st.cache_resource(show_spinner=False)
def recommendations() -> dict:
    return joblib.load(ARTIFACT_DIR / "recommendations.joblib")


@st.cache_resource(show_spinner=False)
def build_info() -> dict:
    return json.loads((ARTIFACT_DIR / "build_info.json").read_text())


def evaluation() -> dict | None:
    p = REPORT_DIR / "evaluation.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource(show_spinner=False, max_entries=24)
def _simulate(key: str, scenario_json: str, reps: int, horizon_h: int):
    return ctx().simulate(Scenario.from_dict(json.loads(scenario_json)), reps=reps, horizon_h=horizon_h)


def simulate(scenario: Scenario, reps: int = 24, horizon_h: int = 168):
    return _simulate(scenario.key(), json.dumps(scenario.to_dict(), default=str), reps, horizon_h)


@st.cache_resource(show_spinner=False, max_entries=8)
def _recommend(key: str, scenario_json: str, reps: int):
    from hospital_twin.recommender.brief import executive_brief
    from hospital_twin.recommender.engine import generate_recommendations

    c = ctx()
    sc = Scenario.from_dict(json.loads(scenario_json))
    ops = operations()
    recs, base, risk = generate_recommendations(c, sc, base=simulate(sc, reps), reps=reps,
                                                roster_summary=ops["roster"].summary(),
                                                elective_plan=ops["elective_plan"])
    brief, src = executive_brief(c.state, risk, recs)
    return {"recs": recs, "risk": risk, "brief": brief, "brief_source": src}


def recommend(scenario: Scenario, reps: int = 24) -> dict:
    return _recommend(scenario.key(), json.dumps(scenario.to_dict(), default=str), reps)
