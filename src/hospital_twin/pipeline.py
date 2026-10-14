"""End-to-end build: data → forecasts → twin projection → optimisation → recommendations."""

from __future__ import annotations

import json
import time

import joblib
import pandas as pd

from .config import ARTIFACT_DIR
from .data.generator import generate_history
from .forecasting.pipeline import run_forecasting
from .optimization.electives import (
    capacity_for_electives,
    fifo_plan,
    generate_waitlist,
    optimize_electives,
)
from .optimization.roster import (
    build_nurse_pool,
    optimize_roster,
    required_from_forecast,
    staffing_demand,
    template_roster,
)
from .recommender.brief import executive_brief
from .recommender.engine import generate_recommendations
from .simulation.scenario import Scenario
from .twin.context import TwinContext, load_context


def plan_operations(ctx: TwinContext, level: str = "hi", time_limit: float = 20.0) -> dict:
    """Roster + elective optimisation for the coming week(s) — the 'optimiser' layer."""
    nurses = build_nurse_pool()
    demand = staffing_demand(required_from_forecast(ctx.forecast, level), ctx.shift_factors)
    roster = optimize_roster(nurses, demand, time_limit=time_limit)
    template = template_roster(nurses, demand)

    snap_el = {}
    for p in ctx.snapshot["inpatients"]:
        if p.get("elective"):
            snap_el[p["unit"]] = snap_el.get(p["unit"], 0) + 1
    cap = capacity_for_electives(ctx.census_forecast(level), ctx.elective_dow, snap_el, ctx.cfg)
    waitlist = generate_waitlist(ctx.start)
    eplan = optimize_electives(waitlist, cap, ctx.cfg, time_limit=time_limit)
    efifo = fifo_plan(waitlist, cap, ctx.cfg)
    return {"nurses": nurses, "demand": demand, "roster": roster, "template": template,
            "capacity": cap, "waitlist": waitlist, "elective_plan": eplan, "elective_fifo": efifo}


def build(days: int = 730, reps: int = 24, verbose: bool = True) -> None:
    t0 = time.time()
    hist = generate_history(days=days, verbose=verbose)
    run_forecasting(hist["daily"], verbose=verbose)
    ctx = load_context()
    if verbose:
        print("Optimising nurse roster and elective schedule…")
    ops = plan_operations(ctx)
    if verbose:
        print(f"  roster: {ops['roster'].summary()}")
        print(f"  electives: {ops['elective_plan'].summary()}")
        print(f"Running twin projection ({reps} Monte-Carlo futures) and verifying recommendations…")
    recs, base, risk = generate_recommendations(
        ctx, Scenario(), roster_summary=ops["roster"].summary(), elective_plan=ops["elective_plan"], reps=reps,
    )
    brief, src = executive_brief(ctx.state, risk, recs)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(base, ARTIFACT_DIR / "baseline_mc.joblib")
    joblib.dump(ops, ARTIFACT_DIR / "operations.joblib")
    joblib.dump({"recs": recs, "risk": risk, "brief": brief, "brief_source": src},
                ARTIFACT_DIR / "recommendations.joblib")
    (ARTIFACT_DIR / "build_info.json").write_text(json.dumps({
        "built_at": pd.Timestamp.now().isoformat(), "seconds": round(time.time() - t0, 1),
        "snapshot": ctx.snapshot["time"], "reps": reps,
    }, indent=1))
    if verbose:
        print(f"\nTwin built in {time.time() - t0:.0f}s. Status: {ctx.state.status}")
        for r in recs:
            flag = "✔" if r.accepted else "✘"
            print(f"  {flag} [{r.severity}] {r.title}  (score {r.score:.1f})")
