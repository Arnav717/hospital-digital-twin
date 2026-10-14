import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from theme import (
    AMBER,
    CYAN,
    GREEN,
    MUTED,
    PINK,
    RED,
    UNIT_COLOR,
    VIOLET,
    card_row,
    hero,
    kpi_card,
    section,
)

import data
from hospital_twin.config import HOSPITAL_LATLON, HOSPITAL_NAME, PARTNER_HOSPITALS
from hospital_twin.optimization.redistribution import (
    excess_from_projection,
    plan_redistribution,
)
from hospital_twin.simulation.scenario import Scenario

data.require_build()
ctx = data.ctx()
ops = data.operations()
hero("Operations Optimizer", "Constraint programming turns forecasts into rosters, theatre lists and transfer plans",
     "OR-Tools CP-SAT · HiGHS LP")

tab1, tab2, tab3 = st.tabs(["👩‍⚕️ Nurse roster", "🔪 Elective theatre list", "🚑 Patient redistribution"])

with tab1:
    ro, rt = ops["roster"], ops["template"]
    so, stt = ro.summary(), rt.summary()
    card_row([
        kpi_card("Required nurse shifts", f"{so['required_shifts']}", "next 7 days · P90 census", CYAN),
        kpi_card("Uncovered shifts", f"{so['short_shifts']}", f"template roster: <b style='color:{RED}'>{stt['short_shifts']}</b>",
                 GREEN if so["short_shifts"] <= stt["short_shifts"] else RED),
        kpi_card("Float-pool placements", f"{so['float_shifts']}",
                 f"+{so['redeployments']} cross-unit moves of ward nurses", VIOLET),
        kpi_card("Agency shifts", f"{so['agency_shifts']}", "premium cover booked", AMBER),
        kpi_card("Over-staffed shifts", f"{so['over_shifts']}", f"template: {stt['over_shifts']}", PINK),
    ])
    cov = ro.coverage.copy()
    cov["slot"] = pd.to_datetime(cov["date"]).dt.strftime("%a %d") + " · " + cov["shift"].str[0]
    tcov = rt.coverage.copy()
    tcov["slot"] = cov["slot"]
    c1, c2 = st.columns(2)
    for col, df, name in ((c1, tcov, "Fixed template roster"), (c2, cov, "Optimised roster")):
        gap = df.assign(gap=df["scheduled"] + df["agency"] - df["required"]).pivot_table(
            index="unit", columns="slot", values="gap", sort=False)
        f = go.Figure(go.Heatmap(z=gap.to_numpy(), x=gap.columns, y=gap.index, zmid=0, zmin=-8, zmax=8,
                                 colorscale=[[0, RED], [0.5, "#111827"], [1, CYAN]],
                                 hovertemplate="%{y} %{x}: %{z:+d} nurses<extra></extra>",
                                 colorbar=dict(title="±")))
        f.update_layout(title=f"{name} — staffing gap (dark = exact match)", height=260, xaxis=dict(tickangle=-60, tickfont=dict(size=9)))
        col.plotly_chart(f, width="stretch")
    unit = st.segmented_control("Unit", ["ED", "ICU", "MED", "SURG"], default="MED")
    u = cov[cov["unit"] == unit]
    f = go.Figure()
    f.add_trace(go.Bar(x=u["slot"], y=u["scheduled"], name="Rostered", marker_color=UNIT_COLOR[unit]))
    f.add_trace(go.Bar(x=u["slot"], y=u["agency"], name="Agency", marker_color=AMBER))
    f.add_trace(go.Bar(x=u["slot"], y=u["short"], name="Uncovered", marker_color=RED))
    f.add_trace(go.Scatter(x=u["slot"], y=u["required"], name="Required (ratio × P90 census)", mode="lines+markers",
                           line=dict(color="white", width=2, dash="dot")))
    f.update_layout(barmode="stack", height=340, title=f"{unit}: demand vs cover per shift")
    st.plotly_chart(f, width="stretch")

    section("Roster grid", "first 40 nurses of the selected unit's pool")
    asg = ro.assignments
    pool = [n for n in ops["nurses"] if unit in n.skills][:40]
    dates = sorted(cov["date"].unique())
    code = {"Day": 1, "Evening": 2, "Night": 3}
    z, txt = [], []
    for n in pool:
        row, trow = [], []
        for d in dates:
            a = asg[(asg["nurse"] == n.nid) & (asg["date"] == d)]
            if len(a):
                row.append(code[a["shift"].iloc[0]])
                trow.append(f"{a['shift'].iloc[0][0]}·{a['unit'].iloc[0]}")
            else:
                row.append(0)
                trow.append("off" if d not in [dates[i] for i in n.days_off] else "req. off")
        z.append(row)
        txt.append(trow)
    f = go.Figure(go.Heatmap(
        z=z, x=[pd.Timestamp(d).strftime("%a %d") for d in dates], y=[f"{n.name} ({n.home})" for n in pool],
        text=txt, texttemplate="%{text}", textfont=dict(size=9), showscale=False, xgap=2, ygap=2,
        colorscale=[[0, "#111827"], [0.33, "#111827"], [0.34, "#0e7490"], [0.66, "#6d28d9"], [1, "#9d174d"]],
        zmin=0, zmax=3, hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
    f.update_layout(height=26 * len(pool) + 60, yaxis=dict(autorange="reversed", tickfont=dict(size=10)))
    st.plotly_chart(f, width="stretch")
    st.caption("Day (teal) · Evening (violet) · Night (magenta). Constraints: ≤1 shift/day, no night→day turnaround, "
               "contracted 4–5 shifts, ≤2 nights, requested days off honoured.")

with tab2:
    ep, ef = ops["elective_plan"], ops["elective_fifo"]
    so, sf = ep.summary(), ef.summary()
    card_row([
        kpi_card("Cases booked", f"{so['scheduled']}", f"FIFO: {sf['scheduled']}", CYAN),
        kpi_card("Urgent (P1) booked", f"{so['p1_scheduled_pct']:.0f}%", f"FIFO: {sf['p1_scheduled_pct']:.0f}%", GREEN),
        kpi_card("Bed overflow days", f"{so['bed_overflow_days']}", f"FIFO: <b style='color:{RED}'>{sf['bed_overflow_days']}</b>", AMBER),
        kpi_card("Theatre utilisation", f"{so['theatre_util_pct']:.0f}%", f"FIFO: {sf['theatre_util_pct']:.0f}%", VIOLET),
        kpi_card("Booked after due date", f"{so['late_cases']}", f"FIFO: {sf['late_cases']}", PINK),
    ])
    s = ep.schedule.dropna(subset=["date"]).copy()
    s["start"] = s["date"] + pd.Timedelta(hours=8)
    s["end"] = s["date"] + pd.to_timedelta(s["icu_days"] + s["ward_days"], unit="D") + pd.Timedelta(hours=12)
    s = s.sort_values(["date", "priority"])
    pcol = {"P1": RED, "P2": AMBER, "P3": CYAN, "P4": MUTED}
    f = px.timeline(s, x_start="start", x_end="end", y="case_id", color="priority", color_discrete_map=pcol,
                    hover_data=["procedure", "specialty", "hours", "icu_days", "ward_days", "due"])
    f.update_yaxes(visible=False, autorange="reversed")
    f.update_layout(height=520, title="Surgery date → expected discharge (bar = hospital stay)")
    st.plotly_chart(f, width="stretch")
    d = ep.daily.copy()
    dfifo = ef.daily
    c1, c2 = st.columns(2)
    f = go.Figure()
    f.add_trace(go.Scatter(x=d["date"], y=d["icu_avail"], name="ICU beds free for electives", fill="tozeroy",
                           line=dict(color="rgba(248,113,113,.6)", width=1), fillcolor="rgba(248,113,113,.10)"))
    f.add_trace(go.Scatter(x=d["date"], y=d["icu_used"], name="Optimised plan", line=dict(color=GREEN, width=3)))
    f.add_trace(go.Scatter(x=d["date"], y=dfifo["icu_used"], name="FIFO plan", line=dict(color=RED, width=2, dash="dash")))
    f.update_layout(title="Post-op ICU beds needed vs available", height=320)
    c1.plotly_chart(f, width="stretch")
    f = go.Figure()
    f.add_trace(go.Scatter(x=d["date"], y=d["surg_avail"], name="Ward beds free for electives", fill="tozeroy",
                           line=dict(color="rgba(167,139,250,.6)", width=1), fillcolor="rgba(167,139,250,.10)"))
    f.add_trace(go.Scatter(x=d["date"], y=d["surg_used"], name="Optimised plan", line=dict(color=GREEN, width=3)))
    f.add_trace(go.Scatter(x=d["date"], y=dfifo["surg_used"], name="FIFO plan", line=dict(color=RED, width=2, dash="dash")))
    f.update_layout(title="Post-op surgical-ward beds needed vs available", height=320)
    c2.plotly_chart(f, width="stretch")
    with st.expander("Theatre list & postponed cases"):
        st.dataframe(s[["date", "case_id", "procedure", "specialty", "priority", "hours", "icu_days", "ward_days", "due"]],
                     hide_index=True, width="stretch")
        st.markdown("**Postponed beyond the planning window**")
        st.dataframe(ep.schedule[ep.schedule["date"].isna()][["case_id", "procedure", "priority", "waited_days", "due"]],
                     hide_index=True, width="stretch")

with tab3:
    sc_dict = st.session_state.get("lab_scenario")
    sc = Scenario.from_dict(sc_dict) if sc_dict else Scenario()
    st.caption(f"Using the projection of **{sc.name}** "
               f"({'from Scenario Lab' if sc_dict else 'run a stress scenario in the Scenario Lab to change this'}).")
    hours = st.slider("Planning window (hours)", 12, 96, 48, step=12)
    mc = data.simulate(sc, 24, 168)
    excess, spare = excess_from_projection(mc.ts_p90, mc.beds, hours)
    plan = plan_redistribution(excess, spare)
    card_row([kpi_card(f"{u} excess (P90)", f"{excess[u]:.0f}", f"spare internal beds {spare[u]:.0f}",
                       UNIT_COLOR[u]) for u in ("ICU", "MED", "SURG")])
    if plan.empty:
        st.success("No unit is projected over capacity in this window — no transfers needed. "
                   "Try the *Mass Casualty Incident* or *ICU Bed Closure* preset in the Scenario Lab.")
    else:
        hlat, hlon = HOSPITAL_LATLON
        f = go.Figure()
        ext = plan[plan["lat"].notna()]
        for r in ext.to_dict("records"):
            f.add_trace(go.Scattermap(lat=[hlat, r["lat"]], lon=[hlon, r["lon"]], mode="lines",
                                      line=dict(width=2 + r["patients"] * 0.8, color=UNIT_COLOR[r["from"]]),
                                      name=f"{r['from']} → {r['to'].split('— ')[-1]}: {r['patients']:.0f}",
                                      hoverinfo="name"))
        f.add_trace(go.Scattermap(lat=[p["lat"] for p in PARTNER_HOSPITALS], lon=[p["lon"] for p in PARTNER_HOSPITALS],
                                  mode="markers+text", text=[p["name"].split("— ")[-1] for p in PARTNER_HOSPITALS],
                                  textposition="top center", marker=dict(size=14, color=CYAN), name="Partner hospitals",
                                  textfont=dict(color="white")))
        f.add_trace(go.Scattermap(lat=[hlat], lon=[hlon], mode="markers+text", text=[HOSPITAL_NAME],
                                  textposition="bottom center", marker=dict(size=22, color=PINK), name=HOSPITAL_NAME,
                                  textfont=dict(color="white")))
        f.update_layout(map=dict(style="carto-darkmatter", center=dict(lat=12.97, lon=80.18), zoom=10.3),
                        height=520, margin=dict(l=0, r=0, t=0, b=0), legend=dict(y=0.98, x=0.01))
        st.plotly_chart(f, width="stretch")
        st.dataframe(plan.drop(columns=["lat", "lon"]), hide_index=True, width="stretch")
    st.caption("Min-cost transport LP: internal outlier beds (clinically compatible wards) are cheapest, partner "
               "hospitals cost grows with distance and acuity, unplaced patients are heavily penalised.")
