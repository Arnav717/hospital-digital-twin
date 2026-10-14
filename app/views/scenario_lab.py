import plotly.graph_objects as go
import streamlit as st
from charts import floor_map, impact_fig, projection_fig, sankey
from theme import (
    AMBER,
    CYAN,
    GREEN,
    MUTED,
    PINK,
    RED,
    VIOLET,
    card_row,
    hero,
    kpi_card,
    rec_card,
    section,
)

import data
from hospital_twin.recommender.engine import METRIC_LABELS, WEIGHTS
from hospital_twin.simulation.runner import compare
from hospital_twin.simulation.scenario import Scenario, Surge, preset_scenarios

data.require_build()
ctx = data.ctx()
PRESETS = preset_scenarios()

hero("Scenario Lab", "Stress-test the hospital before it happens — every run is a Monte-Carlo simulation of the twin",
     f"starting from live state {ctx.start:%a %d %b %H:%M}")

DEFAULTS = dict(arr=1.0, surge_n=0, surge_h=34, icu=0, med=0, surg=0, docs=0, post=0, div=0, early=False,
                lounge=0, los=1.0)


def apply_preset():
    sc = PRESETS[st.session_state["preset"]]
    v = dict(DEFAULTS)
    v.update(arr=sc.arrival_multiplier, icu=sc.bed_delta.get("ICU", 0), med=sc.bed_delta.get("MED", 0),
             surg=sc.bed_delta.get("SURG", 0), docs=sc.doctor_delta.get("Day", 0),
             post=int(sc.elective_postpone_frac * 100), div=int(sc.divert_frac * 100),
             early=sc.early_discharge, lounge=sc.lounge_delta, los=sc.los_multiplier)
    if sc.surges:
        v.update(surge_n=sc.surges[0].patients, surge_h=int(sc.surges[0].hour))
    for k, val in v.items():
        st.session_state[f"s_{k}"] = val


if "s_arr" not in st.session_state:
    st.session_state["preset"] = "Baseline"
    apply_preset()

ctrl, out = st.columns([1, 2.6])
with ctrl:
    st.selectbox("Preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.caption(PRESETS[st.session_state["preset"]].description)
    with st.form("scenario"):
        st.markdown("**Demand**")
        st.slider("ED arrival multiplier", 0.6, 1.8, step=0.05, key="s_arr")
        a, b = st.columns(2)
        a.number_input("Mass-casualty patients", 0, 80, key="s_surge_n")
        b.number_input("…arriving at hour", 0, 160, key="s_surge_h")
        st.slider("Length-of-stay multiplier (case-mix)", 0.8, 1.4, step=0.05, key="s_los")
        st.markdown("**Capacity**")
        a, b, c = st.columns(3)
        a.number_input("ICU beds ±", -12, 12, key="s_icu")
        b.number_input("Med beds ±", -40, 40, key="s_med")
        c.number_input("Surg beds ±", -40, 40, key="s_surg")
        st.slider("ED doctors per shift ±", -3, 3, key="s_docs")
        st.markdown("**Policies**")
        st.slider("Postpone electives (%)", 0, 90, step=5, key="s_post")
        st.slider("Divert ambulance ESI 2–3 (%)", 0, 60, step=5, key="s_div")
        a, b = st.columns(2)
        a.toggle("Discharge before noon", key="s_early")
        b.number_input("Lounge chairs ±", -16, 30, key="s_lounge")
        st.markdown("**Simulation**")
        a, b = st.columns(2)
        reps = a.select_slider("Replications", [8, 16, 24, 32, 48], value=24)
        horizon = b.select_slider("Horizon", [72, 120, 168], value=168, format_func=lambda h: f"{h // 24} days")
        run = st.form_submit_button("▶  Run simulation", type="primary", width="stretch")

S = st.session_state
scenario = Scenario(
    name=S["preset"] if S["preset"] != "Baseline" else "Custom scenario",
    arrival_multiplier=S["s_arr"], los_multiplier=S["s_los"],
    surges=[Surge(hour=float(S["s_surge_h"]), patients=int(S["s_surge_n"]))] if S["s_surge_n"] else [],
    bed_delta={k: v for k, v in {"ICU": S["s_icu"], "MED": S["s_med"], "SURG": S["s_surg"]}.items() if v},
    doctor_delta={k: v for k, v in {"Day": S["s_docs"], "Evening": S["s_docs"],
                                     "Night": int(S["s_docs"] / 2)}.items() if v},
    elective_postpone_frac=S["s_post"] / 100, divert_frac=S["s_div"] / 100,
    early_discharge=S["s_early"], lounge_delta=S["s_lounge"],
)
if run:
    S["lab_run"] = (scenario.to_dict(), reps, horizon)

with out:
    if "lab_run" not in S:
        st.info("Configure a scenario on the left and press **Run simulation**. Try the *Mass Casualty Incident* "
                "or *Monsoon Fever Surge* presets.")
        st.stop()
    sc_dict, reps, horizon = S["lab_run"]
    sc = Scenario.from_dict(sc_dict)
    with st.spinner(f"Simulating {reps} futures × 2 scenarios…"):
        base = data.simulate(Scenario(), reps, horizon)
        alt = data.simulate(sc, reps, horizon)
    S["lab_scenario"] = sc_dict
    cmp = compare(base, alt, ["door_to_doc_mean_min", "boarding_mean_h", "lwbs", "ICU_full_hours",
                              "electives_cancelled", "ICU_occ_peak", "MED_occ_mean", "diverted"]).set_index("metric")

    def delta_card(metric, label, unit, color, fmt="{:.1f}", good_down=True):
        r = cmp.loc[metric]
        d = r["delta"]
        better = (d < 0) == good_down
        dc = GREEN if abs(d) < 1e-9 else (GREEN if better else RED)
        return kpi_card(label, (fmt.format(r["scenario"]) + unit),
                        f"<span style='color:{dc}'>{d:+.1f}{unit}</span> vs baseline {fmt.format(r['baseline'])}{unit}",
                        color)

    card_row([
        delta_card("door_to_doc_mean_min", "Door-to-doctor", " min", CYAN, "{:.0f}"),
        delta_card("boarding_mean_h", "ED boarding", " h", AMBER),
        delta_card("lwbs", "Left w/o being seen", "", PINK, "{:.0f}"),
        delta_card("ICU_full_hours", "Hours ICU full", " h", RED, "{:.0f}"),
        delta_card("electives_cancelled", "Electives cancelled", "", VIOLET, "{:.0f}"),
    ])
    st.plotly_chart(floor_map(alt.sample_ts, alt.beds, ctx.cfg.theatres,
                              max(1, ctx.cfg.lounge_beds + sc.lounge_delta), step=2,
                              title=f"{sc.name} — animated twin"),
                    width="stretch", config={"displayModeBar": False})

section("Occupancy trajectories", "baseline vs scenario · mean with P10–P90 band")
c1, c2 = st.columns(2)
pairs = {"Baseline": (base, MUTED), sc.name: (alt, CYAN)}
c1.plotly_chart(projection_fig(pairs, "ICU", alt.beds["ICU"], "ICU occupied beds"), width="stretch")
c2.plotly_chart(projection_fig(pairs, "ed_occupied", alt.beds["ED"], "ED occupied spaces"), width="stretch")
c1.plotly_chart(projection_fig(pairs, "MED", alt.beds["MED"], "Medical ward"), width="stretch")
c2.plotly_chart(projection_fig(pairs, "ed_waiting", None, "ED waiting room"), width="stretch")

section("Outcome distributions", "each dot is one simulated future")
metrics = ["door_to_doc_mean_min", "boarding_mean_h", "ICU_full_hours", "lwbs"]
cols = st.columns(4)
for c, m in zip(cols, metrics):
    f = go.Figure()
    for label, mc, color in (("Baseline", base, MUTED), ("Scenario", alt, CYAN)):
        f.add_trace(go.Violin(y=mc.kpis[m], name=label, line_color=color, fillcolor=color, opacity=0.45,
                              box_visible=True, meanline_visible=True, points="all", pointpos=0, jitter=0.4,
                              marker=dict(size=4)))
    f.update_layout(title=METRIC_LABELS[m], height=300, showlegend=False)
    c.plotly_chart(f, width="stretch")

section("Patient flow under this scenario", "mean patients per simulated week")
st.plotly_chart(sankey(alt.flows), width="stretch")

section("AI response plan", "recommendations generated and verified for this scenario")
if st.button("🧠 Generate response plan", type="primary"):
    S["lab_recs"] = sc_dict
if S.get("lab_recs") == sc_dict:
    with st.spinner("Detecting risks, generating interventions and re-simulating each one…"):
        R = data.recommend(sc, reps)
    st.markdown(R["brief"])
    for r in R["recs"]:
        if not r.accepted:
            continue
        a, b = st.columns([1.3, 1])
        a.markdown(rec_card(r), unsafe_allow_html=True)
        if r.impact is not None:
            b.plotly_chart(impact_fig(r.impact, METRIC_LABELS, WEIGHTS), width="stretch", key=f"imp_{r.rid}")
