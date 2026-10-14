import pandas as pd
import streamlit as st
from charts import floor_map, projection_fig
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
    occ_color,
    rec_card,
    section,
)

import data
from hospital_twin.config import HOSPITAL_CITY, HOSPITAL_NAME

data.require_build()
ctx = data.ctx()
state = ctx.state
base = data.baseline()
R = data.recommendations()
hourly = ctx.hist["hourly"]

hero("Hospital Digital Twin", f"{HOSPITAL_NAME} · {HOSPITAL_CITY} — predictive operations & decision intelligence",
     f"LIVE · twin synced {state.time:%a %d %b %Y, %H:%M}", state.status)

last24 = hourly.iloc[-48:]
u = state.units
d_arr = state.arrivals_24h - state.arrivals_prev_24h
card_row([
    kpi_card("ED arrivals · 24h", f"{state.arrivals_24h}",
             f"<span style='color:{RED if d_arr > 0 else GREEN}'>{d_arr:+d}</span> vs previous 24h", PINK,
             spark=last24["arrivals"].rolling(3, min_periods=1).mean().to_numpy()),
    kpi_card("ED now", f"{u['ED'].occupied}<span style='font-size:1rem;color:{MUTED}'>/{u['ED'].beds}</span>",
             f"{state.ed_waiting} waiting · <b style='color:{AMBER}'>{state.ed_boarding}</b> boarding",
             PINK, fill=u["ED"].occupancy / 100, spark=last24["ed_occupied"].to_numpy()),
    kpi_card("ICU", f"{u['ICU'].occupancy:.0f}%", f"{u['ICU'].occupied}/{u['ICU'].beds} beds · {u['ICU'].free} free",
             RED, fill=u["ICU"].occupancy / 100, spark=last24["ICU"].to_numpy()),
    kpi_card("Medical ward", f"{u['MED'].occupancy:.0f}%", f"{u['MED'].occupied}/{u['MED'].beds} beds · {u['MED'].free} free",
             CYAN, fill=u["MED"].occupancy / 100, spark=last24["MED"].to_numpy()),
    kpi_card("Surgical ward", f"{u['SURG'].occupancy:.0f}%",
             f"{u['SURG'].occupied}/{u['SURG'].beds} beds · {u['SURG'].electives} elective",
             VIOLET, fill=u["SURG"].occupancy / 100, spark=last24["SURG"].to_numpy()),
])

section("Live floor plan — next 72 hours", "press ▶ to watch the twin simulate the coming days")
left, right = st.columns([2.35, 1])
with left:
    ts = base.sample_ts.iloc[:72]
    st.plotly_chart(floor_map(ts, base.beds, ctx.cfg.theatres, ctx.cfg.lounge_beds, step=1),
                    width="stretch", config={"displayModeBar": False})
with right:
    st.markdown("<div class='kpi-label' style='margin:.2rem 0 .5rem 0'>Top actions · simulation-verified</div>",
                unsafe_allow_html=True)
    shown = [r for r in R["recs"] if r.accepted and r.rid != "demand_alert"][:3]
    for r in shown:
        st.markdown(rec_card(r, compact=True), unsafe_allow_html=True)
    if not shown:
        st.success("No intervention needed — the twin projects safe operations this week.")
    st.page_link("views/recommendations.py", label="Open full AI briefing →", icon="🧠")

section("7-day outlook", "ML forecast · P90 risk")
arr = ctx.champion("ed_arrivals").iloc[:7]
icu = ctx.champion("icu_census").iloc[:7]
med = ctx.champion("med_census").iloc[:7]
cols = st.columns(7)
recent = ctx.hist["daily"]["ed_arrivals"].iloc[-28:].mean()
for c, d in zip(cols, arr.index):
    icu_pct = icu.loc[d, "hi"] / ctx.cfg.beds["ICU"] * 100
    med_pct = med.loc[d, "hi"] / ctx.cfg.beds["MED"] * 100
    risk = max(icu_pct, med_pct)
    col = occ_color(risk)
    c.markdown(
        f"<div class='day' style='background:linear-gradient(180deg,{col}1f,transparent);border-color:{col}55'>"
        f"<div class='d'>{d:%a %d}</div><div class='v'>{arr.loc[d, 'yhat']:.0f}</div>"
        f"<div class='s'>arrivals ({(arr.loc[d, 'yhat'] / recent - 1) * 100:+.0f}%)</div>"
        f"<div class='s' style='margin-top:.3rem'>ICU P90 <b style='color:{occ_color(icu_pct)}'>{icu_pct:.0f}%</b></div>"
        f"<div class='s'>Ward P90 <b style='color:{occ_color(med_pct)}'>{med_pct:.0f}%</b></div></div>",
        unsafe_allow_html=True)

section("Twin projection", f"{len(base.kpis)} Monte-Carlo futures · mean and P10–P90")
c1, c2, c3 = st.columns(3)
c1.plotly_chart(projection_fig({"ICU": (base, RED)}, "ICU", base.beds["ICU"], "ICU occupied beds"), width="stretch")
c2.plotly_chart(projection_fig({"Medical": (base, CYAN)}, "MED", base.beds["MED"], "Medical ward occupied beds"),
                width="stretch")
c3.plotly_chart(projection_fig({"ED": (base, PINK)}, "ed_occupied", base.beds["ED"], "ED occupied spaces"),
                width="stretch")

k = base.kpis.mean()
section("Projected week at a glance")
card_row([
    kpi_card("Door-to-doctor", f"{k['door_to_doc_mean_min']:.0f} min", f"P90 {k['door_to_doc_p90_min']:.0f} min", CYAN),
    kpi_card("ED boarding", f"{k['boarding_mean_h']:.1f} h", f"P90 {k['boarding_p90_h']:.1f} h", AMBER),
    kpi_card("Left without being seen", f"{k['lwbs_rate']:.1f}%", f"{k['lwbs']:.0f} patients", PINK),
    kpi_card("Hours ICU full", f"{k['ICU_full_hours']:.0f} h", f"{(base.kpis['ICU_full_hours'] > 0).mean() * 100:.0f}% of futures", RED),
    kpi_card("Electives cancelled", f"{k['electives_cancelled']:.0f}", f"of {k['electives_listed']:.0f} listed", VIOLET),
])
with st.expander("Executive briefing", expanded=False):
    st.markdown(R["brief"])
    st.caption(f"Generated by: {R['brief_source']} · built {pd.Timestamp(data.build_info()['built_at']):%d %b %H:%M}")
