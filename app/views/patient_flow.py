import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from charts import sankey
from theme import (
    AMBER,
    CYAN,
    GREEN,
    PINK,
    RED,
    VIOLET,
    card_row,
    hero,
    kpi_card,
    section,
)

import data

data.require_build()
ctx = data.ctx()
enc = ctx.hist["encounters"]
daily = ctx.hist["daily"]
hero("Patient Flow Analytics", "Two years of encounters behind the twin — where patients come from, where they go, how long they stay",
     f"{len(enc):,} encounters · {len(daily)} days")

days = st.select_slider("Window", [30, 90, 180, 365, 730], value=90, format_func=lambda d: f"last {d} days")
e = enc[enc["arrival"] >= enc["arrival"].max() - pd.Timedelta(days=days)]
ed = e[~e["elective"]]
d2d = (ed["doc_start"] - ed["arrival"]).dt.total_seconds() / 60
edlos = (ed["ed_depart"] - ed["arrival"]).dt.total_seconds() / 3600
board = (ed["ed_depart"] - ed["dispo_time"]).dt.total_seconds() / 3600
admitted = ed["dispo"].isin(["MED", "SURG", "ICU", "OT"])
card_row([
    kpi_card("ED visits / day", f"{len(ed) / days:.0f}", f"{(ed['ambulance'].mean() * 100):.0f}% by ambulance", PINK),
    kpi_card("Admission rate", f"{admitted.mean() * 100:.1f}%", f"{(ed['dispo'] == 'ICU').mean() * 100:.1f}% to ICU", CYAN),
    kpi_card("Door-to-doctor", f"{d2d.median():.0f} min", f"P90 {d2d.quantile(.9):.0f} min", GREEN),
    kpi_card("ED length of stay", f"{edlos.median():.1f} h", f"P90 {edlos.quantile(.9):.1f} h", AMBER),
    kpi_card("Admitted boarding", f"{board[admitted].mean():.1f} h", f"P90 {board[admitted].quantile(.9):.1f} h", RED),
])

section("Patient journeys", "projected flows through the hospital over the coming week (twin baseline)")
st.plotly_chart(sankey(data.baseline().flows, height=560), width="stretch")

c1, c2 = st.columns(2)
with c1:
    section("Acuity mix")
    esi = ed["esi"].value_counts().sort_index()
    f = go.Figure(go.Pie(labels=[f"ESI {i}" for i in esi.index], values=esi.values, hole=0.62, sort=False,
                         marker=dict(colors=["#ef4444", "#f97316", "#facc15", "#4ade80", "#22d3ee"])))
    f.update_layout(height=320, annotations=[dict(text=f"<b>{len(ed):,}</b><br>visits", showarrow=False,
                                                  font=dict(size=16))])
    st.plotly_chart(f, width="stretch")
with c2:
    section("Length of stay by unit")
    f = go.Figure()
    for u, color in (("ICU", RED), ("MED", CYAN), ("SURG", VIOLET)):
        s = e[e["unit"] == u]
        los = ((s["discharge"] - s["unit_in"]).dt.total_seconds() / 86400).dropna()
        los = los[los < 30]
        f.add_trace(go.Violin(x=los, name=u, line_color=color, fillcolor=color, opacity=0.5, orientation="h",
                              side="positive", meanline_visible=True, points=False, width=1.6))
    f.update_layout(height=320, xaxis_title="days in hospital", showlegend=False)
    st.plotly_chart(f, width="stretch")

section("Occupancy history", "daily mean census vs staffed beds")
f = go.Figure()
for col, u, color in (("icu_census", "ICU", RED), ("med_census", "MED", CYAN), ("surg_census", "SURG", VIOLET),
                      ("ed_census", "ED", PINK)):
    pct = daily[col] / ctx.cfg.beds[u] * 100
    f.add_trace(go.Scatter(x=daily.index, y=pct.rolling(7, min_periods=1).mean(), name=u, line=dict(color=color, width=2)))
f.add_hline(y=95, line=dict(color=RED, dash="dot", width=1), annotation_text="95% — gridlock zone")
f.update_layout(height=360, yaxis_title="occupancy % (7-day mean)", hovermode="x unified")
st.plotly_chart(f, width="stretch")

c1, c2 = st.columns(2)
with c1:
    section("Seasonality", "ED arrivals vs latent demand drivers")
    f = go.Figure()
    f.add_trace(go.Scatter(x=daily.index, y=daily["ed_arrivals"], mode="lines", name="Observed",
                           line=dict(color="rgba(226,232,240,.35)", width=1)))
    f.add_trace(go.Scatter(x=daily.index, y=daily["ed_arrivals"].rolling(28, center=True).mean(), name="28-day trend",
                           line=dict(color=CYAN, width=3)))
    hol = daily[daily["is_holiday"]]
    f.add_trace(go.Scatter(x=hol.index, y=hol["ed_arrivals"], mode="markers", name="Public holiday",
                           marker=dict(color=AMBER, size=7, symbol="diamond")))
    f.update_layout(height=330, hovermode="x unified")
    st.plotly_chart(f, width="stretch")
with c2:
    section("ED waiting vs ward occupancy", "the exit-block mechanism")
    x = daily["med_census"] / ctx.cfg.beds["MED"] * 100
    f = go.Figure(go.Scatter(x=x, y=daily["boarding"], mode="markers",
                             marker=dict(color=daily["ed_arrivals"], colorscale="Plasma", size=7, opacity=0.75,
                                         colorbar=dict(title="arrivals")),
                             hovertemplate="ward %{x:.0f}% · boarding %{y:.1f}<extra></extra>"))
    f.update_layout(height=330, xaxis_title="medical ward occupancy %", yaxis_title="patients boarding in ED (mean)")
    st.plotly_chart(f, width="stretch")
