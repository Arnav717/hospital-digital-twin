import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from charts import forecast_fig
from theme import (
    CYAN,
    GREEN,
    MODEL_COLOR,
    MUTED,
    PINK,
    RED,
    card_row,
    hero,
    kpi_card,
    section,
)

import data
from hospital_twin.config import ARTIFACT_DIR
from hospital_twin.forecasting.models import FEATURE_LABELS, TARGETS

data.require_build()
ctx = data.ctx()
info = ctx.forecast_info
hero("Predictive Analytics", "Machine-learning forecasts of demand and occupancy with calibrated uncertainty",
     "horizon 14 days · champion models auto-selected by back-test")

CAP = {"icu_census": ctx.cfg.beds["ICU"], "med_census": ctx.cfg.beds["MED"],
       "surg_census": ctx.cfg.beds["SURG"], "ed_census": ctx.cfg.beds["ED"]}
c1, c2 = st.columns([3, 1])
target = c1.segmented_control("Target", list(TARGETS), format_func=lambda t: TARGETS[t].split(" (")[0],
                              default="ed_arrivals", label_visibility="collapsed") or "ed_arrivals"
show_models = c2.toggle("Compare all models", value=False)

fc = ctx.forecast[ctx.forecast["target"] == target]
champ = fc[fc["champion"]]
m = pd.DataFrame(info["metrics"])
mt = m[m["target"] == target].set_index("model")
champion = info["champions"][target]
card_row([
    kpi_card("Champion model", champion, "lowest back-test MAE", CYAN),
    kpi_card("MAE", f"{mt.loc[champion, 'MAE']:.1f}", f"MAPE {mt.loc[champion, 'MAPE']:.1f}%", GREEN),
    kpi_card("Skill vs naive", f"{mt.loc[champion, 'Skill vs naive %']:+.0f}%", "MAE reduction", PINK),
    kpi_card("Interval coverage", f"{info['coverage80'][target]:.0f}%", "target 80%", MUTED),
    kpi_card("Next 7 days", f"{champ['yhat'].iloc[:7].mean():.0f}", f"mean / day · peak {champ['hi'].iloc[:7].max():.0f} (P90)", CYAN),
])

st.plotly_chart(forecast_fig(ctx.hist["daily"][target], fc, TARGETS[target], CAP.get(target), show_models),
                width="stretch")

section("Why this forecast?", "SHAP explanations of the gradient-boosting model")
sg = pd.read_parquet(ARTIFACT_DIR / "shap_global.parquet")
sl = pd.read_parquet(ARTIFACT_DIR / "shap_local.parquet")
sg = sg[sg["target"] == target].sort_values("importance").tail(10)
c1, c2 = st.columns(2)
fig = go.Figure(go.Bar(x=sg["importance"], y=sg["feature"].map(FEATURE_LABELS).fillna(sg["feature"]),
                       orientation="h", marker=dict(color=sg["importance"], colorscale=[[0, "#1e3a5f"], [1, CYAN]])))
fig.update_layout(title="Global drivers (mean |SHAP|)", height=380)
c1.plotly_chart(fig, width="stretch")

loc = sl[sl["target"] == target]
days = sorted(loc["date"].unique())
day = c2.select_slider("Explain forecast for", options=days, value=days[0],
                       format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b"))
ld = loc[loc["date"] == day].copy()
ld["abs"] = ld["shap"].abs()
top = ld.sort_values("abs", ascending=False).head(8).iloc[::-1]
rest = ld["shap"].sum() - top["shap"].sum()
labels = ["Baseline"] + [f"{FEATURE_LABELS.get(f, f)} = {v:.0f}" if abs(v) >= 2 else f"{FEATURE_LABELS.get(f, f)} = {v:.2f}"
                         for f, v in zip(top["feature"], top["value"])] + ["Other features", "Forecast"]
base = float(ld["base"].iloc[0])
w = go.Figure(go.Waterfall(
    orientation="h", y=labels, x=[base] + list(top["shap"]) + [rest, 0],
    measure=["absolute"] + ["relative"] * (len(top) + 1) + ["total"],
    increasing=dict(marker=dict(color=RED)), decreasing=dict(marker=dict(color=GREEN)),
    totals=dict(marker=dict(color=CYAN)), connector=dict(line=dict(color="rgba(148,163,184,.3)")),
))
w.update_layout(title=f"Contribution breakdown · {pd.Timestamp(day):%A %d %b}", height=380, showlegend=False)
c2.plotly_chart(w, width="stretch")

section("Demand rhythm", "average ED arrivals by hour of week (history)")
h = ctx.hist["hourly"]["arrivals"]
hm = h.groupby([h.index.dayofweek, h.index.hour]).mean().unstack()
fig = go.Figure(go.Heatmap(z=hm.to_numpy(), x=[f"{i:02d}:00" for i in range(24)],
                           y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                           colorscale=[[0, "#0b1224"], [0.5, "#7c3aed"], [1, "#f472b6"]],
                           hovertemplate="%{y} %{x}: %{z:.1f} arrivals/h<extra></extra>"))
fig.update_layout(height=280, yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")

with st.expander("Model leaderboard (rolling-origin back-test)"):
    t = m[m["target"] == target][["model", "MAE", "RMSE", "MAPE", "Skill vs naive %"]].sort_values("MAE")
    st.dataframe(t.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.1f}%", "Skill vs naive %": "{:+.1f}%"}), hide_index=True, width="stretch")
    bt = pd.read_parquet(ARTIFACT_DIR / "backtest.parquet")
    b = bt[bt["target"] == target]
    err = b.assign(ae=(b["yhat"] - b["y"]).abs()).groupby(["model", "h"])["ae"].mean().reset_index()
    f = go.Figure()
    for mod, g in err.groupby("model"):
        f.add_trace(go.Scatter(x=g["h"], y=g["ae"], name=mod, mode="lines+markers",
                               line=dict(color=MODEL_COLOR.get(mod, MUTED), width=2)))
    f.update_layout(title="Error growth with forecast horizon", xaxis_title="days ahead", yaxis_title="MAE", height=320)
    st.plotly_chart(f, width="stretch")
