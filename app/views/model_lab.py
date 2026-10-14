import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from theme import CYAN, MODEL_COLOR, MUTED, TEXT, hero, section

import data
from hospital_twin.config import ARTIFACT_DIR
from hospital_twin.forecasting.models import TARGETS

data.require_build()
ctx = data.ctx()
info = data.build_info()
hero("Model Lab", "Architecture, validation and evaluation of every layer of the twin",
     f"built {pd.Timestamp(info['built_at']):%d %b %H:%M} in {info['seconds']:.0f}s")

section("System architecture")
st.graphviz_chart(f"""
digraph G {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.5;
  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 fontcolor="{TEXT}" color="#334155" fillcolor="#0f1629" penwidth=1.4];
  edge [color="#64748b" arrowsize=0.7];
  data [label="Hospital data\\n(ED encounters, census,\\nelectives, rosters)" color="#f472b6"];
  prep [label="Pre-processing\\ncalendar · holidays · lags" color="#f472b6"];
  fc [label="ML forecasting\\nSARIMA · LightGBM · Ensemble\\nback-test · SHAP · intervals" color="#22d3ee"];
  twin [label="Digital twin state\\nlive snapshot of every\\nbed, queue & patient" color="#22d3ee"];
  sim [label="Simulation engine\\nSimPy discrete-event\\nMonte-Carlo scenarios" color="#a78bfa"];
  opt [label="Optimisation\\nCP-SAT rosters & theatre lists\\nLP patient transfers" color="#fbbf24"];
  ai [label="AI recommendation engine\\nrisk detection · playbook ·\\nsimulation-verified impact" color="#34d399"];
  ui [label="Command centre\\n& REST API" color="#34d399"];
  data -> prep -> fc -> twin -> sim -> ai -> ui;
  fc -> opt -> ai; sim -> opt [style=dashed]; ai -> sim [label=" what-if" fontcolor="{MUTED}" fontsize=9 style=dashed];
}}
""", width="stretch")

ev = data.evaluation()
section("Forecast back-test", "10 rolling origins × 14-day horizon")
bt = pd.read_parquet(ARTIFACT_DIR / "backtest.parquet")
m = pd.DataFrame(ctx.forecast_info["metrics"])
f = go.Figure()
for mod in ["Seasonal naive", "SARIMA", "LightGBM", "Ensemble"]:
    g = m[m["model"] == mod]
    f.add_trace(go.Bar(x=[TARGETS[t].split(" (")[0] for t in g["target"]], y=g["MAPE"], name=mod,
                       marker_color=MODEL_COLOR[mod]))
f.update_layout(barmode="group", height=340, yaxis_title="MAPE %", title="Accuracy by model and target")
st.plotly_chart(f, width="stretch")
target = st.selectbox("Back-test detail", list(TARGETS), format_func=TARGETS.get)
champ = ctx.forecast_info["champions"][target]
b = bt[(bt["target"] == target) & (bt["model"] == champ)]
f = go.Figure()
y = ctx.hist["daily"][target]
y = y[y.index >= b["date"].min() - pd.Timedelta(days=14)]
f.add_trace(go.Scatter(x=y.index, y=y, name="Observed", line=dict(color=TEXT, width=1.5)))
for o, g in b.groupby("origin"):
    f.add_trace(go.Scatter(x=g["date"], y=g["yhat"], mode="lines", line=dict(color=CYAN, width=2),
                           showlegend=False, hovertemplate=f"origin {pd.Timestamp(o):%d %b}: %{{y:.1f}}<extra></extra>"))
f.update_layout(height=340, title=f"{champ}: each cyan segment is a 14-day forecast from a different origin")
st.plotly_chart(f, width="stretch")

if ev:
    section("Twin validation", "replaying 8 weeks of observed demand through the simulator")
    st.dataframe(pd.DataFrame(ev["validation"]), hide_index=True, width="stretch")
    section("Optimisation vs traditional practice")
    c1, c2 = st.columns(2)
    c1.dataframe(pd.DataFrame(ev["roster"]).drop(columns=["status"]), hide_index=True, width="stretch")
    c2.dataframe(pd.DataFrame(ev["electives"]).drop(columns=["status"]), hide_index=True, width="stretch")
    section("Ablation", "reactive operations vs simulation-verified AI plan, across stress scenarios")
    st.dataframe(pd.DataFrame(ev["ablation"]), hide_index=True, width="stretch")
else:
    st.info("Run `uv run hospital-twin evaluate` to add twin validation, optimisation benchmarks and the ablation study.")

section("Technology")
st.markdown("""
| Layer | Technology |
|---|---|
| Synthetic hospital & simulation | SimPy discrete-event engine, NumPy, calibrated log-normal service/LOS models |
| Forecasting | statsmodels SARIMAX, LightGBM (recursive multi-step), ensemble, rolling-origin CV |
| Explainability | SHAP TreeExplainer, residual-calibrated 80% intervals |
| Optimisation | Google OR-Tools CP-SAT (rostering, theatre scheduling), SciPy HiGHS LP (transfers) |
| Decision support | rule playbook + paired Monte-Carlo verification, optional Claude executive briefing |
| Interfaces | Streamlit + Plotly command centre, FastAPI REST API |
""")
