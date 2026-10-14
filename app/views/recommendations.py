import pandas as pd
import streamlit as st
from charts import impact_fig
from theme import AMBER, BLUE, GREEN, RED, card_row, hero, kpi_card, rec_card, section

import data
from hospital_twin.recommender.engine import METRIC_LABELS, WEIGHTS

data.require_build()
ctx = data.ctx()
R = data.recommendations()
recs = R["recs"]
hero("AI Decision Support", "Risks detected from forecasts, interventions proposed from the playbook, "
     "every action verified in the twin before it is recommended",
     f"{sum(r.accepted for r in recs)} recommended · {sum(not r.accepted for r in recs)} rejected", ctx.state.status)

sev = pd.Series([r.severity for r in recs if r.accepted]).value_counts()
risk = R["risk"]
card_row([
    kpi_card("Critical", f"{sev.get('critical', 0)}", "act within 12 h", RED),
    kpi_card("High", f"{sev.get('high', 0)}", "act within 24–48 h", AMBER),
    kpi_card("Medium / info", f"{sev.get('medium', 0) + sev.get('info', 0)}", "monitor", BLUE),
    kpi_card("ICU full risk", f"{risk['p_icu_full'] * 100:.0f}%", f"of simulated futures · {risk['icu_full_h']:.0f} h mean", RED),
    kpi_card("Demand change", f"{risk['arrivals_change_pct']:+.1f}%", "next 7 days vs last 4 weeks", GREEN),
])

section("Executive briefing", "Claude-written if ANTHROPIC_API_KEY is set, otherwise templated")
with st.container(border=True):
    st.markdown(R["brief"])
    st.caption(f"source: {R['brief_source']}")

section("Recommended actions", "ranked by severity, then simulated benefit")
for r in [r for r in recs if r.accepted]:
    a, b = st.columns([1.35, 1])
    a.markdown(rec_card(r), unsafe_allow_html=True)
    with b:
        if r.impact is not None:
            st.plotly_chart(impact_fig(r.impact, METRIC_LABELS, WEIGHTS), width="stretch", key=f"i_{r.rid}")
            st.caption(f"Benefit score {r.score:.1f} · paired Monte-Carlo, {len(data.baseline().kpis)} futures")
        elif r.evidence:
            st.json({k: v for k, v in r.evidence.items() if not isinstance(v, float) or v == v}, expanded=False)

rej = [r for r in recs if not r.accepted]
if rej:
    section("Considered but rejected", "the twin showed no net benefit — transparency over black-box advice")
    for r in rej:
        a, b = st.columns([1.35, 1])
        a.markdown(rec_card(r), unsafe_allow_html=True)
        if r.impact is not None:
            b.plotly_chart(impact_fig(r.impact, METRIC_LABELS, WEIGHTS), width="stretch", key=f"r_{r.rid}")

with st.expander("How recommendations are produced"):
    st.markdown("""
1. **Detect** – the champion forecasts (P90) and the baseline twin projection are scanned for ICU saturation,
   ward gridlock, ED crowding, boarding and elective cancellations.
2. **Propose** – each trigger maps to operational levers from a hospital playbook (surge beds, discharge drive,
   extra physicians, ambulance diversion, elective re-planning, nurse redeployment).
3. **Verify** – every lever is applied to the digital twin and re-simulated with *common random numbers*, so the
   difference against baseline isolates the effect of the action (90% CI shown as error bars).
4. **Rank** – a weighted benefit score (boarding hours, ICU-full hours, cancellations, waiting time, LWBS …) minus
   an action-cost term decides acceptance. Individually beneficial actions are combined and re-simulated as an
   integrated plan to capture interactions.
""")
