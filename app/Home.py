"""Streamlit entrypoint: `uv run hospital-twin app` (or `streamlit run app/Home.py`)."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Hospital Digital Twin", page_icon="🏥", layout="wide",
                   initial_sidebar_state="expanded")

from theme import setup_page  # noqa: E402

setup_page()

pages = [
    st.Page("views/command_center.py", title="Command Center", icon="🛰️", default=True),
    st.Page("views/forecasts.py", title="Forecasts", icon="📈"),
    st.Page("views/scenario_lab.py", title="Scenario Lab", icon="🧪"),
    st.Page("views/optimizer.py", title="Optimizer", icon="⚙️"),
    st.Page("views/recommendations.py", title="AI Recommendations", icon="🧠"),
    st.Page("views/patient_flow.py", title="Patient Flow", icon="🔀"),
    st.Page("views/model_lab.py", title="Model Lab", icon="🔬"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown(
        "<div style='font-family:Space Grotesk;font-size:1.25rem;font-weight:700;"
        "background:linear-gradient(90deg,#22d3ee,#a78bfa);-webkit-background-clip:text;color:transparent'>"
        "🏥 MediTwin</div><div style='color:#94a3b8;font-size:.8rem;margin-bottom:.6rem'>"
        "AI-powered hospital digital twin</div>", unsafe_allow_html=True)

nav.run()

with st.sidebar:
    st.markdown("---")
    st.caption("Forecast → Simulate → Optimise → Recommend")
    st.caption("Shresth Kumar Gupta · Arnav Kamra · SCOPE, VIT Chennai")
