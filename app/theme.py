"""Visual system: CSS, Plotly template and small HTML components."""

from __future__ import annotations

import html

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG = "#070b18"
PANEL = "#0f1629"
GRID = "rgba(148,163,184,0.10)"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
CYAN = "#22d3ee"
VIOLET = "#a78bfa"
PINK = "#f472b6"
AMBER = "#fbbf24"
GREEN = "#34d399"
RED = "#f87171"
BLUE = "#60a5fa"
UNIT_COLOR = {"ED": PINK, "ICU": RED, "MED": CYAN, "SURG": VIOLET, "OT": AMBER, "LOUNGE": GREEN}
SEVERITY_COLOR = {"critical": RED, "high": AMBER, "medium": BLUE, "info": GREEN}
MODEL_COLOR = {"Seasonal naive": "#64748b", "SARIMA": BLUE, "LightGBM": GREEN, "Ensemble": PINK}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3, h4, .hero-title, .kpi-value {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }}
.stApp {{
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,0.10), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(167,139,250,0.10), transparent 60%),
    radial-gradient(800px 600px at 50% 120%, rgba(244,114,182,0.06), transparent 60%),
    {BG};
}}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0b1224 0%, #070b18 100%); border-right: 1px solid rgba(148,163,184,0.08); }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; max-width: 1500px; }}
.hero {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  padding: 1.2rem 1.5rem; border-radius: 20px; margin-bottom: 1.1rem;
  background: linear-gradient(120deg, rgba(34,211,238,0.10), rgba(167,139,250,0.08) 50%, rgba(244,114,182,0.08));
  border: 1px solid rgba(148,163,184,0.12); box-shadow: 0 10px 40px rgba(0,0,0,0.35); }}
.hero-title {{ font-size: 2.0rem; font-weight: 700; margin: 0;
  background: linear-gradient(90deg, #e0f2fe, {CYAN} 40%, {VIOLET} 75%, {PINK});
  -webkit-background-clip: text; background-clip: text; color: transparent; }}
.hero-sub {{ color: {MUTED}; font-size: 0.95rem; margin-top: 0.15rem; }}
.live {{ display:inline-flex; align-items:center; gap:.5rem; font-size:.8rem; color:{TEXT};
  padding:.35rem .8rem; border-radius:999px; background:rgba(15,22,41,.8); border:1px solid rgba(148,163,184,.15); }}
.dot {{ width:9px; height:9px; border-radius:50%; background:{GREEN}; box-shadow:0 0 0 0 rgba(52,211,153,.7); animation:pulse 1.8s infinite; }}
@keyframes pulse {{ 0% {{ box-shadow:0 0 0 0 rgba(52,211,153,.7);}} 70% {{ box-shadow:0 0 0 10px rgba(52,211,153,0);}} 100% {{ box-shadow:0 0 0 0 rgba(52,211,153,0);}} }}
.pill {{ display:inline-block; padding:.3rem .8rem; border-radius:999px; font-weight:600; font-size:.78rem; letter-spacing:.08em; }}
.card {{ background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
  border: 1px solid rgba(148,163,184,0.12); border-radius: 18px; padding: 1rem 1.1rem;
  box-shadow: 0 6px 24px rgba(0,0,0,0.25); backdrop-filter: blur(8px); height: 100%; }}
.kpi-label {{ color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.09em; }}
.kpi-value {{ font-size:1.9rem; font-weight:700; line-height:1.15; margin-top:.15rem; }}
.kpi-sub {{ color:{MUTED}; font-size:.8rem; }}
.bar {{ height:6px; border-radius:6px; background:rgba(148,163,184,.15); overflow:hidden; margin-top:.55rem; }}
.bar > span {{ display:block; height:100%; border-radius:6px; }}
.section {{ display:flex; align-items:center; gap:.6rem; margin:1.3rem 0 .6rem 0; }}
.section h3 {{ margin:0; font-size:1.15rem; }}
.section .line {{ flex:1; height:1px; background:linear-gradient(90deg, rgba(148,163,184,.25), transparent); }}
.section .tag {{ color:{MUTED}; font-size:.78rem; }}
.rec {{ position:relative; padding:1rem 1.1rem 1rem 1.3rem; border-radius:16px; margin-bottom:.8rem;
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
  border:1px solid rgba(148,163,184,0.12); overflow:hidden; }}
.rec::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--sev); box-shadow:0 0 18px var(--sev); }}
.rec-title {{ font-family:'Space Grotesk'; font-weight:600; font-size:1.05rem; }}
.rec-meta {{ color:{MUTED}; font-size:.78rem; margin:.15rem 0 .5rem 0; }}
.rec-impact {{ font-size:.85rem; color:{TEXT}; }}
.rec-impact span {{ display:inline-block; margin:.15rem .4rem .15rem 0; padding:.15rem .55rem; border-radius:8px; background:rgba(52,211,153,.10); border:1px solid rgba(52,211,153,.25); }}
.chip {{ display:inline-block; padding:.12rem .55rem; border-radius:8px; font-size:.72rem; font-weight:600; margin-right:.3rem; }}
.day {{ text-align:center; padding:.6rem .3rem; border-radius:14px; border:1px solid rgba(148,163,184,.12); }}
.day .d {{ color:{MUTED}; font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; }}
.day .v {{ font-family:'Space Grotesk'; font-size:1.25rem; font-weight:700; }}
.day .s {{ font-size:.72rem; color:{MUTED}; }}
div[data-testid="stMetric"] {{ background: rgba(255,255,255,0.03); border:1px solid rgba(148,163,184,.12); border-radius:14px; padding:.7rem .9rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; }}
.stTabs [data-baseweb="tab"] {{ background:rgba(255,255,255,.03); border-radius:10px; padding:.4rem 1rem; }}
.stTabs [aria-selected="true"] {{ background:rgba(34,211,238,.12) !important; }}
.brief {{ padding:1.1rem 1.3rem; border-radius:18px; border:1px solid rgba(34,211,238,.25);
  background: linear-gradient(135deg, rgba(34,211,238,.08), rgba(167,139,250,.06)); }}
</style>
"""


def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    _plotly_template()


def _plotly_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        colorway=[CYAN, VIOLET, PINK, AMBER, GREEN, BLUE, RED],
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=CYAN, font=dict(color=TEXT)),
        title=dict(font=dict(family="Space Grotesk, sans-serif", size=15)),
    )
    pio.templates["twin"] = t
    pio.templates.default = "twin"


def hero(title: str, subtitle: str, live_text: str, status: str | None = None) -> None:
    pill = ""
    if status:
        c = {"STABLE": GREEN, "STRAINED": AMBER, "CRITICAL": RED}.get(status, CYAN)
        pill = f'<span class="pill" style="background:{c}22;color:{c};border:1px solid {c}66">{status}</span>'
    st.markdown(
        f"""<div class="hero"><div><div class="hero-title">{html.escape(title)}</div>
        <div class="hero-sub">{html.escape(subtitle)}</div></div>
        <div style="display:flex;gap:.6rem;align-items:center">{pill}
        <span class="live"><span class="dot"></span>{html.escape(live_text)}</span></div></div>""",
        unsafe_allow_html=True,
    )


def section(title: str, tag: str = "") -> None:
    st.markdown(f'<div class="section"><h3>{html.escape(title)}</h3><div class="line"></div>'
                f'<span class="tag">{html.escape(tag)}</span></div>', unsafe_allow_html=True)


def sparkline(values, color: str, w: int = 120, h: int = 34) -> str:
    v = np.asarray(values, dtype=float)
    if v.size < 2 or not np.isfinite(v).all():
        return ""
    lo, hi = v.min(), v.max()
    rng = hi - lo or 1
    xs = np.linspace(2, w - 2, v.size)
    ys = h - 3 - (v - lo) / rng * (h - 6)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"2,{h} " + pts + f" {w - 2},{h}"
    gid = f"g{abs(hash((color, v.tobytes()))) % 10**8}"
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><defs><linearGradient id="{gid}" x1="0" x2="0" y1="0" y2="1">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity=".45"/><stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
            f'</linearGradient></defs><polygon points="{area}" fill="url(#{gid})"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/></svg>')


def kpi_card(label: str, value: str, sub: str = "", color: str = CYAN, fill: float | None = None,
             spark=None) -> str:
    bar = ""
    if fill is not None:
        f = max(0.0, min(1.0, fill))
        c = GREEN if f < 0.85 else AMBER if f < 0.95 else RED
        bar = f'<div class="bar"><span style="width:{f * 100:.0f}%;background:linear-gradient(90deg,{c}aa,{c})"></span></div>'
    sp = f'<div style="margin-top:.35rem">{sparkline(spark, color)}</div>' if spark is not None else ""
    return (f'<div class="card"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value" style="color:{color}">{value}</div>'
            f'<div class="kpi-sub">{sub}</div>{bar}{sp}</div>')


def card_row(cards: list[str]) -> None:
    cols = st.columns(len(cards))
    for c, h in zip(cols, cards):
        c.markdown(h, unsafe_allow_html=True)


def occ_color(pct: float) -> str:
    return GREEN if pct < 85 else AMBER if pct < 95 else RED


def rec_card(r, compact: bool = False) -> str:
    color = SEVERITY_COLOR.get(r.severity, BLUE)
    impact = "".join(f"<span>{html.escape(x)}</span>" for x in r.headline_impact(3 if not compact else 2))
    actions = "" if compact else "<ul style='margin:.4rem 0 .3rem 1rem;color:#cbd5e1;font-size:.86rem'>" + "".join(
        f"<li>{html.escape(a)}</li>" for a in r.actions) + "</ul>"
    rationale = "" if compact else f"<div style='font-size:.88rem;color:#cbd5e1'>{html.escape(r.rationale)}</div>"
    status = "" if r.accepted else f"<span class='chip' style='background:{MUTED}22;color:{MUTED}'>REJECTED BY SIMULATION</span>"
    return (f'<div class="rec" style="--sev:{color}"><div class="rec-title">{r.icon} {html.escape(r.title)}</div>'
            f'<div class="rec-meta"><span class="chip" style="background:{color}22;color:{color}">{r.severity.upper()}</span>'
            f'<span class="chip" style="background:rgba(148,163,184,.12);color:#cbd5e1">{html.escape(r.category)}</span>'
            f'{status} owner · {html.escape(r.owner)}</div>{rationale}{actions}'
            f'<div class="rec-impact">{impact}</div></div>')
