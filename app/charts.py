"""Plotly figure builders for the command centre."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from theme import (
    AMBER,
    BLUE,
    CYAN,
    GREEN,
    MODEL_COLOR,
    MUTED,
    PINK,
    RED,
    TEXT,
    UNIT_COLOR,
    VIOLET,
    occ_color,
)

FREE = "#1b2540"
BLOCKED = "#c084fc"

ROOMS = {
    "ED": dict(x0=1, y0=35, x1=31, y1=61, cols=9, size=15, label="Emergency Dept"),
    "ICU": dict(x0=33, y0=35, x1=53, y1=61, cols=6, size=15, label="ICU"),
    "OT": dict(x0=55, y0=35, x1=71, y1=61, cols=3, size=26, label="Theatres"),
    "LOUNGE": dict(x0=73, y0=35, x1=99, y1=61, cols=6, size=13, label="Discharge Lounge"),
    "MED": dict(x0=1, y0=1, x1=57, y1=32, cols=20, size=10, label="Medical Ward"),
    "SURG": dict(x0=59, y0=1, x1=99, y1=32, cols=15, size=10, label="Surgical Ward"),
}


def _grid(room: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    cols = room["cols"]
    rows = max(1, math.ceil(n / cols))
    xs = np.linspace(room["x0"] + 2.2, room["x1"] - 2.2, cols)
    ys = np.linspace(room["y1"] - 6, room["y0"] + 2.5, rows) if rows > 1 else np.array([room["y1"] - 6])
    pts = [(xs[i % cols], ys[i // cols]) for i in range(n)]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def floor_map(ts: pd.DataFrame, beds: dict, theatres: int, lounge: int, step: int = 2,
              title: str = "") -> go.Figure:
    """Animated top-down hospital floor plan driven by a twin timeseries."""
    caps = {"ED": beds["ED"], "ICU": beds["ICU"], "MED": beds["MED"], "SURG": beds["SURG"],
            "OT": theatres, "LOUNGE": max(lounge, 1)}
    pos = {u: _grid(ROOMS[u], caps[u]) for u in ROOMS}
    X = np.concatenate([pos[u][0] for u in ROOMS])
    Y = np.concatenate([pos[u][1] for u in ROOMS])
    S = np.concatenate([np.full(caps[u], ROOMS[u]["size"]) for u in ROOMS])
    wait_x = np.linspace(2.5, 30, 24)

    def colors(r) -> list[str]:
        occ = {
            "ED": [(PINK, int(r.ed_occupied - r.ed_boarding)), (AMBER, int(r.ed_boarding))],
            "ICU": [(RED, int(r.ICU - r.icu_blocked)), (BLOCKED, int(r.icu_blocked))],
            "MED": [(CYAN, int(r.MED))], "SURG": [(VIOLET, int(r.SURG))],
            "OT": [(AMBER, int(r.ot_busy))], "LOUNGE": [(GREEN, int(r.lounge))],
        }
        out = []
        for u in ROOMS:
            cs = []
            for c, n in occ[u]:
                cs += [c] * max(0, n)
            cs = (cs + [FREE] * caps[u])[: caps[u]]
            out += cs
        return out

    def annotations(r) -> list[dict]:
        vals = {"ED": r.ed_occupied, "ICU": r.ICU, "MED": r.MED, "SURG": r.SURG, "OT": r.ot_busy, "LOUNGE": r.lounge}
        ann = []
        for u, room in ROOMS.items():
            pct = vals[u] / caps[u] * 100
            col = occ_color(pct) if u in ("ED", "ICU", "MED", "SURG") else MUTED
            ann.append(dict(x=room["x0"] + 1.2, y=room["y1"] - 1.6, xanchor="left", yanchor="middle",
                            showarrow=False, font=dict(size=12, color=TEXT, family="Space Grotesk"),
                            text=f"<b>{room['label']}</b>  <span style='color:{col}'>{int(vals[u])}/{caps[u]}</span>"))
        ann.append(dict(x=1.2, y=64.6, xanchor="left", showarrow=False, font=dict(size=11, color=MUTED),
                        text=f"ED waiting room · <b style='color:{PINK}'>{int(r.ed_waiting)}</b> waiting"))
        q = int(r.MED_queue + r.SURG_queue + r.ICU_queue)
        ann.append(dict(x=99, y=64.6, xanchor="right", showarrow=False, font=dict(size=11, color=MUTED),
                        text=f"{r.time:%a %d %b · %H:%M} · patients awaiting a bed: <b style='color:{AMBER}'>{q}</b>"))
        return ann

    def wait_trace(r):
        n = int(min(r.ed_waiting, 24))
        return go.Scatter(x=wait_x[:n], y=np.full(n, 62.4), mode="markers", hoverinfo="skip",
                          marker=dict(size=9, color=PINK, symbol="circle", line=dict(width=0)), showlegend=False)

    rows = list(ts.iloc[::step].itertuples())
    r0 = rows[0]
    shapes = [dict(type="rect", x0=r["x0"], y0=r["y0"], x1=r["x1"], y1=r["y1"], layer="below",
                   line=dict(color=UNIT_COLOR[u], width=1.5), fillcolor="rgba(255,255,255,0.025)")
              for u, r in ROOMS.items()]
    shapes.append(dict(type="rect", x0=1, y0=61.3, x1=31, y1=63.6, layer="below",
                       line=dict(color="rgba(244,114,182,.35)", width=1, dash="dot")))
    hover = [f"{u} bed {i + 1}" for u in ROOMS for i in range(caps[u])]
    bed = go.Scatter(x=X, y=Y, mode="markers", text=hover, hoverinfo="text", showlegend=False,
                     marker=dict(size=S, symbol="square", color=colors(r0), line=dict(width=0)))
    fig = go.Figure(data=[bed, wait_trace(r0)])
    fig.frames = [go.Frame(name=str(i), data=[go.Scatter(marker=dict(color=colors(r), size=S)), wait_trace(r)],
                           layout=go.Layout(annotations=annotations(r)))
                  for i, r in enumerate(rows)]
    steps = [dict(method="animate", label=f"{r.time:%a %H:%M}",
                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for i, r in enumerate(rows)]
    legend = [(PINK, "ED patient"), (AMBER, "Boarding / OT busy"), (RED, "ICU patient"), (BLOCKED, "ICU exit-blocked"),
              (CYAN, "Medical"), (VIOLET, "Surgical"), (GREEN, "Lounge"), (FREE, "Free")]
    for c, name in legend:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(size=10, symbol="square", color=c)))
    fig.update_layout(
        title=title, height=560, shapes=shapes, annotations=annotations(r0),
        xaxis=dict(range=[0, 100], visible=False, fixedrange=True),
        yaxis=dict(range=[-1, 66], visible=False, fixedrange=True, scaleanchor=None),
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center", font=dict(size=11)),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.11, xanchor="left", showactive=False,
                          bgcolor="#0f1629", bordercolor="rgba(148,163,184,.3)", font=dict(color=TEXT),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=280, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, x=0.13, len=0.87, y=-0.06, pad=dict(t=0, b=0), steps=steps,
                      currentvalue=dict(visible=True, prefix="⏱ ", xanchor="right", font=dict(size=12, color=CYAN)),
                      bgcolor="#1e293b", activebgcolor=CYAN, bordercolor="rgba(0,0,0,0)", tickcolor="rgba(0,0,0,0)",
                      font=dict(size=1, color="rgba(0,0,0,0)"))],
    )
    return fig


NODE_COLOR = {
    "Walk-in": BLUE, "Ambulance": RED, "ESI 1": "#ef4444", "ESI 2": "#f97316", "ESI 3": "#facc15",
    "ESI 4": "#4ade80", "ESI 5": "#22d3ee", "Discharged from ED": GREEN, "Left without being seen": "#64748b",
    "Medical Ward": CYAN, "Surgical Ward": VIOLET, "ICU": RED, "Emergency Surgery": AMBER,
    "Elective list": "#e879f9", "Elective Surgery": AMBER, "Discharge Lounge": GREEN, "Discharged home": "#10b981",
    "Diverted to partner": "#64748b", "Cancelled — no bed": "#64748b", "Cancelled — theatre overrun": "#64748b",
    "Postponed": "#94a3b8",
}


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def sankey(flows: dict, title: str = "", height: int = 520) -> go.Figure:
    flows = {k: v for k, v in flows.items() if v >= 0.5}
    nodes = []
    for s, t in flows:
        for n in (s, t):
            if n not in nodes:
                nodes.append(n)
    idx = {n: i for i, n in enumerate(nodes)}
    colors = [NODE_COLOR.get(n, MUTED) for n in nodes]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, color=colors, pad=14, thickness=16, line=dict(width=0),
                  hovertemplate="%{label}: %{value:.0f}<extra></extra>"),
        link=dict(source=[idx[s] for s, _ in flows], target=[idx[t] for _, t in flows],
                  value=list(flows.values()), color=[_rgba(NODE_COLOR.get(s, "#94a3b8"), 0.28) for s, _ in flows],
                  hovertemplate="%{source.label} → %{target.label}: %{value:.0f}<extra></extra>"),
    ))
    fig.update_layout(title=title, height=height, font=dict(size=12, color=TEXT))
    return fig


def forecast_fig(history: pd.Series, fc: pd.DataFrame, title: str, capacity: float | None = None,
                 show_models: bool = False, days_back: int = 120) -> go.Figure:
    h = history.iloc[-days_back:]
    champ = fc[fc["champion"]].set_index("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h.rolling(7, center=True, min_periods=1).mean(), name="7-day trend",
                             line=dict(color="rgba(148,163,184,.55)", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=h.index, y=h, name="Observed", mode="lines",
                             line=dict(color=TEXT, width=1.6)))
    fig.add_trace(go.Scatter(x=list(champ.index) + list(champ.index[::-1]),
                             y=list(champ["hi"]) + list(champ["lo"][::-1]), fill="toself",
                             fillcolor="rgba(34,211,238,0.16)", line=dict(width=0), name="80% interval",
                             hoverinfo="skip"))
    xs = [h.index[-1]] + list(champ.index)
    fig.add_trace(go.Scatter(x=xs, y=[h.iloc[-1]] + list(champ["yhat"]), name=f"Forecast ({champ['model'].iloc[0]})",
                             mode="lines+markers", line=dict(color=CYAN, width=3), marker=dict(size=6)))
    if show_models:
        for m, g in fc[~fc["champion"]].groupby("model"):
            g = g.set_index("date")
            fig.add_trace(go.Scatter(x=g.index, y=g["yhat"], name=m, mode="lines",
                                     line=dict(color=MODEL_COLOR.get(m, MUTED), width=1.4, dash="dash")))
    if capacity:
        fig.add_hline(y=capacity, line=dict(color=RED, dash="dash", width=1.2),
                      annotation_text=f"capacity {capacity:.0f}", annotation_font_color=RED)
    fig.add_vline(x=h.index[-1], line=dict(color="rgba(148,163,184,.4)", width=1))
    fig.update_layout(title=title, height=420, hovermode="x unified")
    return fig


def projection_fig(mcs: dict, col: str, capacity: float | None, title: str, height: int = 330) -> go.Figure:
    """mcs: {label: (MCResult, color)} — mean line + P10–P90 band per scenario."""
    fig = go.Figure()
    for label, (mc, color) in mcs.items():
        t = mc.ts_mean["time"]
        fig.add_trace(go.Scatter(x=list(t) + list(t[::-1]), y=list(mc.ts_p90[col]) + list(mc.ts_p10[col][::-1]),
                                 fill="toself", fillcolor=_rgba(color, 0.15), line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=t, y=mc.ts_mean[col], name=label, line=dict(color=color, width=2.4)))
    if capacity:
        fig.add_hline(y=capacity, line=dict(color=RED, dash="dash", width=1.2))
    fig.update_layout(title=title, height=height, hovermode="x unified")
    return fig


def gauge(value: float, title: str, color: str, maxv: float = 100) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number=dict(suffix="%", font=dict(size=30, family="Space Grotesk")),
        title=dict(text=title, font=dict(size=13, color=MUTED)),
        gauge=dict(axis=dict(range=[0, maxv], tickcolor=MUTED, tickwidth=1), bar=dict(color=color, thickness=0.28),
                   bgcolor="rgba(255,255,255,0.03)", borderwidth=0,
                   steps=[dict(range=[0, 85], color="rgba(52,211,153,0.10)"),
                          dict(range=[85, 95], color="rgba(251,191,36,0.14)"),
                          dict(range=[95, maxv], color="rgba(248,113,113,0.18)")],
                   threshold=dict(line=dict(color=RED, width=2), thickness=0.8, value=95)),
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=0))
    return fig


def impact_fig(impact: pd.DataFrame, labels: dict, weights: dict | None = None) -> go.Figure:
    d = impact[impact["baseline"].abs() + impact["scenario"].abs() > 1e-6].copy()
    d["pct"] = np.where(d["baseline"].abs() > 1e-9, d["delta"] / d["baseline"].abs() * 100, 0)
    d["lo"] = np.where(d["baseline"].abs() > 1e-9, d["ci_lo"] / d["baseline"].abs() * 100, 0)
    d["hi"] = np.where(d["baseline"].abs() > 1e-9, d["ci_hi"] / d["baseline"].abs() * 100, 0)
    d = d.sort_values("pct")
    d["label"] = d["metric"].map(labels)
    w = d["metric"].map(weights or {}).fillna(1.0)
    colors = [GREEN if good > 0 else RED for good in -d["delta"] * w]
    fig = go.Figure(go.Bar(
        x=d["pct"], y=d["label"], orientation="h", marker=dict(color=colors, opacity=0.85),
        error_x=dict(type="data", symmetric=False, array=d["hi"] - d["pct"], arrayminus=d["pct"] - d["lo"],
                     color="rgba(226,232,240,.5)", thickness=1),
        customdata=np.stack([d["baseline"], d["scenario"]], axis=1),
        hovertemplate="%{y}: %{customdata[0]:.2f} → %{customdata[1]:.2f} (%{x:+.0f}%)<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="rgba(148,163,184,.5)", width=1))
    fig.update_layout(height=40 + 26 * len(d), xaxis_title="change vs baseline (%) · 90% CI",
                      margin=dict(l=10, r=10, t=10, b=40))
    return fig
