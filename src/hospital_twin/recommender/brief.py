"""Executive briefing: templated, optionally rewritten by Claude when an API key is set."""

from __future__ import annotations

import json
import os

from .engine import Recommendation

MODEL = os.environ.get("HOSPITAL_TWIN_LLM_MODEL", "claude-opus-5-5")


def template_brief(state, risk: dict, recs: list[Recommendation]) -> str:
    acc = [r for r in recs if r.accepted and r.rid != "demand_alert"]
    lines = [
        f"**Status: {state.status}** as of {state.time:%a %d %b %Y, %H:%M}.",
        f"ED has {state.ed_waiting} waiting, {state.ed_treatment} in treatment and {state.ed_boarding} boarding; "
        f"ICU {state.units['ICU'].occupied}/{state.units['ICU'].beds}, medical ward "
        f"{state.units['MED'].occupancy:.0f}% and surgical ward {state.units['SURG'].occupancy:.0f}% occupied.",
        f"Next 7 days: ~{risk['arrivals_next7']:.0f} ED arrivals ({risk['arrivals_change_pct']:+.1f}% vs last 4 weeks); "
        f"ICU forecast to peak at {risk['icu_fc_peak_hi']:.0f} beds (P90). The twin projects the ICU full for "
        f"{risk['icu_full_h']:.0f} h and mean ED boarding of {risk['boarding_mean']:.1f} h if nothing changes.",
    ]
    if acc:
        lines.append("")
        lines.append("**Recommended actions (simulation-verified):**")
        for r in acc[:5]:
            imp = "; ".join(r.headline_impact(2))
            lines.append(f"- {r.icon} **{r.title}** — owner: {r.owner}" + (f". Expected: {imp}" if imp else ""))
    return "\n".join(lines)


def llm_brief(state, risk: dict, recs: list[Recommendation]) -> str | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        payload = {
            "status": state.status, "time": str(state.time),
            "units": {k: {"occupied": u.occupied, "beds": u.beds} for k, u in state.units.items()},
            "ed": {"waiting": state.ed_waiting, "boarding": state.ed_boarding},
            "risk": {k: (str(v) if not isinstance(v, (int, float)) else round(v, 2)) for k, v in risk.items()},
            "recommendations": [
                {"title": r.title, "severity": r.severity, "owner": r.owner, "accepted": r.accepted,
                 "impact": r.headline_impact(3), "rationale": r.rationale} for r in recs
            ],
        }
        msg = anthropic.Anthropic().messages.create(
            model=MODEL, max_tokens=700,
            system=("You are the operations analyst of a hospital digital twin. Write a crisp executive "
                    "briefing (max 180 words, markdown, bullet points allowed) for the hospital COO. Use only "
                    "the numbers provided. Lead with the single most important decision."),
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception as e:  # network / auth problems fall back to the template
        return f"_(LLM briefing unavailable: {type(e).__name__})_\n\n" + template_brief(state, risk, recs)


def executive_brief(state, risk: dict, recs: list[Recommendation]) -> tuple[str, str]:
    """Returns (markdown, source) where source is 'claude' or 'template'."""
    text = llm_brief(state, risk, recs)
    if text:
        return text, "claude"
    return template_brief(state, risk, recs), "template"
