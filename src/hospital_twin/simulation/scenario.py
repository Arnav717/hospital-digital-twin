"""What-if scenario definitions for the digital twin."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Surge:
    """A mass-casualty style burst of high-acuity arrivals."""

    hour: float  # hours after simulation start
    patients: int
    duration: float = 2.0


@dataclass
class Scenario:
    name: str = "Baseline"
    description: str = "Expected demand from the forecast, current capacity and rosters."
    arrival_multiplier: float = 1.0
    surges: list[Surge] = field(default_factory=list)
    bed_delta: dict[str, int] = field(default_factory=dict)  # e.g. {"ICU": +4}
    doctor_delta: dict[str, int] = field(default_factory=dict)  # per shift
    elective_postpone_frac: float = 0.0
    # Optimised theatre list: per day index, cases as [icu_days, ward_days, theatre_hours].
    elective_plan: list[list[list[float]]] | None = None
    divert_frac: float = 0.0  # share of ambulance ESI 2-3 traffic diverted
    early_discharge: bool = False  # discharge-before-noon programme
    lounge_delta: int = 0
    los_multiplier: float = 1.0  # e.g. 1.1 for a sicker case-mix

    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        d = dict(d)
        d["surges"] = [Surge(**s) for s in d.get("surges", [])]
        return cls(**d)

    def with_(self, **changes) -> "Scenario":
        d = self.to_dict()
        d.update(changes)
        if "surges" in changes:
            d["surges"] = [asdict(s) if isinstance(s, Surge) else s for s in changes["surges"]]
        return Scenario.from_dict(d)


def preset_scenarios() -> dict[str, Scenario]:
    """Ready-made stress tests shown in the Scenario Lab."""
    return {
        "Baseline": Scenario(),
        "Monsoon Fever Surge": Scenario(
            name="Monsoon Fever Surge",
            description="Dengue / viral fever wave: +25% ED arrivals and a sicker case-mix.",
            arrival_multiplier=1.25,
            los_multiplier=1.1,
        ),
        "Mass Casualty Incident": Scenario(
            name="Mass Casualty Incident",
            description="Highway pile-up: 35 high-acuity patients arrive within 2 hours on day 2.",
            surges=[Surge(hour=34.0, patients=35, duration=2.0)],
        ),
        "ICU Bed Closure": Scenario(
            name="ICU Bed Closure",
            description="6 ICU beds closed for infection-control deep cleaning.",
            bed_delta={"ICU": -6},
        ),
        "Staff Shortage": Scenario(
            name="Staff Shortage",
            description="Sickness wave: 2 fewer ED doctors on every shift.",
            doctor_delta={"Day": -2, "Evening": -2, "Night": -1},
        ),
        "Festival Weekend": Scenario(
            name="Festival Weekend",
            description="Diwali weekend: +15% arrivals, burns & trauma surge on night 1.",
            arrival_multiplier=1.15,
            surges=[Surge(hour=20.0, patients=20, duration=4.0)],
        ),
    }
