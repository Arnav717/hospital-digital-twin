"""Static configuration of the modelled hospital.

All times inside the simulation are expressed in hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

HOSPITAL_NAME = "Aarogya Multispecialty Hospital"
HOSPITAL_CITY = "Chennai"
HOSPITAL_LATLON = (12.9716, 80.2200)

# Inpatient units that hold beds (ED handled separately).
INPATIENT_UNITS = ("ICU", "MED", "SURG")
ALL_UNITS = ("ED", "ICU", "MED", "SURG")
UNIT_NAMES = {
    "ED": "Emergency Department",
    "ICU": "Intensive Care Unit",
    "MED": "Medical Ward",
    "SURG": "Surgical Ward",
    "OT": "Operating Theatres",
    "LOUNGE": "Discharge Lounge",
}

SHIFTS = ("Day", "Evening", "Night")
# Shift start hours: Day 07-15, Evening 15-23, Night 23-07.
SHIFT_START = {"Day": 7, "Evening": 15, "Night": 23}


def shift_of_hour(hour: int) -> str:
    if 7 <= hour < 15:
        return "Day"
    if 15 <= hour < 23:
        return "Evening"
    return "Night"


@dataclass(frozen=True)
class LogNormal:
    """Log-normal distribution parametrised by mean and coefficient of variation."""

    mean: float
    cv: float

    @property
    def sigma(self) -> float:
        import math

        return math.sqrt(math.log(1 + self.cv**2))

    @property
    def mu(self) -> float:
        import math

        return math.log(self.mean) - self.sigma**2 / 2


@dataclass(frozen=True)
class HospitalConfig:
    beds: dict[str, int] = field(
        default_factory=lambda: {"ED": 36, "ICU": 24, "MED": 180, "SURG": 120}
    )
    theatres: int = 7
    lounge_beds: int = 16
    triage_nurses: int = 3
    # ED physicians on duty per shift.
    ed_doctors: dict[str, int] = field(
        default_factory=lambda: {"Day": 7, "Evening": 7, "Night": 4}
    )
    # Patients per nurse (day / evening); nights are relaxed by night_ratio_factor.
    nurse_ratio: dict[str, float] = field(
        default_factory=lambda: {"ED": 4.0, "ICU": 2.0, "MED": 6.0, "SURG": 6.0}
    )
    night_ratio_factor: float = 1.25

    # Emergency Severity Index mix (ESI 1..5).
    esi_mix: tuple[float, ...] = (0.02, 0.12, 0.40, 0.36, 0.10)
    mass_casualty_mix: tuple[float, ...] = (0.25, 0.45, 0.25, 0.05, 0.0)
    # Probability of arriving by ambulance per ESI level.
    ambulance_prob: tuple[float, ...] = (0.9, 0.6, 0.3, 0.08, 0.02)
    # Disposition probabilities per ESI: HOME, MED, SURG, ICU, OT.
    disposition: dict[int, tuple[float, ...]] = field(
        default_factory=lambda: {
            1: (0.05, 0.20, 0.00, 0.55, 0.20),
            2: (0.40, 0.35, 0.10, 0.08, 0.07),
            3: (0.73, 0.18, 0.06, 0.01, 0.02),
            4: (0.97, 0.02, 0.01, 0.00, 0.00),
            5: (1.00, 0.00, 0.00, 0.00, 0.00),
        }
    )
    # Mean patience (hours) before leaving without being seen, per ESI.
    lwbs_patience: tuple[float, ...] = (1e9, 1e9, 8.0, 4.0, 3.0)

    triage_time: LogNormal = LogNormal(0.12, 0.4)
    doctor_time: tuple[LogNormal, ...] = (
        LogNormal(1.0, 0.5),
        LogNormal(0.8, 0.5),
        LogNormal(0.6, 0.5),
        LogNormal(0.35, 0.5),
        LogNormal(0.25, 0.5),
    )
    ed_workup: tuple[LogNormal, ...] = (
        LogNormal(3.0, 0.6),
        LogNormal(3.0, 0.6),
        LogNormal(2.5, 0.6),
        LogNormal(1.2, 0.6),
        LogNormal(0.6, 0.6),
    )
    los_hours: dict[str, LogNormal] = field(
        default_factory=lambda: {
            "MED": LogNormal(5.0 * 24, 0.9),
            "SURG": LogNormal(4.0 * 24, 0.8),
            "SURG_ELECTIVE": LogNormal(2.8 * 24, 0.6),
            "ICU": LogNormal(3.2 * 24, 1.0),
            "ICU_POSTOP": LogNormal(1.5 * 24, 0.7),
            "STEPDOWN": LogNormal(4.0 * 24, 0.8),
        }
    )
    surgery_time: LogNormal = LogNormal(2.0, 0.5)
    emergency_surgery_time: LogNormal = LogNormal(2.2, 0.5)
    transport_time: LogNormal = LogNormal(2.0, 0.6)

    icu_stepdown_prob: float = 0.75
    emergency_postop_icu_prob: float = 0.25
    elective_postop_icu_prob: float = 0.15
    elective_daily_mean: float = 22.0
    # Hours an admitted boarder waits before accepting an outlier bed on another ward.
    overflow_after: float = 3.0
    # Ward discharges are released inside this window (hour of day).
    discharge_window: tuple[int, int] = (10, 18)
    elective_bed_reserve: int = 2

    def required_nurses(self, unit: str, census: float, shift: str) -> int:
        import math

        ratio = self.nurse_ratio[unit]
        if shift == "Night":
            ratio *= self.night_ratio_factor
        return max(2, math.ceil(census / ratio))


DEFAULT_CONFIG = HospitalConfig()

# Fictional partner hospitals used for patient redistribution.
PARTNER_HOSPITALS = [
    {"name": "Partner Hospital — Tambaram", "lat": 12.9249, "lon": 80.1000, "km": 14.0,
     "free": {"ICU": 4, "MED": 18, "SURG": 10}},
    {"name": "Partner Hospital — Guindy", "lat": 13.0067, "lon": 80.2206, "km": 6.5,
     "free": {"ICU": 2, "MED": 9, "SURG": 6}},
    {"name": "Partner Hospital — Porur", "lat": 13.0382, "lon": 80.1565, "km": 12.0,
     "free": {"ICU": 3, "MED": 12, "SURG": 12}},
    {"name": "Partner Hospital — Sholinganallur", "lat": 12.9010, "lon": 80.2279, "km": 9.0,
     "free": {"ICU": 1, "MED": 14, "SURG": 8}},
]
