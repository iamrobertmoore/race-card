"""Turn sessions into model rows. Every feature is something the athlete's own export contains."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["distance_km", "elev_per_km", "age_days", "load_7h", "load_42h", "sport_km_42", "is_race"]


def _rolling_hours(dates: pd.Series, hours: pd.Series, at: pd.Timestamp, days: int) -> float:
    m = (dates < at) & (dates >= at - pd.Timedelta(days=days))
    return float(hours[m].sum())


def training_rows(sessions: pd.DataFrame, sport: str, ref: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows for one sport, using only sessions strictly before `ref`. Target: speed in km/h."""
    before = sessions[sessions.date < ref]
    hours = before.moving_s / 3600.0
    rows = []
    for _, s in before[before.sport == sport].iterrows():
        at = s.date
        same = (before.sport == sport) & (before.date < at) & (before.date >= at - pd.Timedelta(days=42))
        rows.append({
            "distance_km": s.distance_km,
            "elev_per_km": s.elev_m / max(s.distance_km, 0.1),
            "age_days": (ref - at).days,
            "load_7h": _rolling_hours(before.date, hours, at, 7),
            "load_42h": _rolling_hours(before.date, hours, at, 42),
            "sport_km_42": float(before.distance_km[same].sum()),
            "is_race": int(s.is_race),
            "_kmh": s.distance_km / (s.moving_s / 3600.0),
        })
    X = pd.DataFrame(rows, columns=FEATURES + ["_kmh"])
    return X[FEATURES], X["_kmh"].to_numpy()


def race_row(sessions: pd.DataFrame, sport: str, ref: pd.Timestamp, distance_km: float, elev_m: float) -> pd.DataFrame:
    """The row for race day: today's fitness, the course's distance and climbing, race effort."""
    before = sessions[sessions.date < ref]
    hours = before.moving_s / 3600.0
    same = (before.sport == sport) & (before.date >= ref - pd.Timedelta(days=42))
    return pd.DataFrame([{
        "distance_km": distance_km,
        "elev_per_km": elev_m / max(distance_km, 0.1),
        "age_days": 0,
        "load_7h": _rolling_hours(before.date, hours, ref, 7),
        "load_42h": _rolling_hours(before.date, hours, ref, 42),
        "sport_km_42": float(before.distance_km[same].sum()),
        "is_race": 1,
    }], columns=FEATURES)
