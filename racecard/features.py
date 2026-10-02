"""Turn sessions into model rows. Every feature is something the athlete's own export contains."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["distance_km", "elev_per_km", "age_days", "load_7h", "load_42h", "sport_km_42", "is_race"]


def _loads_at(sessions: pd.DataFrame, at: np.ndarray, sport_of: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training hours in the 7 and 42 days before each time in `at`, and same-sport km in 42 days."""
    t = sessions.date.values.astype("datetime64[s]").astype(np.int64)
    order = np.argsort(t)
    t, hrs, km, sp = t[order], (sessions.moving_s.values / 3600.0)[order], sessions.distance_km.values[order], sessions.sport.values[order]
    ch = np.concatenate([[0], np.cumsum(hrs)])
    a = at.astype("datetime64[s]").astype(np.int64)
    i_now = np.searchsorted(t, a, side="left")
    i7 = np.searchsorted(t, a - 7 * 86400, side="left")
    i42 = np.searchsorted(t, a - 42 * 86400, side="left")
    l7, l42 = ch[i_now] - ch[i7], ch[i_now] - ch[i42]
    skm = np.zeros(len(a))
    if sport_of is not None:
        for s in np.unique(sport_of):
            ck = np.concatenate([[0], np.cumsum(np.where(sp == s, km, 0.0))])
            m = sport_of == s
            skm[m] = ck[i_now[m]] - ck[i42[m]]
    return l7, l42, skm


def session_table(sessions: pd.DataFrame) -> pd.DataFrame:
    """Per-session features that don't depend on the prediction date. Cached on the frame."""
    cached = sessions.attrs.get("_feat")
    if cached is not None and len(cached) == len(sessions):
        return cached
    l7, l42, skm = _loads_at(sessions, sessions.date.values, sessions.sport.values)
    t = pd.DataFrame({
        "date": sessions.date.values, "sport": sessions.sport.values,
        "distance_km": sessions.distance_km.values,
        "elev_per_km": sessions.elev_m.values / np.maximum(sessions.distance_km.values, 0.1),
        "load_7h": l7, "load_42h": l42, "sport_km_42": skm, "is_race": sessions.is_race.values,
        "_kmh": sessions.distance_km.values / (sessions.moving_s.values / 3600.0),
    })
    sessions.attrs["_feat"] = t
    return t


def training_rows(sessions: pd.DataFrame, sport: str, ref: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows for one sport, using only sessions strictly before `ref`. Target: speed in km/h."""
    t = session_table(sessions)
    t = t[(t.sport == sport) & (t.date < ref)].copy()
    t["age_days"] = (ref - t.date).dt.days
    return t[FEATURES].reset_index(drop=True), t["_kmh"].to_numpy()


def race_row(sessions: pd.DataFrame, sport: str, ref: pd.Timestamp, distance_km: float, elev_m: float) -> pd.DataFrame:
    """The row for race day: today's fitness, the course's distance and climbing, race effort."""
    l7, l42, skm = _loads_at(sessions, np.array([np.datetime64(ref)]), np.array([sport]))
    return pd.DataFrame([{
        "distance_km": distance_km, "elev_per_km": elev_m / max(distance_km, 0.1), "age_days": 0,
        "load_7h": float(l7[0]), "load_42h": float(l42[0]), "sport_km_42": float(skm[0]), "is_race": 1,
    }], columns=FEATURES)
