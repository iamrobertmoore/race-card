"""Read a Strava archive or a Garmin Connect CSV into one sessions table.

Columns out: date (UTC-naive datetime), sport ("swim" | "bike" | "run"), name, distance_km,
moving_s, elev_m, avg_hr, is_race. Anything that isn't a swim, ride or run is dropped.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SPORT_MAP = {
    "run": "run", "trail run": "run", "virtual run": "run", "virtualrun": "run", "treadmill running": "run",
    "running": "run", "trail running": "run", "track running": "run", "treadmill run": "run",
    "ride": "bike", "virtual ride": "bike", "virtualride": "bike", "gravel ride": "bike", "road cycling": "bike",
    "cycling": "bike", "indoor cycling": "bike", "virtual cycling": "bike", "gravel cycling": "bike",
    "swim": "swim", "swimming": "swim", "pool swim": "swim", "lap swimming": "swim",
    "open water swimming": "swim", "open water swim": "swim",
}

RACE_WORDS = re.compile(r"\b(?:race|70\.3|ironman|triathlon|parkrun|half marathon|marathon|10k race|t100|sprint tri|olympic tri)\b", re.I)


def _num(s: pd.Series) -> pd.Series:
    """Numbers that may arrive as '1,234.5', '1.234,5', '--' or blank."""
    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (int, float, np.number)):
            return float(v)
        t = str(v).strip().replace(" ", "")
        if t in ("", "--", "-"):
            return np.nan
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):
            t = t.replace(",", "")
        elif re.fullmatch(r"-?\d+,\d+", t):
            t = t.replace(",", ".")
        try:
            return float(t)
        except ValueError:
            return np.nan
    return s.map(conv)


def _hms(s: pd.Series) -> pd.Series:
    """'1:02:03', '02:03', '3723' or '3723.0' -> seconds."""
    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        t = str(v).strip()
        if t in ("", "--"):
            return np.nan
        if ":" in t:
            parts = [float(p) for p in t.split(":")]
            sec = 0.0
            for p in parts:
                sec = sec * 60 + p
            return sec
        try:
            return float(t.replace(",", ""))
        except ValueError:
            return np.nan
    return s.map(conv)


def _sport(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(SPORT_MAP)


def read_strava_activities(fh) -> pd.DataFrame:
    raw = pd.read_csv(fh, dtype=str, keep_default_na=False)
    cols = list(raw.columns)
    # Strava repeats several headers: the first Distance is km, the later one ("Distance.1") metres.
    dist_m = _num(raw["Distance.1"]) if "Distance.1" in cols else _num(raw["Distance"]) * 1000
    moving = _hms(raw["Moving Time"]) if "Moving Time" in cols else _hms(raw["Elapsed Time"])
    hr_col = "Average Heart Rate" if "Average Heart Rate" in cols else None
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Activity Date"], format="mixed", errors="coerce"),
        "sport": _sport(raw["Activity Type"]),
        "name": raw.get("Activity Name", pd.Series([""] * len(raw))).astype(str),
        "distance_km": dist_m / 1000.0,
        "moving_s": moving,
        "elev_m": _num(raw["Elevation Gain"]) if "Elevation Gain" in cols else np.nan,
        "avg_hr": _num(raw[hr_col]) if hr_col else np.nan,
    })
    return _finish(out)


def read_garmin_csv(fh) -> pd.DataFrame:
    raw = pd.read_csv(fh, dtype=str, keep_default_na=False)
    cols = list(raw.columns)
    sport = _sport(raw["Activity Type"])
    dist = _num(raw["Distance"])
    # Garmin reports swims in metres and everything else in km (metric accounts).
    dist_km = np.where(sport == "swim", dist / 1000.0, dist)
    time_col = "Moving Time" if "Moving Time" in cols else "Time"
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], errors="coerce"),
        "sport": sport,
        "name": raw.get("Title", pd.Series([""] * len(raw))).astype(str),
        "distance_km": dist_km,
        "moving_s": _hms(raw[time_col]),
        "elev_m": _num(raw["Total Ascent"]) if "Total Ascent" in cols else np.nan,
        "avg_hr": _num(raw["Avg HR"]) if "Avg HR" in cols else np.nan,
    })
    return _finish(out)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["date", "sport", "distance_km", "moving_s"])
    df = df[(df.distance_km > 0) & (df.moving_s > 60)].copy()
    df["elev_m"] = df["elev_m"].fillna(0.0)
    df["is_race"] = df["name"].str.contains(RACE_WORDS).astype(int)
    df = _drop_implausible(df)
    return df.sort_values("date").reset_index(drop=True)


# Speeds (km/h) outside these are GPS glitches or mislabelled activities, not training.
PLAUSIBLE_KMH = {"swim": (1.0, 5.5), "bike": (8.0, 60.0), "run": (5.0, 25.0)}


def _drop_implausible(df: pd.DataFrame) -> pd.DataFrame:
    kmh = df.distance_km / (df.moving_s / 3600.0)
    lo = df.sport.map(lambda s: PLAUSIBLE_KMH[s][0])
    hi = df.sport.map(lambda s: PLAUSIBLE_KMH[s][1])
    return df[(kmh >= lo) & (kmh <= hi)]


def load(path: str | Path) -> pd.DataFrame:
    """A Strava archive (.zip), its activities.csv, or a Garmin Connect Activities CSV."""
    p = Path(path).expanduser()
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            name = next((n for n in z.namelist() if n.lower().endswith("activities.csv")), None)
            if not name:
                raise ValueError(f"{p.name} has no activities.csv. Is it a Strava archive?")
            return read_strava_activities(io.BytesIO(z.read(name)))
    head = p.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
    if "Activity ID" in head and "Activity Date" in head:
        return read_strava_activities(p)
    if "Activity Type" in head and "Date" in head:
        return read_garmin_csv(p)
    raise ValueError(f"Don't recognise {p.name}. Expected a Strava archive or a Garmin Connect CSV.")


def mark_races(df: pd.DataFrame, races: list[tuple[str, str]] | None) -> pd.DataFrame:
    """Override the name heuristic with the athlete's own list of (date, name fragment)."""
    if not races:
        return df
    df = df.copy()
    df["is_race"] = 0
    for day, frag in races:
        m = (df.date.dt.strftime("%Y-%m-%d") == day) & df.name.str.contains(re.escape(frag), case=False)
        df.loc[m, "is_race"] = 1
    return df
