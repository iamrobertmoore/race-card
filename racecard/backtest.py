"""For every past race leg: train only on the sessions before race day, predict it, compare."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import race_row, training_rows
from .model import MAX_ROWS, fit_sport, seconds_at, speed_quantiles


# Only race legs about the length of a 70.3 leg are checked: that's the question the card answers.
LONG = {"swim": (1.0, 3.0), "bike": (60.0, 120.0), "run": (15.0, 25.0)}


def _baselines(sessions: pd.DataFrame, races: pd.DataFrame, r, ref: pd.Timestamp) -> dict:
    """The same leg guessed four simpler ways, from the same training. km/h, or None if there's nothing to go on."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    X, y = training_rows(sessions, r.sport, ref)
    X, y = X.iloc[-MAX_ROWS:], y[-MAX_ROWS:]
    row = race_row(sessions, r.sport, ref, r.distance_km, r.elev_m)[X.columns]
    prev = races[(races.sport == r.sport) & (races.date < ref)]
    recent = sessions[(sessions.sport == r.sport) & (sessions.date < ref) & (sessions.date >= ref - pd.Timedelta(days=42))]
    return {
        # What most of us actually do: assume it'll go like last time.
        "last_race": float(prev.iloc[-1].distance_km / (prev.iloc[-1].moving_s / 3600)) if len(prev) else None,
        # The usual tabular model, trained on exactly the same rows.
        "boosted_trees": float(HistGradientBoostingRegressor(random_state=0).fit(X, y).predict(row)[0]),
        "straight_line": float(LinearRegression().fit(X, y).predict(row)[0]),
        "recent_training": float(np.median(recent.distance_km / (recent.moving_s / 3600))) if len(recent) else None,
    }


def backtest(sessions: pd.DataFrame) -> list[dict]:
    rows = []
    races = sessions[sessions.is_race == 1]
    races = races[[LONG[s][0] <= d <= LONG[s][1] for s, d in zip(races.sport, races.distance_km)]]
    for _, r in races.iterrows():
        ref = r.date.normalize()
        m, n = fit_sport(sessions, r.sport, ref)
        if m is None:
            continue
        kq = speed_quantiles(m, sessions, r.sport, ref, [(r.distance_km, r.elev_m)])[0]
        lo, med, hi = (seconds_at(r.distance_km, kq, q) for q in (0.1, 0.5, 0.9))
        rows.append({
            "date": ref.date().isoformat(), "name": r["name"], "sport": r.sport,
            "distance_km": round(float(r.distance_km), 2), "actual_s": int(round(r.moving_s)),
            "pred_s": int(round(med)), "lo_s": int(round(lo)), "hi_s": int(round(hi)),
            "error_pct": round((med - r.moving_s) / r.moving_s * 100, 1),
            "inside_80": bool(lo <= r.moving_s <= hi), "n_train": n,
            "baseline_error_pct": {k: (None if v is None or v <= 0 else
                                       round((r.distance_km / v * 3600 - r.moving_s) / r.moving_s * 100, 1))
                                   for k, v in _baselines(sessions, races, r, ref).items()},
        })
    return rows
