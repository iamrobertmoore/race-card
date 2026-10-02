"""For every past race leg: train only on the sessions before race day, predict it, compare."""
from __future__ import annotations

import pandas as pd

from .model import fit_sport, seconds_at, speed_quantiles


# Only race legs about the length of a 70.3 leg are checked: that's the question the card answers.
LONG = {"swim": (1.0, 3.0), "bike": (60.0, 120.0), "run": (15.0, 25.0)}


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
        })
    return rows
