"""For every past race leg: train only on the sessions before race day, predict it, compare."""
from __future__ import annotations

import pandas as pd

from .model import fit_sport, seconds_at, speed_quantiles


def backtest(sessions: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in sessions[sessions.is_race == 1].iterrows():
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
