"""For every past race leg: train only on the sessions before race day, predict it, compare."""
from __future__ import annotations

import pandas as pd

from .model import predict_leg


def backtest(sessions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in sessions[sessions.is_race == 1].iterrows():
        ref = r.date.normalize()
        p = predict_leg(sessions, r.sport, ref, r.distance_km, r.elev_m)
        if p is None:
            continue
        lo, med, hi = p.seconds_at(0.1), p.median_s, p.seconds_at(0.9)
        rows.append({
            "date": ref.date().isoformat(), "name": r["name"], "sport": r.sport,
            "distance_km": round(r.distance_km, 2), "actual_s": round(r.moving_s),
            "pred_s": round(med), "lo_s": round(lo), "hi_s": round(hi),
            "error_pct": round((med - r.moving_s) / r.moving_s * 100, 1),
            "inside_80": bool(lo <= r.moving_s <= hi), "n_train": p.n_train,
        })
    return pd.DataFrame(rows)
