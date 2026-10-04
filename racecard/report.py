"""Everything the race card shows, computed once, as plain JSON the page reads."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from . import backtest as bt
from .model import LEVELS, fit_sport, page_pctl, page_samples, seconds_at, speed_quantiles
from .races import COURSES, DEFAULT_TRANSITIONS_S, LEGS, leg_elev


def weekly_hours(sessions: pd.DataFrame, ref: pd.Timestamp, weeks: int = 52) -> list[dict]:
    start = ref - pd.Timedelta(weeks=weeks)
    s = sessions[(sessions.date >= start) & (sessions.date < ref)].copy()
    s["wk"] = ((s.date - start).dt.days // 7).clip(0, weeks - 1)
    out = []
    for w in range(weeks):
        g = s[s.wk == w]
        out.append({sp: round(float(g[g.sport == sp].moving_s.sum()) / 3600, 2) for sp in ("swim", "bike", "run")})
    return out


def course_odds(legs: list[dict], goal_s: int) -> dict:
    """Chance of the goal and the finish range, from the same simulated race days the page draws."""
    f = page_samples([(l["km"], l["kmh_q"]) for l in legs], DEFAULT_TRANSITIONS_S)
    return {"p_goal": round(float((f < goal_s).mean()), 4), "median_s": round(page_pctl(f, .5)),
            "lo_s": round(page_pctl(f, .1)), "hi_s": round(page_pctl(f, .9))}


def build(sessions: pd.DataFrame, *, athlete: str, built_for: str, goal_s: int, ref: pd.Timestamp | None = None,
          run_backtest: bool = True) -> dict:
    ref = ref or (pd.Timestamp(dt.date.today()) + pd.Timedelta(days=1))
    ids = list(COURSES)
    per_sport = {}
    for sport, km in LEGS:
        m, n = fit_sport(sessions, sport, ref)
        if m is None:
            raise SystemExit(f"Not enough {sport} sessions to predict the {sport} (need 12, found {n}).")
        qs = speed_quantiles(m, sessions, sport, ref, [(km, leg_elev(COURSES[c], sport)) for c in ids])
        per_sport[sport] = (qs, n)
    courses = []
    for i, cid in enumerate(ids):
        c = COURSES[cid]
        legs = [{"sport": sp, "km": km, "kmh_q": [round(v, 4) for v in per_sport[sp][0][i]], "n_train": per_sport[sp][1],
                 "median_s": round(seconds_at(km, per_sport[sp][0][i], .5)),
                 "lo_s": round(seconds_at(km, per_sport[sp][0][i], .1)), "hi_s": round(seconds_at(km, per_sport[sp][0][i], .9))}
                for sp, km in LEGS]
        courses.append({"id": cid, **c, "legs": legs, **course_odds(legs, goal_s)})
    courses.sort(key=lambda c: -c["p_goal"])
    return {
        "athlete": athlete, "built_for": built_for, "goal_s": goal_s, "ref": ref.date().isoformat(),
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "levels": LEVELS,
        "transitions_s": DEFAULT_TRANSITIONS_S,
        "sessions": {"n": int(len(sessions)), "first": sessions.date.min().date().isoformat(),
                     "last": sessions.date.max().date().isoformat(),
                     "by_sport": {k: int(v) for k, v in sessions.sport.value_counts().items()}},
        "weekly": weekly_hours(sessions, ref),
        "courses": courses,
        "backtest": bt.backtest(sessions) if run_backtest else [], "backtest_ran": run_backtest,
        "note": None,
    }
