"""racecard <export> --race weymouth-70.3 [--name Sam] [--goal 5:30] [--races races.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from . import card, explain, ingest
from .model import finish_distribution, predict_leg
from .races import COURSES, DEFAULT_TRANSITIONS_S


def _parse_hm(s: str | None) -> float | None:
    if not s:
        return None
    h, m = s.split(":")
    return int(h) * 3600 + int(m) * 60


def build(export: str, race: str, name: str, goal: str | None, races_file: str | None, out: str, skip_backtest: bool = False) -> dict:
    sessions = ingest.load(export)
    if races_file:
        sessions = ingest.mark_races(sessions, [tuple(x) for x in json.loads(Path(races_file).read_text())])
    course = COURSES[race]
    ref = pd.Timestamp(dt.date.today()) + pd.Timedelta(days=1)
    legs = []
    for sport, km, elev in course["legs"]:
        p = predict_leg(sessions, sport, ref, km, elev)
        if p is None:
            raise SystemExit(f"Not enough {sport} sessions to predict the {sport} (need 12).")
        legs.append(p)
    finish = finish_distribution(legs, DEFAULT_TRANSITIONS_S)
    widths = {l.sport: l.seconds_at(.9) - l.seconds_at(.1) for l in legs}
    facts = {
        "race": course["name"], "athlete": name,
        "finish": card.hm(np.percentile(finish, 50)), "finish_low": card.hm(np.percentile(finish, 10)),
        "finish_high": card.hm(np.percentile(finish, 90)),
        "legs": {l.sport: {"predicted": card.hms(l.median_s), "low": card.hms(l.seconds_at(.1)), "high": card.hms(l.seconds_at(.9)),
                           "sessions_used": l.n_train} for l in legs},
        "widest_leg": max(widths, key=widths.get), "goal": goal, "course": course["notes"],
    }
    text, source = explain.note(facts)
    rows = [] if skip_backtest else bt.backtest(sessions).to_dict("records")
    html = card.render(who=name, course=course, legs=legs, finish=finish, transitions_s=DEFAULT_TRANSITIONS_S,
                       note_text=text, note_source=source, goal_s=_parse_hm(goal), backtest_rows=rows,
                       data_label=f"{len(sessions)} sessions in your export", generated=dt.datetime.now().strftime("%d %b %Y %H:%M"))
    Path(out).write_text(html, encoding="utf-8")
    return {"facts": facts, "note_source": source, "backtest": rows, "out": out}


def main() -> None:
    a = argparse.ArgumentParser(prog="racecard", description="A race-day prediction from your own training export, made on your machine.")
    a.add_argument("export", help="Strava archive (.zip or activities.csv) or Garmin Connect Activities CSV")
    a.add_argument("--race", default="weymouth-70.3", choices=sorted(COURSES))
    a.add_argument("--name", default="you")
    a.add_argument("--goal", help="goal finish time, h:mm")
    a.add_argument("--races", help="JSON list of [date, name fragment] marking your past races")
    a.add_argument("--out", default="race-card.html")
    a.add_argument("--no-open", action="store_true")
    args = a.parse_args()
    r = build(args.export, args.race, args.name, args.goal, args.races, args.out)
    print(f"Race card written to {r['out']} (note by {r['note_source']}).")
    if not args.no_open:
        webbrowser.open(Path(r["out"]).resolve().as_uri())


if __name__ == "__main__":
    main()
