"""racecard <export> [--name Sam] [--for Sam] [--goal 5:00] [--races races.json] [--out race-card.html]"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from . import card, explain, ingest, report


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 3600 + int(m) * 60


def facts_for(data: dict, course_id: str = "weymouth") -> dict:
    c = next((c for c in data["courses"] if c["id"] == course_id), data["courses"][0])
    best = data["courses"][0]
    widths = {l["sport"]: l["hi_s"] - l["lo_s"] for l in c["legs"]}
    return {
        "athlete": data["built_for"], "goal": card.hm(data["goal_s"]), "race": c["name"],
        "chance_of_goal_percent": round(c["p_goal"] * 100), "predicted_finish": card.hm(c["median_s"]),
        "finish_range_80": [card.hm(c["lo_s"]), card.hm(c["hi_s"])],
        "legs": {l["sport"]: {"predicted": card.hms(l["median_s"]), "range_80": [card.hms(l["lo_s"]), card.hms(l["hi_s"])]} for l in c["legs"]},
        "bike_climbing_m": c["bike_m"], "widest_range_leg": max(widths, key=widths.get),
        "best_course": best["name"], "best_course_chance_percent": round(best["p_goal"] * 100),
    }


def build(export: str, name: str, built_for: str, goal: str, races_file: str | None, out: str,
          backtest: bool = True, use_gemma: bool = True) -> dict:
    sessions = ingest.load(export)
    if races_file:
        sessions = ingest.mark_races(sessions, [tuple(x) for x in json.loads(Path(races_file).read_text())])
    data = report.build(sessions, athlete=name, built_for=built_for, goal_s=_hm(goal), run_backtest=backtest)
    if use_gemma:
        f = facts_for(data)
        text, source = explain.note(f)
        data["note"] = {"text": text, "source": source, "facts": f}
    Path(out).write_text(card.render(data), encoding="utf-8")
    return data


def main() -> None:
    a = argparse.ArgumentParser(prog="racecard", description="A race-day prediction from your own training export, made on your machine.")
    a.add_argument("export", help="Strava archive (.zip or activities.csv) or Garmin Connect Activities CSV")
    a.add_argument("--name", default="You", help="whose training this is")
    a.add_argument("--for", dest="built_for", help="who the card is for (defaults to --name)")
    a.add_argument("--goal", default="5:00", help="goal finish time, h:mm")
    a.add_argument("--races", help="JSON list of [date, name fragment] marking your past races")
    a.add_argument("--out", default="race-card.html")
    a.add_argument("--no-backtest", action="store_true")
    a.add_argument("--no-gemma", action="store_true")
    a.add_argument("--no-open", action="store_true")
    args = a.parse_args()
    d = build(args.export, args.name, args.built_for or args.name, args.goal, args.races, args.out,
              backtest=not args.no_backtest, use_gemma=not args.no_gemma)
    src = d["note"]["source"] if d.get("note") else "off"
    print(f"Race card written to {args.out}  (Gemma note: {src})")
    if not args.no_open:
        webbrowser.open(Path(args.out).resolve().as_uri())


if __name__ == "__main__":
    main()
