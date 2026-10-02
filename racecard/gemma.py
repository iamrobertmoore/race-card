"""Gemma 4 on your own machine, through Ollama. Standard library only, so it runs on a stock Mac.

Two jobs:
  find_races()  reads every activity title and picks out the ones that were races. Race Card needs
                to know which sessions were all-out efforts, and nobody wants to tag 15 years by hand.
  write_note()  turns the computed numbers into a short plan, and is rejected if it invents a number.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import time
import urllib.request

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"
DEFAULT_NAME = re.compile(r"^(morning|afternoon|evening|night|lunch)\s+(run|ride|swim|walk|workout)$", re.I)

RACE_SYSTEM = (
    "You label an athlete's activity log. Each numbered line is: date | type | distance | title | what else "
    "they did that day. Decide which lines were races the athlete competed in.\n"
    "RACES: a triathlon, duathlon or aquathlon leg (a swim, ride and run on the same day with an event name, "
    "e.g. '70.3 Weymouth bike', 'Outlaw Half Swim', 'T100 run'), a running race (parkrun, 10k, 10 mile, half "
    "marathon, marathon, a named road race), an open-water swim race, a swimrun, a DNF in a race.\n"
    "NOT RACES: training, intervals, tempo or 'race pace' work, bricks, Masters swim sessions, warm-ups, "
    "'pre race' shakeouts and openers, course recces, sportives and charity rides, holidays, titles that are "
    "only a date or a place, and parkruns run pushing a buggy or with a child.\n"
    "A full-distance or middle-distance event name on a swim, ride or run is a race even if the title is short. "
    "Answer only JSON: {\"races\": [line numbers that were races]}. Use [] if none."
)
RACE_TYPES = {"Run", "Ride", "Swim"}


def _chat(system: str, user: str, *, json_mode: bool, timeout: float = 180.0) -> str | None:
    # think: False because Gemma 4 reasons before answering by default, which made each call ~45s.
    body = {"model": MODEL, "stream": False, "think": False, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["message"]["content"]
    except Exception:
        return None


def ollama_ready() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
        return any(n.startswith(MODEL.split(":")[0]) for n in names)
    except Exception:
        return False


def read_activity_titles(path: str) -> list[dict]:
    """Date, type, title and distance from a Strava activities.csv, without pandas."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    head = rows[0]
    i_date, i_name, i_type = head.index("Activity Date"), head.index("Activity Name"), head.index("Activity Type")
    dist_idx = [i for i, h in enumerate(head) if h == "Distance"]
    out = []
    for r in rows[1:]:
        try:
            d = dt.datetime.strptime(r[i_date].strip(), "%b %d, %Y, %I:%M:%S %p").date().isoformat()
        except ValueError:
            continue
        try:
            km = float(r[dist_idx[-1]].replace(",", "")) / 1000 if len(dist_idx) > 1 else float(r[dist_idx[0]].replace(",", ""))
        except (ValueError, IndexError):
            km = 0.0
        out.append({"date": d, "type": r[i_type], "name": r[i_name].strip(), "km": round(km, 2)})
    return out


def _ask(chunk: list[dict]) -> set[int] | None:
    lines = "\n".join(f"{i + 1}. {a['date']} | {a['type']} | {a['km']} km | {a['name']} | same day: {a['same_day']}"
                      for i, a in enumerate(chunk))
    ans = _chat(RACE_SYSTEM, lines, json_mode=True)
    try:
        return {int(n) for n in json.loads(ans or "").get("races", [])}
    except (ValueError, TypeError, AttributeError):
        return None


def find_races(acts: list[dict], batch: int = 25, progress=None) -> list[dict]:
    """Outdoor swims, rides and runs with a title the athlete wrote. Titles Strava generates itself
    ('Morning Run') carry no information, and virtual sessions are never race-day efforts, so both
    are skipped before Gemma sees anything."""
    by_day: dict[str, set[str]] = {}
    for a in acts:
        by_day.setdefault(a["date"], set()).add(a["type"])
    cands = [dict(a, same_day=", ".join(sorted(by_day[a["date"]] - {a["type"]})) or "nothing else")
             for a in acts if a["type"] in RACE_TYPES and a["name"] and not DEFAULT_NAME.match(a["name"])]
    found = []
    for start in range(0, len(cands), batch):
        chunk = cands[start:start + batch]
        nums = _ask(chunk)
        if nums is None:  # a garbled answer: try the two halves on their own before giving up
            half = len(chunk) // 2
            a, b = _ask(chunk[:half]) or set(), _ask(chunk[half:]) or set()
            nums = a | {n + half for n in b}
        found += [{k: chunk[n - 1][k] for k in ("date", "type", "name", "km")} for n in sorted(nums) if 1 <= n <= len(chunk)]
        if progress:
            progress(min(start + batch, len(cands)), len(cands), len(found))
    return found


def score(found: list[dict], key: list[list[str]]) -> dict:
    """Compare Gemma's races with a hand-checked list of [date, title]."""
    k = {(d, n.strip()) for d, n in key}
    g = {(a["date"], a["name"]) for a in found}
    tp = k & g
    return {"gemma_found": len(g), "answer_key": len(k), "agree": len(tp),
            "precision": round(len(tp) / max(len(g), 1), 3), "recall": round(len(tp) / max(len(k), 1), 3),
            "gemma_only": sorted(g - k), "key_only": sorted(k - g)}


NOTE_SYSTEM = (
    "You write a short race-day note for one athlete, from facts you are given. "
    "Use only the numbers in the facts, written exactly as given. Do not invent paces, times, distances or percentages. "
    "Plain British English, second person, no headings, no lists, at most 80 words. "
    "Say the chance of the goal at this race, compare it with the best course in the facts, "
    "and name the single leg the facts say would move the chance most."
)


def numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[:.]\d+)*", text))


def write_note(facts: dict, tries: int = 3) -> tuple[str | None, list[str]]:
    """Returns (note, invented_numbers). note is None if every try invented something or Ollama is down."""
    allowed = numbers_in(json.dumps(facts)) | {str(i) for i in range(11)} | {"70.3"}
    bad: list[str] = []
    for _ in range(tries):
        t = _chat(NOTE_SYSTEM, json.dumps(facts, ensure_ascii=False, indent=1), json_mode=False)
        if t is None:
            return None, []
        t = t.strip().strip('"')
        bad = sorted(numbers_in(t) - allowed)
        if not bad:
            return t, []
    return None, bad
