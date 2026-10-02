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
    "You label an athlete's activity log. For each numbered line decide if it was a race the athlete "
    "competed in: a triathlon or duathlon leg, a running race (parkrun, 10k, half marathon, marathon), "
    "an open-water swim race, a time trial event. NOT races: training sessions, intervals, tempo or "
    "'race pace' work, warm-ups, 'pre race' shakeouts and openers, course recces, virtual or Zwift "
    "events, recovery, and parkruns or runs done pushing a buggy or running with a child. "
    "Answer only JSON: {\"races\": [line numbers that were races]}."
)


def _chat(system: str, user: str, *, json_mode: bool, timeout: float = 180.0) -> str | None:
    body = {"model": MODEL, "stream": False, "options": {"temperature": 0},
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


def find_races(acts: list[dict], batch: int = 40, progress=None) -> list[dict]:
    """Titles Strava generates itself ('Morning Run') carry no information and are skipped."""
    cands = [a for a in acts if a["name"] and not DEFAULT_NAME.match(a["name"])]
    found = []
    for start in range(0, len(cands), batch):
        chunk = cands[start:start + batch]
        lines = "\n".join(f"{i + 1}. {a['date']} | {a['type']} | {a['km']} km | {a['name']}" for i, a in enumerate(chunk))
        ans = _chat(RACE_SYSTEM, lines, json_mode=True)
        try:
            nums = {int(n) for n in json.loads(ans or "{}").get("races", [])}
        except (ValueError, TypeError, AttributeError):
            nums = set()
        found += [chunk[n - 1] for n in sorted(nums) if 1 <= n <= len(chunk)]
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
