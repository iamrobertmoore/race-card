#!/usr/bin/env python3
"""The two Gemma steps, run on your own machine with Ollama. Python standard library only.

  python3 scripts/on_device.py races path/to/activities.csv [--key races.json] [--out gemma-races.json]
  python3 scripts/on_device.py note docs/index.html [--course weymouth]

Needs Ollama running with Gemma 4:  ollama pull gemma4:e4b
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from racecard import gemma  # noqa: E402  (stdlib-only module)

DATA_RE = re.compile(r'(<script id="data" type="application/json">)(.*?)(</script>)', re.S)


def hm(s: float) -> str:
    m = int(round(s / 60))
    return f"{m // 60}:{m % 60:02d}"


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def chance(course: dict, levels: list, goal: int, trans: int, gains=(0, 0, 0), n: int = 6000) -> float:
    rnd = random.Random(7)
    hit = 0
    for _ in range(n):
        tot = trans
        for i, leg in enumerate(course["legs"]):
            u = levels[0] + rnd.random() * (levels[-1] - levels[0])
            tot += leg["km"] / (_interp(u, levels, leg["kmh_q"]) * (1 + gains[i])) * 3600
        hit += tot < goal
    return hit / n


def cmd_races(a) -> None:
    if not gemma.ollama_ready():
        sys.exit("Ollama isn't running with Gemma 4. Open Ollama, then: ollama pull gemma4:e4b")
    acts = gemma.read_activity_titles(a.csv)
    t0 = time.time()

    def prog(done, total, found):
        print(f"\r  Gemma has read {done:>5} of {total} titles · {found} races so far · {time.time() - t0:5.0f}s", end="", flush=True)

    print(f"Reading {len(acts)} activities with {gemma.MODEL} on this machine. Nothing leaves it.")
    found = gemma.find_races(acts, progress=prog)
    print()
    out = {"model": gemma.MODEL, "seconds": round(time.time() - t0), "activities": len(acts), "races": found}
    if a.key:
        out["score"] = gemma.score(found, json.loads(Path(a.key).read_text()))
        s = out["score"]
        print(f"  Against my hand-checked list: {s['agree']} of {s['answer_key']} found "
              f"(recall {s['recall']:.0%}), {s['gemma_found']} flagged (precision {s['precision']:.0%}).")
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"  Saved to {a.out} in {out['seconds']}s.")


def cmd_note(a) -> None:
    if not gemma.ollama_ready():
        sys.exit("Ollama isn't running with Gemma 4. Open Ollama, then: ollama pull gemma4:e4b")
    html = Path(a.page).read_text(encoding="utf-8")
    m = DATA_RE.search(html)
    d = json.loads(m.group(2).replace("<\\/", "</"))
    c = next(c for c in d["courses"] if c["id"] == a.course)
    best = d["courses"][0]
    lv, goal, tr = d["levels"], d["goal_s"], d["transitions_s"]
    base = chance(c, lv, goal, tr)
    lift = {}
    for i, leg in enumerate(c["legs"]):
        g = [0, 0, 0]
        g[i] = 0.03
        lift[leg["sport"]] = round(chance(c, lv, goal, tr, tuple(g)) * 100)
    facts = {
        "athlete": d["built_for"], "goal": hm(goal), "race": c["name"],
        "chance_percent": round(base * 100), "predicted_finish": hm(c["median_s"]),
        "best_course": best["name"], "best_course_chance_percent": round(best["p_goal"] * 100),
        "chance_percent_if_3_percent_faster": lift,
    }
    print("Facts given to Gemma:\n" + json.dumps(facts, indent=1, ensure_ascii=False))
    t0 = time.time()
    note, bad = gemma.write_note(facts)
    if note is None:
        sys.exit(f"Gemma's note invented numbers {bad} three times; page left unchanged." if bad else "No answer from Ollama.")
    print(f"\nGemma wrote, in {time.time() - t0:.0f}s:\n  {note}")
    d["note"] = {"text": note, "source": "gemma", "model": gemma.MODEL, "facts": facts}
    blob = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    Path(a.page).write_text(html[:m.start(2)] + blob + html[m.end(2):], encoding="utf-8")
    print(f"Note saved into {a.page}.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("races")
    r.add_argument("csv")
    r.add_argument("--key")
    r.add_argument("--out", default="gemma-races.json")
    n = sub.add_parser("note")
    n.add_argument("page")
    n.add_argument("--course", default="weymouth")
    a = p.parse_args()
    {"races": cmd_races, "note": cmd_note}[a.cmd](a)


if __name__ == "__main__":
    main()
