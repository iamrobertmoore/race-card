"""Gemma 4, running locally through Ollama, turns the numbers into a plan a person would read.

The model only sees numbers we computed, and anything it writes that contains a number we didn't
give it is rejected. If Ollama isn't running, a plain template says the same thing without the prose.
"""
from __future__ import annotations

import json
import re
import urllib.request

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"

SYSTEM = (
    "You write a short race-day note for one athlete, from facts you are given. "
    "Use only the numbers in the facts, written exactly as given. Do not invent paces, times, distances or percentages. "
    "Plain British English, second person, no headings, no bullet points, at most 90 words. "
    "Say what the prediction is, how sure it is, and the single thing in the facts most worth working on."
)


def facts_text(f: dict) -> str:
    return json.dumps(f, indent=1, ensure_ascii=False)


def numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[:.]\d+)*", text))


def allowed_numbers(f: dict) -> set[str]:
    return numbers_in(facts_text(f)) | {str(i) for i in range(0, 11)} | {"70.3"}


def validate(text: str, f: dict) -> list[str]:
    """Numbers in the note that aren't in the facts."""
    return sorted(numbers_in(text) - allowed_numbers(f))


def ask_gemma(f: dict, timeout: float = 120.0) -> str | None:
    body = json.dumps({
        "model": MODEL, "stream": False, "options": {"temperature": 0.4},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": facts_text(f)}],
    }).encode()
    try:
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["message"]["content"].strip()
    except Exception:
        return None


def template(f: dict) -> str:
    return (f"If you raced {f['race']} today, your training says about {f['finish']} "
            f"(80% range {f['finish_low']} to {f['finish_high']}). "
            f"The leg with the widest range is the {f['widest_leg']}, so that is where consistency would pay most.")


def note(f: dict, tries: int = 3) -> tuple[str, str]:
    """Returns (text, source) where source is 'gemma' or 'template'."""
    for _ in range(tries):
        t = ask_gemma(f)
        if t is None:
            break
        if not validate(t, f):
            return t, "gemma"
    return template(f), "template"
