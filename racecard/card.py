"""Write the race card: one self-contained HTML file with the computed JSON inside."""
from __future__ import annotations

import json
from importlib.resources import files


def hm(sec: float) -> str:
    m = int(round(sec / 60.0))
    return f"{m // 60}:{m % 60:02d}"


def hms(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render(data: dict) -> str:
    tpl = files("racecard").joinpath("web/template.html").read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return tpl.replace("/*__DATA__*/", blob)
