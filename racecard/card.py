"""Write the race card as one self-contained HTML file."""
from __future__ import annotations

import html
import json

import numpy as np


def hm(sec: float) -> str:
    sec = int(round(sec / 60.0)) * 60
    return f"{sec // 3600}:{sec % 3600 // 60:02d}"


def hms(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


LEG_NAMES = {"swim": "Swim", "bike": "Bike", "run": "Run"}

CSS = """
:root{--bg:#f4f2ec;--ink:#151a17;--muted:#5d655f;--green:#17463a;--green2:#2f7a62;--orange:#c75a22;--line:#dcd8cd;--card:#fbfaf6}
@media (prefers-color-scheme:dark){:root{--bg:#121513;--ink:#eef0ec;--muted:#a3aba5;--green:#7fc4a7;--green2:#5fae8c;--orange:#f08a52;--line:#2b302d;--card:#1a1e1b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 Geist,ui-sans-serif,system-ui,sans-serif}
.mono{font-family:"Geist Mono",ui-monospace,monospace}
.wrap{max-width:1080px;margin:0 auto;padding:40px 20px 80px}
.chip{display:inline-flex;gap:8px;align-items:center;font-size:14px;border:1px solid var(--line);background:var(--card);padding:6px 12px;border-radius:999px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green2)}
h1{font-size:clamp(36px,6vw,64px);line-height:1;letter-spacing:-.03em;margin:18px 0 18px}
h1 em{font-style:normal;color:var(--orange)}
.lede{font-size:19px;color:var(--muted);max-width:640px}
.card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:26px 28px;margin-top:28px}
.lab{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.big{font-size:clamp(56px,10vw,84px);font-weight:700;letter-spacing:-.04em;margin:4px 0 0}
.band{position:relative;height:44px;margin:22px 0 6px}.axis{position:absolute;left:0;right:0;top:21px;height:2px;background:var(--line)}
.int{position:absolute;top:11px;height:22px;border-radius:11px;background:color-mix(in srgb,var(--green2) 35%,transparent)}
.med{position:absolute;top:8px;width:28px;height:28px;margin-left:-14px;border-radius:50%;background:var(--green);border:4px solid var(--card)}
.goal{position:absolute;top:0;bottom:0;width:2px;background:var(--orange)}.goal span{position:absolute;left:8px;top:-4px;font-size:12px;color:var(--orange);white-space:nowrap}
.ticks{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}
.legs{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:20px}@media(max-width:640px){.legs{grid-template-columns:1fr}}
.leg{border:1px solid var(--line);border-radius:14px;padding:14px 16px}.leg b{display:block;font-size:26px}.leg i{font-style:normal;font-size:13px;color:var(--muted)}
.note{margin-top:20px;border-left:3px solid var(--orange);padding:4px 0 4px 14px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:500}
.ok{color:var(--green2)}.miss{color:var(--orange)}
.fine{font-size:13.5px;color:var(--muted)}
@media(max-width:640px){.rn{display:none}table{font-size:13px}th,td{padding:7px 4px}}
"""


def render(*, who: str, course: dict, legs: list, finish: np.ndarray, transitions_s: float,
           note_text: str, note_source: str, goal_s: float | None, backtest_rows: list[dict],
           data_label: str, generated: str) -> str:
    med, lo, hi = (float(np.percentile(finish, p)) for p in (50, 10, 90))
    span_lo = min(lo, goal_s or lo) - 10 * 60
    span_hi = max(hi, goal_s or hi) + 10 * 60
    pos = lambda s: (s - span_lo) / (span_hi - span_lo) * 100
    ticks = np.linspace(span_lo, span_hi, 4)
    e = html.escape
    leg_html = "".join(
        f'<div class="leg"><i>{LEG_NAMES[l.sport]} {l.distance_km:g} km</i><b>{hm(l.median_s) if l.sport != "swim" else hms(l.median_s)[2:]}</b>'
        f'<i class="mono">{(hm if l.sport != "swim" else (lambda s: hms(s)[2:]))(l.seconds_at(.1))} to {(hm if l.sport != "swim" else (lambda s: hms(s)[2:]))(l.seconds_at(.9))} · {l.n_train} sessions</i></div>'
        for l in legs)
    goal_html = f'<div class="goal" style="left:{pos(goal_s):.1f}%"><span>goal {hm(goal_s)}</span></div>' if goal_s else ""
    bt = ""
    if backtest_rows:
        inside = sum(r["inside_80"] for r in backtest_rows)
        rows = "".join(
            f'<tr><td class="mono">{r["date"]}</td><td class="rn">{e(r["name"])}</td><td>{LEG_NAMES[r["sport"]]}</td>'
            f'<td class="mono">{hms(r["pred_s"])}</td><td class="mono">{hms(r["actual_s"])}</td>'
            f'<td class="mono">{r["error_pct"]:+.1f}%</td><td class="{"ok" if r["inside_80"] else "miss"}">{"inside" if r["inside_80"] else "outside"}</td></tr>'
            for r in backtest_rows)
        bt = (f'<div class="card"><div class="lab">Checked against past races</div>'
              f'<p>Each leg below was predicted using only the training before that race day, then compared with what happened. '
              f'{inside} of {len(backtest_rows)} landed inside the 80% range.</p>'
              f'<div style="overflow-x:auto"><table><tr><th>Date</th><th class="rn">Race</th><th>Leg</th><th>Predicted</th><th>Actual</th><th>Off by</th><th>80% range</th></tr>{rows}</table></div></div>')
    src = "Written by Gemma 4 on this machine, from the numbers above only." if note_source == "gemma" else "Ollama wasn't running, so this is the plain template rather than Gemma's note."
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Race Card · {e(course['name'])}</title>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;600;700&family=Geist+Mono&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><div class="wrap">
<span class="chip"><span class="dot"></span>Made on this machine · nothing uploaded</span>
<h1>If {e(who)} raced {e(course['name'])} today:<br><em>about {hm(med)}.</em></h1>
<p class="lede">Predicted from {e(data_label)} with TabPFN v2, an open tabular model, running locally. The range is what the model is 80% sure of.</p>
<div class="card"><div class="lab">Predicted finish · {e(course['name'])}</div>
<div class="big">{hm(med)}</div><div class="mono fine">80% range {hm(lo)} to {hm(hi)} · includes {round(transitions_s/60)} min for transitions</div>
<div class="band"><div class="axis"></div><div class="int" style="left:{pos(lo):.1f}%;width:{pos(hi)-pos(lo):.1f}%"></div>{goal_html}<div class="med" style="left:{pos(med):.1f}%"></div></div>
<div class="ticks mono">{''.join(f'<span>{hm(t)}</span>' for t in ticks)}</div>
<div class="legs">{leg_html}</div>
<div class="note">{e(note_text)}</div><p class="fine">{src}</p></div>
{bt}
<p class="fine">Course: {e(course['notes'])} Source: {e(course['source'])}. Generated {e(generated)}. TabPFN v2 is licensed for non-commercial use; this card is for one person's training, not advice.</p>
</div></body></html>"""
