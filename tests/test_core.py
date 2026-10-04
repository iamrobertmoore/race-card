"""Fast checks that need no model download. Run: python -m pytest -q"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from racecard import features, gemma, ingest, model  # noqa: E402

STRAVA_HEAD = ["Activity ID", "Activity Date", "Activity Name", "Activity Type", "Elapsed Time", "Distance",
               "Max Heart Rate", "Moving Time", "Distance", "Elevation Gain", "Average Heart Rate"]


def strava_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(STRAVA_HEAD)
    w.writerows(rows)
    return io.StringIO(buf.getvalue())


def test_strava_repeated_headers_and_comma_metres():
    # Strava's first Distance is km (swims in metres with a thousands comma); the second is metres.
    rows = [[1, "Jul 26, 2026, 7:00:00 AM", "T100 swim", "Swim", 2200, "2,192", "", 2115, "2192.0", "0", ""],
            [2, "Jul 26, 2026, 8:00:00 AM", "T100 bike", "Ride", 8600, "78.80", "", 8446, "78800.0", "650", "151"],
            [3, "Jul 27, 2026, 7:00:00 AM", "Morning Walk", "Walk", 900, "2.0", "", 900, "2000.0", "5", ""]]
    s = ingest.read_strava_activities(strava_csv(rows))
    assert list(s.sport) == ["swim", "bike"]          # walks dropped
    assert abs(s.distance_km.iloc[0] - 2.192) < 1e-6  # metres column used, not "2,192 km"
    assert s.elev_m.iloc[1] == 650


def test_implausible_speeds_are_dropped():
    rows = [[1, "Jul 9, 2023, 7:00:00 AM", "Oysterman tri swim", "Swim", 611, "1,080", "", 611, "1080", "0", ""]]
    assert len(ingest.read_strava_activities(strava_csv(rows))) == 0   # 6.4 km/h swimming is a GPS glitch


def test_garmin_swims_are_metres():
    g = io.StringIO('Activity Type,Date,Title,Distance,Time,Total Ascent,Avg HR\n'
                    'Pool Swim,2026-09-01 07:00:00,Pool,"1,500",00:30:00,--,--\n'
                    'Running,2026-09-02 07:00:00,Run,10.02,00:45:10,40,150\n')
    s = ingest.read_garmin_csv(g)
    assert s.distance_km.round(2).tolist() == [1.5, 10.02]


def _sessions():
    d = pd.date_range("2026-01-01", periods=60, freq="D")
    return pd.DataFrame({"date": d, "sport": ["run", "bike", "swim"] * 20, "name": "x",
                         "distance_km": [10.0, 40.0, 2.0] * 20, "moving_s": [3000.0, 5400.0, 2400.0] * 20,
                         "elev_m": [50.0, 400.0, 0.0] * 20, "avg_hr": np.nan, "is_race": 0})


def test_training_rows_only_use_the_past():
    s = _sessions()
    ref = pd.Timestamp("2026-02-01")
    X, y = features.training_rows(s, "run", ref)
    assert len(X) == len([1 for dd, sp in zip(s.date, s.sport) if sp == "run" and dd < ref])
    assert (X.age_days > 0).all()
    assert np.allclose(y, 12.0)  # 10 km in 50 min


def test_loads_count_the_right_window():
    s = _sessions()
    r = features.race_row(s, "bike", pd.Timestamp("2026-03-01"), 90, 1043)
    # 7 days before 1 Mar: sessions on 22-28 Feb, one a day, hours alternate 50/90/40 min
    week = s[(s.date >= "2026-02-22") & (s.date < "2026-03-01")]
    assert abs(r.load_7h.iloc[0] - week.moving_s.sum() / 3600) < 1e-9
    assert abs(r.elev_per_km.iloc[0] - 1043 / 90) < 1e-9


def test_finish_gets_faster_with_gains():
    q = [30 + i * 0.5 for i in range(19)]
    legs = [(90.0, q)]
    a = model.finish_samples(legs, 0)
    b = model.finish_samples(legs, 0, gains={0: 0.05})
    assert np.median(b) < np.median(a)


def test_gemma_note_validator_catches_invented_numbers():
    facts = {"chance_percent": 31, "goal": "5:00"}
    allowed = gemma.numbers_in(json.dumps(facts))
    assert gemma.numbers_in("a 31% chance of 5:00") <= allowed | {"70.3"}
    assert "4:45" in gemma.numbers_in("you could do 4:45") - allowed


def test_race_score():
    found = [{"date": "2022-09-18", "name": "70.3 Weymouth bike"}, {"date": "2022-09-14", "name": "4x1k @70.3 pace"}]
    key = [["2022-09-18", "70.3 Weymouth bike"], ["2022-09-18", "70.3 Weymouth run"]]
    s = gemma.score(found, key)
    assert (s["agree"], s["precision"], s["recall"]) == (1, 0.5, 0.5)


def test_baselines_only_use_the_past():
    from racecard import backtest
    s = _sessions()
    s.loc[s.index[-1], "is_race"] = 1
    races = s[s.is_race == 1]
    r = races.iloc[-1]
    ref = r.date.normalize()
    b = backtest._baselines(s, races, r, ref)
    assert set(b) == {"last_race", "boosted_trees", "straight_line", "recent_training"}
    assert b["last_race"] is None  # no earlier race to copy
    assert all(v is None or v > 0 for v in b.values())


def test_page_samples_match_the_browser():
    # Same generator, seed and order as the page. First draws of mulberry32(7), checked in the browser.
    from racecard.model import _mulberry32
    r = _mulberry32(7)
    first = [round(next(r), 6) for _ in range(3)]
    assert first == [0.011705, 0.061958, 0.976908]
    legs = [(1.9, [3.0] * 19), (90.0, [30.0] * 19), (21.1, [12.0] * 19)]
    s = model.page_samples(legs, 420)
    assert len(s) == model.PAGE_N and np.allclose(s, 420 + 1.9 / 3 * 3600 + 3 * 3600 + 21.1 / 12 * 3600)


def test_titles_skip_garmin_and_read_strava_zips(tmp_path):
    import zipfile
    from racecard import cli
    g = tmp_path / "garmin.csv"
    g.write_text("Activity Type,Date,Title,Distance,Time\nRunning,2026-01-01 07:00:00,Parkrun,5.0,00:22:00\n")
    assert cli._titles(str(g)) is None
    rows = [["1", "Jan 1, 2026, 7:00:00 AM", "Canterbury 10", "Run", "3600", "16.1", "", "3500", "16100", "40", ""]]
    csv_text = strava_csv(rows).getvalue()
    z = tmp_path / "export.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("export_1/activities.csv", csv_text)
    t = cli._titles(str(z))
    assert t and t[0]["name"] == "Canterbury 10"
