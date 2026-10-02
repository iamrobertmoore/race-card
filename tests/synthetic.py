"""A made-up athlete in Strava's activities.csv format. For tests only; never shown as real."""
import numpy as np
import pandas as pd


def strava_csv(path, seed=1, days=540):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-03-01 07:00")
    rows, aid = [], 1000
    for d in range(days):
        day = start + pd.Timedelta(days=d)
        fit = 1 + 0.12 * d / days  # slowly getting fitter
        for sport, p in (("Run", 0.45), ("Ride", 0.35), ("Swim", 0.25)):
            if rng.random() > p:
                continue
            if sport == "Run":
                dist = rng.uniform(5, 18); kmh = 10.5 * fit * (1 - 0.006 * dist) * rng.normal(1, .04); elev = dist * rng.uniform(2, 15)
            elif sport == "Ride":
                dist = rng.uniform(25, 100); elev = dist * rng.uniform(0.5, 16); kmh = 29 * fit * (1 - 0.001 * dist) * (1 - 0.018 * elev / dist) * rng.normal(1, .04)
            else:
                dist = rng.uniform(1.5, 3.2); kmh = 3.0 * fit * rng.normal(1, .04); elev = 0
            aid += 1
            rows.append(_row(aid, day, f"Morning {sport}", sport, dist, dist / kmh * 3600, elev))
    for when, label in (("2025-09-14", "70.3 race"), ("2026-07-26", "T100 race")):
        day = pd.Timestamp(when + " 07:00"); f = 1 + 0.12 * (day - start).days / days
        for sport, dist, kmh, elev in (("Swim", 1.9, 3.2 * f, 0), ("Ride", 90, 30 * f * .95, 900), ("Run", 21.1, 11.2 * f * .93, 30)):
            aid += 1
            rows.append(_row(aid, day, f"{label} {sport.lower()}", sport, dist, dist / kmh * 3600, elev))
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(aid, day, name, sport, dist_km, secs, elev):
    return {"Activity ID": aid, "Activity Date": day.strftime("%b %-d, %Y, %-I:%M:%S %p"), "Activity Name": name,
            "Activity Type": sport, "Elapsed Time": round(secs * 1.05), "Distance": round(dist_km, 2),
            "Max Heart Rate": "", "Moving Time": round(secs), "Distance.1": round(dist_km * 1000, 1),
            "Elevation Gain": round(elev, 1), "Average Heart Rate": ""}
