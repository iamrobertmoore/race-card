"""IRONMAN 70.3 courses Race Card can predict. Climbing from IRONMAN's own Strava course routes
(IM) or finishers.com (F), checked 2 Oct 2026. Where a run's climbing isn't published it is treated
as flat and marked, because a made-up number would be worse."""

COURSES = {
    "weymouth": {"name": "IRONMAN 70.3 Weymouth", "where": "UK", "when": "12 Sep 2027", "swim": "sea",
                 "bike_m": 1043, "run_m": 24, "run_known": True, "src": "IM route 3311384455140007166; finishers.com"},
    "swansea": {"name": "IRONMAN 70.3 Swansea", "where": "UK", "when": "11 Jul 2027", "swim": "sea dock",
                "bike_m": 1106, "run_m": 0, "run_known": False, "src": "IM route 3311387332027418870"},
    "mallorca": {"name": "IRONMAN 70.3 Alcúdia-Mallorca", "where": "Spain", "when": "8 May 2027", "swim": "sea",
                 "bike_m": 882, "run_m": 0, "run_known": False, "src": "IM route 3314355818279989220"},
    "cascais": {"name": "IRONMAN 70.3 Portugal-Cascais", "where": "Portugal", "when": "17 Oct 2026", "swim": "sea",
                "bike_m": 771, "run_m": 141, "run_known": True, "src": "IM routes 3340350705873445894, 3333094951988521874"},
    "turkiye": {"name": "IRONMAN 70.3 Türkiye", "where": "Belek, Türkiye", "when": "1 Nov 2026", "swim": "sea",
                "bike_m": 115, "run_m": 0, "run_known": False, "src": "IM route 3327306152809908766"},
    "la-quinta": {"name": "IRONMAN 70.3 La Quinta", "where": "California, USA", "when": "6 Dec 2026", "swim": "lake",
                  "bike_m": 166, "run_m": 0, "run_known": False, "src": "IM route 3313979455385330672"},
    "erkner": {"name": "IRONMAN 70.3 Erkner", "where": "Germany", "when": "Sep 2027", "swim": "lake",
               "bike_m": 129, "run_m": 65, "run_known": True, "src": "finishers.com"},
    "westfriesland": {"name": "IRONMAN 70.3 Westfriesland", "where": "Netherlands", "when": "20 Jun 2027", "swim": "lake",
                      "bike_m": 55, "run_m": 0, "run_known": False, "src": "IM route 3326512018760900022"},
}
LEGS = [("swim", 1.9), ("bike", 90.0), ("run", 21.1)]
DEFAULT_TRANSITIONS_S = 7 * 60  # T1 + T2 when the athlete's own races don't tell us


def leg_elev(course: dict, sport: str) -> float:
    return {"swim": 0.0, "bike": float(course["bike_m"]), "run": float(course["run_m"])}[sport]
