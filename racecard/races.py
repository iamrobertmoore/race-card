"""Courses Race Card knows. Distances and climbing from the organiser or finishers.com."""
COURSES = {
    "weymouth-70.3": {
        "name": "IRONMAN 70.3 Weymouth",
        "legs": [("swim", 1.9, 0.0), ("bike", 90.0, 1043.0), ("run", 21.1, 24.0)],
        "source": "finishers.com/en/event/ironman-70-3-weymouth, read 2 Oct 2026",
        "notes": "Sea swim. Hilly bike. Flat run along the front.",
    },
}
DEFAULT_TRANSITIONS_S = 7 * 60  # T1 + T2 when the athlete's own races don't tell us
