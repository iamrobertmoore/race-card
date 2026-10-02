"""TabPFN v2 (open weights, no account) predicts each leg's speed as a distribution."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import race_row, training_rows

LEVELS = np.round(np.linspace(0.05, 0.95, 19), 2).tolist()
MIN_ROWS = 12
MAX_ROWS = 1000  # TabPFN v2 is fine with more, but CPU time grows; keep the most recent


def _regressor():
    warnings.filterwarnings("ignore", message=".*CPU with more than.*")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    # v2 weights download from Hugging Face without an account, so anyone can run this.
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)


@dataclass
class LegPrediction:
    sport: str
    distance_km: float
    elev_m: float
    n_train: int
    kmh_quantiles: list[float] = field(default_factory=list)  # at LEVELS

    def seconds_at(self, q: float) -> float:
        # Faster speed -> shorter time, so the time quantile q is the speed quantile 1-q.
        kmh = float(np.interp(1 - q, LEVELS, self.kmh_quantiles))
        return self.distance_km / kmh * 3600.0

    @property
    def median_s(self) -> float:
        return self.seconds_at(0.5)


def predict_leg(sessions: pd.DataFrame, sport: str, ref: pd.Timestamp, distance_km: float, elev_m: float) -> LegPrediction | None:
    X, y = training_rows(sessions, sport, ref)
    if len(X) < MIN_ROWS:
        return None
    if len(X) > MAX_ROWS:
        X, y = X.iloc[-MAX_ROWS:], y[-MAX_ROWS:]
    m = _regressor()
    m.fit(X, y)
    q = m.predict(race_row(sessions, sport, ref, distance_km, elev_m), output_type="quantiles", quantiles=LEVELS)
    kq = sorted(float(v[0]) for v in q)  # enforce monotone
    return LegPrediction(sport, distance_km, elev_m, len(X), kq)


def finish_distribution(legs: list[LegPrediction], transitions_s: float, n: int = 20000, seed: int = 0) -> np.ndarray:
    """Sum of legs by sampling each leg's distribution independently. Independence understates the
    range a little (a bad day is usually bad in all three), and the post says so."""
    rng = np.random.default_rng(seed)
    total = np.full(n, float(transitions_s))
    for leg in legs:
        u = rng.uniform(LEVELS[0], LEVELS[-1], n)
        kmh = np.interp(u, LEVELS, leg.kmh_quantiles)
        total += leg.distance_km / kmh * 3600.0
    return total
