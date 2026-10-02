"""TabPFN v2 (open weights, no account) predicts each leg's speed as a distribution."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from .features import race_row, training_rows

LEVELS = np.round(np.linspace(0.05, 0.95, 19), 2).tolist()
MIN_ROWS = 12
MAX_ROWS = 1000


def _regressor():
    warnings.filterwarnings("ignore", message=".*CPU with more than.*")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    # v2 weights download from Hugging Face without an account, so anyone can run this.
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)


def fit_sport(sessions: pd.DataFrame, sport: str, ref: pd.Timestamp):
    X, y = training_rows(sessions, sport, ref)
    if len(X) < MIN_ROWS:
        return None, len(X)
    if len(X) > MAX_ROWS:
        X, y = X.iloc[-MAX_ROWS:], y[-MAX_ROWS:]
    m = _regressor()
    m.fit(X, y)
    return m, len(X)


def speed_quantiles(model, sessions, sport, ref, legs: list[tuple[float, float]]) -> list[list[float]]:
    """km/h at LEVELS for each (distance_km, elev_m) asked, in one predict call."""
    rows = pd.concat([race_row(sessions, sport, ref, d, e) for d, e in legs], ignore_index=True)
    q = model.predict(rows, output_type="quantiles", quantiles=LEVELS)
    q = np.array(q)  # (levels, rows)
    return [sorted(float(v) for v in q[:, i]) for i in range(len(legs))]


def seconds_at(distance_km: float, kmh_q: list[float], q: float) -> float:
    """Time quantile q is speed quantile 1-q."""
    return distance_km / float(np.interp(1 - q, LEVELS, kmh_q)) * 3600.0


def finish_samples(legs: list[tuple[float, list[float]]], transitions_s: float, n: int = 20000, seed: int = 0,
                   gains: dict | None = None) -> np.ndarray:
    """Independent draws per leg. `gains` is fractional speed-up per leg index, for the sliders."""
    rng = np.random.default_rng(seed)
    total = np.full(n, float(transitions_s))
    for i, (km, kq) in enumerate(legs):
        u = rng.uniform(LEVELS[0], LEVELS[-1], n)
        kmh = np.interp(u, LEVELS, kq) * (1 + (gains or {}).get(i, 0.0))
        total += km / kmh * 3600.0
    return total
