"""Butterfly (Durrleman) and calendar no-arbitrage checks.

These operate on discretized (k, w) total-variance slices, not on SVIParams
directly, so they can be applied both to fitted curves and to raw market
quotes (see README Results: "share of snapshots with ... violations in raw
quotes").
"""

from __future__ import annotations

import numpy as np


def durrleman_g(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Durrleman's density-positivity function g(k). g(k) >= 0 everywhere
    is equivalent to a non-negative risk-neutral density (no butterfly
    arbitrage). k must be sorted ascending.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    wp = np.gradient(w, k)
    wpp = np.gradient(wp, k)
    return (1 - k * wp / (2 * w)) ** 2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2


def has_butterfly_arbitrage(k: np.ndarray, w: np.ndarray, tol: float = -1e-8) -> bool:
    """True if Durrleman's condition is violated anywhere on the slice."""
    return bool(np.any(durrleman_g(k, w) < tol))


def has_calendar_arbitrage(
    expiries: np.ndarray, w_by_expiry: list[np.ndarray], tol: float = 1e-8
) -> bool:
    """True if total variance decreases at any shared k as expiry increases.

    `w_by_expiry[i]` is the total-variance slice (evaluated on a common k
    grid) for `expiries[i]`.
    """
    order = np.argsort(expiries)
    sorted_w = [np.asarray(w_by_expiry[i], dtype=float) for i in order]
    for earlier, later in zip(sorted_w[:-1], sorted_w[1:]):
        if np.any(later < earlier - tol):
            return True
    return False
