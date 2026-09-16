"""Vega-weighted per-expiry raw SVI fitting.

Raw SVI is non-convex in its parameters and has local minima, so recovering
a good (or, on a noiseless synthetic smile, the exact) fit needs multiple
random restarts, not a single local optimization.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from btc_options_mm.surface.svi import SVIParams, total_variance


def _bounds(k: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_lo, k_hi = float(np.min(k)), float(np.max(k))
    k_range = max(k_hi - k_lo, 1e-3)
    w_max = max(float(np.max(w)), 1e-6)

    lower = np.array([-2.0 * w_max, 1e-8, -0.999, k_lo - k_range, 1e-6])
    upper = np.array([2.0 * w_max, 10.0 * w_max / k_range, 0.999, k_hi + k_range, k_range])
    return lower, upper


def _initial_guess(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    m0 = float(k[np.argmin(w)])
    a0 = float(np.min(w))
    b0 = float((np.max(w) - np.min(w)) / max(np.max(k) - np.min(k), 1e-3)) or 0.1
    return np.array([a0, max(b0, 1e-3), 0.0, m0, 0.1])


def fit_svi_slice(
    log_moneyness: np.ndarray,
    total_variance_obs: np.ndarray,
    weights: np.ndarray | None = None,
    n_restarts: int = 20,
    seed: int = 0,
) -> SVIParams:
    """Fit raw SVI params to one expiry's (log-moneyness, total variance) slice.

    `weights` should be vega (or another liquidity proxy) per point; defaults
    to uniform weighting when not given.
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_variance_obs, dtype=float)
    if weights is None:
        weights = np.ones_like(w)
    sqrt_weights = np.sqrt(np.asarray(weights, dtype=float))

    lower, upper = _bounds(k, w)
    heuristic_x0 = np.clip(_initial_guess(k, w), lower, upper)

    def residuals(x: np.ndarray) -> np.ndarray:
        params = SVIParams(*x)
        return sqrt_weights * (total_variance(k, params) - w)

    rng = np.random.default_rng(seed)
    best_result = None
    best_cost = np.inf

    starts = [heuristic_x0]
    for _ in range(n_restarts - 1):
        starts.append(rng.uniform(lower, upper))

    for x0 in starts:
        result = least_squares(residuals, x0=x0, bounds=(lower, upper), method="trf")
        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result

    return SVIParams(*best_result.x)
