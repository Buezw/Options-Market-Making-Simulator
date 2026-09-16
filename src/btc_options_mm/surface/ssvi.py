"""Surface-SVI (SSVI): one (rho, eta, gamma) skew fit jointly across every
expiry, riding on an ATM total-variance term structure theta(T) taken
directly from the data, instead of 5 free parameters per expiry like raw
SVI (see `surface.fit`). Gatheral & Jacquier (2014).

w(k, theta) = theta/2 * (1 + rho*phi(theta)*k + sqrt((phi(theta)*k+rho)^2 + (1-rho^2)))
phi(theta) = eta / theta^gamma                                    (power-law)

Sufficient conditions for the whole surface to be free of calendar and
butterfly arbitrage everywhere (Gatheral & Jacquier, Theorem 4.2 and
Corollary 4.1): theta(T) non-decreasing in T, 0 < gamma <= 1/2, eta > 0, and
eta * (1 + |rho|) <= 2. `is_arbitrage_free` checks the parameter conditions;
theta(T) monotonicity is the caller's responsibility (it comes straight from
observed ATM variances, which should already be calendar-consistent).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SSVIParams:
    rho: float
    eta: float
    gamma: float


def phi(theta: np.ndarray, params: SSVIParams) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    return params.eta / theta**params.gamma


def total_variance(k: np.ndarray, theta: float, params: SSVIParams) -> np.ndarray:
    """SSVI total variance at log-moneyness `k` for one expiry whose ATM
    total variance is `theta`."""
    k = np.asarray(k, dtype=float)
    p = phi(theta, params)
    return theta / 2 * (1 + params.rho * p * k + np.sqrt((p * k + params.rho) ** 2 + (1 - params.rho**2)))


def is_arbitrage_free(params: SSVIParams) -> bool:
    return 0 < params.gamma <= 0.5 and params.eta > 0 and params.eta * (1 + abs(params.rho)) <= 2


def fit_ssvi(
    k_by_expiry: dict[float, np.ndarray],
    w_by_expiry: dict[float, np.ndarray],
    theta_by_expiry: dict[float, float],
    weights_by_expiry: dict[float, np.ndarray] | None = None,
    n_restarts: int = 10,
    seed: int = 0,
) -> SSVIParams:
    """Jointly fit (rho, eta, gamma) across every expiry in `k_by_expiry`,
    holding each expiry's ATM total variance fixed at `theta_by_expiry[T]`
    (as observed/interpolated from the data, not fit).
    """
    expiries = sorted(k_by_expiry)
    if weights_by_expiry is None:
        weights_by_expiry = {T: np.ones_like(np.asarray(k_by_expiry[T])) for T in expiries}

    lower = np.array([-0.999, 1e-6, 1e-6])
    upper = np.array([0.999, 10.0, 0.5])

    def residuals(x: np.ndarray) -> np.ndarray:
        params = SSVIParams(*x)
        parts = [
            np.sqrt(np.asarray(weights_by_expiry[T], dtype=float))
            * (total_variance(k_by_expiry[T], theta_by_expiry[T], params) - np.asarray(w_by_expiry[T], dtype=float))
            for T in expiries
        ]
        return np.concatenate(parts)

    rng = np.random.default_rng(seed)
    starts = [np.clip(np.array([0.0, 1.0, 0.3]), lower, upper)]
    starts += [rng.uniform(lower, upper) for _ in range(n_restarts - 1)]

    best_result, best_cost = None, np.inf
    for x0 in starts:
        result = least_squares(residuals, x0=x0, bounds=(lower, upper), method="trf")
        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result

    return SSVIParams(*best_result.x)
