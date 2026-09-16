"""Hagan et al. (2002) SABR implied-vol approximation, lognormal (beta=1),
used only as a comparison model against the per-expiry raw-SVI fit in
`surface.fit` (README roadmap "extensions": SABR comparison) -- not a
replacement for it.

sigma_B(k, T) = alpha * z/x(z) * [1 + (rho*nu*alpha/4 + (2-3*rho^2)/24*nu^2) * T]
z = -(nu/alpha) * k,   x(z) = ln( (sqrt(1-2*rho*z+z^2) + z - rho) / (1-rho) )

beta is fixed at 1: appropriate for a strictly-positive underlying, avoids
the well-documented alpha/beta/rho identifiability issues from fitting beta
freely, and keeps the fit to 3 parameters (alpha, rho, nu) -- comparable in
spirit to SVI's shape parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

_Z_EPS = 1e-7


@dataclass(frozen=True)
class SABRParams:
    alpha: float
    rho: float
    nu: float


def implied_vol(k: np.ndarray, T: float, params: SABRParams) -> np.ndarray:
    """Hagan's lognormal (beta=1) SABR implied vol at log-moneyness k=ln(K/F)."""
    k = np.asarray(k, dtype=float)
    alpha, rho, nu = params.alpha, params.rho, params.nu

    z = -(nu / alpha) * k
    x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    # z/x(z) -> 1 as z -> 0 (matches the ATM formula); avoid the 0/0 there.
    z_over_x = np.divide(z, x, out=np.ones_like(z, dtype=float), where=np.abs(z) >= _Z_EPS)

    correction = 1 + (rho * nu * alpha / 4 + (2 - 3 * rho**2) / 24 * nu**2) * T
    return alpha * z_over_x * correction


def fit_sabr_slice(
    log_moneyness: np.ndarray,
    vol_obs: np.ndarray,
    T: float,
    weights: np.ndarray | None = None,
    n_restarts: int = 20,
    seed: int = 0,
) -> SABRParams:
    """Fit (alpha, rho, nu) to one expiry's (log-moneyness, implied vol) smile."""
    k = np.asarray(log_moneyness, dtype=float)
    v = np.asarray(vol_obs, dtype=float)
    if weights is None:
        weights = np.ones_like(v)
    sqrt_weights = np.sqrt(np.asarray(weights, dtype=float))

    lower = np.array([1e-4, -0.999, 1e-4])
    upper = np.array([5.0, 0.999, 5.0])
    alpha0 = float(np.clip(np.median(v), lower[0], upper[0]))

    def residuals(x: np.ndarray) -> np.ndarray:
        params = SABRParams(*x)
        return sqrt_weights * (implied_vol(k, T, params) - v)

    rng = np.random.default_rng(seed)
    starts = [np.clip(np.array([alpha0, 0.0, 0.5]), lower, upper)]
    starts += [rng.uniform(lower, upper) for _ in range(n_restarts - 1)]

    best_result, best_cost = None, np.inf
    for x0 in starts:
        result = least_squares(residuals, x0=x0, bounds=(lower, upper), method="trf")
        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result

    return SABRParams(*best_result.x)
