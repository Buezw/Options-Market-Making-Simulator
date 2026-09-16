"""Implied volatility solver: Newton's method with a Brent fallback.

Newton's method converges fast when vega is not too small. Deep ITM/OTM
options and very short expiries have small vega, where Newton is unstable,
so those cases fall back to Brent's method on a wide bracket.
"""

from __future__ import annotations

from scipy.optimize import brentq

from btc_options_mm.pricing.black76 import price
from btc_options_mm.pricing.greeks import vega

_MAX_NEWTON_ITER = 50
_TOL = 1e-8
# Below this vega, a Newton step is numerically unstable (tiny denominator),
# so hand off to bisection well before precision degrades.
_MIN_VEGA = 1e-3
_SIGMA_LO, _SIGMA_HI = 1e-4, 5.0


def implied_vol(
    target_price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    initial_guess: float = 0.5,
) -> float:
    """Solve for sigma such that price(F, K, T, sigma, r, option_type) == target_price."""
    sigma = initial_guess
    for _ in range(_MAX_NEWTON_ITER):
        model_price = price(F, K, T, sigma, r, option_type)
        diff = model_price - target_price
        if abs(diff) < _TOL:
            return sigma

        v = vega(F, K, T, sigma, r)
        if v < _MIN_VEGA:
            break

        sigma -= diff / v
        if sigma <= _SIGMA_LO or sigma >= _SIGMA_HI:
            break

    def objective(s: float) -> float:
        return price(F, K, T, s, r, option_type) - target_price

    lo_val, hi_val = objective(_SIGMA_LO), objective(_SIGMA_HI)
    if lo_val * hi_val > 0:
        raise ValueError(
            f"target_price {target_price} not bracketed by sigma in "
            f"[{_SIGMA_LO}, {_SIGMA_HI}] for F={F}, K={K}, T={T}, r={r}, "
            f"option_type={option_type!r}"
        )
    return brentq(objective, _SIGMA_LO, _SIGMA_HI, xtol=_TOL)
