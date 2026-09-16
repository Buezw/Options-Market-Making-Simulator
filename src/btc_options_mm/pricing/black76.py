"""Black-76 pricing on a forward.

All inputs are in the option's own quote currency: forward F, strike K,
time to expiry T (years), volatility sigma (annualized), and risk-free
rate r. Deribit's BTC-denominated (inverse) quoting convention is handled
in the data loading layer, not here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / sqrt_t
    d2 = d1 - sqrt_t
    return d1, d2


def call_price(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-76 European call price."""
    d1, d2 = _d1_d2(F, K, T, sigma)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def put_price(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-76 European put price."""
    d1, d2 = _d1_d2(F, K, T, sigma)
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def price(F: float, K: float, T: float, sigma: float, r: float, option_type: str) -> float:
    """Dispatch to call_price / put_price by option_type ('call' or 'put')."""
    if option_type == "call":
        return call_price(F, K, T, sigma, r)
    if option_type == "put":
        return put_price(F, K, T, sigma, r)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
