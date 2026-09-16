"""Analytic Black-76 Greeks, for both calls and puts."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from btc_options_mm.pricing.black76 import _d1_d2


def delta(F: float, K: float, T: float, sigma: float, r: float, option_type: str) -> float:
    d1, _ = _d1_d2(F, K, T, sigma)
    disc = np.exp(-r * T)
    if option_type == "call":
        return disc * norm.cdf(d1)
    if option_type == "put":
        return -disc * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def gamma(F: float, K: float, T: float, sigma: float, r: float) -> float:
    """Same for calls and puts."""
    d1, _ = _d1_d2(F, K, T, sigma)
    return np.exp(-r * T) * norm.pdf(d1) / (F * sigma * np.sqrt(T))


def vega(F: float, K: float, T: float, sigma: float, r: float) -> float:
    """Same for calls and puts. Price change per 1.0 (100 vol points) change in sigma."""
    d1, _ = _d1_d2(F, K, T, sigma)
    return np.exp(-r * T) * F * norm.pdf(d1) * np.sqrt(T)


def theta(F: float, K: float, T: float, sigma: float, r: float, option_type: str) -> float:
    """Price change per year of time decay (i.e. d(price)/d(-T))."""
    d1, d2 = _d1_d2(F, K, T, sigma)
    disc = np.exp(-r * T)
    carry_term = -disc * F * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        opt_price = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        opt_price = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return carry_term + r * opt_price


def rho(F: float, K: float, T: float, sigma: float, r: float, option_type: str) -> float:
    """Price change per 1.0 (100%) change in the risk-free rate."""
    d1, d2 = _d1_d2(F, K, T, sigma)
    disc = np.exp(-r * T)
    if option_type == "call":
        return -T * disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    if option_type == "put":
        return -T * disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
