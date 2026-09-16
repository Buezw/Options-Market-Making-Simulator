"""Raw SVI parametrization of total implied variance.

w(k) = a + b(rho(k-m) + sqrt((k-m)^2 + sigma^2))

where k = ln(K/F) is log-moneyness and w = sigma_impl^2 * T is total variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.a, self.b, self.rho, self.m, self.sigma)


def total_variance(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """Raw SVI total variance w(k) for log-moneyness array k."""
    k = np.asarray(k, dtype=float)
    centered = k - params.m
    return params.a + params.b * (
        params.rho * centered + np.sqrt(centered**2 + params.sigma**2)
    )
