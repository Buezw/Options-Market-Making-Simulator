"""Fixed-interval, fixed-band and Whalley-Wilmott hedging rules.

Each rule exposes a `decide(t, spot, delta, gamma)` method that returns the
underlying trade size needed to re-hedge, or None if the rule doesn't want to
trade right now. Rules only see the portfolio's aggregate delta/gamma, not
how they were computed, so they can be driven by a real Portfolio later or
by a synthetic path in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class HedgeRule(Protocol):
    def decide(self, t: float, spot: float, delta: float, gamma: float) -> float | None: ...


@dataclass
class FixedInterval:
    """Re-hedge to zero delta every `interval` seconds, regardless of size."""

    interval: float
    _last_hedge_t: float = field(default=float("-inf"), init=False, repr=False)

    def decide(self, t: float, spot: float, delta: float, gamma: float) -> float | None:
        if t - self._last_hedge_t < self.interval:
            return None
        self._last_hedge_t = t
        return -delta


@dataclass
class FixedBand:
    """Re-hedge only once |delta| exceeds `band`, trading back to the edge."""

    band: float

    def decide(self, t: float, spot: float, delta: float, gamma: float) -> float | None:
        if abs(delta) <= self.band:
            return None
        target = self.band if delta > 0 else -self.band
        return target - delta


def whalley_wilmott_band(gamma: float, spot: float, risk_aversion: float, cost: float) -> float:
    """Half-width of the no-transaction delta band (Whalley & Wilmott, 1997).

    h = (1.5 * cost * Gamma_dollar^2 / risk_aversion)^(1/3) / spot,
    where Gamma_dollar = gamma * spot^2. Widens with gamma and cost, shrinks
    as risk aversion grows (a more risk-averse trader hedges tighter).
    """
    gamma_dollar = gamma * spot**2
    return (1.5 * cost * gamma_dollar**2 / risk_aversion) ** (1 / 3) / spot


@dataclass
class WhalleyWilmott:
    """Utility-based band that widens with gamma and per-trade cost."""

    risk_aversion: float
    cost: float

    def decide(self, t: float, spot: float, delta: float, gamma: float) -> float | None:
        band = whalley_wilmott_band(gamma, spot, self.risk_aversion, self.cost)
        if abs(delta) <= band:
            return None
        target = band if delta > 0 else -band
        return target - delta
