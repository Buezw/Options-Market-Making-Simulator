"""Inventory tracking in Greeks space, by vega bucket, with risk limits."""

from __future__ import annotations

from dataclasses import dataclass, field

MATURITY_BUCKETS = ("<7d", "7-30d", "30-90d", "90d+")
_BUCKET_EDGES_YEARS = (7 / 365, 30 / 365, 90 / 365)


def maturity_bucket(T: float) -> str:
    if T < _BUCKET_EDGES_YEARS[0]:
        return MATURITY_BUCKETS[0]
    if T < _BUCKET_EDGES_YEARS[1]:
        return MATURITY_BUCKETS[1]
    if T < _BUCKET_EDGES_YEARS[2]:
        return MATURITY_BUCKETS[2]
    return MATURITY_BUCKETS[3]


@dataclass
class Position:
    qty: float = 0.0
    T: float = 0.0
    unit_delta: float = 0.0
    unit_gamma: float = 0.0
    unit_vega: float = 0.0

    @property
    def delta(self) -> float:
        return self.qty * self.unit_delta

    @property
    def gamma(self) -> float:
        return self.qty * self.unit_gamma

    @property
    def vega(self) -> float:
        return self.qty * self.unit_vega


@dataclass(frozen=True)
class RiskLimits:
    max_net_vega: float = 0.0
    max_net_gamma: float = 0.0
    max_position_per_contract: float = 0.0


@dataclass
class Inventory:
    positions: dict[str, Position] = field(default_factory=dict)

    def _get(self, instrument: str) -> Position:
        return self.positions.setdefault(instrument, Position())

    def apply_fill(self, instrument: str, qty_delta: float) -> None:
        self._get(instrument).qty += qty_delta

    def mark(self, instrument: str, T: float, unit_delta: float, unit_gamma: float, unit_vega: float) -> None:
        """Refresh an instrument's per-unit Greeks (called as the market moves)."""
        pos = self._get(instrument)
        pos.T = T
        pos.unit_delta = unit_delta
        pos.unit_gamma = unit_gamma
        pos.unit_vega = unit_vega

    def net_delta(self) -> float:
        return sum(p.delta for p in self.positions.values())

    def net_gamma(self) -> float:
        return sum(p.gamma for p in self.positions.values())

    def net_vega_by_bucket(self) -> dict[str, float]:
        totals = {bucket: 0.0 for bucket in MATURITY_BUCKETS}
        for pos in self.positions.values():
            totals[maturity_bucket(pos.T)] += pos.vega
        return totals

    def _gamma_flags(self, limits: RiskLimits) -> dict[str, bool]:
        gamma = self.net_gamma()
        over = limits.max_net_gamma > 0 and abs(gamma) > limits.max_net_gamma
        return {"pause_bid": over and gamma > 0, "pause_ask": over and gamma < 0}

    def _position_flags(self, instrument: str, limits: RiskLimits) -> dict[str, bool]:
        qty = self.positions[instrument].qty if instrument in self.positions else 0.0
        over = limits.max_position_per_contract > 0 and abs(qty) > limits.max_position_per_contract
        return {"pause_bid": over and qty > 0, "pause_ask": over and qty < 0}

    def risk_flags(self, instrument: str, expiry: float, limits: RiskLimits) -> dict[str, bool]:
        """Which side(s) of `instrument`'s quote should be paused.

        A breach pauses the side that would make it worse: net long over a
        limit pauses the bid (stop buying more), net short pauses the ask.
        `expiry` is used for the bucket lookup so this works even before
        `instrument` has a tracked position.
        """
        bucket_vega = self.net_vega_by_bucket()[maturity_bucket(expiry)]
        vega_over = limits.max_net_vega > 0 and abs(bucket_vega) > limits.max_net_vega
        bucket_flags = {"pause_bid": vega_over and bucket_vega > 0, "pause_ask": vega_over and bucket_vega < 0}
        gamma_flags = self._gamma_flags(limits)
        position_flags = self._position_flags(instrument, limits)
        return {
            "pause_bid": bucket_flags["pause_bid"] or gamma_flags["pause_bid"] or position_flags["pause_bid"],
            "pause_ask": bucket_flags["pause_ask"] or gamma_flags["pause_ask"] or position_flags["pause_ask"],
        }
