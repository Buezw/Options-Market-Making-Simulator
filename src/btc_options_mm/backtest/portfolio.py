"""Portfolio state: positions, mark-to-market, aggregate Greeks."""

from __future__ import annotations

from dataclasses import dataclass, field

from btc_options_mm.pricing import greeks
from btc_options_mm.pricing.instrument import OptionSpec


@dataclass
class Portfolio:
    cash: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)

    def apply_trade(self, instrument: str, qty: float, price: float, fee: float = 0.0) -> None:
        self.positions[instrument] = self.positions.get(instrument, 0.0) + qty
        self.cash -= qty * price + fee

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """Cash plus the value of every position at `prices`."""
        return self.cash + sum(
            qty * prices[instrument] for instrument, qty in self.positions.items() if qty != 0
        )

    def aggregate_greeks(
        self,
        specs: dict[str, OptionSpec],
        F: float,
        vols: dict[str, float],
        r: float = 0.0,
    ) -> dict[str, float]:
        """Sum delta/gamma/vega/theta across option positions.

        `specs` and `vols` are keyed by instrument name. A position without a
        matching entry in `specs` (e.g. the hedge future/perpetual) is
        skipped here -- its delta is just its qty, which the caller adds in
        separately since it isn't priced off the vol surface.
        """
        totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for instrument, qty in self.positions.items():
            if qty == 0 or instrument not in specs:
                continue
            spec = specs[instrument]
            sigma = vols[instrument]
            totals["delta"] += qty * greeks.delta(F, spec.strike, spec.expiry, sigma, r, spec.option_type)
            totals["gamma"] += qty * greeks.gamma(F, spec.strike, spec.expiry, sigma, r)
            totals["vega"] += qty * greeks.vega(F, spec.strike, spec.expiry, sigma, r)
            totals["theta"] += qty * greeks.theta(F, spec.strike, spec.expiry, sigma, r, spec.option_type)
        return totals
