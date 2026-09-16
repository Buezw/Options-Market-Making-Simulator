"""Poisson and trade-replay fill models.

See README methodology section 4. Both models decide whether a resting
`Quote` (from `mm.quoter`) gets filled, but on different information: the
Poisson model reacts to elapsed time and distance from fair value, the
replay model reacts to an actual recorded (or synthetic) market trade
printing through our price. A run only uses one of the two, so they aren't
forced into a single call signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from btc_options_mm.mm.quoter import Quote


@dataclass(frozen=True)
class Fill:
    instrument: str
    side: str  # "buy" or "sell", from the market maker's own book
    price: float
    size: float


@dataclass
class PoissonFillModel:
    """Fill intensity lambda(delta) = A * exp(-kappa * delta), where delta is
    the quote's distance from fair value (same units A/kappa are calibrated
    in, e.g. vol points). Fill probability over an elapsed `dt` is
    1 - exp(-lambda * dt).
    """

    A: float
    kappa: float
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def try_fill(self, quote: Quote, side: str, distance: float, dt: float) -> Fill | None:
        price = quote.bid_price if side == "bid" else quote.ask_price
        if price is None or distance < 0 or dt <= 0:
            return None

        intensity = self.A * np.exp(-self.kappa * distance)
        p_fill = 1.0 - np.exp(-intensity * dt)
        if self.rng.random() >= p_fill:
            return None

        fill_side = "buy" if side == "bid" else "sell"
        return Fill(instrument=quote.instrument, side=fill_side, price=price, size=quote.size)


@dataclass
class TradeReplayFillModel:
    """Our quote fills when a market trade prints through it -- optimistic,
    per the README's "queue position is not modeled" limitation.
    """

    def try_fill(
        self,
        quote: Quote,
        trade_price: float,
        trade_amount: float,
        trade_direction: str,
    ) -> Fill | None:
        if trade_direction == "buy" and quote.ask_price is not None and trade_price >= quote.ask_price:
            size = min(trade_amount, quote.size)
            return Fill(instrument=quote.instrument, side="sell", price=quote.ask_price, size=size)
        if trade_direction == "sell" and quote.bid_price is not None and trade_price <= quote.bid_price:
            size = min(trade_amount, quote.size)
            return Fill(instrument=quote.instrument, side="buy", price=quote.bid_price, size=size)
        return None
