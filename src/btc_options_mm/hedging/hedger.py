"""Executes hedge trades against the portfolio, with fees and slippage.

Pairs with `hedging.strategies`: a `HedgeRule` decides *whether* and *how
much* to trade; `Hedger` actually executes that trade against a `Portfolio`,
charging a fee plus half the hedge instrument's bid-ask spread as slippage
(README methodology section 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from btc_options_mm.backtest.portfolio import Portfolio


class HedgeRule(Protocol):
    def decide(self, t: float, spot: float, delta: float, gamma: float) -> float | None: ...


@dataclass(frozen=True)
class HedgeTrade:
    instrument: str
    qty: float
    price: float
    cost: float  # fee + slippage paid


@dataclass
class Hedger:
    instrument: str
    rule: HedgeRule
    fee_rate: float = 0.0
    spread: float = 0.0

    def maybe_hedge(
        self,
        t: float,
        spot: float,
        delta: float,
        gamma: float,
        portfolio: Portfolio,
    ) -> HedgeTrade | None:
        qty = self.rule.decide(t, spot, delta, gamma)
        if not qty:
            return None

        direction = 1.0 if qty > 0 else -1.0
        slippage_price = spot + direction * self.spread / 2
        notional = abs(qty) * slippage_price
        fee = notional * self.fee_rate
        slippage_cost = abs(qty) * self.spread / 2
        cost = fee + slippage_cost

        portfolio.apply_trade(self.instrument, qty, slippage_price, fee)
        return HedgeTrade(instrument=self.instrument, qty=qty, price=slippage_price, cost=cost)
