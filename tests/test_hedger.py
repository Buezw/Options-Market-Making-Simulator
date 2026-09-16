from dataclasses import dataclass

import pytest

from btc_options_mm.backtest.portfolio import Portfolio
from btc_options_mm.hedging.hedger import Hedger
from btc_options_mm.hedging.strategies import FixedBand


@dataclass
class FixedTradeRule:
    """Test double: always decides to trade a fixed amount."""

    qty: float | None

    def decide(self, t, spot, delta, gamma):
        return self.qty


def test_no_trade_when_rule_returns_none():
    hedger = Hedger(instrument="BTC-PERPETUAL", rule=FixedTradeRule(None))
    pf = Portfolio()
    trade = hedger.maybe_hedge(0.0, 50_000.0, delta=1.0, gamma=0.0, portfolio=pf)

    assert trade is None
    assert pf.positions == {}


def test_no_trade_when_rule_returns_zero():
    hedger = Hedger(instrument="BTC-PERPETUAL", rule=FixedTradeRule(0.0))
    pf = Portfolio()
    trade = hedger.maybe_hedge(0.0, 50_000.0, delta=1.0, gamma=0.0, portfolio=pf)
    assert trade is None


def test_executes_trade_with_fee_and_slippage():
    hedger = Hedger(instrument="BTC-PERPETUAL", rule=FixedTradeRule(2.0), fee_rate=0.001, spread=10.0)
    pf = Portfolio(cash=0.0)

    trade = hedger.maybe_hedge(0.0, spot=50_000.0, delta=-2.0, gamma=0.0, portfolio=pf)

    expected_price = 50_000.0 + 5.0  # buying 2.0 -> pays half the spread above spot
    expected_fee = 2.0 * expected_price * 0.001
    expected_slippage_cost = 2.0 * 5.0
    assert trade.price == pytest.approx(expected_price)
    assert trade.cost == pytest.approx(expected_fee + expected_slippage_cost)
    assert pf.positions["BTC-PERPETUAL"] == pytest.approx(2.0)
    assert pf.cash == pytest.approx(-2.0 * expected_price - expected_fee)


def test_sell_trade_prices_below_spot():
    hedger = Hedger(instrument="BTC-PERPETUAL", rule=FixedTradeRule(-1.0), fee_rate=0.0, spread=10.0)
    pf = Portfolio()
    trade = hedger.maybe_hedge(0.0, spot=50_000.0, delta=1.0, gamma=0.0, portfolio=pf)

    assert trade.price == pytest.approx(50_000.0 - 5.0)
    assert pf.positions["BTC-PERPETUAL"] == pytest.approx(-1.0)


def test_integrates_with_real_hedge_rule_bringing_delta_to_band_edge():
    rule = FixedBand(band=0.1)
    hedger = Hedger(instrument="BTC-PERPETUAL", rule=rule)
    pf = Portfolio()

    trade = hedger.maybe_hedge(0.0, spot=50_000.0, delta=0.5, gamma=0.0, portfolio=pf)
    assert trade.qty == pytest.approx(0.1 - 0.5)
