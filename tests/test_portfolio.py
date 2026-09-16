import pytest

from btc_options_mm.backtest.portfolio import Portfolio
from btc_options_mm.pricing import greeks
from btc_options_mm.pricing.instrument import OptionSpec

F = 50_000.0
T = 30 / 365
SIGMA = 0.6
CALL = OptionSpec(name="CALL", strike=F, expiry=T, option_type="call")


def test_apply_trade_updates_position_and_cash():
    pf = Portfolio(cash=1000.0)
    pf.apply_trade("BTC-PERPETUAL", qty=2.0, price=50_000.0, fee=5.0)

    assert pf.positions["BTC-PERPETUAL"] == pytest.approx(2.0)
    assert pf.cash == pytest.approx(1000.0 - 2.0 * 50_000.0 - 5.0)


def test_apply_trade_accumulates_across_calls():
    pf = Portfolio()
    pf.apply_trade("A", 3.0, 10.0)
    pf.apply_trade("A", -1.0, 12.0)
    assert pf.positions["A"] == pytest.approx(2.0)


def test_mark_to_market_sums_cash_and_position_values():
    pf = Portfolio(cash=100.0)
    pf.apply_trade("A", 2.0, 50.0)  # cash now 100 - 100 = 0
    value = pf.mark_to_market({"A": 60.0})
    assert value == pytest.approx(0.0 + 2.0 * 60.0)


def test_aggregate_greeks_matches_manual_calculation():
    pf = Portfolio()
    pf.apply_trade(CALL.name, 3.0, price=0.0)
    specs = {CALL.name: CALL}
    vols = {CALL.name: SIGMA}

    result = pf.aggregate_greeks(specs, F, vols)

    expected_delta = 3.0 * greeks.delta(F, CALL.strike, CALL.expiry, SIGMA, 0.0, "call")
    expected_gamma = 3.0 * greeks.gamma(F, CALL.strike, CALL.expiry, SIGMA, 0.0)
    expected_vega = 3.0 * greeks.vega(F, CALL.strike, CALL.expiry, SIGMA, 0.0)
    expected_theta = 3.0 * greeks.theta(F, CALL.strike, CALL.expiry, SIGMA, 0.0, "call")

    assert result["delta"] == pytest.approx(expected_delta)
    assert result["gamma"] == pytest.approx(expected_gamma)
    assert result["vega"] == pytest.approx(expected_vega)
    assert result["theta"] == pytest.approx(expected_theta)


def test_aggregate_greeks_skips_positions_without_a_spec():
    pf = Portfolio()
    pf.apply_trade(CALL.name, 1.0, price=0.0)
    pf.apply_trade("BTC-PERPETUAL", 5.0, price=50_000.0)  # no matching spec

    result = pf.aggregate_greeks({CALL.name: CALL}, F, {CALL.name: SIGMA})
    expected_delta = 1.0 * greeks.delta(F, CALL.strike, CALL.expiry, SIGMA, 0.0, "call")
    assert result["delta"] == pytest.approx(expected_delta)


def test_aggregate_greeks_skips_zero_quantity_positions():
    pf = Portfolio()
    pf.apply_trade(CALL.name, 1.0, price=0.0)
    pf.apply_trade(CALL.name, -1.0, price=0.0)  # net zero
    result = pf.aggregate_greeks({CALL.name: CALL}, F, {CALL.name: SIGMA})
    assert result == {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
