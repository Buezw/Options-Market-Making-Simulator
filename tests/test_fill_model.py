import numpy as np
import pytest

from btc_options_mm.mm.fill_model import PoissonFillModel, TradeReplayFillModel
from btc_options_mm.mm.quoter import Quote

QUOTE = Quote(instrument="TEST", bid_price=100.0, ask_price=110.0, bid_vol=0.5, ask_vol=0.55, size=2.0)


def test_poisson_fill_rate_matches_theoretical_probability():
    A, kappa, distance, dt = 5.0, 2.0, 0.5, 0.1
    expected_p = 1 - np.exp(-A * np.exp(-kappa * distance) * dt)

    model = PoissonFillModel(A=A, kappa=kappa, rng=np.random.default_rng(0))
    n_trials = 20_000
    n_fills = sum(1 for _ in range(n_trials) if model.try_fill(QUOTE, "bid", distance, dt) is not None)

    empirical_p = n_fills / n_trials
    tol = 4 * np.sqrt(expected_p * (1 - expected_p) / n_trials)
    assert empirical_p == pytest.approx(expected_p, abs=tol)


def test_poisson_fill_returns_buy_on_bid_and_sell_on_ask():
    model = PoissonFillModel(A=1e6, kappa=0.0, rng=np.random.default_rng(1))  # near-certain fill
    bid_fill = model.try_fill(QUOTE, "bid", distance=0.0, dt=1.0)
    ask_fill = model.try_fill(QUOTE, "ask", distance=0.0, dt=1.0)

    assert bid_fill is not None and bid_fill.side == "buy" and bid_fill.price == 100.0
    assert ask_fill is not None and ask_fill.side == "sell" and ask_fill.price == 110.0


def test_poisson_fill_none_when_side_is_paused():
    paused_quote = Quote(instrument="TEST", bid_price=None, ask_price=110.0, bid_vol=None, ask_vol=0.55, size=2.0)
    model = PoissonFillModel(A=1e6, kappa=0.0, rng=np.random.default_rng(2))
    assert model.try_fill(paused_quote, "bid", distance=0.0, dt=1.0) is None


def test_replay_fills_ask_when_buy_trade_prints_through():
    model = TradeReplayFillModel()
    fill = model.try_fill(QUOTE, trade_price=111.0, trade_amount=1.0, trade_direction="buy")
    assert fill is not None
    assert fill.side == "sell"
    assert fill.price == 110.0
    assert fill.size == 1.0


def test_replay_fills_bid_when_sell_trade_prints_through():
    model = TradeReplayFillModel()
    fill = model.try_fill(QUOTE, trade_price=99.0, trade_amount=5.0, trade_direction="sell")
    assert fill is not None
    assert fill.side == "buy"
    assert fill.price == 100.0
    assert fill.size == 2.0  # capped at quote size


def test_replay_no_fill_when_trade_does_not_cross():
    model = TradeReplayFillModel()
    assert model.try_fill(QUOTE, trade_price=105.0, trade_amount=1.0, trade_direction="buy") is None
    assert model.try_fill(QUOTE, trade_price=105.0, trade_amount=1.0, trade_direction="sell") is None


def test_replay_no_fill_when_side_is_paused():
    paused_quote = Quote(instrument="TEST", bid_price=None, ask_price=110.0, bid_vol=None, ask_vol=0.55, size=2.0)
    model = TradeReplayFillModel()
    assert model.try_fill(paused_quote, trade_price=90.0, trade_amount=1.0, trade_direction="sell") is None
