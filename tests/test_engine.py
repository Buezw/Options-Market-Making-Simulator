import numpy as np
import pytest

from btc_options_mm.backtest.engine import BacktestEngine
from btc_options_mm.backtest.events import HedgeCheckEvent, TickerEvent, TradeEvent
from btc_options_mm.backtest.portfolio import Portfolio
from btc_options_mm.hedging.hedger import Hedger
from btc_options_mm.hedging.strategies import FixedBand
from btc_options_mm.mm.fill_model import TradeReplayFillModel
from btc_options_mm.mm.inventory import Inventory, RiskLimits
from btc_options_mm.mm.quoter import Quoter
from btc_options_mm.pricing import black76, greeks
from btc_options_mm.pricing.instrument import OptionSpec
from btc_options_mm.surface.interpolate import Slice, Surface
from btc_options_mm.surface.svi import SVIParams

F0 = 50_000.0
T = 30 / 365
FLAT_VOL = 0.6
HALF_SPREAD = 0.02
HEDGE_INSTRUMENT = "BTC-PERPETUAL"


def flat_surface() -> Surface:
    params = SVIParams(a=FLAT_VOL**2 * T, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    return Surface([Slice(expiry=T, params=params)])


def make_engine(fill_model=None) -> BacktestEngine:
    spec = OptionSpec(name="CALL", strike=F0, expiry=T, option_type="call")
    quoter = Quoter(surface=flat_surface(), base_half_spread_vol=HALF_SPREAD, quote_size=2.0)
    return BacktestEngine(
        specs={"CALL": spec},
        quoter=quoter,
        inventory=Inventory(),
        portfolio=Portfolio(),
        fill_model=fill_model or TradeReplayFillModel(),
        hedger=Hedger(instrument=HEDGE_INSTRUMENT, rule=FixedBand(band=0.0), fee_rate=0.0, spread=0.0),
        risk_limits=RiskLimits(),
    )


def test_hand_traced_fill_then_hedge_then_attribution():
    engine = make_engine()
    spec = engine.specs["CALL"]

    ask_price = black76.price(F0, spec.strike, spec.expiry, FLAT_VOL + HALF_SPREAD, 0.0, "call")
    fair_price = black76.price(F0, spec.strike, spec.expiry, FLAT_VOL, 0.0, "call")
    option_delta = greeks.delta(F0, spec.strike, spec.expiry, FLAT_VOL, 0.0, "call")

    events = [
        TickerEvent(timestamp=0, instrument=HEDGE_INSTRUMENT, mark_price=F0, index_price=F0),
        HedgeCheckEvent(timestamp=0),  # baseline checkpoint, no position yet
        TradeEvent(timestamp=1, instrument="CALL", price=ask_price, amount=2.0, direction="buy"),
        HedgeCheckEvent(timestamp=2),  # hedges and produces the first attribution
    ]

    result = engine.run(events)

    # One fill: a buy trade at our ask lifts it, so we sold 2.0 calls.
    assert len(result.fills) == 1
    fill = result.fills[0]
    assert fill.side == "sell"
    assert fill.price == pytest.approx(ask_price)
    assert engine.portfolio.positions["CALL"] == pytest.approx(-2.0)

    # The hedge brings total delta to exactly zero (FixedBand with band=0).
    expected_hedge_qty = 2.0 * option_delta  # we're short 2 calls -> short delta -2*option_delta -> hedge buys it back
    assert len(result.hedge_trades) == 1
    assert engine.portfolio.positions[HEDGE_INSTRUMENT] == pytest.approx(expected_hedge_qty)

    # No market move happened (F and vol both unchanged) and the position
    # only existed during this one interval, so the entire realized PnL is
    # exactly the spread captured on the fill -- nothing left unexplained.
    expected_edge = 2.0 * (ask_price - fair_price)
    assert len(result.attributions) == 1
    attribution = result.attributions[0]
    assert attribution.edge == pytest.approx(expected_edge)
    assert attribution.delta_pnl == pytest.approx(0.0)
    assert attribution.gamma_pnl == pytest.approx(0.0)
    assert attribution.theta_pnl == pytest.approx(0.0)
    assert attribution.vega_pnl == pytest.approx(0.0)
    assert attribution.hedge_cost == pytest.approx(0.0)
    assert attribution.residual == pytest.approx(0.0, abs=1e-8)
    assert result.pnl_series == pytest.approx([expected_edge])


def test_no_fill_when_trade_does_not_cross_quote():
    engine = make_engine()
    spec = engine.specs["CALL"]
    fair_price = black76.price(F0, spec.strike, spec.expiry, FLAT_VOL, 0.0, "call")

    events = [
        TickerEvent(timestamp=0, instrument=HEDGE_INSTRUMENT, mark_price=F0, index_price=F0),
        HedgeCheckEvent(timestamp=0),
        # A buy print right at fair value sits strictly below our ask (fair + half-spread), so it shouldn't fill.
        TradeEvent(timestamp=1, instrument="CALL", price=fair_price, amount=2.0, direction="buy"),
    ]
    result = engine.run(events)
    assert result.fills == []
    assert engine.portfolio.positions == {}


def test_surface_refit_recovers_known_smile_from_ticker_stream():
    from btc_options_mm.backtest.events import SurfaceRefitEvent

    true_params = SVIParams(a=0.03, b=0.35, rho=-0.3, m=0.0, sigma=0.25)
    true_surface = Surface([Slice(expiry=T, params=true_params)])
    strikes = np.linspace(0.7, 1.3, 9) * F0
    specs = {
        f"CALL-{i}": OptionSpec(name=f"CALL-{i}", strike=K, expiry=T, option_type="call")
        for i, K in enumerate(strikes)
    }

    seed_surface = Surface([Slice(expiry=T, params=SVIParams(a=0.02, b=0.0, rho=0.0, m=0.0, sigma=0.2))])
    engine = BacktestEngine(
        specs=specs,
        quoter=Quoter(surface=seed_surface, base_half_spread_vol=0.0),
        inventory=Inventory(),
        portfolio=Portfolio(),
        fill_model=TradeReplayFillModel(),
        hedger=Hedger(instrument=HEDGE_INSTRUMENT, rule=FixedBand(band=1e9)),  # never hedges
        risk_limits=RiskLimits(),
    )

    events = [TickerEvent(timestamp=0, instrument=HEDGE_INSTRUMENT, mark_price=F0, index_price=F0)]
    for i, (name, spec) in enumerate(specs.items()):
        k = float(np.log(spec.strike / F0))
        true_vol = true_surface.implied_vol(k, T)
        mark_price = black76.price(F0, spec.strike, T, true_vol, 0.0, "call")
        events.append(TickerEvent(timestamp=1 + i, instrument=name, mark_price=mark_price, index_price=F0))
    events.append(SurfaceRefitEvent(timestamp=100))

    engine.run(events)

    for spec in specs.values():
        k = float(np.log(spec.strike / F0))
        assert engine.quoter.surface.implied_vol(k, T) == pytest.approx(true_surface.implied_vol(k, T), abs=1e-4)
