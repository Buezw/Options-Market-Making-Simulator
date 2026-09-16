import pytest

from btc_options_mm.mm.inventory import Inventory, RiskLimits
from btc_options_mm.mm.quoter import Quoter
from btc_options_mm.pricing import black76
from btc_options_mm.pricing.instrument import OptionSpec
from btc_options_mm.surface.interpolate import Slice, Surface
from btc_options_mm.surface.svi import SVIParams

F = 50_000.0
T = 30 / 365
FLAT_VOL = 0.6


def flat_surface(vol: float = FLAT_VOL, T: float = T) -> Surface:
    # b=0 kills the smile/skew terms, leaving total variance = a everywhere.
    params = SVIParams(a=vol**2 * T, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    return Surface([Slice(expiry=T, params=params)])


def make_spec(strike: float = F) -> OptionSpec:
    return OptionSpec(name="TEST", strike=strike, expiry=T, option_type="call")


def test_fair_value_matches_flat_surface_vol_with_zero_spread():
    quoter = Quoter(surface=flat_surface(), base_half_spread_vol=0.0)
    quote = quoter.quote(make_spec(), F, Inventory())

    expected_price = black76.price(F, F, T, FLAT_VOL, 0.0, "call")
    assert quote.bid_vol == pytest.approx(FLAT_VOL, abs=1e-9)
    assert quote.ask_vol == pytest.approx(FLAT_VOL, abs=1e-9)
    assert quote.bid_price == pytest.approx(expected_price, rel=1e-6)
    assert quote.ask_price == pytest.approx(expected_price, rel=1e-6)


def test_base_half_spread_widens_symmetric_quote():
    quoter = Quoter(surface=flat_surface(), base_half_spread_vol=0.02)
    quote = quoter.quote(make_spec(), F, Inventory())

    assert quote.ask_vol - quote.bid_vol == pytest.approx(0.04, abs=1e-9)
    assert (quote.ask_vol + quote.bid_vol) / 2 == pytest.approx(FLAT_VOL, abs=1e-9)


def test_vega_spread_coef_widens_spread_further():
    spec = make_spec()
    base = Quoter(surface=flat_surface(), base_half_spread_vol=0.02)
    with_vega = Quoter(surface=flat_surface(), base_half_spread_vol=0.02, vega_spread_coef=1e-6)

    base_quote = base.quote(spec, F, Inventory())
    vega_quote = with_vega.quote(spec, F, Inventory())
    assert (vega_quote.ask_vol - vega_quote.bid_vol) > (base_quote.ask_vol - base_quote.bid_vol)


def test_inventory_skew_shifts_vol_down_when_long_vega_in_bucket():
    quoter = Quoter(surface=flat_surface(), base_half_spread_vol=0.02, inventory_skew_coef=0.001)
    spec = make_spec()
    inv = Inventory()
    inv.mark(spec.name, T=spec.expiry, unit_delta=0.5, unit_gamma=0.001, unit_vega=10.0)
    inv.apply_fill(spec.name, 5)  # long vega in this bucket

    long_quote = quoter.quote(spec, F, inv)
    flat_quote = quoter.quote(spec, F, Inventory())

    long_mid = (long_quote.bid_vol + long_quote.ask_vol) / 2
    flat_mid = (flat_quote.bid_vol + flat_quote.ask_vol) / 2
    assert long_mid < flat_mid


def test_risk_limit_pauses_bid_when_breached():
    quoter = Quoter(surface=flat_surface(), base_half_spread_vol=0.02)
    spec = make_spec()
    inv = Inventory()
    inv.mark(spec.name, T=spec.expiry, unit_delta=0.5, unit_gamma=0.001, unit_vega=10.0)
    inv.apply_fill(spec.name, 100)  # far over the limit below
    limits = RiskLimits(max_net_vega=5.0)

    quote = quoter.quote(spec, F, inv, limits)
    assert quote.bid_price is None
    assert quote.bid_vol is None
    assert quote.ask_price is not None
