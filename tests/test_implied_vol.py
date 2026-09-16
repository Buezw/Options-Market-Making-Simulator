import pytest

from btc_options_mm.pricing.black76 import price
from btc_options_mm.pricing.implied_vol import implied_vol

FORWARDS = [80.0, 100.0, 120.0]
STRIKES = [70.0, 90.0, 100.0, 110.0, 130.0]
MATURITIES = [0.1, 1.0, 2.0]
VOLS = [0.2, 0.5, 1.0, 1.5]
RATES = [0.0, 0.05]
OPTION_TYPES = ["call", "put"]

def _is_resolvable(F, K, T, sigma, r, option_type) -> bool:
    """Exclude combos where price is numerically indistinguishable from a
    nearby-vol price in float64: deep ITM options at low vol carry a time
    value too small relative to intrinsic value to invert reliably. Real IV
    pipelines filter these out too (see README: "prices below intrinsic
    value... are dropped"); no solver can recover vol from a price that has
    lost the information in floating point.
    """
    base = price(F, K, T, sigma, r, option_type)
    bumped = price(F, K, T, sigma + 0.05, r, option_type)
    return abs(bumped - base) > 1e-6


CASES = [
    (F, K, T, sigma, r, option_type)
    for F in FORWARDS
    for K in STRIKES
    for T in MATURITIES
    for sigma in VOLS
    for r in RATES
    for option_type in OPTION_TYPES
    if _is_resolvable(F, K, T, sigma, r, option_type)
]


@pytest.mark.parametrize("F,K,T,sigma,r,option_type", CASES)
def test_round_trip_recovers_input_vol(F, K, T, sigma, r, option_type):
    target = price(F, K, T, sigma, r, option_type)
    recovered = implied_vol(target, F, K, T, r, option_type)
    assert recovered == pytest.approx(sigma, abs=1e-5)


def test_round_trip_low_vega_forces_brent_fallback():
    # Deep OTM: vega at the default initial guess (sigma=0.5) is ~6e-3, well
    # below the Newton-to-Brent handoff threshold, so this exercises the
    # Brent fallback path from the very first iteration.
    F, K, T, r, sigma = 100.0, 250.0, 0.2, 0.0, 0.4
    target = price(F, K, T, sigma, r, "call")
    recovered = implied_vol(target, F, K, T, r, "call")
    assert recovered == pytest.approx(sigma, abs=1e-4)


def test_unreachable_price_raises():
    # A price above the no-arbitrage upper bound cannot be matched by any vol.
    F, K, T, r = 100.0, 100.0, 1.0, 0.0
    with pytest.raises(ValueError):
        implied_vol(target_price=F * 2, F=F, K=K, T=T, r=r, option_type="call")
