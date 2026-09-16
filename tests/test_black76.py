import numpy as np
import pytest

from btc_options_mm.pricing.black76 import call_price, put_price

FORWARDS = [80.0, 100.0, 120.0]
STRIKES = [70.0, 90.0, 100.0, 110.0, 130.0]
MATURITIES = [0.01, 0.1, 1.0, 2.0]
VOLS = [0.2, 0.5, 1.0]
RATES = [0.0, 0.05]


@pytest.mark.parametrize("F", FORWARDS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("T", MATURITIES)
@pytest.mark.parametrize("sigma", VOLS)
@pytest.mark.parametrize("r", RATES)
def test_put_call_parity(F, K, T, sigma, r):
    c = call_price(F, K, T, sigma, r)
    p = put_price(F, K, T, sigma, r)
    assert c - p == pytest.approx(np.exp(-r * T) * (F - K), abs=1e-8)


@pytest.mark.parametrize("F", FORWARDS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("T", MATURITIES)
@pytest.mark.parametrize("sigma", VOLS)
@pytest.mark.parametrize("r", RATES)
def test_prices_are_nonnegative_and_bounded(F, K, T, sigma, r):
    disc = np.exp(-r * T)
    c = call_price(F, K, T, sigma, r)
    p = put_price(F, K, T, sigma, r)
    assert c >= -1e-9
    assert p >= -1e-9
    assert c <= disc * F + 1e-9
    assert p <= disc * K + 1e-9


def test_deep_itm_call_approaches_intrinsic():
    F, K, T, sigma, r = 1000.0, 10.0, 0.5, 0.3, 0.0
    c = call_price(F, K, T, sigma, r)
    assert c == pytest.approx(F - K, rel=1e-3)


def test_deep_otm_call_approaches_zero():
    F, K, T, sigma, r = 10.0, 1000.0, 0.1, 0.3, 0.0
    c = call_price(F, K, T, sigma, r)
    assert c == pytest.approx(0.0, abs=1e-6)
