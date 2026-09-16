import pytest

from btc_options_mm.pricing.black76 import price
from btc_options_mm.pricing.greeks import delta, gamma, rho, theta, vega

FORWARDS = [80.0, 100.0, 120.0]
STRIKES = [80.0, 100.0, 120.0]
MATURITIES = [0.1, 1.0]
VOLS = [0.3, 0.8]
RATES = [0.0, 0.05]
OPTION_TYPES = ["call", "put"]

CASES = [
    (F, K, T, sigma, r, option_type)
    for F in FORWARDS
    for K in STRIKES
    for T in MATURITIES
    for sigma in VOLS
    for r in RATES
    for option_type in OPTION_TYPES
]


def _central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


@pytest.mark.parametrize("F,K,T,sigma,r,option_type", CASES)
def test_delta_matches_finite_difference(F, K, T, sigma, r, option_type):
    h = 1e-4 * F
    fd = _central_diff(lambda f: price(f, K, T, sigma, r, option_type), F, h)
    assert delta(F, K, T, sigma, r, option_type) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("F,K,T,sigma,r,option_type", CASES)
def test_gamma_matches_finite_difference(F, K, T, sigma, r, option_type):
    h = 1e-3 * F

    def d(f):
        return delta(f, K, T, sigma, r, option_type)

    fd = _central_diff(d, F, h)
    assert gamma(F, K, T, sigma, r) == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("F,K,T,sigma,r,option_type", CASES)
def test_vega_matches_finite_difference(F, K, T, sigma, r, option_type):
    h = 1e-5
    fd = _central_diff(lambda s: price(F, K, T, s, r, option_type), sigma, h)
    assert vega(F, K, T, sigma, r) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("F,K,T,sigma,r,option_type", CASES)
def test_theta_matches_finite_difference(F, K, T, sigma, r, option_type):
    h = 1e-5
    # theta is d(price)/d(-T): price increases as T decreases (more time value)
    fd = -_central_diff(lambda t: price(F, K, t, sigma, r, option_type), T, h)
    assert theta(F, K, T, sigma, r, option_type) == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("F,K,T,sigma,r,option_type", CASES)
def test_rho_matches_finite_difference(F, K, T, sigma, r, option_type):
    h = 1e-5
    fd = _central_diff(lambda rate: price(F, K, T, sigma, rate, option_type), r, h)
    assert rho(F, K, T, sigma, r, option_type) == pytest.approx(fd, abs=1e-4)
