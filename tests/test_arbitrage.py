import numpy as np

from btc_options_mm.surface.arbitrage import has_butterfly_arbitrage, has_calendar_arbitrage
from btc_options_mm.surface.svi import SVIParams, total_variance


def test_smooth_svi_curve_has_no_butterfly_arbitrage():
    params = SVIParams(a=0.04, b=0.4, rho=-0.3, m=0.0, sigma=0.2)
    k = np.linspace(-1.5, 1.5, 200)
    w = total_variance(k, params)
    assert has_butterfly_arbitrage(k, w) is False


def test_constructed_dip_has_butterfly_arbitrage():
    params = SVIParams(a=0.04, b=0.4, rho=-0.3, m=0.0, sigma=0.2)
    k = np.linspace(-1.5, 1.5, 200)
    w = total_variance(k, params)
    # Punch a sharp dip near the money: total variance must stay smooth and
    # (roughly) convex for the implied density to stay non-negative.
    dip = np.exp(-((k - 0.0) ** 2) / (2 * 0.01**2)) * 0.3
    w_bad = w - dip
    assert has_butterfly_arbitrage(k, w_bad) is True


def test_non_decreasing_total_variance_has_no_calendar_arbitrage():
    k = np.linspace(-1.0, 1.0, 50)
    w_short = total_variance(k, SVIParams(a=0.02, b=0.3, rho=-0.2, m=0.0, sigma=0.2))
    w_long = total_variance(k, SVIParams(a=0.06, b=0.4, rho=-0.2, m=0.0, sigma=0.2))
    expiries = np.array([0.1, 0.5])
    assert has_calendar_arbitrage(expiries, [w_short, w_long]) is False


def test_decreasing_total_variance_has_calendar_arbitrage():
    k = np.linspace(-1.0, 1.0, 50)
    w_short = total_variance(k, SVIParams(a=0.06, b=0.4, rho=-0.2, m=0.0, sigma=0.2))
    w_long = total_variance(k, SVIParams(a=0.02, b=0.3, rho=-0.2, m=0.0, sigma=0.2))
    expiries = np.array([0.1, 0.5])
    # Longer-dated slice has *lower* total variance than the shorter one: a
    # calendar-spread arbitrage.
    assert has_calendar_arbitrage(expiries, [w_short, w_long]) is True


def test_calendar_check_sorts_by_expiry_not_input_order():
    k = np.linspace(-1.0, 1.0, 50)
    w_short = total_variance(k, SVIParams(a=0.02, b=0.3, rho=-0.2, m=0.0, sigma=0.2))
    w_long = total_variance(k, SVIParams(a=0.06, b=0.4, rho=-0.2, m=0.0, sigma=0.2))
    expiries = np.array([0.5, 0.1])  # long expiry listed first
    assert has_calendar_arbitrage(expiries, [w_long, w_short]) is False
