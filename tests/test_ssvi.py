import numpy as np
import pytest

from btc_options_mm.surface.arbitrage import has_calendar_arbitrage
from btc_options_mm.surface.ssvi import SSVIParams, fit_ssvi, is_arbitrage_free, total_variance

TRUE_PARAMS = SSVIParams(rho=-0.3, eta=1.0, gamma=0.3)
THETA_BY_EXPIRY = {0.1: 0.02, 0.25: 0.04, 0.5: 0.08}


def _synthetic_data():
    k_grid = np.linspace(-1.5, 1.5, 25)
    k_by_expiry = {T: k_grid for T in THETA_BY_EXPIRY}
    w_by_expiry = {T: total_variance(k_grid, theta, TRUE_PARAMS) for T, theta in THETA_BY_EXPIRY.items()}
    return k_by_expiry, w_by_expiry


def test_fit_recovers_known_global_params_from_synthetic_multi_expiry_data():
    k_by_expiry, w_by_expiry = _synthetic_data()

    fitted = fit_ssvi(k_by_expiry, w_by_expiry, THETA_BY_EXPIRY, n_restarts=15, seed=0)

    assert fitted.rho == pytest.approx(TRUE_PARAMS.rho, abs=1e-3)
    assert fitted.eta == pytest.approx(TRUE_PARAMS.eta, abs=1e-3)
    assert fitted.gamma == pytest.approx(TRUE_PARAMS.gamma, abs=1e-3)


def test_fit_matches_curve_exactly_on_noiseless_synthetic_data():
    k_by_expiry, w_by_expiry = _synthetic_data()
    fitted = fit_ssvi(k_by_expiry, w_by_expiry, THETA_BY_EXPIRY, n_restarts=15, seed=1)

    for T, k in k_by_expiry.items():
        w_fitted = total_variance(k, THETA_BY_EXPIRY[T], fitted)
        np.testing.assert_allclose(w_fitted, w_by_expiry[T], atol=1e-6)


def test_true_params_are_arbitrage_free():
    assert is_arbitrage_free(TRUE_PARAMS)


def test_is_arbitrage_free_rejects_violations():
    assert not is_arbitrage_free(SSVIParams(rho=0.0, eta=3.0, gamma=0.3))  # eta*(1+|rho|) > 2
    assert not is_arbitrage_free(SSVIParams(rho=0.0, eta=1.0, gamma=0.6))  # gamma > 1/2
    assert not is_arbitrage_free(SSVIParams(rho=0.0, eta=-1.0, gamma=0.3))  # eta <= 0


def test_surface_is_calendar_arbitrage_free_by_construction():
    k_grid = np.linspace(-1.5, 1.5, 25)
    expiries = np.array(sorted(THETA_BY_EXPIRY))
    w_by_expiry = [total_variance(k_grid, THETA_BY_EXPIRY[T], TRUE_PARAMS) for T in expiries]

    assert not has_calendar_arbitrage(expiries, w_by_expiry)
