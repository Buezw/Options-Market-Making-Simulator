import numpy as np
import pytest

from btc_options_mm.surface.sabr import SABRParams, fit_sabr_slice, implied_vol

TRUE_PARAMS = SABRParams(alpha=0.5, rho=-0.3, nu=0.8)
T = 0.25


def test_atm_value_matches_hagan_closed_form():
    # z/x(z) -> 1 exactly at k=0, so this checks the epsilon-handling branch.
    expected = TRUE_PARAMS.alpha * (
        1 + (TRUE_PARAMS.rho * TRUE_PARAMS.nu * TRUE_PARAMS.alpha / 4 + (2 - 3 * TRUE_PARAMS.rho**2) / 24 * TRUE_PARAMS.nu**2) * T
    )
    assert float(implied_vol(np.array([0.0]), T, TRUE_PARAMS)[0]) == pytest.approx(expected, rel=1e-9)


def test_implied_vol_continuous_through_the_atm_epsilon_branch():
    near_zero = float(implied_vol(np.array([1e-6]), T, TRUE_PARAMS)[0])
    at_zero = float(implied_vol(np.array([0.0]), T, TRUE_PARAMS)[0])
    assert near_zero == pytest.approx(at_zero, abs=1e-4)


def test_implied_vol_positive_across_a_wide_smile():
    k = np.linspace(-1.5, 1.5, 41)
    vols = implied_vol(k, T, TRUE_PARAMS)
    assert np.all(vols > 0)


def test_fit_recovers_known_params_from_synthetic_smile():
    k = np.linspace(-1.0, 1.0, 21)
    vols = implied_vol(k, T, TRUE_PARAMS)

    fitted = fit_sabr_slice(k, vols, T, n_restarts=20, seed=0)

    assert fitted.alpha == pytest.approx(TRUE_PARAMS.alpha, abs=1e-3)
    assert fitted.rho == pytest.approx(TRUE_PARAMS.rho, abs=1e-3)
    assert fitted.nu == pytest.approx(TRUE_PARAMS.nu, abs=1e-3)

    vols_fitted = implied_vol(k, T, fitted)
    np.testing.assert_allclose(vols_fitted, vols, atol=1e-6)
