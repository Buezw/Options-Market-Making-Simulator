import numpy as np
import pytest

from btc_options_mm.surface.fit import fit_svi_slice
from btc_options_mm.surface.svi import SVIParams, total_variance

KNOWN_PARAMS = [
    SVIParams(a=0.04, b=0.4, rho=-0.3, m=0.0, sigma=0.2),
    SVIParams(a=0.02, b=0.6, rho=0.2, m=-0.1, sigma=0.15),
    SVIParams(a=0.10, b=0.3, rho=-0.6, m=0.05, sigma=0.3),
]


@pytest.mark.parametrize("true_params", KNOWN_PARAMS)
def test_fit_recovers_known_params_from_synthetic_smile(true_params):
    k = np.linspace(-1.5, 1.5, 31)
    w = total_variance(k, true_params)

    fitted = fit_svi_slice(k, w, n_restarts=20, seed=0)

    assert fitted.a == pytest.approx(true_params.a, abs=1e-3)
    assert fitted.b == pytest.approx(true_params.b, abs=1e-3)
    assert fitted.rho == pytest.approx(true_params.rho, abs=1e-3)
    assert fitted.m == pytest.approx(true_params.m, abs=1e-3)
    assert fitted.sigma == pytest.approx(true_params.sigma, abs=1e-3)

    w_fitted = total_variance(k, fitted)
    np.testing.assert_allclose(w_fitted, w, atol=1e-6)


def test_fit_matches_curve_even_with_vega_like_weights():
    true_params = SVIParams(a=0.05, b=0.5, rho=-0.4, m=0.0, sigma=0.25)
    k = np.linspace(-1.2, 1.2, 25)
    w = total_variance(k, true_params)
    # Vega-like weight: heaviest near the money, lighter in the wings.
    weights = np.exp(-2 * k**2)

    fitted = fit_svi_slice(k, w, weights=weights, n_restarts=20, seed=1)
    w_fitted = total_variance(k, fitted)
    np.testing.assert_allclose(w_fitted, w, atol=1e-6)
