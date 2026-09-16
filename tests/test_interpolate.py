import numpy as np
import pytest

from btc_options_mm.surface.arbitrage import has_calendar_arbitrage
from btc_options_mm.surface.interpolate import Slice, Surface
from btc_options_mm.surface.svi import SVIParams, total_variance

SHORT = SVIParams(a=0.02, b=0.3, rho=-0.2, m=0.0, sigma=0.2)
MID = SVIParams(a=0.05, b=0.4, rho=-0.3, m=0.0, sigma=0.25)
LONG = SVIParams(a=0.10, b=0.5, rho=-0.4, m=0.0, sigma=0.3)


def make_surface() -> Surface:
    return Surface(
        [
            Slice(expiry=0.5, params=LONG),
            Slice(expiry=0.1, params=SHORT),
            Slice(expiry=0.25, params=MID),
        ]
    )


def test_sorts_slices_by_expiry():
    surface = make_surface()
    assert [s.expiry for s in surface.slices] == [0.1, 0.25, 0.5]


@pytest.mark.parametrize("k", [-0.5, 0.0, 0.3])
def test_matches_listed_slice_exactly(k):
    surface = make_surface()
    for expiry, params in [(0.1, SHORT), (0.25, MID), (0.5, LONG)]:
        expected = float(total_variance(np.array([k]), params)[0])
        assert surface.total_variance(k, expiry) == pytest.approx(expected)


@pytest.mark.parametrize("k", [-0.5, 0.0, 0.3])
def test_interpolated_point_lies_between_neighbors(k):
    surface = make_surface()
    w_short = float(total_variance(np.array([k]), SHORT)[0])
    w_mid = float(total_variance(np.array([k]), MID)[0])

    w_interp = surface.total_variance(k, 0.18)
    assert min(w_short, w_mid) <= w_interp <= max(w_short, w_mid)


def test_extrapolates_flat_outside_listed_range():
    surface = make_surface()
    k = 0.1
    w_at_shortest = float(total_variance(np.array([k]), SHORT)[0])
    w_at_longest = float(total_variance(np.array([k]), LONG)[0])

    assert surface.total_variance(k, 0.01) == pytest.approx(w_at_shortest)
    assert surface.total_variance(k, 5.0) == pytest.approx(w_at_longest)


def test_implied_vol_matches_sqrt_w_over_t():
    surface = make_surface()
    w = surface.total_variance(0.0, 0.25)
    assert surface.implied_vol(0.0, 0.25) == pytest.approx(np.sqrt(w / 0.25))


def test_preserves_calendar_condition_at_intermediate_tenors():
    surface = make_surface()
    k_grid = np.linspace(-1.0, 1.0, 21)
    sample_expiries = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    w_by_expiry = [
        np.array([surface.total_variance(k, T) for k in k_grid]) for T in sample_expiries
    ]

    assert not has_calendar_arbitrage(sample_expiries, w_by_expiry)


def test_rejects_empty_slice_list():
    with pytest.raises(ValueError):
        Surface([])


def test_implied_vol_floors_at_a_small_positive_value_for_a_degenerate_fit():
    # A pathological slice that goes negative in total variance somewhere
    # (b large enough, rho pushed to -1) shouldn't hand callers vol <= 0.
    bad_params = SVIParams(a=-0.5, b=0.4, rho=-0.999, m=0.0, sigma=0.05)
    surface = Surface([Slice(expiry=0.1, params=bad_params)])
    assert surface.implied_vol(0.0, 0.1) > 0
