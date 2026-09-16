import numpy as np
import pytest

from btc_options_mm.analytics.metrics import compute_metrics, max_drawdown, sharpe_ratio
from btc_options_mm.analytics.pnl_attribution import PnLAttribution


def test_sharpe_ratio_matches_hand_calculation():
    pnl = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = np.mean(pnl) / np.std(pnl, ddof=1)
    assert sharpe_ratio(pnl) == pytest.approx(expected)


def test_sharpe_ratio_zero_when_no_variance():
    assert sharpe_ratio([2.0, 2.0, 2.0]) == 0.0


def test_sharpe_ratio_zero_for_short_series():
    assert sharpe_ratio([1.0]) == 0.0
    assert sharpe_ratio([]) == 0.0


def test_max_drawdown_matches_hand_calculation():
    # equity curve: 10, 5, -15, 0, 30 -> drawdowns: 0, 5, 25, 10, 0
    pnl = [10.0, -5.0, -20.0, 15.0, 30.0]
    assert max_drawdown(pnl) == pytest.approx(25.0)


def test_max_drawdown_zero_for_monotonically_increasing_pnl():
    assert max_drawdown([1.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_max_drawdown_empty_series():
    assert max_drawdown([]) == 0.0


def _attribution(edge: float) -> PnLAttribution:
    return PnLAttribution(
        edge=edge,
        delta_pnl=0.0,
        gamma_pnl=0.0,
        theta_pnl=0.0,
        vega_pnl=0.0,
        hedge_cost=0.0,
        residual=0.0,
    )


def test_compute_metrics_aggregates_all_fields():
    pnl_series = [10.0, -5.0, -20.0, 15.0, 30.0]
    attributions = [_attribution(1.0), _attribution(2.0), _attribution(3.0)]
    hedge_costs = [0.5, 0.7]

    metrics = compute_metrics(pnl_series, attributions, hedge_costs, n_fills=6)

    assert metrics.total_pnl == pytest.approx(30.0)
    assert metrics.sharpe == pytest.approx(sharpe_ratio(pnl_series))
    assert metrics.max_drawdown == pytest.approx(25.0)
    assert metrics.n_hedges == 2
    assert metrics.mean_hedge_cost == pytest.approx(0.6)
    assert metrics.edge_per_contract == pytest.approx(6.0 / 6)


def test_compute_metrics_handles_no_hedges_or_fills():
    metrics = compute_metrics([1.0, 2.0], [], [], n_fills=0)
    assert metrics.n_hedges == 0
    assert metrics.mean_hedge_cost == 0.0
    assert metrics.edge_per_contract == 0.0
