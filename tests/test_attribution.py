import numpy as np
import pytest

from btc_options_mm.analytics.pnl_attribution import attribute_pnl, gamma_theta_pnl
from btc_options_mm.pricing import black76, greeks


def test_attribute_pnl_basic_formula():
    result = attribute_pnl(
        actual_pnl=10.0,
        delta=0.5,
        gamma=0.02,
        theta=-1.0,
        vega=5.0,
        dF=2.0,
        dt=1 / 365,
        dsigma=0.01,
        edge=0.1,
        hedge_cost=0.05,
    )
    assert result.delta_pnl == pytest.approx(0.5 * 2.0)
    assert result.gamma_pnl == pytest.approx(0.5 * 0.02 * 2.0**2)
    assert result.theta_pnl == pytest.approx(-1.0 * (1 / 365))
    assert result.vega_pnl == pytest.approx(5.0 * 0.01)
    expected_explained = 0.1 + result.delta_pnl + result.gamma_pnl + result.theta_pnl + result.vega_pnl - 0.05
    assert result.explained == pytest.approx(expected_explained)
    assert result.residual == pytest.approx(10.0 - expected_explained)


def _simulate_gbm_forward_path(F0, sigma_realized, T0, n_steps, seed):
    """Local test helper: simulate a driftless lognormal forward path.

    Not production code -- the real engine gets F(t) from recorded market
    data. This only exists to validate the attribution formulas against a
    ground truth we control, per README section 6.
    """
    rng = np.random.default_rng(seed)
    dt = T0 / n_steps
    forwards = np.empty(n_steps + 1)
    forwards[0] = F0
    z = rng.standard_normal(n_steps)
    for i in range(n_steps):
        forwards[i + 1] = forwards[i] * np.exp(
            -0.5 * sigma_realized**2 * dt + sigma_realized * np.sqrt(dt) * z[i]
        )
    return forwards, dt


def test_unhedged_taylor_residual_is_small_per_step():
    F0, K, T0, sigma, r = 100.0, 100.0, 1.0, 0.3, 0.0
    n_steps = 252
    forwards, dt = _simulate_gbm_forward_path(F0, sigma, T0, n_steps, seed=0)

    # Stop well before expiry: as T -> 0 an ATM option's gamma diverges, so
    # second-order Taylor truncation error necessarily blows up too. That's
    # a property of the expansion near expiry, not a bug in the attribution.
    min_t_remaining = 0.15
    max_residual = 0.0
    for i in range(n_steps):
        F, F_next = forwards[i], forwards[i + 1]
        T, T_next = T0 - i * dt, T0 - (i + 1) * dt
        if T_next <= min_t_remaining:
            break

        price0 = black76.call_price(F, K, T, sigma, r)
        price1 = black76.call_price(F_next, K, T_next, sigma, r)
        actual_pnl = price1 - price0

        d = greeks.delta(F, K, T, sigma, r, "call")
        g = greeks.gamma(F, K, T, sigma, r)
        th = greeks.theta(F, K, T, sigma, r, "call")
        v = greeks.vega(F, K, T, sigma, r)

        result = attribute_pnl(
            actual_pnl=actual_pnl,
            delta=d,
            gamma=g,
            theta=th,
            vega=v,
            dF=F_next - F,
            dt=dt,
            dsigma=0.0,
        )
        max_residual = max(max_residual, abs(result.residual))

    # Second-order Taylor truncation error over a daily step on a $100
    # underlying should be tiny relative to typical daily option price moves.
    assert max_residual < 0.05


def test_hedged_pnl_matches_gamma_theta_relation_on_gbm_path():
    F0, K, T0, implied_vol, r = 100.0, 100.0, 1.0, 0.25, 0.0
    realized_vol = 0.40
    n_steps = 2000
    forwards, dt = _simulate_gbm_forward_path(F0, realized_vol, T0, n_steps, seed=42)

    actual_hedged_pnl = 0.0
    predicted_pnl = 0.0
    for i in range(n_steps):
        F, F_next = forwards[i], forwards[i + 1]
        T, T_next = T0 - i * dt, T0 - (i + 1) * dt
        if T_next <= 0:
            break

        price0 = black76.call_price(F, K, T, implied_vol, r)
        price1 = black76.call_price(F_next, K, T_next, implied_vol, r)
        d = greeks.delta(F, K, T, implied_vol, r, "call")
        g = greeks.gamma(F, K, T, implied_vol, r)

        hedge_qty = -d
        step_pnl = (price1 - price0) + hedge_qty * (F_next - F)
        actual_hedged_pnl += step_pnl
        predicted_pnl += gamma_theta_pnl(g, F, realized_vol, implied_vol, dt)

    # Statistical (Monte Carlo) comparison over many steps of one path;
    # seeded for determinism. Tolerance is generous relative to the scale of
    # the effect being measured (realized >> implied vol here).
    assert actual_hedged_pnl == pytest.approx(predicted_pnl, rel=0.15)
