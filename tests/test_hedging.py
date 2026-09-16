import numpy as np
import pytest

from btc_options_mm.hedging.strategies import (
    FixedBand,
    FixedInterval,
    WhalleyWilmott,
    whalley_wilmott_band,
)


def simulate(rule, n_steps=2000, dt=1.0, gamma=0.01, seed=0):
    """Drive a hedge rule over a synthetic path and report activity."""
    rng = np.random.default_rng(seed)
    spot = 50_000.0
    delta = 0.0
    n_hedges = 0
    max_abs_delta = 0.0
    for i in range(n_steps):
        t = i * dt
        d_spot = spot * 0.0005 * rng.standard_normal()
        spot += d_spot
        delta += gamma * d_spot
        trade = rule.decide(t, spot, delta, gamma)
        if trade is not None:
            delta += trade
            n_hedges += 1
        max_abs_delta = max(max_abs_delta, abs(delta))
    return n_hedges, max_abs_delta


def test_fixed_interval_hedges_on_schedule_regardless_of_size():
    rule = FixedInterval(interval=10.0)
    n_hedges, _ = simulate(rule, n_steps=100, dt=1.0)
    assert n_hedges == 10


def test_fixed_interval_brings_delta_exactly_to_zero():
    rule = FixedInterval(interval=1.0)
    rng = np.random.default_rng(1)
    spot, delta, gamma = 50_000.0, 0.0, 0.01
    for i in range(5):
        delta += gamma * spot * 0.001 * rng.standard_normal()
        trade = rule.decide(float(i), spot, delta, gamma)
        assert trade == pytest.approx(-delta)
        delta += trade
        assert delta == pytest.approx(0.0)


def test_fixed_band_does_not_hedge_within_band():
    rule = FixedBand(band=0.5)
    assert rule.decide(0.0, 50_000.0, 0.3, 0.01) is None
    assert rule.decide(0.0, 50_000.0, -0.5, 0.01) is None


def test_fixed_band_hedges_to_band_edge_when_breached():
    rule = FixedBand(band=0.5)
    assert rule.decide(0.0, 50_000.0, 0.8, 0.01) == pytest.approx(0.5 - 0.8)
    assert rule.decide(0.0, 50_000.0, -1.2, 0.01) == pytest.approx(-0.5 + 1.2)


def test_whalley_wilmott_band_widens_with_gamma_and_cost_shrinks_with_risk_aversion():
    base = whalley_wilmott_band(gamma=0.01, spot=50_000.0, risk_aversion=1.0, cost=0.0005)
    wider_gamma = whalley_wilmott_band(gamma=0.05, spot=50_000.0, risk_aversion=1.0, cost=0.0005)
    wider_cost = whalley_wilmott_band(gamma=0.01, spot=50_000.0, risk_aversion=1.0, cost=0.005)
    tighter = whalley_wilmott_band(gamma=0.01, spot=50_000.0, risk_aversion=10.0, cost=0.0005)

    assert wider_gamma > base
    assert wider_cost > base
    assert tighter < base


def test_whalley_wilmott_decide_matches_band():
    rule = WhalleyWilmott(risk_aversion=1.0, cost=0.0005)
    band = whalley_wilmott_band(gamma=0.01, spot=50_000.0, risk_aversion=1.0, cost=0.0005)

    assert rule.decide(0.0, 50_000.0, band * 0.9, 0.01) is None
    assert rule.decide(0.0, 50_000.0, band * 1.5, 0.01) == pytest.approx(band - band * 1.5)


def test_band_rules_trade_less_often_than_fixed_interval_and_respect_their_band():
    interval_rule = FixedInterval(interval=1.0)
    band_rule = FixedBand(band=0.02)
    ww_rule = WhalleyWilmott(risk_aversion=5.0, cost=0.0005)

    n_interval, _ = simulate(interval_rule, seed=7)
    n_band, max_band_delta = simulate(band_rule, seed=7)
    n_ww, max_ww_delta = simulate(ww_rule, seed=7)

    assert n_band < n_interval
    assert n_ww < n_interval
    assert max_band_delta <= 0.02 + 1e-9
