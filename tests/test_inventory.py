import pytest

from btc_options_mm.mm.inventory import Inventory, RiskLimits, maturity_bucket


def test_maturity_bucket_edges():
    assert maturity_bucket(1 / 365) == "<7d"
    assert maturity_bucket(10 / 365) == "7-30d"
    assert maturity_bucket(60 / 365) == "30-90d"
    assert maturity_bucket(200 / 365) == "90d+"


def test_net_delta_and_gamma_sum_across_positions():
    inv = Inventory()
    inv.mark("A", T=10 / 365, unit_delta=0.5, unit_gamma=0.01, unit_vega=2.0)
    inv.apply_fill("A", 4)  # long 4
    inv.mark("B", T=60 / 365, unit_delta=-0.3, unit_gamma=0.02, unit_vega=3.0)
    inv.apply_fill("B", -2)  # short 2

    assert inv.net_delta() == pytest.approx(4 * 0.5 + (-2) * -0.3)
    assert inv.net_gamma() == pytest.approx(4 * 0.01 + (-2) * 0.02)


def test_net_vega_by_bucket_groups_by_maturity():
    inv = Inventory()
    inv.mark("A", T=10 / 365, unit_delta=0.0, unit_gamma=0.0, unit_vega=2.0)
    inv.apply_fill("A", 3)
    inv.mark("B", T=15 / 365, unit_delta=0.0, unit_gamma=0.0, unit_vega=1.0)
    inv.apply_fill("B", 5)
    inv.mark("C", T=60 / 365, unit_delta=0.0, unit_gamma=0.0, unit_vega=4.0)
    inv.apply_fill("C", 1)

    buckets = inv.net_vega_by_bucket()
    assert buckets["7-30d"] == pytest.approx(3 * 2.0 + 5 * 1.0)
    assert buckets["30-90d"] == pytest.approx(4.0)
    assert buckets["<7d"] == pytest.approx(0.0)


def test_apply_fill_accumulates_qty():
    inv = Inventory()
    inv.apply_fill("A", 3)
    inv.apply_fill("A", -1)
    assert inv.positions["A"].qty == pytest.approx(2)


def test_risk_flags_pauses_bid_when_long_vega_bucket_breached():
    inv = Inventory()
    inv.mark("A", T=10 / 365, unit_delta=0.0, unit_gamma=0.0, unit_vega=2.0)
    inv.apply_fill("A", 10)  # net vega = 20, long
    limits = RiskLimits(max_net_vega=5.0)

    flags = inv.risk_flags("A", expiry=10 / 365, limits=limits)
    assert flags == {"pause_bid": True, "pause_ask": False}


def test_risk_flags_pauses_ask_when_short_vega_bucket_breached():
    inv = Inventory()
    inv.mark("A", T=10 / 365, unit_delta=0.0, unit_gamma=0.0, unit_vega=2.0)
    inv.apply_fill("A", -10)  # net vega = -20, short
    limits = RiskLimits(max_net_vega=5.0)

    flags = inv.risk_flags("A", expiry=10 / 365, limits=limits)
    assert flags == {"pause_bid": False, "pause_ask": True}


def test_risk_flags_no_pause_within_limits():
    inv = Inventory()
    inv.mark("A", T=10 / 365, unit_delta=0.0, unit_gamma=0.0, unit_vega=2.0)
    inv.apply_fill("A", 1)
    limits = RiskLimits(max_net_vega=5.0, max_net_gamma=5.0, max_position_per_contract=5.0)

    flags = inv.risk_flags("A", expiry=10 / 365, limits=limits)
    assert flags == {"pause_bid": False, "pause_ask": False}


def test_risk_flags_gamma_limit_pauses_across_all_instruments():
    inv = Inventory()
    inv.mark("A", T=10 / 365, unit_delta=0.0, unit_gamma=1.0, unit_vega=0.0)
    inv.apply_fill("A", 10)  # net gamma = 10, long
    limits = RiskLimits(max_net_gamma=5.0)

    flags_a = inv.risk_flags("A", expiry=10 / 365, limits=limits)
    flags_b = inv.risk_flags("B", expiry=200 / 365, limits=limits)  # different instrument/bucket
    assert flags_a == {"pause_bid": True, "pause_ask": False}
    assert flags_b == {"pause_bid": True, "pause_ask": False}


def test_risk_flags_per_contract_position_limit():
    inv = Inventory()
    inv.mark("A", T=10 / 365, unit_delta=0.0, unit_gamma=0.0, unit_vega=0.0)
    inv.apply_fill("A", -8)
    limits = RiskLimits(max_position_per_contract=5.0)

    flags = inv.risk_flags("A", expiry=10 / 365, limits=limits)
    assert flags == {"pause_bid": False, "pause_ask": True}


def test_risk_flags_zero_limit_means_unset_no_pause():
    inv = Inventory()
    inv.apply_fill("A", 1_000_000)
    limits = RiskLimits()  # all zero -> no limits configured

    flags = inv.risk_flags("A", expiry=10 / 365, limits=limits)
    assert flags == {"pause_bid": False, "pause_ask": False}
