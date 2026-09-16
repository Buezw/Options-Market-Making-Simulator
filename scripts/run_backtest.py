#!/usr/bin/env python
"""Run the full event-driven market-making backtest.

Usage:
    python scripts/run_backtest.py --config configs/backtest.yaml

Wires configs/backtest.yaml into the loader, quoter, inventory, fill model,
hedger and backtest engine, prints a summary, and writes figures to
reports/figures/. Does NOT touch reports/report.md -- that stays a manual
write-up (see README Results: numbers go in once a run is backed by enough
recorded data to be meaningful, not just because the CLI ran).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Callable, Iterable, Iterator

import numpy as np
import yaml

from btc_options_mm.analytics.metrics import compute_metrics
from btc_options_mm.analytics.plots import plot_pnl_attribution, plot_surface_3d
from btc_options_mm.backtest.engine import BacktestEngine
from btc_options_mm.backtest.events import Event, HedgeCheckEvent, SurfaceRefitEvent, TickerEvent
from btc_options_mm.backtest.portfolio import Portfolio
from btc_options_mm.data.loader import load_range, parse_instrument_name
from btc_options_mm.hedging.hedger import Hedger
from btc_options_mm.hedging.strategies import FixedBand, FixedInterval, WhalleyWilmott
from btc_options_mm.mm.fill_model import PoissonFillModel, TradeReplayFillModel
from btc_options_mm.mm.inventory import Inventory, RiskLimits
from btc_options_mm.mm.quoter import Quoter
from btc_options_mm.pricing.implied_vol import implied_vol
from btc_options_mm.pricing.instrument import OptionSpec
from btc_options_mm.surface.fit import fit_svi_slice
from btc_options_mm.surface.interpolate import Slice, Surface

_MIN_FIT_POINTS = 5


def _fallback_surface() -> Surface:
    """An uncalibrated flat 50%-vol placeholder, used only when there isn't
    enough real data yet to fit anything."""
    T = 30 / 365
    params = fit_svi_slice(np.array([0.0, 0.1]), np.array([0.5**2 * T, 0.5**2 * T]), n_restarts=1, seed=0)
    return Surface([Slice(expiry=T, params=params)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default="data/raw")
    return parser.parse_args()


def _parse_duration_seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    for suffix, scale in (("min", 60.0), ("h", 3600.0), ("s", 1.0)):
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * scale
    return float(s)


def _build_surface(events: list[Event], currency: str, as_of: datetime) -> Surface:
    """Bootstrap an initial surface by fitting every expiry with enough
    resolvable option quotes across the whole loaded range. Falls back to a
    single flat, uncalibrated slice (loudly flagged) if there isn't enough
    data yet -- so the pipeline still runs mechanically rather than crashing.
    """
    hedge_instrument = f"{currency.upper()}-PERPETUAL"
    forward: float | None = None
    samples: dict[float, list[tuple[float, float, float]]] = defaultdict(list)

    for event in events:
        if not isinstance(event, TickerEvent):
            continue
        if event.instrument == hedge_instrument:
            forward = event.mark_price
            continue
        if forward is None:
            continue
        spec = parse_instrument_name(event.instrument, as_of)
        if spec is None:
            continue
        try:
            vol = implied_vol(event.mark_price, forward, spec.strike, spec.expiry, 0.0, spec.option_type)
        except ValueError:
            continue
        k = float(np.log(spec.strike / forward))
        weight = 1.0
        samples[spec.expiry].append((k, vol**2 * spec.expiry, weight))

    slices = []
    for expiry, pts in samples.items():
        if len(pts) < _MIN_FIT_POINTS:
            continue
        ks, ws, weights = zip(*pts)
        slices.append(Slice(expiry=expiry, params=fit_svi_slice(np.array(ks), np.array(ws), weights=np.array(weights), seed=0)))

    if not slices:
        print("WARNING: not enough data to fit any expiry -- using an uncalibrated flat 50% vol surface")
        return _fallback_surface()
    return Surface(slices)


def _build_specs(events: list[Event], currency: str, as_of: datetime) -> dict[str, OptionSpec]:
    specs: dict[str, OptionSpec] = {}
    for event in events:
        if not isinstance(event, TickerEvent) or event.instrument in specs:
            continue
        spec = parse_instrument_name(event.instrument, as_of)
        if spec is not None:
            specs[event.instrument] = spec
    return specs


def _build_hedge_rule(cfg: dict):
    rule = cfg["rule"]
    if rule == "interval":
        return FixedInterval(interval=_parse_duration_seconds(cfg["interval"]))
    if rule == "band":
        return FixedBand(band=float(cfg["band"]))
    if rule == "whalley_wilmott":
        return WhalleyWilmott(risk_aversion=float(cfg["risk_aversion"]), cost=float(cfg["cost"]))
    raise ValueError(f"Unknown hedging.rule: {rule!r}")


def _build_fill_model(fill_model_name: str, poisson_cfg: dict):
    if fill_model_name == "replay":
        return TradeReplayFillModel()
    if fill_model_name == "poisson":
        return PoissonFillModel(A=float(poisson_cfg["A"]), kappa=float(poisson_cfg["kappa"]))
    raise ValueError(f"Unknown fill_model: {fill_model_name!r}")


def _inject_periodic(
    events: Iterable[Event], start_ts: float, end_ts: float, interval: float, factory: Callable[[float], Event]
) -> Iterator[Event]:
    """Merge periodic events (from `factory(t)`) into an already
    timestamp-ordered stream, keeping the result ordered."""
    it = iter(events)
    current = next(it, None)
    next_periodic = start_ts
    while current is not None or next_periodic <= end_ts:
        if current is not None and (next_periodic > end_ts or current.timestamp <= next_periodic):
            yield current
            current = next(it, None)
        else:
            yield factory(next_periodic)
            next_periodic += interval


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    start = date.fromisoformat(str(config["data"]["start"]))
    end = date.fromisoformat(str(config["data"]["end"]))
    currency = config.get("currency", "BTC")
    as_of = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)

    events = list(load_range(args.root, currency, start, end, min_open_interest=float(config["data"]["min_open_interest"])))
    if not events:
        print(f"No recorded data for {currency} between {start} and {end} under {args.root} -- nothing to run.")
        return

    surface = _build_surface(events, currency, as_of)
    specs = _build_specs(events, currency, as_of)
    if not specs:
        print("No option instruments found in the loaded data -- nothing to quote.")
        return

    quoting = config["quoting"]
    quoter = Quoter(
        surface=surface,
        base_half_spread_vol=float(quoting["base_half_spread_vol"]),
        vega_spread_coef=float(quoting["vega_spread_coef"]),
        gamma_spread_coef=float(quoting["gamma_spread_coef"]),
        inventory_skew_coef=float(quoting["inventory_skew_coef"]),
        quote_size=float(quoting["quote_size"]),
    )
    risk_limits = RiskLimits(**{k: float(v) for k, v in config["risk_limits"].items()})
    costs = config["costs"]
    hedger = Hedger(
        instrument=config["hedging"]["instrument"],
        rule=_build_hedge_rule(config["hedging"]),
        fee_rate=float(costs["future_fee"]),
        spread=float(costs.get("hedge_spread", 0.0)),
    )
    fill_model = _build_fill_model(config["fill_model"], config.get("poisson", {}))

    engine = BacktestEngine(
        specs=specs,
        quoter=quoter,
        inventory=Inventory(),
        portfolio=Portfolio(),
        fill_model=fill_model,
        hedger=hedger,
        risk_limits=risk_limits,
        option_fee=float(costs["option_fee"]),
    )

    start_ts = events[0].timestamp
    end_ts = events[-1].timestamp
    refit_interval = _parse_duration_seconds(config["surface"]["refit_interval"])
    check_interval = _parse_duration_seconds(config["hedging"]["check_interval"])

    stream = _inject_periodic(events, start_ts, end_ts, refit_interval, lambda t: SurfaceRefitEvent(timestamp=t))
    stream = _inject_periodic(stream, start_ts, end_ts, check_interval, lambda t: HedgeCheckEvent(timestamp=t))

    result = engine.run(stream)
    metrics = compute_metrics(
        result.pnl_series,
        result.attributions,
        [t.cost for t in result.hedge_trades],
        n_fills=len(result.fills),
    )

    span_days = (end - start).days + 1
    print(f"{currency} {start} -> {end} ({span_days}d), {len(events)} events, {len(specs)} option instruments")
    if span_days < 7 or len(result.pnl_series) < 20:
        print("WARNING: this is a smoke run, not enough data/steps for meaningful metrics yet.")
    print(
        f"total_pnl={metrics.total_pnl:.4f} sharpe={metrics.sharpe:.4f} "
        f"max_drawdown={metrics.max_drawdown:.4f} n_hedges={metrics.n_hedges} "
        f"mean_hedge_cost={metrics.mean_hedge_cost:.6f} edge_per_contract={metrics.edge_per_contract:.6f} "
        f"n_fills={len(result.fills)}"
    )

    plot_surface_3d(engine.quoter.surface, "reports/figures/surface_3d.png")
    plot_pnl_attribution(result.pnl_series, result.attributions, "reports/figures/pnl_attribution.png")
    print("Wrote reports/figures/surface_3d.png and reports/figures/pnl_attribution.png")


if __name__ == "__main__":
    main()
