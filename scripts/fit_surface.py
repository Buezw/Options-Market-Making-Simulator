#!/usr/bin/env python
"""Fit and plot the SVI surface for one day.

Usage:
    python scripts/fit_surface.py --date 2026-10-01
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np

from btc_options_mm.analytics.plots import plot_surface_3d
from btc_options_mm.backtest.events import TickerEvent
from btc_options_mm.data.loader import load_range, parse_instrument_name
from btc_options_mm.pricing.implied_vol import implied_vol
from btc_options_mm.surface.arbitrage import has_butterfly_arbitrage, has_calendar_arbitrage
from btc_options_mm.surface.fit import fit_svi_slice
from btc_options_mm.surface.interpolate import Slice, Surface

_MIN_FIT_POINTS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--out", default="reports/figures/surface_3d.png")
    return parser.parse_args()


def _collect_smile_samples(
    events, currency: str, as_of: datetime
) -> dict[float, list[tuple[float, float, float]]]:
    """Walk a day's events once, tracking the live forward from the
    perpetual's ticker, and bucket every resolvable option quote into
    (log-moneyness, total-variance, vega-weight) samples per expiry.
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
        w = vol**2 * spec.expiry
        weight = 1.0
        samples[spec.expiry].append((k, w, weight))

    return samples


def main() -> None:
    args = parse_args()
    day = date.fromisoformat(args.date)
    as_of = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)

    events = list(load_range(args.root, args.currency, day, day))
    if not events:
        print(f"No recorded data for {args.currency} on {args.date} under {args.root}")
        return

    samples_by_expiry = _collect_smile_samples(events, args.currency, as_of)

    slices = []
    for expiry, samples in samples_by_expiry.items():
        if len(samples) < _MIN_FIT_POINTS:
            continue
        ks, ws, weights = zip(*samples)
        params = fit_svi_slice(np.array(ks), np.array(ws), weights=np.array(weights), seed=0)
        slices.append(Slice(expiry=expiry, params=params))

    if not slices:
        print("Not enough option samples with a live forward to fit any expiry yet.")
        return

    surface = Surface(slices)
    k_grid = np.linspace(-1.0, 1.0, 41)
    for s in slices:
        w = np.array([surface.total_variance(k, s.expiry) for k in k_grid])
        if has_butterfly_arbitrage(k_grid, w):
            print(f"WARNING: butterfly arbitrage in fitted expiry T={s.expiry:.4f}")

    expiries = np.array([s.expiry for s in surface.slices])
    w_by_expiry = [np.array([surface.total_variance(k, T) for k in k_grid]) for T in expiries]
    if has_calendar_arbitrage(expiries, w_by_expiry):
        print("WARNING: calendar arbitrage across fitted expiries")

    out_path = plot_surface_3d(surface, args.out)
    print(f"Fitted {len(slices)} expiries from {len(events)} events, wrote {out_path}")


if __name__ == "__main__":
    main()
