"""Drives book updates, trades, surface refits and hedge checks in order.

No lookahead: events are consumed strictly in timestamp order, and every
decision (quotes, fills, hedges) only ever sees state built from events
already processed. `HedgeCheckEvent`s double as the PnL-attribution
checkpoint boundary: between two consecutive checks, `attribute_pnl` (README
section 6) decomposes the portfolio's actual mark-to-market PnL using the
Greeks as they stood at the start of the interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from btc_options_mm.analytics.pnl_attribution import PnLAttribution, attribute_pnl
from btc_options_mm.backtest.events import (
    BookUpdateEvent,
    Event,
    HedgeCheckEvent,
    SurfaceRefitEvent,
    TickerEvent,
    TradeEvent,
)
from btc_options_mm.backtest.portfolio import Portfolio
from btc_options_mm.hedging.hedger import Hedger, HedgeTrade
from btc_options_mm.mm.fill_model import Fill, PoissonFillModel, TradeReplayFillModel
from btc_options_mm.mm.inventory import Inventory, RiskLimits
from btc_options_mm.mm.quoter import Quote, Quoter
from btc_options_mm.pricing import black76, greeks
from btc_options_mm.pricing.implied_vol import implied_vol
from btc_options_mm.pricing.instrument import OptionSpec
from btc_options_mm.surface.fit import fit_svi_slice
from btc_options_mm.surface.interpolate import Slice, Surface

_MIN_REFIT_POINTS = 5


@dataclass
class BacktestResult:
    pnl_series: list[float] = field(default_factory=list)
    attributions: list[PnLAttribution] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    hedge_trades: list[HedgeTrade] = field(default_factory=list)


@dataclass
class BacktestEngine:
    specs: dict[str, OptionSpec]
    quoter: Quoter
    inventory: Inventory
    portfolio: Portfolio
    fill_model: PoissonFillModel | TradeReplayFillModel
    hedger: Hedger
    risk_limits: RiskLimits
    option_fee: float = 0.0
    r: float = 0.0

    forward: float | None = field(default=None, init=False)
    vols: dict[str, float] = field(default_factory=dict, init=False)
    smile_buffer: dict[float, list[tuple[float, float, float]]] = field(default_factory=dict, init=False)
    last_poisson_check: dict[str, float] = field(default_factory=dict, init=False)
    edge_accum: float = field(default=0.0, init=False)
    hedge_cost_accum: float = field(default=0.0, init=False)
    last_checkpoint: dict | None = field(default=None, init=False)

    def run(self, events: Iterable[Event]) -> BacktestResult:
        result = BacktestResult()
        for event in events:
            if isinstance(event, TickerEvent):
                self._on_ticker(event, result)
            elif isinstance(event, BookUpdateEvent):
                self._on_book_update(event)
            elif isinstance(event, SurfaceRefitEvent):
                self._on_refit(event)
            elif isinstance(event, TradeEvent):
                self._on_trade(event, result)
            elif isinstance(event, HedgeCheckEvent):
                self._on_hedge_check(event, result)
            else:
                raise TypeError(f"Unknown event type: {type(event)!r}")
        return result

    # -- market state -----------------------------------------------------

    def _on_ticker(self, event: TickerEvent, result: BacktestResult) -> None:
        if event.instrument == self.hedger.instrument:
            self.forward = event.mark_price
            return
        spec = self.specs.get(event.instrument)
        if spec is None or self.forward is None:
            return  # no live forward yet (the hedge instrument hasn't ticked) -- nothing pricing-related can happen
        self._buffer_smile_sample(event.instrument, spec, event.mark_price)
        if isinstance(self.fill_model, PoissonFillModel):
            self._poisson_check(event.instrument, spec, event.timestamp, result)

    def _on_book_update(self, event: BookUpdateEvent) -> None:
        # Only used as a forward proxy for the hedge instrument; this engine
        # doesn't yet consume option book depth (see README Limitations).
        if event.instrument == self.hedger.instrument:
            self.forward = (event.best_bid + event.best_ask) / 2

    def _current_vol(self, instrument: str, spec: OptionSpec) -> float:
        k = float(np.log(spec.strike / self.forward))
        vol = self.quoter.surface.implied_vol(k, spec.expiry)
        self.vols[instrument] = vol
        return vol

    def _quote(self, instrument: str, spec: OptionSpec) -> Quote:
        vol = self._current_vol(instrument, spec)
        unit_delta = greeks.delta(self.forward, spec.strike, spec.expiry, vol, self.r, spec.option_type)
        unit_gamma = greeks.gamma(self.forward, spec.strike, spec.expiry, vol, self.r)
        unit_vega = greeks.vega(self.forward, spec.strike, spec.expiry, vol, self.r)
        self.inventory.mark(instrument, spec.expiry, unit_delta, unit_gamma, unit_vega)
        return self.quoter.quote(spec, self.forward, self.inventory, self.risk_limits)

    # -- surface refit ------------------------------------------------------

    def _buffer_smile_sample(self, instrument: str, spec: OptionSpec, mark_price: float) -> None:
        try:
            sample_vol = implied_vol(mark_price, self.forward, spec.strike, spec.expiry, self.r, spec.option_type)
        except ValueError:
            return  # unresolvable quote (crossed/below intrinsic/etc.) -- dropped, per README section 2
        k = float(np.log(spec.strike / self.forward))
        w = sample_vol**2 * spec.expiry
        weight = greeks.vega(self.forward, spec.strike, spec.expiry, sample_vol, self.r)
        self.smile_buffer.setdefault(spec.expiry, []).append((k, w, weight))

    def _on_refit(self, event: SurfaceRefitEvent) -> None:
        fitted = {s.expiry: s.params for s in self.quoter.surface.slices}
        for expiry, samples in self.smile_buffer.items():
            if len(samples) < _MIN_REFIT_POINTS:
                continue
            ks, ws, weights = zip(*samples)
            fitted[expiry] = fit_svi_slice(np.array(ks), np.array(ws), weights=np.array(weights), n_restarts=5, seed=0)
        self.quoter.surface = Surface([Slice(expiry=T, params=p) for T, p in fitted.items()])
        self.smile_buffer.clear()

    # -- fills ---------------------------------------------------------------

    def _on_trade(self, event: TradeEvent, result: BacktestResult) -> None:
        spec = self.specs.get(event.instrument)
        if spec is None or self.forward is None or not isinstance(self.fill_model, TradeReplayFillModel):
            return
        quote = self._quote(event.instrument, spec)
        fill = self.fill_model.try_fill(quote, event.price, event.amount, event.direction)
        if fill is not None:
            self._apply_fill(event.instrument, spec, fill, result)

    def _poisson_check(self, instrument: str, spec: OptionSpec, t: float, result: BacktestResult) -> None:
        quote = self._quote(instrument, spec)
        last_t = self.last_poisson_check.get(instrument, t)
        dt = t - last_t
        self.last_poisson_check[instrument] = t
        if dt <= 0 or quote.bid_vol is None or quote.ask_vol is None:
            return
        distance = (quote.ask_vol - quote.bid_vol) / 2
        for side in ("bid", "ask"):
            fill = self.fill_model.try_fill(quote, side, distance, dt)
            if fill is not None:
                self._apply_fill(instrument, spec, fill, result)

    def _apply_fill(self, instrument: str, spec: OptionSpec, fill: Fill, result: BacktestResult) -> None:
        qty_delta = fill.size if fill.side == "buy" else -fill.size
        fee = abs(qty_delta) * fill.price * self.option_fee
        self.portfolio.apply_trade(instrument, qty_delta, fill.price, fee)
        self.inventory.apply_fill(instrument, qty_delta)

        fair_vol = self.vols.get(instrument) or self._current_vol(instrument, spec)
        fair_price = black76.price(self.forward, spec.strike, spec.expiry, fair_vol, self.r, spec.option_type)
        self.edge_accum += qty_delta * (fair_price - fill.price) - fee
        result.fills.append(fill)

    # -- hedging and PnL attribution -----------------------------------------

    def _on_hedge_check(self, event: HedgeCheckEvent, result: BacktestResult) -> None:
        if self.forward is None:
            return  # nothing priced yet
        vols = {name: self._current_vol(name, spec) for name, spec in self.specs.items()}
        opt_greeks = self.portfolio.aggregate_greeks(self.specs, self.forward, vols, self.r)
        hedge_qty = self.portfolio.positions.get(self.hedger.instrument, 0.0)
        total_delta = opt_greeks["delta"] + hedge_qty

        trade = self.hedger.maybe_hedge(event.timestamp, self.forward, total_delta, opt_greeks["gamma"], self.portfolio)
        if trade is not None:
            self.hedge_cost_accum += trade.cost
            result.hedge_trades.append(trade)

        self._checkpoint(event.timestamp, total_delta, opt_greeks, vols, result)

    def _weighted_avg_vol(self, vols: dict[str, float]) -> float:
        weights = {name: abs(self.portfolio.positions.get(name, 0.0)) for name in vols}
        total_weight = sum(weights.values())
        if total_weight == 0:
            return float(np.mean(list(vols.values()))) if vols else 0.0
        return sum(vols[name] * weights[name] for name in vols) / total_weight

    def _checkpoint(
        self,
        t: float,
        total_delta: float,
        opt_greeks: dict[str, float],
        vols: dict[str, float],
        result: BacktestResult,
    ) -> None:
        current_prices = {
            name: black76.price(self.forward, spec.strike, spec.expiry, vols[name], self.r, spec.option_type)
            for name, spec in self.specs.items()
        }
        current_prices[self.hedger.instrument] = self.forward
        current_value = self.portfolio.mark_to_market(current_prices)
        avg_vol = self._weighted_avg_vol(vols)

        if self.last_checkpoint is not None:
            prev = self.last_checkpoint
            attribution = attribute_pnl(
                current_value - prev["value"],
                prev["delta"],
                prev["gamma"],
                prev["theta"],
                prev["vega"],
                dF=self.forward - prev["F"],
                dt=t - prev["t"],
                dsigma=avg_vol - prev["avg_vol"],
                edge=self.edge_accum,
                hedge_cost=self.hedge_cost_accum,
            )
            result.attributions.append(attribution)
            result.pnl_series.append(current_value - prev["value"])
            self.edge_accum = 0.0
            self.hedge_cost_accum = 0.0

        self.last_checkpoint = {
            "t": t,
            "F": self.forward,
            "value": current_value,
            "avg_vol": avg_vol,
            "delta": total_delta,
            "gamma": opt_greeks["gamma"],
            "theta": opt_greeks["theta"],
            "vega": opt_greeks["vega"],
        }
