"""Backtest summary metrics: Sharpe, drawdown, hedge cost, edge captured."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from btc_options_mm.analytics.pnl_attribution import PnLAttribution


@dataclass(frozen=True)
class BacktestMetrics:
    total_pnl: float
    sharpe: float
    max_drawdown: float
    n_hedges: int
    mean_hedge_cost: float
    edge_per_contract: float


def sharpe_ratio(pnl_series: Sequence[float]) -> float:
    """Mean over stdev of per-period PnL.

    Not annualized -- the periodicity is whatever `pnl_series` is sampled at
    (README reports "Sharpe (daily)", i.e. this run on daily PnL).
    """
    pnl = np.asarray(pnl_series, dtype=float)
    if pnl.size < 2:
        return 0.0
    std = np.std(pnl, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(pnl) / std)


def max_drawdown(pnl_series: Sequence[float]) -> float:
    """Largest peak-to-trough decline in cumulative PnL, as a positive number."""
    pnl = np.asarray(pnl_series, dtype=float)
    if pnl.size == 0:
        return 0.0
    equity = np.cumsum(pnl)
    running_max = np.maximum.accumulate(equity)
    return float(np.max(running_max - equity))


def compute_metrics(
    pnl_series: Sequence[float],
    attributions: Sequence[PnLAttribution],
    hedge_costs: Sequence[float],
    n_fills: int,
) -> BacktestMetrics:
    """Summarize one backtest run.

    `pnl_series` is per-step realized PnL, `attributions` the matching
    per-step `PnLAttribution` breakdown, `hedge_costs` the cost paid on each
    executed hedge trade, and `n_fills` the number of option fills (for
    edge-per-contract).
    """
    total_edge = float(sum(a.edge for a in attributions))
    return BacktestMetrics(
        total_pnl=float(np.sum(pnl_series)) if len(pnl_series) else 0.0,
        sharpe=sharpe_ratio(pnl_series),
        max_drawdown=max_drawdown(pnl_series),
        n_hedges=len(hedge_costs),
        mean_hedge_cost=float(np.mean(hedge_costs)) if len(hedge_costs) else 0.0,
        edge_per_contract=total_edge / n_fills if n_fills else 0.0,
    )
