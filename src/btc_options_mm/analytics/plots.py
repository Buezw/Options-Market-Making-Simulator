"""Figures for reports/figures/: surface plots, hedging comparison, PnL attribution."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from btc_options_mm.analytics.metrics import BacktestMetrics
from btc_options_mm.analytics.pnl_attribution import PnLAttribution
from btc_options_mm.surface.interpolate import Surface


def plot_surface_3d(
    surface: Surface,
    out_path: str | Path,
    k_range: tuple[float, float] = (-1.0, 1.0),
    n_k: int = 41,
) -> Path:
    """3D implied-vol surface (log-moneyness x expiry x vol) across the fitted expiries."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ks = np.linspace(*k_range, n_k)
    expiries = np.array([s.expiry for s in surface.slices])
    K, T = np.meshgrid(ks, expiries)
    vols = np.array([[surface.implied_vol(k, T_) for k in ks] for T_ in expiries])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K, T, vols, cmap="viridis")
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("expiry T (years)")
    ax.set_zlabel("implied vol")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_hedging_comparison(metrics_by_rule: dict[str, BacktestMetrics], out_path: str | Path) -> Path:
    """Bar charts comparing hedge rules on mean cost and trade count."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rules = list(metrics_by_rule)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(rules, [metrics_by_rule[r].mean_hedge_cost for r in rules])
    axes[0].set_title("Mean hedge cost")
    axes[1].bar(rules, [metrics_by_rule[r].n_hedges for r in rules])
    axes[1].set_title("Number of hedges")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_pnl_attribution(
    pnl_series: Sequence[float],
    attributions: Sequence[PnLAttribution],
    out_path: str | Path,
) -> Path:
    """Cumulative total PnL vs. its edge and gamma+theta components."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(pnl_series):
        ax.plot(np.cumsum(pnl_series), label="Total PnL")
    if attributions:
        ax.plot(np.cumsum([a.edge for a in attributions]), label="Cumulative edge")
        ax.plot(
            np.cumsum([a.gamma_pnl + a.theta_pnl for a in attributions]),
            label="Cumulative gamma+theta",
        )
    ax.set_xlabel("step")
    ax.set_ylabel("PnL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
