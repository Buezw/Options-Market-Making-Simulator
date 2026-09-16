"""Taylor-expansion PnL decomposition: edge, delta, gamma, theta, vega, hedge cost.

See README methodology section 6. Greeks (delta, gamma, theta, vega) are
evaluated at the start of the step; dF, dt, dsigma are the step's changes in
forward, time-to-expiry consumed, and implied vol.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PnLAttribution:
    edge: float
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    hedge_cost: float
    residual: float

    @property
    def explained(self) -> float:
        return (
            self.edge
            + self.delta_pnl
            + self.gamma_pnl
            + self.theta_pnl
            + self.vega_pnl
            - self.hedge_cost
        )


def attribute_pnl(
    actual_pnl: float,
    delta: float,
    gamma: float,
    theta: float,
    vega: float,
    dF: float,
    dt: float,
    dsigma: float,
    edge: float = 0.0,
    hedge_cost: float = 0.0,
) -> PnLAttribution:
    """Decompose one step's actual PnL into edge, Greeks terms and a residual."""
    delta_pnl = delta * dF
    gamma_pnl = 0.5 * gamma * dF**2
    theta_pnl = theta * dt
    vega_pnl = vega * dsigma
    explained = edge + delta_pnl + gamma_pnl + theta_pnl + vega_pnl - hedge_cost
    residual = actual_pnl - explained
    return PnLAttribution(
        edge=edge,
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        theta_pnl=theta_pnl,
        vega_pnl=vega_pnl,
        hedge_cost=hedge_cost,
        residual=residual,
    )


def gamma_theta_pnl(gamma: float, F: float, realized_vol: float, implied_vol: float, dt: float) -> float:
    """Expected PnL of a delta-hedged position from gamma and theta combined:

    PnL ~= 0.5 * Gamma * F^2 * (realized_vol^2 - implied_vol^2) * dt
    """
    return 0.5 * gamma * F**2 * (realized_vol**2 - implied_vol**2) * dt
