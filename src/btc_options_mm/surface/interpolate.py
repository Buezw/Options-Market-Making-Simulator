"""Total-variance interpolation across listed expiries.

Between two listed expiries, total variance is interpolated linearly in T at
fixed log-moneyness k: w(k, T) = w1(k) + (w2(k) - w1(k)) * (T - T1) / (T2 - T1).
Since w1(k) <= w2(k) everywhere when the two slices are calendar-consistent,
the interpolated value stays between them, so this preserves the calendar
no-arbitrage condition. Outside the listed range, the surface is held flat
at the nearest slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from btc_options_mm.surface.svi import SVIParams, total_variance

# A fitted raw-SVI slice can dip to zero or negative total variance in some
# region without being flagged (that's exactly what `surface.arbitrage`
# checks for, separately) -- floor the returned vol so a bad fit degrades
# quoting there instead of handing downstream pricing/Greeks a sigma<=0,
# which they reject outright.
_MIN_VOL = 1e-4


@dataclass(frozen=True)
class Slice:
    expiry: float
    params: SVIParams


@dataclass
class Surface:
    slices: list[Slice]

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("Surface needs at least one expiry slice")
        self.slices = sorted(self.slices, key=lambda s: s.expiry)

    def total_variance(self, k: float, T: float) -> float:
        expiries = [s.expiry for s in self.slices]
        if T <= expiries[0]:
            return float(total_variance(np.array([k]), self.slices[0].params)[0])
        if T >= expiries[-1]:
            return float(total_variance(np.array([k]), self.slices[-1].params)[0])

        idx = np.searchsorted(expiries, T)
        lower, upper = self.slices[idx - 1], self.slices[idx]
        w_lower = float(total_variance(np.array([k]), lower.params)[0])
        w_upper = float(total_variance(np.array([k]), upper.params)[0])
        frac = (T - lower.expiry) / (upper.expiry - lower.expiry)
        return w_lower + frac * (w_upper - w_lower)

    def implied_vol(self, k: float, T: float) -> float:
        w = self.total_variance(k, T)
        vol = float(np.sqrt(max(w, 0.0) / T))
        return max(vol, _MIN_VOL)
