"""Surface-based fair value, vega/gamma-aware spreads, inventory skew.

See README methodology section 4. Fair value comes from the fitted vol
surface; half-spread widens with the option's vega and gamma; quoted vol is
skewed by current inventory (long vega/delta in a bucket shifts that
bucket's vol down, to attract buyers and encourage selling to us); hard
risk limits pause a side entirely via `Inventory.risk_flags`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from btc_options_mm.mm.inventory import Inventory, RiskLimits, maturity_bucket
from btc_options_mm.pricing import black76, greeks
from btc_options_mm.pricing.instrument import OptionSpec
from btc_options_mm.surface.interpolate import Surface


@dataclass(frozen=True)
class Quote:
    instrument: str
    bid_price: float | None
    ask_price: float | None
    bid_vol: float | None
    ask_vol: float | None
    size: float


@dataclass
class Quoter:
    surface: Surface
    base_half_spread_vol: float
    vega_spread_coef: float = 0.0
    gamma_spread_coef: float = 0.0
    inventory_skew_coef: float = 0.0
    quote_size: float = 1.0
    r: float = 0.0

    def quote(
        self,
        spec: OptionSpec,
        F: float,
        inventory: Inventory,
        limits: RiskLimits | None = None,
    ) -> Quote:
        k = float(np.log(spec.strike / F))
        fair_vol = self.surface.implied_vol(k, spec.expiry)

        vega = greeks.vega(F, spec.strike, spec.expiry, fair_vol, self.r)
        gamma = greeks.gamma(F, spec.strike, spec.expiry, fair_vol, self.r)
        half_spread = (
            self.base_half_spread_vol
            + self.vega_spread_coef * vega
            + self.gamma_spread_coef * gamma
        )

        bucket_vega = inventory.net_vega_by_bucket()[maturity_bucket(spec.expiry)]
        skew = self.inventory_skew_coef * (bucket_vega + inventory.net_delta())
        skewed_vol = fair_vol - skew

        bid_vol = skewed_vol - half_spread
        ask_vol = skewed_vol + half_spread

        pause_bid = pause_ask = False
        if limits is not None:
            flags = inventory.risk_flags(spec.name, spec.expiry, limits)
            pause_bid, pause_ask = flags["pause_bid"], flags["pause_ask"]

        bid_price = (
            None
            if pause_bid or bid_vol <= 0
            else black76.price(F, spec.strike, spec.expiry, bid_vol, self.r, spec.option_type)
        )
        ask_price = (
            None
            if pause_ask
            else black76.price(F, spec.strike, spec.expiry, ask_vol, self.r, spec.option_type)
        )

        return Quote(
            instrument=spec.name,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_vol=None if pause_bid or bid_vol <= 0 else bid_vol,
            ask_vol=None if pause_ask else ask_vol,
            size=self.quote_size,
        )
