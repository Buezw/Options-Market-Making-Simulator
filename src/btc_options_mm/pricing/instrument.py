"""Strike/expiry/type descriptor for one option.

Shared across quoting, inventory and the backtest engine so they all agree
on what identifies a contract, instead of passing strike/expiry/type as
separate loose arguments everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionSpec:
    name: str
    strike: float
    expiry: float
    option_type: str  # "call" or "put"
