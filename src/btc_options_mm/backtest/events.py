"""Event types processed by the backtest engine, in timestamp order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class BookUpdateEvent:
    timestamp: float
    instrument: str
    best_bid: float
    best_ask: float


@dataclass(frozen=True)
class TickerEvent:
    timestamp: float
    instrument: str
    mark_price: float
    index_price: float


@dataclass(frozen=True)
class TradeEvent:
    timestamp: float
    instrument: str
    price: float
    amount: float
    direction: str  # "buy" or "sell", taker's side


@dataclass(frozen=True)
class SurfaceRefitEvent:
    timestamp: float


@dataclass(frozen=True)
class HedgeCheckEvent:
    timestamp: float


Event = Union[BookUpdateEvent, TickerEvent, TradeEvent, SurfaceRefitEvent, HedgeCheckEvent]
