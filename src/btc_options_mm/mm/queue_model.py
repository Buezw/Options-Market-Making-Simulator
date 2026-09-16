"""Queue-position aware replay fill model.

`mm.fill_model.TradeReplayFillModel` fills our whole quote the instant a
trade prints through it -- optimistic, since it ignores whatever size was
already resting ahead of us at that price level (README Limitations:
"queue position is not modeled"). This model instead tracks, per
instrument/side, how much size is estimated to be resting ahead of our own
order at its current price, and only fills us once that's been worked
through by observed trade volume (and thinned by observed cancellations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from btc_options_mm.mm.fill_model import Fill
from btc_options_mm.mm.quoter import Quote


@dataclass
class _QueuePosition:
    price: float
    ahead: float  # estimated size resting ahead of us at `price`


@dataclass
class QueuePositionFillModel:
    _queues: dict[tuple[str, str], _QueuePosition] = field(default_factory=dict, init=False)

    def sync_quote(self, instrument: str, side: str, price: float, resting_size_at_price: float) -> None:
        """Call whenever our quote is (re)posted. Moving to a new price joins
        the back of that level's queue; staying at the same price keeps our
        existing position in line.
        """
        key = (instrument, side)
        existing = self._queues.get(key)
        if existing is None or existing.price != price:
            self._queues[key] = _QueuePosition(price=price, ahead=max(resting_size_at_price, 0.0))

    def on_book_update(self, instrument: str, side: str, price: float, resting_size_at_price: float) -> None:
        """A level's resting size shrank without a matching trade print --
        treat it as cancellations ahead of us thinning the queue."""
        key = (instrument, side)
        pos = self._queues.get(key)
        if pos is None or pos.price != price:
            return
        pos.ahead = min(pos.ahead, max(resting_size_at_price, 0.0))

    def try_fill(self, quote: Quote, side: str, trade_price: float, trade_amount: float, trade_direction: str) -> Fill | None:
        price = quote.bid_price if side == "bid" else quote.ask_price
        if price is None:
            return None
        crosses = (side == "bid" and trade_direction == "sell" and trade_price <= price) or (
            side == "ask" and trade_direction == "buy" and trade_price >= price
        )
        if not crosses:
            return None

        key = (quote.instrument, side)
        pos = self._queues.get(key)
        if pos is None or pos.price != price:
            pos = _QueuePosition(price=price, ahead=0.0)
            self._queues[key] = pos

        remaining_after_queue = trade_amount - pos.ahead
        pos.ahead = max(pos.ahead - trade_amount, 0.0)
        if remaining_after_queue <= 0:
            return None

        fill_size = min(remaining_after_queue, quote.size)
        fill_side = "buy" if side == "bid" else "sell"
        return Fill(instrument=quote.instrument, side=fill_side, price=price, size=fill_size)
