import pytest

from btc_options_mm.mm.queue_model import QueuePositionFillModel
from btc_options_mm.mm.quoter import Quote

QUOTE = Quote(instrument="TEST", bid_price=100.0, ask_price=110.0, bid_vol=0.5, ask_vol=0.55, size=2.0)


def test_no_fill_when_trade_does_not_cross():
    model = QueuePositionFillModel()
    model.sync_quote("TEST", "bid", price=100.0, resting_size_at_price=0.0)
    assert model.try_fill(QUOTE, "bid", trade_price=105.0, trade_amount=10.0, trade_direction="sell") is None


def test_no_fill_while_trade_volume_is_smaller_than_queue_ahead():
    model = QueuePositionFillModel()
    model.sync_quote("TEST", "bid", price=100.0, resting_size_at_price=10.0)

    fill = model.try_fill(QUOTE, "bid", trade_price=99.0, trade_amount=4.0, trade_direction="sell")
    assert fill is None
    assert model._queues[("TEST", "bid")].ahead == pytest.approx(6.0)


def test_partial_fill_once_trade_volume_exceeds_queue_ahead():
    model = QueuePositionFillModel()
    model.sync_quote("TEST", "bid", price=100.0, resting_size_at_price=10.0)

    fill = model.try_fill(QUOTE, "bid", trade_price=99.0, trade_amount=13.0, trade_direction="sell")
    assert fill is not None
    assert fill.side == "buy"
    assert fill.price == 100.0
    assert fill.size == pytest.approx(2.0)  # 13 - 10 ahead = 3, capped at quote.size=2
    assert model._queues[("TEST", "bid")].ahead == pytest.approx(0.0)


def test_full_quote_size_fill_once_queue_already_exhausted():
    model = QueuePositionFillModel()
    model.sync_quote("TEST", "bid", price=100.0, resting_size_at_price=0.0)

    fill = model.try_fill(QUOTE, "bid", trade_price=99.0, trade_amount=5.0, trade_direction="sell")
    assert fill is not None
    assert fill.size == pytest.approx(2.0)  # capped at quote.size


def test_cancellations_via_book_update_shrink_the_queue():
    model = QueuePositionFillModel()
    model.sync_quote("TEST", "bid", price=100.0, resting_size_at_price=10.0)
    model.on_book_update("TEST", "bid", price=100.0, resting_size_at_price=3.0)

    assert model._queues[("TEST", "bid")].ahead == pytest.approx(3.0)
    assert model.try_fill(QUOTE, "bid", trade_price=99.0, trade_amount=3.0, trade_direction="sell") is None

    fill = model.try_fill(QUOTE, "bid", trade_price=99.0, trade_amount=1.0, trade_direction="sell")
    assert fill is not None
    assert fill.size == pytest.approx(1.0)


def test_book_update_never_increases_the_queue():
    model = QueuePositionFillModel()
    model.sync_quote("TEST", "bid", price=100.0, resting_size_at_price=3.0)
    model.on_book_update("TEST", "bid", price=100.0, resting_size_at_price=50.0)  # book grew -- ignore
    assert model._queues[("TEST", "bid")].ahead == pytest.approx(3.0)


def test_moving_to_a_new_price_resets_the_queue():
    model = QueuePositionFillModel()
    model.sync_quote("TEST", "bid", price=100.0, resting_size_at_price=10.0)
    model.sync_quote("TEST", "bid", price=101.0, resting_size_at_price=2.0)

    pos = model._queues[("TEST", "bid")]
    assert pos.price == pytest.approx(101.0)
    assert pos.ahead == pytest.approx(2.0)
