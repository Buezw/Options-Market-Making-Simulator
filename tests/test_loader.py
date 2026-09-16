import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from btc_options_mm.backtest.events import BookUpdateEvent, TickerEvent, TradeEvent
from btc_options_mm.data.loader import load_range, parse_instrument_name


def _write(root, table, day: str, rows: list[dict]) -> None:
    partition = root / table / f"date={day}"
    partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(partition / "part-0.parquet", engine="pyarrow", index=False)


def test_load_range_merges_and_orders_by_exchange_timestamp(tmp_path):
    _write(tmp_path, "ticker", "2026-01-01", [
        {
            "recorded_at": 1.0, "instrument_name": "BTC-PERPETUAL", "timestamp": 3000,
            "mark_price": 50_000.0, "index_price": 50_000.0, "best_bid_price": None,
            "best_ask_price": None, "last_price": None, "open_interest": 100.0,
        },
        {
            "recorded_at": 1.0, "instrument_name": "BTC-PERPETUAL", "timestamp": 1000,
            "mark_price": 49_900.0, "index_price": 49_900.0, "best_bid_price": None,
            "best_ask_price": None, "last_price": None, "open_interest": 100.0,
        },
    ])
    _write(tmp_path, "trades", "2026-01-01", [
        {
            "recorded_at": 1.0, "instrument_name": "BTC-PERPETUAL", "timestamp": 2000,
            "trade_id": "1", "price": 49_950.0, "amount": 1.5, "direction": "buy",
        },
    ])

    events = list(load_range(tmp_path, "BTC", date(2026, 1, 1), date(2026, 1, 1), ticker_sample_interval=0))

    assert [e.timestamp for e in events] == [1.0, 2.0, 3.0]
    assert isinstance(events[0], TickerEvent) and events[0].mark_price == 49_900.0
    assert isinstance(events[1], TradeEvent) and events[1].price == 49_950.0
    assert isinstance(events[2], TickerEvent) and events[2].mark_price == 50_000.0


def test_load_range_parses_book_levels_from_json(tmp_path):
    _write(tmp_path, "orderbook", "2026-01-01", [
        {
            "recorded_at": 1.0, "instrument_name": "BTC-PERPETUAL", "timestamp": 1000,
            "change_id": 1, "type": "snapshot",
            "bids_json": json.dumps([[49_999.0, 2.0], [49_998.0, 1.0]]),
            "asks_json": json.dumps([[50_001.0, 3.0]]),
        },
    ])
    events = list(load_range(tmp_path, "BTC", date(2026, 1, 1), date(2026, 1, 1)))
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, BookUpdateEvent)
    assert event.best_bid == 49_999.0
    assert event.best_ask == 50_001.0


def test_load_range_drops_book_rows_missing_a_side(tmp_path):
    _write(tmp_path, "orderbook", "2026-01-01", [
        {
            "recorded_at": 1.0, "instrument_name": "BTC-PERPETUAL", "timestamp": 1000,
            "change_id": 1, "type": "snapshot",
            "bids_json": json.dumps([]),
            "asks_json": json.dumps([[50_001.0, 3.0]]),
        },
    ])
    assert list(load_range(tmp_path, "BTC", date(2026, 1, 1), date(2026, 1, 1))) == []


def test_load_range_filters_by_currency(tmp_path):
    _write(tmp_path, "ticker", "2026-01-01", [
        {
            "recorded_at": 1.0, "instrument_name": "ETH-PERPETUAL", "timestamp": 1000,
            "mark_price": 3_000.0, "index_price": 3_000.0, "best_bid_price": None,
            "best_ask_price": None, "last_price": None, "open_interest": 100.0,
        },
    ])
    assert list(load_range(tmp_path, "BTC", date(2026, 1, 1), date(2026, 1, 1))) == []
    assert len(list(load_range(tmp_path, "ETH", date(2026, 1, 1), date(2026, 1, 1)))) == 1


def test_load_range_filters_by_min_open_interest(tmp_path):
    _write(tmp_path, "ticker", "2026-01-01", [
        {
            "recorded_at": 1.0, "instrument_name": "BTC-PERPETUAL", "timestamp": 1000,
            "mark_price": 50_000.0, "index_price": 50_000.0, "best_bid_price": None,
            "best_ask_price": None, "last_price": None, "open_interest": 5.0,
        },
    ])
    events = list(load_range(tmp_path, "BTC", date(2026, 1, 1), date(2026, 1, 1), min_open_interest=10.0))
    assert events == []


def test_load_range_skips_ticker_rows_with_no_mark_price(tmp_path):
    _write(tmp_path, "ticker", "2026-01-01", [
        {
            "recorded_at": 1.0, "instrument_name": "BTC-PERPETUAL", "timestamp": 1000,
            "mark_price": None, "index_price": 50_000.0, "best_bid_price": None,
            "best_ask_price": None, "last_price": None, "open_interest": 100.0,
        },
    ])
    assert list(load_range(tmp_path, "BTC", date(2026, 1, 1), date(2026, 1, 1))) == []


def test_load_range_throttles_ticker_rows_per_instrument(tmp_path):
    _write(tmp_path, "ticker", "2026-01-01", [
        {
            "recorded_at": 1.0, "instrument_name": "BTC-PERPETUAL", "timestamp": ts,
            "mark_price": 50_000.0, "index_price": 50_000.0, "best_bid_price": None,
            "best_ask_price": None, "last_price": None, "open_interest": 100.0,
        }
        for ts in (1_000, 1_500, 2_000, 7_000, 7_400)  # two 5s buckets: {1000,1500,2000} and {7000,7400}
    ])
    events = list(load_range(tmp_path, "BTC", date(2026, 1, 1), date(2026, 1, 1), ticker_sample_interval=5.0))
    assert [e.timestamp for e in events] == [1.0, 7.0]


def test_load_range_ticker_sample_interval_zero_disables_throttling(tmp_path):
    _write(tmp_path, "ticker", "2026-01-01", [
        {
            "recorded_at": 1.0, "instrument_name": "BTC-PERPETUAL", "timestamp": ts,
            "mark_price": 50_000.0, "index_price": 50_000.0, "best_bid_price": None,
            "best_ask_price": None, "last_price": None, "open_interest": 100.0,
        }
        for ts in (1_000, 1_500, 2_000)
    ])
    events = list(load_range(tmp_path, "BTC", date(2026, 1, 1), date(2026, 1, 1), ticker_sample_interval=0))
    assert len(events) == 3


def test_load_range_converts_option_mark_price_from_coin_to_usd(tmp_path):
    _write(tmp_path, "ticker", "2026-01-01", [
        {
            "recorded_at": 1.0, "instrument_name": "BTC-27JUN26-50000-C", "timestamp": 1000,
            "mark_price": 0.05, "index_price": 50_000.0, "best_bid_price": None,
            "best_ask_price": None, "last_price": None, "open_interest": 100.0,
        },
        {
            "recorded_at": 1.0, "instrument_name": "BTC-PERPETUAL", "timestamp": 2000,
            "mark_price": 50_000.0, "index_price": 50_000.0, "best_bid_price": None,
            "best_ask_price": None, "last_price": None, "open_interest": 100.0,
        },
    ])
    events = list(load_range(tmp_path, "BTC", date(2026, 1, 1), date(2026, 1, 1)))

    option_event = next(e for e in events if e.instrument == "BTC-27JUN26-50000-C")
    perp_event = next(e for e in events if e.instrument == "BTC-PERPETUAL")
    assert option_event.mark_price == pytest.approx(0.05 * 50_000.0)
    assert perp_event.mark_price == pytest.approx(50_000.0)  # already USD, not multiplied again


def test_load_range_drops_option_ticker_with_no_index_price(tmp_path):
    _write(tmp_path, "ticker", "2026-01-01", [
        {
            "recorded_at": 1.0, "instrument_name": "BTC-27JUN26-50000-C", "timestamp": 1000,
            "mark_price": 0.05, "index_price": None, "best_bid_price": None,
            "best_ask_price": None, "last_price": None, "open_interest": 100.0,
        },
    ])
    assert list(load_range(tmp_path, "BTC", date(2026, 1, 1), date(2026, 1, 1))) == []


def test_parse_instrument_name_option():
    as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
    call = parse_instrument_name("BTC-27JUN26-50000-C", as_of)
    assert call is not None
    assert call.strike == 50_000.0
    assert call.option_type == "call"
    assert call.expiry > 0

    put = parse_instrument_name("BTC-27JUN26-50000-P", as_of)
    assert put.option_type == "put"


def test_parse_instrument_name_returns_none_for_futures_and_perpetual():
    as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_instrument_name("BTC-PERPETUAL", as_of) is None
    assert parse_instrument_name("BTC-27JUN26", as_of) is None


def test_parse_instrument_name_is_currency_agnostic():
    as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
    spec = parse_instrument_name("ETH-27JUN26-3000-P", as_of)
    assert spec is not None
    assert spec.strike == 3_000.0
    assert spec.option_type == "put"
    assert parse_instrument_name("ETH-PERPETUAL", as_of) is None
