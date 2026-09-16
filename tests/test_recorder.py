import json

import pandas as pd
import pytest

from btc_options_mm.data.recorder import (
    _build_channels,
    _flush_all,
    normalize_book_message,
    normalize_ticker_message,
    normalize_trade_message,
    write_partition,
)

SAMPLE_BOOK = {
    "type": "snapshot",
    "timestamp": 1_700_000_000_000,
    "instrument_name": "BTC-PERPETUAL",
    "change_id": 12345,
    "bids": [[65000.5, 10.0], [65000.0, 5.0]],
    "asks": [[65001.0, 8.0], [65001.5, 12.0]],
}

SAMPLE_TICKER = {
    "timestamp": 1_700_000_000_000,
    "instrument_name": "BTC-PERPETUAL",
    "mark_price": 65000.7,
    "index_price": 65000.1,
    "best_bid_price": 65000.5,
    "best_ask_price": 65001.0,
    "last_price": 65000.8,
    "open_interest": 123456.0,
}

SAMPLE_TRADE = {
    "trade_id": "BTC-987654",
    "instrument_name": "BTC-PERPETUAL",
    "timestamp": 1_700_000_000_500,
    "price": 65000.5,
    "amount": 100.0,
    "direction": "buy",
}


def test_normalize_book_message():
    row = normalize_book_message(SAMPLE_BOOK)
    assert row["instrument_name"] == "BTC-PERPETUAL"
    assert row["timestamp"] == 1_700_000_000_000
    assert row["change_id"] == 12345
    assert row["type"] == "snapshot"
    assert json.loads(row["bids_json"]) == SAMPLE_BOOK["bids"]
    assert json.loads(row["asks_json"]) == SAMPLE_BOOK["asks"]
    assert "recorded_at" in row


def test_normalize_book_message_defaults_missing_bids_asks():
    data = {"instrument_name": "BTC-PERPETUAL", "timestamp": 1}
    row = normalize_book_message(data)
    assert json.loads(row["bids_json"]) == []
    assert json.loads(row["asks_json"]) == []


def test_normalize_ticker_message():
    row = normalize_ticker_message(SAMPLE_TICKER)
    assert row["instrument_name"] == "BTC-PERPETUAL"
    assert row["mark_price"] == 65000.7
    assert row["index_price"] == 65000.1
    assert row["open_interest"] == 123456.0


def test_normalize_trade_message():
    row = normalize_trade_message(SAMPLE_TRADE)
    assert row["trade_id"] == "BTC-987654"
    assert row["price"] == 65000.5
    assert row["amount"] == 100.0
    assert row["direction"] == "buy"


def test_write_partition_round_trips_through_parquet(tmp_path):
    rows = [normalize_ticker_message(SAMPLE_TICKER), normalize_ticker_message(SAMPLE_TICKER)]
    path = write_partition(rows, "ticker", tmp_path)

    assert path.exists()
    assert path.parent.name.startswith("date=")
    assert path.parent.parent.name == "ticker"

    df = pd.read_parquet(path)
    assert len(df) == 2
    assert set(rows[0].keys()) <= set(df.columns)
    assert df.iloc[0]["mark_price"] == 65000.7


def test_write_partition_raises_on_empty_rows(tmp_path):
    with pytest.raises(ValueError):
        write_partition([], "ticker", tmp_path)


def test_flush_all_drops_batch_and_logs_on_write_failure(tmp_path, monkeypatch, caplog):
    # A write failure (e.g. disk full / quota hit) must not crash a
    # long-running background recorder or accumulate unboundedly in memory --
    # it should log and drop that interval's batch, then carry on.
    def failing_write_partition(rows, table, out_dir):
        raise OSError("no space left on device")

    monkeypatch.setattr("btc_options_mm.data.recorder.write_partition", failing_write_partition)

    buffers = {
        "orderbook": [normalize_ticker_message(SAMPLE_TICKER)],
        "ticker": [],
        "trades": [],
    }
    with caplog.at_level("ERROR"):
        _flush_all(buffers, tmp_path)

    assert buffers["orderbook"] == []
    assert "Failed to write" in caplog.text


def test_flush_all_writes_normally_when_no_failure(tmp_path):
    buffers = {
        "orderbook": [],
        "ticker": [normalize_ticker_message(SAMPLE_TICKER)],
        "trades": [],
    }
    _flush_all(buffers, tmp_path)

    assert buffers["ticker"] == []
    files = list((tmp_path / "ticker").glob("date=*/*.parquet"))
    assert len(files) == 1
    assert pd.read_parquet(files[0]).iloc[0]["mark_price"] == 65000.7


def test_build_channels_rejects_invalid_depth():
    # Deribit's grouped book channel only accepts depth in {1, 10, 20}; any
    # other value is silently dropped by the exchange rather than erroring,
    # so this needs to be caught locally instead.
    with pytest.raises(ValueError):
        _build_channels("BTC", depth=5)
