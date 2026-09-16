"""Continuous order book, ticker and trade recorder, writing partitioned parquet.

Subscribes to Deribit's public WebSocket channels for every active option
and future in a currency, buffers incoming messages, and periodically
flushes them to `{out_dir}/{table}/date=YYYY-MM-DD/part-*.parquet`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from btc_options_mm.data.deribit_client import get_instruments, run_forever_with_reconnect

TABLES = ("orderbook", "ticker", "trades")

# Deribit's grouped book channel (book.{instrument}.none.{depth}.100ms) only
# accepts these depths -- anything else is silently dropped from the
# subscription (Deribit returns an empty result instead of an error).
VALID_BOOK_DEPTHS = (1, 10, 20)


def normalize_book_message(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "recorded_at": time.time(),
        "instrument_name": data["instrument_name"],
        "timestamp": data["timestamp"],
        "change_id": data.get("change_id"),
        "type": data.get("type"),
        "bids_json": json.dumps(data.get("bids", [])),
        "asks_json": json.dumps(data.get("asks", [])),
    }


def normalize_ticker_message(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "recorded_at": time.time(),
        "instrument_name": data["instrument_name"],
        "timestamp": data["timestamp"],
        "mark_price": data.get("mark_price"),
        "index_price": data.get("index_price"),
        "best_bid_price": data.get("best_bid_price"),
        "best_ask_price": data.get("best_ask_price"),
        "last_price": data.get("last_price"),
        "open_interest": data.get("open_interest"),
    }


def normalize_trade_message(trade: dict[str, Any]) -> dict[str, Any]:
    return {
        "recorded_at": time.time(),
        "instrument_name": trade["instrument_name"],
        "timestamp": trade["timestamp"],
        "trade_id": trade.get("trade_id"),
        "price": trade.get("price"),
        "amount": trade.get("amount"),
        "direction": trade.get("direction"),
    }


def write_partition(rows: list[dict[str, Any]], table: str, out_dir: str | Path) -> Path:
    """Write buffered rows for one table to a new UTC-date-partitioned parquet file."""
    if not rows:
        raise ValueError("no rows to write")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    partition_dir = Path(out_dir) / table / f"date={date_str}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    path = partition_dir / f"part-{uuid.uuid4().hex}.parquet"
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)
    return path


def _build_channels(currency: str, depth: int) -> list[str]:
    if depth not in VALID_BOOK_DEPTHS:
        raise ValueError(f"depth must be one of {VALID_BOOK_DEPTHS}, got {depth}")
    instruments = get_instruments(currency, "option") + get_instruments(currency, "future")
    channels = []
    for inst in instruments:
        name = inst["instrument_name"]
        channels.append(f"book.{name}.none.{depth}.100ms")
        channels.append(f"ticker.{name}.100ms")
        channels.append(f"trades.{name}.100ms")
    return channels


def _flush_all(buffers: dict[str, list[dict[str, Any]]], out_dir: str | Path) -> None:
    for table, rows in buffers.items():
        if rows:
            write_partition(list(rows), table, out_dir)
            rows.clear()


async def _flush_loop(
    buffers: dict[str, list[dict[str, Any]]], out_dir: str | Path, flush_interval: float
) -> None:
    while True:
        await asyncio.sleep(flush_interval)
        _flush_all(buffers, out_dir)


def _make_dispatcher(
    buffers: dict[str, list[dict[str, Any]]],
) -> Callable[[dict[str, Any]], None]:
    def on_message(params: dict[str, Any]) -> None:
        channel = params["channel"]
        data = params["data"]
        if channel.startswith("book."):
            buffers["orderbook"].append(normalize_book_message(data))
        elif channel.startswith("ticker."):
            buffers["ticker"].append(normalize_ticker_message(data))
        elif channel.startswith("trades."):
            for trade in data:
                buffers["trades"].append(normalize_trade_message(trade))

    return on_message


async def record(
    currency: str,
    depth: int,
    out_dir: str | Path,
    flush_interval: float = 5.0,
    duration: float | None = None,
) -> None:
    """Record order books, tickers and trades for every active option/future
    in `currency` until cancelled, or for `duration` seconds if given.
    """
    channels = _build_channels(currency, depth)
    buffers: dict[str, list[dict[str, Any]]] = {table: [] for table in TABLES}

    record_task = asyncio.create_task(
        run_forever_with_reconnect(channels, _make_dispatcher(buffers))
    )
    flush_task = asyncio.create_task(_flush_loop(buffers, out_dir, flush_interval))

    try:
        if duration is not None:
            await asyncio.sleep(duration)
        else:
            await record_task
    finally:
        record_task.cancel()
        flush_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await record_task
        with contextlib.suppress(asyncio.CancelledError):
            await flush_task
        _flush_all(buffers, out_dir)
