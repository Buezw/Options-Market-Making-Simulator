"""Continuous order book, ticker and trade recorder, writing partitioned parquet.

Subscribes to Deribit's public WebSocket channels for every active option
and future in a currency, buffers incoming messages, and periodically
flushes them to `{out_dir}/{table}/date=YYYY-MM-DD/part-*.parquet`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from btc_options_mm.data.deribit_client import get_instruments, run_forever_with_reconnect

logger = logging.getLogger(__name__)

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
    """Write and clear every non-empty buffer.

    A write failure (disk full/quota hit, permissions, etc.) is logged and
    that batch is dropped rather than left to accumulate: on a long-running
    background recorder, losing one flush interval's data is far preferable
    to either an unbounded memory buffer or a crashed process that stops
    recording entirely until someone notices and restarts it. Deliberately
    broad except -- this is the isolation boundary for exactly that.
    """
    for table, rows in buffers.items():
        if not rows:
            continue
        try:
            write_partition(list(rows), table, out_dir)
        except Exception:
            logger.exception(
                "Failed to write %d buffered %s rows -- dropping this batch", len(rows), table
            )
        finally:
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

    # A plain `kill <pid>` sends SIGTERM, which Python does not turn into a
    # catchable exception by default -- the process would die on the spot,
    # skipping the finally block below and losing whatever's buffered since
    # the last flush. Route both SIGTERM and SIGINT through stop_event so a
    # background daemon killed the ordinary way still shuts down cleanly.
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, ValueError):
            loop.add_signal_handler(sig, stop_event.set)

    wait_tasks = [record_task, asyncio.create_task(stop_event.wait())]
    if duration is not None:
        wait_tasks.append(asyncio.create_task(asyncio.sleep(duration)))

    try:
        done, pending = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if record_task in done:
            record_task.result()  # re-raise if it ended on an actual error
    finally:
        record_task.cancel()
        flush_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await record_task
        with contextlib.suppress(asyncio.CancelledError):
            await flush_task
        _flush_all(buffers, out_dir)
