"""Timestamp alignment and filtering of recorded market data.

Reads the recorder's partitioned parquet (`{table}/date=YYYY-MM-DD/*.parquet`
for orderbook/ticker/trades -- see `data.recorder`), decodes the JSON-encoded
book levels, parses option instrument names into `OptionSpec`s, and merges
everything into a single timestamp-ordered stream of `backtest.events.Event`s
using Deribit's exchange `timestamp` (converted from milliseconds to
seconds), not the recorder's `recorded_at` ingestion time, so replay
reflects when things actually happened on the exchange.

Deribit quotes option premiums in BTC/ETH (inverse contracts), not USD, so
an option's `mark_price` is converted to USD terms here (multiplied by that
tick's `index_price`) before it reaches anything that prices in USD, like
`pricing.black76` -- see that module's docstring. The future/perpetual's own
mark price is already in USD and is left alone.

Rows with unusable core fields (a ticker with no mark price, a book snapshot
missing a side) are dropped rather than passed through as NaN, similar in
spirit to the quote filtering described in README section 2.

The recorder subscribes at 100ms resolution across every listed instrument,
so a full day of ticker rows is easily in the tens of millions -- far finer
than anything downstream needs. A surface fit is a cross-sectional fit
across strikes at one point in time, not a time series, so one snapshot per
instrument per refit window is already enough signal; extra ticks in
between are redundant, near-duplicate points on the same smile, not new
information. `ticker_sample_interval` downsamples to at most one ticker row
per instrument per that many seconds before any per-row work (like an IV
solve) touches it, which is where the real cost is.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from btc_options_mm.backtest.events import BookUpdateEvent, Event, TickerEvent, TradeEvent
from btc_options_mm.pricing.instrument import OptionSpec

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_OPTION_RE = re.compile(
    r"^(?P<currency>[A-Z]+)-(?P<day>\d{1,2})(?P<mon>[A-Z]{3})(?P<yr>\d{2})"
    r"-(?P<strike>\d+(?:\.\d+)?)-(?P<type>[CP])$"
)
_EXPIRY_HOUR_UTC = 8  # Deribit options expire at 08:00 UTC


def parse_instrument_name(name: str, as_of: datetime) -> OptionSpec | None:
    """Parse a Deribit option instrument name (e.g. "BTC-27JUN25-50000-C")
    into an `OptionSpec`, with `expiry` as a year-fraction from `as_of`.

    Returns None for futures and the perpetual (e.g. "BTC-PERPETUAL",
    "BTC-27JUN25"), which aren't options.
    """
    match = _OPTION_RE.match(name)
    if match is None:
        return None
    expiry_dt = datetime(
        2000 + int(match["yr"]), _MONTHS[match["mon"]], int(match["day"]), _EXPIRY_HOUR_UTC, tzinfo=timezone.utc
    )
    T = (expiry_dt - as_of).total_seconds() / (365 * 86400)
    option_type = "call" if match["type"] == "C" else "put"
    return OptionSpec(name=name, strike=float(match["strike"]), expiry=T, option_type=option_type)


def _is_option(instrument_name: str) -> bool:
    return _OPTION_RE.match(instrument_name) is not None


def _level_price(level: list) -> float:
    # Grouped book levels are [price, amount]; be lenient about a leading
    # action tag ([action, price, amount]) in case of a raw-style feed.
    return float(level[0]) if len(level) == 2 else float(level[1])


def _date_range(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _read_table(root: Path, table: str, start: date, end: date) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path, engine="pyarrow")
        for d in _date_range(start, end)
        for path in sorted((root / table / f"date={d.isoformat()}").glob("*.parquet"))
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _filter_currency_and_throttle(df: pd.DataFrame, currency: str, sample_interval: float) -> pd.DataFrame:
    """Vectorized pre-filtering shared by every table: keep only rows for
    `currency`, then downsample to at most one row per instrument per
    `sample_interval` seconds (0 disables). Runs before the per-row Python
    loop below, where the real per-row cost (JSON parsing, IV solving) is.
    """
    if df.empty:
        return df
    df = df.loc[df["instrument_name"].str.upper().str.startswith(f"{currency.upper()}-")]
    if df.empty or sample_interval <= 0:
        return df
    bucket = (df["timestamp"] // (sample_interval * 1000)).astype("int64")
    return df.loc[df.groupby(["instrument_name", bucket]).cumcount() == 0]


def load_range(
    root: str | Path,
    currency: str,
    start: date,
    end: date,
    min_open_interest: float = 0.0,
    ticker_sample_interval: float = 60.0,
    book_sample_interval: float = 60.0,
) -> Iterator[Event]:
    """Stream `Event`s for `currency` between `start` and `end` (inclusive
    UTC dates), merged in ascending exchange-timestamp order.

    `min_open_interest` drops ticker samples for instruments whose reported
    open interest is below it (0 disables the filter). `ticker_sample_interval`
    and `book_sample_interval` (seconds, 0 disables either) downsample ticker
    and orderbook rows to at most one per instrument per interval -- see the
    module docstring.
    """
    root = Path(root)
    tickers = _filter_currency_and_throttle(_read_table(root, "ticker", start, end), currency, ticker_sample_interval)
    books = _filter_currency_and_throttle(_read_table(root, "orderbook", start, end), currency, book_sample_interval)
    trades = _read_table(root, "trades", start, end)
    trades = trades.loc[trades["instrument_name"].str.upper().str.startswith(f"{currency.upper()}-")] if not trades.empty else trades

    staged: list[tuple[float, Event]] = []

    for row in tickers.itertuples(index=False):
        if pd.isna(row.mark_price):
            continue
        if pd.notna(row.open_interest) and row.open_interest < min_open_interest:
            continue

        mark_price = float(row.mark_price)
        if _is_option(row.instrument_name):
            if pd.isna(row.index_price):
                continue  # can't convert BTC/ETH-denominated premium to USD -- drop it
            mark_price *= float(row.index_price)

        staged.append((
            float(row.timestamp),
            TickerEvent(
                timestamp=float(row.timestamp) / 1000.0,
                instrument=row.instrument_name,
                mark_price=mark_price,
                index_price=float(row.index_price) if pd.notna(row.index_price) else float("nan"),
            ),
        ))

    for row in books.itertuples(index=False):
        bids = json.loads(row.bids_json) if row.bids_json else []
        asks = json.loads(row.asks_json) if row.asks_json else []
        if not bids or not asks:
            continue
        staged.append((
            float(row.timestamp),
            BookUpdateEvent(
                timestamp=float(row.timestamp) / 1000.0,
                instrument=row.instrument_name,
                best_bid=_level_price(bids[0]),
                best_ask=_level_price(asks[0]),
            ),
        ))

    for row in trades.itertuples(index=False):
        staged.append((
            float(row.timestamp),
            TradeEvent(
                timestamp=float(row.timestamp) / 1000.0,
                instrument=row.instrument_name,
                price=float(row.price),
                amount=float(row.amount),
                direction=row.direction,
            ),
        ))

    staged.sort(key=lambda pair: pair[0])
    for _, event in staged:
        yield event
