#!/usr/bin/env python
"""Continuously record Deribit order books and trades to parquet.

Usage:
    python scripts/record_deribit.py --currency BTC --depth 10 --out data/raw
"""

from __future__ import annotations

import argparse
import asyncio

from btc_options_mm.data.recorder import record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (default: run indefinitely)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(record(args.currency, args.depth, args.out, duration=args.duration))


if __name__ == "__main__":
    main()
