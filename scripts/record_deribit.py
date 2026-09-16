#!/usr/bin/env python
"""Continuously record Deribit order books and trades to parquet.

Usage:
    python scripts/record_deribit.py --currency BTC --depth 10 --out data/raw
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--out", default="data/raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("Deribit recorder not yet implemented")


if __name__ == "__main__":
    main()
