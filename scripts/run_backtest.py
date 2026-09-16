#!/usr/bin/env python
"""Run the full event-driven market-making backtest.

Usage:
    python scripts/run_backtest.py --config configs/backtest.yaml
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("Backtest runner not yet implemented")


if __name__ == "__main__":
    main()
