#!/usr/bin/env python
"""Fit and plot the SVI surface for one day.

Usage:
    python scripts/fit_surface.py --date 2026-10-01
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("Surface fitting script not yet implemented")


if __name__ == "__main__":
    main()
