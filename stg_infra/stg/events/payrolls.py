"""Nonfarm payrolls implied mean recovery from Kalshi above/below markets.

Series: KXPAYROLLS-YYMM  (monthly nonfarm payrolls, thresholds in raw job counts)

The implied mean is expressed in the same units as the thresholds (jobs added).
"""

from __future__ import annotations

import logging

import polars as pl

from stg.events.implied import compute_threshold_series

log = logging.getLogger(__name__)


def compute_payrolls_series(
    markets: pl.DataFrame,
    trades: pl.DataFrame,
) -> pl.DataFrame:
    """Daily implied nonfarm payrolls mean (units: jobs added)."""
    return compute_threshold_series(
        markets, trades,
        event_pattern=r"^(KX)?PAYROLLS-\d{2}[A-Z]{3}$",
        series_type="payrolls",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    from stg.events._cli import run_and_save
    
    run_and_save({"Payrolls": compute_payrolls_series}, "kalshi/payrolls_implied_mean.parquet")
