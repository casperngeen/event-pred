"""Unemployment rate implied mean recovery from Kalshi above/below markets.

Series: KXU3-YYMM  (U3 unemployment rate, monthly)
"""

from __future__ import annotations

import logging

import polars as pl

from stg.events.implied import compute_threshold_series

log = logging.getLogger(__name__)


def compute_unemployment_series(
    markets: pl.DataFrame,
    trades: pl.DataFrame,
) -> pl.DataFrame:
    """Daily implied unemployment rate mean."""
    return compute_threshold_series(
        markets, trades,
        event_pattern=r"^(KX)?U3-\d{2}[A-Z]{3}$",
        series_type="unemployment",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    from stg.events._cli import run_and_save
    
    run_and_save({"Unemployment": compute_unemployment_series}, "kalshi/unemployment_implied_mean.parquet")