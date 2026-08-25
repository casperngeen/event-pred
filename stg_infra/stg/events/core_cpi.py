"""Core CPI implied mean recovery from Kalshi above/below markets.

Covers two series:
  - Core CPI MoM : CPICORE-YYMM   (month-on-month)
  - Core CPI YoY : CPICOREYOY-YYMM (year-on-year)
"""

from __future__ import annotations

import logging

import polars as pl

from stg.events.implied import compute_threshold_series

log = logging.getLogger(__name__)


def compute_core_cpi_mom_series(
    markets: pl.DataFrame,
    trades: pl.DataFrame,
) -> pl.DataFrame:
    """Daily implied Core CPI MoM mean."""
    return compute_threshold_series(
        markets, trades,
        event_pattern=r"^(KXCPICORE|CPICORE)-\d{2}[A-Z]{3}$",
        series_type="core_cpi_mom",
    )


def compute_core_cpi_yoy_series(
    markets: pl.DataFrame,
    trades: pl.DataFrame,
) -> pl.DataFrame:
    """Daily implied Core CPI YoY mean."""
    return compute_threshold_series(
        markets, trades,
        event_pattern=r"^(KXCPICOREYOY|CPICOREYOY)-\d{2}[A-Z]{3}$",
        series_type="core_cpi_yoy",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    from stg.events._cli import run_and_save

    run_and_save(
        {"Core CPI MoM": compute_core_cpi_mom_series, "Core CPI YoY": compute_core_cpi_yoy_series},
        "kalshi/core_cpi_implied_mean.parquet",
    )
