"""CPI implied mean recovery from Kalshi above/below markets.

Covers two series:
  - CPI MoM  : CPI-YYMM / KXCPI-YYMM  (headline month-on-month)
  - CPI YoY  : CPIYOY-YYMM             (headline year-on-year)

Uses KalshiOHLCV to reconstruct daily OHLCV per submarket, then recovers
the implied PDF from daily close prices (each close ≈ P(CPI > threshold))
and computes the implied mean per event per day.
"""

from __future__ import annotations

import logging

import polars as pl

from stg.events.implied import compute_threshold_series

log = logging.getLogger(__name__)


def compute_cpi_mom_series(
    markets: pl.DataFrame,
    trades: pl.DataFrame,
) -> pl.DataFrame:
    """Daily implied headline CPI MoM mean."""
    return compute_threshold_series(
        markets, trades,
        event_pattern=r"^(KXCPI|CPI)-\d{2}[A-Z]{3}$",
        series_type="cpi_mom",
    )


def compute_cpi_yoy_series(
    markets: pl.DataFrame,
    trades: pl.DataFrame,
) -> pl.DataFrame:
    """Daily implied headline CPI YoY mean."""
    return compute_threshold_series(
        markets, trades,
        event_pattern=r"^(KXCPIYOY|CPIYOY)-\d{2}[A-Z]{3}$",
        series_type="cpi_yoy",
    )


# Keep the old name as an alias so existing call sites don't break
compute_daily_implied_means = compute_cpi_mom_series


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    from stg.events._cli import run_and_save
    
    run_and_save(
        {"CPI MoM": compute_cpi_mom_series, "CPI YoY": compute_cpi_yoy_series},
        "kalshi/cpi_implied_mean.parquet",
    )