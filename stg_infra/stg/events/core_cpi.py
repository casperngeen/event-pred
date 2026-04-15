"""Core CPI implied mean recovery from Kalshi above/below markets.

Covers two series:
  - Core CPI MoM : CPICORE-YYMM   (month-on-month)
  - Core CPI YoY : CPICOREYOY-YYMM (year-on-year)
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "stg"))

from stg_infra.stg.events.implied import compute_threshold_series
from stg_infra.stg.events.config import DATA_DIR, DATE_START, DATE_END

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

    log.info("Loading data...")
    markets = pl.read_parquet(str(DATA_DIR / "markets/*.parquet"))
    trades  = pl.read_parquet(str(DATA_DIR / "trades/*.parquet"))

    mom = compute_core_cpi_mom_series(markets, trades)
    yoy = compute_core_cpi_yoy_series(markets, trades)
    results = pl.concat([df for df in [mom, yoy] if not df.is_empty()])

    print(f"\nCore CPI MoM rows: {len(mom)}")
    print(f"Core CPI YoY rows: {len(yoy)}")
    print(f"Total rows:        {len(results)}")
    print(f"Events covered:    {results['event_ticker'].n_unique()}")
    print(results.filter(pl.col("implied_mean").is_not_null()).head(10))

    out = Path("kalshi/core_cpi_implied_mean.parquet")
    results.write_parquet(str(out))
    log.info("Saved to %s", out)
