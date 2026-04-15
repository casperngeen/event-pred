"""CPI implied mean recovery from Kalshi above/below markets.

Covers two series:
  - CPI MoM  : CPI-YYMM / KXCPI-YYMM  (headline month-on-month)
  - CPI YoY  : CPIYOY-YYMM             (headline year-on-year)

Uses KalshiOHLCV to reconstruct daily OHLCV per submarket, then recovers
the implied PDF from daily close prices (each close ≈ P(CPI > threshold))
and computes the implied mean per event per day.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "stg"))

from stg_infra.stg.events.implied import compute_threshold_series
from stg_infra.stg.events.config import DATA_DIR

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

    log.info("Loading data...")
    markets = pl.read_parquet(str(DATA_DIR / "markets/*.parquet"))
    trades  = pl.read_parquet(str(DATA_DIR / "trades/*.parquet"))

    mom = compute_cpi_mom_series(markets, trades)
    yoy = compute_cpi_yoy_series(markets, trades)
    results = pl.concat([df for df in [mom, yoy] if not df.is_empty()])

    print(f"\nCPI MoM rows:   {len(mom)}")
    print(f"CPI YoY rows:   {len(yoy)}")
    print(f"Total rows:     {len(results)}")
    print(f"Events covered: {results['event_ticker'].n_unique()}")

    out = Path("kalshi/cpi_implied_mean.parquet")
    results.write_parquet(str(out))
    log.info("Saved to %s", out)
