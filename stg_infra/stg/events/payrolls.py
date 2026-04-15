"""Nonfarm payrolls implied mean recovery from Kalshi above/below markets.

Series: KXPAYROLLS-YYMM  (monthly nonfarm payrolls, thresholds in raw job counts)

The implied mean is expressed in the same units as the thresholds (jobs added).
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

    log.info("Loading data...")
    markets = pl.read_parquet(str(DATA_DIR / "markets/*.parquet"))
    trades  = pl.read_parquet(str(DATA_DIR / "trades/*.parquet"))

    results = compute_payrolls_series(markets, trades)

    print(f"\nTotal rows:     {len(results)}")
    print(f"Events covered: {results['event_ticker'].n_unique()}")
    print(results.filter(pl.col("implied_mean").is_not_null()).head(10))

    out = Path("kalshi/payrolls_implied_mean.parquet")
    results.write_parquet(str(out))
    log.info("Saved to %s", out)
