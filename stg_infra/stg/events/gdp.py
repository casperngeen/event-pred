"""GDP growth implied mean recovery from Kalshi above/below markets.

Series: GDP-YYMM / KXGDP-YYMM  (quarterly advance GDP growth, annualised %)

Note: GDP Annual (KXGDPYEAR-*) uses a different bucket format (B-prefix)
and is not covered here.
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


def compute_gdp_series(
    markets: pl.DataFrame,
    trades: pl.DataFrame,
) -> pl.DataFrame:
    """Daily implied GDP growth mean (annualised %, quarterly releases)."""
    return compute_threshold_series(
        markets, trades,
        event_pattern=r"^(KX)?GDP-\d{2}[A-Z]{3}\d{2}$",
        series_type="gdp",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    log.info("Loading data...")
    markets = pl.read_parquet(str(DATA_DIR / "markets/*.parquet"))
    trades  = pl.read_parquet(str(DATA_DIR / "trades/*.parquet"))

    results = compute_gdp_series(markets, trades)

    print(f"\nTotal rows:     {len(results)}")
    print(f"Events covered: {results['event_ticker'].n_unique()}")
    print(results.filter(pl.col("implied_mean").is_not_null()).head(10))

    out = Path("kalshi/gdp_implied_mean.parquet")
    results.write_parquet(str(out))
    log.info("Saved to %s", out)
