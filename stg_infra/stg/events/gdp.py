"""GDP growth implied mean recovery from Kalshi above/below markets.

Series: GDP-YYMM / KXGDP-YYMM  (quarterly advance GDP growth, annualised %)

Note: GDP Annual (KXGDPYEAR-*) uses a different bucket format (B-prefix)
and is not covered here.
"""

from __future__ import annotations

import logging

import polars as pl

from stg.events.implied import compute_threshold_series

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

    from stg.events._cli import run_and_save
    
    run_and_save({"GDP": compute_gdp_series}, "kalshi/gdp_implied_mean.parquet")
