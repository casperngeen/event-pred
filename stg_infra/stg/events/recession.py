"""Recession probability recovery from Kalshi binary markets.

Series: RECSS-YYMM / RECSSNBER-YY / KXRECSSNBER-YY

Each event has a single binary contract; its daily close price is directly
the market-implied probability of a US recession. No PDF recovery needed.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "stg"))
from stg.io.kalshi import KalshiOHLCV

from stg_infra.stg.events.config import DATA_DIR, DATE_START, DATE_END

log = logging.getLogger(__name__)

_EVENT_PATTERN = r"^(KX)?(RECSS|RECSSNBER)-"


def compute_recession_series(
    markets: pl.DataFrame,
    trades: pl.DataFrame,
) -> pl.DataFrame:
    """Daily market-implied recession probability (0–100 scale, close price).

    Returns a DataFrame with columns:
        event_ticker, date, implied_prob, series_type, resolved_value
    """
    rec_markets = markets.filter(pl.col("event_ticker").str.contains(_EVENT_PATTERN))
    rec_tickers = rec_markets["ticker"].unique().to_list()
    rec_trades  = trades.filter(pl.col("ticker").is_in(rec_tickers))

    if rec_trades.is_empty():
        return pl.DataFrame()

    log.info("Building daily OHLCV for %d recession tickers...", len(rec_tickers))
    daily = (
        KalshiOHLCV.build_daily(rec_trades, rec_markets)
        .filter((pl.col("date") >= DATE_START) & (pl.col("date") <= DATE_END))
        .select(["event_ticker", "ticker", "date", "close"])
        .rename({"close": "implied_prob"})
        .with_columns(pl.lit("recession_prob").alias("series_type"))
    )

    # Resolved value: 100 if yes (recession occurred), 0 if no
    def _resolved(event: str) -> float | None:
        rows = rec_markets.filter(pl.col("event_ticker") == event)
        yes = rows.filter(pl.col("result") == "yes")
        no  = rows.filter(pl.col("result") == "no")
        if not yes.is_empty():
            return 100.0
        if not no.is_empty():
            return 0.0
        return None

    events   = daily["event_ticker"].unique().to_list()
    resolved = {e: _resolved(e) for e in events}

    return (
        daily
        .with_columns(
            pl.col("event_ticker")
            .map_elements(lambda e: resolved.get(e), return_dtype=pl.Float64)
            .alias("resolved_value")
        )
        .sort(["event_ticker", "date"])
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    log.info("Loading data...")
    markets = pl.read_parquet(str(DATA_DIR / "markets/*.parquet"))
    trades  = pl.read_parquet(str(DATA_DIR / "trades/*.parquet"))

    results = compute_recession_series(markets, trades)

    print(f"\nTotal rows:     {len(results)}")
    print(f"Events covered: {results['event_ticker'].n_unique()}")
    print(results.head(15))

    out = Path("kalshi/recession_prob.parquet")
    results.write_parquet(str(out))
    log.info("Saved to %s", out)
