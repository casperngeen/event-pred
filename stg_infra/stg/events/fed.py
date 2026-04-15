"""FED implied rate recovery from Kalshi markets.

Handles two market structures:

1. FED-YYMM (old format) — "Above X%" contracts for the fed funds rate level.
   Uses the shared threshold-based PDF recovery from implied.py.

2. FEDDECISION-YYMM / KXFEDDECISION-YYMM (newer format) — categorical
   contracts for the rate change decision (Cut/Hold/Hike bps).
   Computes a probability-weighted implied rate change.
"""

from __future__ import annotations

import re
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "stg"))
from stg.io.kalshi import KalshiOHLCV

from stg_infra.stg.events.implied import parse_threshold, build_daily_implied_means
from stg_infra.stg.events.config import DATA_DIR, DATE_START, DATE_END

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FEDDECISION: categorical bps mapping
# ---------------------------------------------------------------------------

_DECISION_BPS: dict[str, float] = {
    "C26": -50.0,   # Cut >25bps
    "C25": -25.0,   # Cut 25bps
    "H0":    0.0,   # Hold
    "H25":  25.0,   # Hike 25bps
    "H26":  50.0,   # Hike >25bps
}

_DECISION_RE = re.compile(r"-([A-Z0-9]+)$")


def _parse_decision_suffix(ticker: str) -> Optional[str]:
    m = _DECISION_RE.search(ticker)
    if m is None:
        return None
    suffix = m.group(1)
    return suffix if suffix in _DECISION_BPS else None


# ---------------------------------------------------------------------------
# FED-YYMM: "Above X%" rate level contracts
# ---------------------------------------------------------------------------

def compute_fed_level_series(
    markets: pl.DataFrame,
    trades: pl.DataFrame,
) -> pl.DataFrame:
    """Daily implied fed funds rate level for FED-YYMM events."""
    fed_markets = markets.filter(
        pl.col("event_ticker").str.contains(r"^FED-\d{2}[A-Z]{3}$")
    )
    fed_tickers = fed_markets["ticker"].unique().to_list()
    fed_trades  = trades.filter(pl.col("ticker").is_in(fed_tickers))

    if fed_trades.is_empty():
        return pl.DataFrame()

    log.info("Building daily OHLCV for %d FED-level tickers...", len(fed_tickers))
    daily = (
        KalshiOHLCV.build_daily(fed_trades, fed_markets)
        .filter((pl.col("date") >= DATE_START) & (pl.col("date") <= DATE_END))
        .with_columns(
            pl.col("ticker")
            .map_elements(parse_threshold, return_dtype=pl.Float64)
            .alias("threshold")
        )
        .filter(pl.col("threshold").is_not_null())
    )

    return build_daily_implied_means(daily, fed_markets, series_type="fed_level")


# ---------------------------------------------------------------------------
# FEDDECISION: categorical cut/hold/hike contracts
# ---------------------------------------------------------------------------

def compute_fed_decision_series(
    markets: pl.DataFrame,
    trades: pl.DataFrame,
) -> pl.DataFrame:
    """Daily implied rate change (bps) for FEDDECISION/KXFEDDECISION events."""
    dec_markets = markets.filter(
        pl.col("event_ticker").str.contains(r"^(KXFEDDECISION|FEDDECISION)-")
    )
    dec_tickers = dec_markets["ticker"].unique().to_list()
    dec_trades  = trades.filter(pl.col("ticker").is_in(dec_tickers))

    if dec_trades.is_empty():
        return pl.DataFrame()

    log.info("Building daily OHLCV for %d FEDDECISION tickers...", len(dec_tickers))
    daily = (
        KalshiOHLCV.build_daily(dec_trades, dec_markets)
        .filter((pl.col("date") >= DATE_START) & (pl.col("date") <= DATE_END))
        .with_columns(
            pl.col("ticker")
            .map_elements(_parse_decision_suffix, return_dtype=pl.String)
            .alias("decision")
        )
        .filter(pl.col("decision").is_not_null())
    )

    def _group_decision_mean(df: pl.DataFrame) -> pl.DataFrame:
        decisions  = df["decision"].to_list()
        prices     = np.clip(df["close"].to_numpy() / 100.0, 0.0, None)
        total      = prices.sum()
        probs      = prices / total if total > 0 else np.ones(len(prices)) / len(prices)
        bps_values = np.array([_DECISION_BPS[d] for d in decisions])
        mean = float(np.dot(probs, bps_values))
        std  = float(np.sqrt(np.dot(probs, (bps_values - mean) ** 2)))
        return df.head(1).select(["event_ticker", "date"]).with_columns([
            pl.lit(mean).alias("implied_mean"),
            pl.lit(std).alias("implied_std"),
            pl.lit(len(decisions)).cast(pl.Int32).alias("n_submarkets"),
            pl.lit("decision_bps").alias("series_type"),
        ])

    result = (
        daily
        .group_by(["event_ticker", "date"])
        .map_groups(_group_decision_mean)
        .sort(["event_ticker", "date"])
    )

    def _resolved_decision(event: str) -> Optional[float]:
        mkts = dec_markets.filter(pl.col("event_ticker") == event)
        yes_rows = (
            mkts
            .with_columns(
                pl.col("ticker")
                .map_elements(_parse_decision_suffix, return_dtype=pl.String)
                .alias("decision")
            )
            .filter(pl.col("result") == "yes")
            .filter(pl.col("decision").is_not_null())
        )
        if yes_rows.is_empty():
            return None
        return _DECISION_BPS.get(yes_rows["decision"][0])

    events   = result["event_ticker"].unique().to_list()
    resolved = {e: _resolved_decision(e) for e in events}
    return result.with_columns(
        pl.col("event_ticker")
        .map_elements(lambda e: resolved.get(e), return_dtype=pl.Float64)
        .alias("resolved_value")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    log.info("Loading data...")
    markets = pl.read_parquet(str(DATA_DIR / "markets/*.parquet"))
    trades  = pl.read_parquet(str(DATA_DIR / "trades/*.parquet"))

    level    = compute_fed_level_series(markets, trades)
    decision = compute_fed_decision_series(markets, trades)

    all_results = pl.concat([df for df in [level, decision] if not df.is_empty()])

    print(f"\nLevel series rows:    {len(level)}")
    print(f"Decision series rows: {len(decision)}")
    print(f"Total rows:           {len(all_results)}")
    print(f"Events covered:       {all_results['event_ticker'].n_unique()}")

    print("\nLevel sample:")
    print(level.filter(pl.col("implied_mean").is_not_null()).head(10))

    print("\nDecision sample:")
    print(decision.filter(pl.col("implied_mean").is_not_null()).head(10))

    out = Path("kalshi/fed_implied_mean.parquet")
    all_results.write_parquet(str(out))
    log.info("Saved to %s", out)
