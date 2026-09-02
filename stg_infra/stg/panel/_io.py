"""Shared, in-sample-guarded loaders for the raw Kalshi archive.

Every ad-hoc script under ``analysis/exploratory_2026_08/`` re-implemented this
block with its own ``IS_CUT = datetime(2026, 1, 1)``. It lives here once now, and
routes the wall through :mod:`stg.splits` so it cannot drift.

Default layout (run from ``event-pred/``)::

    data/markets/*.parquet
    data/trades/*.parquet

Override the root with ``$KALSHI_ARCHIVE_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from stg.splits import OOS_START, assert_no_oos

ARCHIVE_DIR = Path(os.environ.get("KALSHI_ARCHIVE_DIR", "data"))

_MARKET_COLS = (
    "ticker", "event_ticker", "market_type", "title", "yes_sub_title",
    "no_sub_title", "status", "result", "open_time", "close_time",
    "volume", "open_interest", "_fetched_at",
)
_TRADE_COLS = (
    "trade_id", "ticker", "count", "yes_price", "no_price", "taker_side",
    "created_time",
)


def markets_glob() -> str:
    return str(ARCHIVE_DIR / "markets" / "*.parquet")


def trades_glob() -> str:
    return str(ARCHIVE_DIR / "trades" / "*.parquet")


def load_markets(is_only: bool = True) -> pl.DataFrame:
    """One metadata row per ticker (latest fetch), optionally IS-only.

    ``is_only`` filters on ``close_time`` — a market that *closes* in 2026 is
    out of sample even if it opened earlier.
    """
    mk = (
        pl.scan_parquet(markets_glob())
        .select(_MARKET_COLS)
        .sort("_fetched_at", descending=True)
        .unique(subset=["ticker"], keep="first")
    )
    if is_only:
        mk = mk.filter(pl.col("close_time") < OOS_START)
    out = mk.with_columns(
        pl.col("ticker").str.replace(r"^KX", "").str.split("-").list.first()
        .alias("series_raw")
    ).collect()
    if is_only:
        assert_no_oos(out, time_col="close_time")
    return out


def scan_trades(is_only: bool = True) -> pl.LazyFrame:
    """Lazy trades scan, IS-only by default. Collect after filtering to tickers."""
    tr = pl.scan_parquet(trades_glob()).select(_TRADE_COLS)
    if is_only:
        tr = tr.filter(pl.col("created_time") < OOS_START)
    return tr
