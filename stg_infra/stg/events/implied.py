"""Shared utilities for recovering implied distributions from Kalshi markets.

Provides ticker parsing, PDF recovery, and implied mean/std computation
for markets that follow the "Above X%" threshold structure (CPI, FED, etc.).
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import polars as pl

_THRESHOLD_RE = re.compile(
    r"-T(-\d+\.?\d*)$"     # new format negative: T-0.1
    r"|"
    r"-TN(\d+\.?\d*)$"     # old format negative: TN0.3
    r"|"
    r"-T(\d+\.?\d*)$"      # positive: T0.4
)

def parse_threshold(ticker: str) -> Optional[float]:
    """Extract the numeric threshold from an 'Above X%' submarket ticker."""
    m = _THRESHOLD_RE.search(ticker)
    if m is None:
        return None
    new_neg, old_neg, pos = m.group(1), m.group(2), m.group(3)
    if new_neg is not None:
        return float(new_neg)
    if old_neg is not None:
        return -float(old_neg)
    return float(pos)

def recover_pdf(
    thresholds: np.ndarray,
    above_probs: np.ndarray,
    tail_width: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover a discrete PDF from P(X > threshold) values.

    Parameters
    ----------
    thresholds  : (N,) sorted ascending
    above_probs : (N,) P(value > threshold_i) in [0, 1]
    tail_width  : width of open tail buckets (defaults to median spacing)

    Returns
    -------
    midpoints : (N+1,) bucket midpoints
    probs     : (N+1,) probability mass per bucket, sums to 1

    Buckets:
        0      : value ≤ thresholds[0]              (left tail)
        1..N-1 : thresholds[i-1] < value ≤ thresholds[i]
        N      : value > thresholds[-1]              (right tail)
    """
    n = len(thresholds)
    spacing = float(np.median(np.diff(thresholds))) if n > 1 else 0.1
    if tail_width is None:
        tail_width = spacing

    raw = np.empty(n + 1)
    raw[0] = 1.0 - above_probs[0]
    raw[1:n] = above_probs[:-1] - above_probs[1:]
    raw[n] = above_probs[-1]

    raw = np.clip(raw, 0.0, None)
    total = raw.sum()
    probs = raw / total if total > 0 else np.ones(n + 1) / (n + 1)

    midpoints = np.empty(n + 1)
    midpoints[0] = thresholds[0] - tail_width / 2.0
    midpoints[1:n] = (thresholds[:-1] + thresholds[1:]) / 2.0
    midpoints[n] = thresholds[-1] + tail_width / 2.0

    return midpoints, probs


def pdf_implied_mean(thresholds: np.ndarray, above_probs: np.ndarray) -> float:
    midpoints, probs = recover_pdf(thresholds, above_probs)
    return float(np.dot(midpoints, probs))


def pdf_implied_std(thresholds: np.ndarray, above_probs: np.ndarray) -> float:
    midpoints, probs = recover_pdf(thresholds, above_probs)
    mean = float(np.dot(midpoints, probs))
    return float(np.sqrt(np.dot(probs, (midpoints - mean) ** 2)))


def resolved_value(event_markets: pl.DataFrame) -> Optional[float]:
    """Infer the resolved value from submarket yes/no results.

    Handles three cases:
    - Both yes and no: midpoint of (highest yes threshold, lowest no threshold)
    - Only no: resolved value was below the lowest no threshold
    - Only yes: resolved value was above the highest yes threshold
    """
    rows = (
        event_markets
        .with_columns(
            pl.col("ticker")
            .map_elements(parse_threshold, return_dtype=pl.Float64)
            .alias("threshold")
        )
        .filter(pl.col("threshold").is_not_null())
        .filter(pl.col("result").is_in(["yes", "no"]))
        .sort("threshold")
    )
    if rows.is_empty():
        return None

    spacing = 0.1
    yes_rows = rows.filter(pl.col("result") == "yes")
    no_rows  = rows.filter(pl.col("result") == "no")

    if not yes_rows.is_empty() and not no_rows.is_empty():
        hy = yes_rows["threshold"].max()
        ln = no_rows["threshold"].min()
        if hy is None or ln is None:
            return None
        highest_yes, lowest_no = float(hy), float(ln) # type: ignore
        if lowest_no <= highest_yes:
            return None
        return (highest_yes + lowest_no) / 2.0

    if yes_rows.is_empty():
        ln = no_rows["threshold"].min()
        return float(ln) - spacing / 2.0 if ln is not None else None # type: ignore

    hy = yes_rows["threshold"].max()
    return float(hy) + spacing / 2.0 if hy is not None else None # type: ignore

def compute_threshold_series(
    markets: pl.DataFrame,
    trades: pl.DataFrame,
    event_pattern: str,
    series_type: str,
    date_start: "pl.Expr | None" = None,
    date_end: "pl.Expr | None" = None,
) -> pl.DataFrame:
    """Build a daily implied-mean series for any threshold-based event type.

    Filters markets/trades by ``event_pattern``, builds daily OHLCV,
    parses thresholds, and returns the PDF-derived implied mean/std.

    Parameters
    ----------
    markets, trades  : raw Kalshi parquet DataFrames
    event_pattern    : regex matched against ``event_ticker``
    series_type      : label stored in the ``series_type`` column
    date_start/end   : override the module-level DATE_START/DATE_END;
                       pass a ``pl.date(...)`` expression if needed
    """
    # import here to avoid circular dependency (KalshiOHLCV → stg)
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent.parent / "stg"))
    from stg.io.kalshi import KalshiOHLCV

    from stg_infra.stg.events.config import DATE_START as _DS, DATE_END as _DE
    ds = date_start if date_start is not None else _DS
    de = date_end   if date_end   is not None else _DE

    evt_markets = markets.filter(pl.col("event_ticker").str.contains(event_pattern))
    tickers     = evt_markets["ticker"].unique().to_list()
    evt_trades  = trades.filter(pl.col("ticker").is_in(tickers))

    if evt_trades.is_empty():
        return pl.DataFrame()

    daily = (
        KalshiOHLCV.build_daily(evt_trades, evt_markets)
        .filter((pl.col("date") >= ds) & (pl.col("date") <= de))
        .with_columns(
            pl.col("ticker")
            .map_elements(parse_threshold, return_dtype=pl.Float64)
            .alias("threshold")
        )
        .filter(pl.col("threshold").is_not_null())
    )

    return build_daily_implied_means(daily, evt_markets, series_type=series_type)


def build_daily_implied_means(
    daily_ohlcv: pl.DataFrame,
    markets: pl.DataFrame,
    series_type: str,
) -> pl.DataFrame:
    """Compute daily implied mean/std from a daily OHLCV DataFrame.

    Expects `daily_ohlcv` to already be filtered to the relevant tickers
    and date range, and to have a `threshold` column parsed from the ticker.

    Parameters
    ----------
    daily_ohlcv : output of KalshiOHLCV.build_daily with `threshold` column
    markets     : markets DataFrame for the same events (used for resolved values)
    series_type : label to store in the `series_type` column

    Returns
    -------
    DataFrame with columns:
        event_ticker, date, implied_mean, implied_std,
        n_submarkets, series_type, resolved_value
    """
    def _group_mean(df: pl.DataFrame) -> pl.DataFrame:
        thr = df["threshold"].to_numpy()
        prb = df["close"].to_numpy() / 100.0
        idx = np.argsort(thr)
        thr, prb = thr[idx], prb[idx]

        if len(thr) < 2:
            return df.head(1).select(["event_ticker", "date"]).with_columns([
                pl.lit(None).cast(pl.Float64).alias("implied_mean"),
                pl.lit(None).cast(pl.Float64).alias("implied_std"),
                pl.lit(len(thr)).cast(pl.Int32).alias("n_submarkets"),
                pl.lit(series_type).alias("series_type"),
            ])

        midpoints, probs = recover_pdf(thr, prb)
        mean = float(np.dot(midpoints, probs))
        std  = float(np.sqrt(np.dot(probs, (midpoints - mean) ** 2)))
        return df.head(1).select(["event_ticker", "date"]).with_columns([
            pl.lit(mean).alias("implied_mean"),
            pl.lit(std).alias("implied_std"),
            pl.lit(len(thr)).cast(pl.Int32).alias("n_submarkets"),
            pl.lit(series_type).alias("series_type"),
        ])

    result = (
        daily_ohlcv
        .group_by(["event_ticker", "date"])
        .map_groups(_group_mean)
        .sort(["event_ticker", "date"])
    )

    events = result["event_ticker"].unique().to_list()
    resolved = {
        e: resolved_value(markets.filter(pl.col("event_ticker") == e))
        for e in events
    }
    return result.with_columns(
        pl.col("event_ticker")
        .map_elements(lambda e: resolved.get(e), return_dtype=pl.Float64)
        .alias("resolved_value")
    )
