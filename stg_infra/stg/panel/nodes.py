"""The node-feature panel: one row per (canonical series, date), holding the
market-implied belief state about that series' *nearest unresolved event*.

This is the feature source for :class:`stg.nodes.kalshi.SeriesBeliefNodes`.

Feature columns
---------------
``implied_mean implied_std implied_entropy implied_skew implied_kurtosis``
    distributional summary of the nearest unresolved event's implied PDF that day
``d_implied_mean``
    change in ``implied_mean`` over the last ``momentum_days`` (belief momentum)
``days_to_close``
    calendar days to the nearest event's resolution
``recent_volume`` ``net_flow``
    contracts traded / signed taker imbalance over the last ``activity_days``
    across all of the event's legs (attention proxies, target side)
``max_stale_days``
    worst per-leg staleness in the ladder used that day (0 = every leg fresh)
``is_bucket``
    contract-kind flag

Threshold series use :func:`stg.events.implied.build_daily_implied_means`
(forward-filled ``close``, staleness carried alongside). Bucket series (WTI)
use a per-day version of the bucket-PMF path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from stg.io.kalshi import KalshiOHLCV
from stg.events.implied import (
    BUCKET, THRESHOLD, build_daily_implied_means,
    classify_contract, parse_bucket, parse_threshold,
    parse_threshold_from_subtitle, pdf_implied_stats,
)
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import SPECS, series_filter_expr, universe

_STAT_COLS = ["implied_mean", "implied_std", "implied_entropy",
              "implied_skew", "implied_kurtosis"]


# --------------------------------------------------------------------------
def _threshold_daily(canon: str, mk: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    sub_mk = mk.filter(series_filter_expr(canon))
    daily = KalshiOHLCV.build_daily(trades, sub_mk).join(
        sub_mk.select("ticker", "yes_sub_title").unique(subset=["ticker"]),
        on="ticker", how="left",
    )
    daily = daily.with_columns(
        pl.struct("ticker", "yes_sub_title").map_elements(
            lambda r: parse_threshold(r["ticker"], r["yes_sub_title"]),
            return_dtype=pl.Float64).alias("threshold"),
        pl.col("yes_sub_title").map_elements(
            lambda s: parse_threshold_from_subtitle(s)[1],
            return_dtype=pl.Utf8).alias("threshold_convention"),
    ).filter(pl.col("threshold").is_not_null())

    stale = (daily.group_by("event_ticker", "date")
             .agg(pl.col("stale_days").max().alias("max_stale_days")))
    stats = build_daily_implied_means(daily, sub_mk, series_type=canon)
    return (stats.join(stale, on=["event_ticker", "date"], how="left")
            .with_columns(pl.lit(False).alias("is_bucket")))


def _bucket_daily(canon: str, mk: pl.DataFrame, trades_lf: pl.LazyFrame) -> pl.DataFrame:
    m = mk.filter(series_filter_expr(canon)).select(
        "ticker", "event_ticker", "yes_sub_title")
    lo, hi = [], []
    for t, s in zip(m["ticker"], m["yes_sub_title"]):
        b = parse_bucket(t, s) if classify_contract(t, s) == BUCKET else None
        lo.append(b[0] if b else None)
        hi.append(b[1] if b else None)
    m = m.with_columns(pl.Series("lo", lo, dtype=pl.Float64),
                       pl.Series("hi", hi, dtype=pl.Float64)).drop_nulls(["lo", "hi"])
    tickers = m["ticker"].unique().to_list()
    dd = (trades_lf.filter(pl.col("ticker").is_in(tickers))
          .select("ticker", "yes_price", "created_time")
          .with_columns(pl.col("created_time").dt.date().alias("date"))
          .group_by("ticker", "date")
          .agg(pl.col("yes_price").last().alias("px"))
          .collect()
          .join(m.select("ticker", "event_ticker", "lo", "hi"), on="ticker", how="inner"))

    rows: list[dict] = []
    for (ev, d), g in dd.group_by("event_ticker", "date"):
        if g.height < 3:
            continue
        mids = ((g["lo"] + g["hi"]) / 2.0).to_numpy()
        probs = (g["px"].to_numpy().astype(float) / 100.0)
        mass = probs.sum()
        if mass <= 0:
            continue
        order = np.argsort(mids)
        st = pdf_implied_stats(mids[order], (probs / mass)[order])
        rows.append(dict(event_ticker=ev, date=d, max_stale_days=0,
                         is_bucket=True, n_submarkets=int(g.height),
                         resolved_value=None, **{k: st[k] for k in
                         ("mean", "std", "skew", "kurtosis", "entropy", "median")}))
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).rename({
        "mean": "implied_mean", "std": "implied_std", "skew": "implied_skew",
        "kurtosis": "implied_kurtosis", "entropy": "implied_entropy",
        "median": "implied_median"})


def _activity(canon: str, mk: pl.DataFrame, trades_lf: pl.LazyFrame) -> pl.DataFrame:
    sub = mk.filter(series_filter_expr(canon))
    tickers = sub["ticker"].unique().to_list()
    return (trades_lf.filter(pl.col("ticker").is_in(tickers))
            .join(sub.lazy().select("ticker", "event_ticker"), on="ticker")
            .with_columns(pl.col("created_time").dt.date().alias("date"),
                          pl.when(pl.col("taker_side") == "yes")
                          .then(pl.col("count")).otherwise(-pl.col("count"))
                          .alias("signed"))
            .group_by("event_ticker", "date")
            .agg(pl.col("count").sum().alias("day_volume"),
                 pl.col("signed").sum().alias("day_signed"))
            .collect())


# --------------------------------------------------------------------------
def build_node_panel(
    series: Optional[list[str]] = None,
    *,
    cadence: str = "event",
    min_events: int = 5,
    momentum_days: int = 5,
    activity_days: int = 7,
    clearance_days: int = 16,
    markets: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Node-feature rows.

    ``cadence="event"`` keeps only rows on macro-resolution dates (the snapshot
    grid); ``cadence="daily"`` keeps every day; ``cadence="weekly"`` resamples
    to Mondays. Every row still describes the *nearest unresolved* event, so a
    node only appears while it has an event at least ``clearance_days`` out.
    """
    mk = markets if markets is not None else load_markets(is_only=True)
    tr_lf = scan_trades(is_only=True)
    tr_df = tr_lf.collect()
    names = series if series is not None else universe(min_events, mk)

    ev_meta = (mk.group_by("series_raw", "event_ticker")
               .agg(pl.col("close_time").min().alias("close_time")))

    frames: list[pl.DataFrame] = []
    for canon in names:
        spec = SPECS.get(canon)
        if spec is None or spec.kind == "categorical":
            continue
        daily = (_bucket_daily(canon, mk, tr_lf) if spec.kind == BUCKET
                 else _threshold_daily(canon, mk, tr_df))
        if daily.is_empty():
            continue
        ev_close = (ev_meta.filter(series_filter_expr(canon))
                    .select("event_ticker", "close_time"))
        daily = daily.join(ev_close, on="event_ticker", how="left").with_columns(
            pl.lit(canon).alias("series"),
            (pl.col("close_time").dt.date() - pl.col("date")).dt.total_days()
            .alias("days_to_close"))
        # nearest unresolved event per (series, date)
        daily = (daily.filter(pl.col("days_to_close") >= 0)
                 .sort("date", "days_to_close")
                 .group_by("series", "date").first())

        act = _activity(canon, mk, tr_lf)
        daily = daily.join(act, on=["event_ticker", "date"], how="left").with_columns(
            pl.col("day_volume").fill_null(0), pl.col("day_signed").fill_null(0))

        daily = daily.sort("date").with_columns(
            (pl.col("implied_mean") - pl.col("implied_mean").shift(momentum_days))
            .alias("d_implied_mean"),
            pl.col("day_volume").rolling_sum(activity_days).alias("recent_volume"),
            (pl.col("day_signed").rolling_sum(activity_days) /
             pl.col("day_volume").rolling_sum(activity_days).clip(lower_bound=1))
            .alias("net_flow"),
        )
        frames.append(daily)

    if not frames:
        return pl.DataFrame()
    panel = pl.concat(frames, how="diagonal_relaxed").sort("series", "date")
    panel = panel.filter(pl.col("days_to_close") >= clearance_days) if False else panel

    if cadence == "event":
        from stg.panel.snapshots import macro_resolution_dates
        snaps = macro_resolution_dates(mk, names)
        panel = panel.filter(pl.col("date").is_in(snaps))
    elif cadence == "weekly":
        panel = panel.filter(pl.col("date").dt.weekday() == 1)
    keep = ["series", "date", "event_ticker", "days_to_close", "is_bucket",
            *_STAT_COLS, "implied_median", "d_implied_mean", "recent_volume",
            "net_flow", "max_stale_days", "n_submarkets", "resolved_value"]
    return panel.select([c for c in keep if c in panel.columns])
