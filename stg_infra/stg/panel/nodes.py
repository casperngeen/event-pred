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
    worst per-leg staleness across the **whole ladder** that day — a
    ladder-completeness diagnostic, not a contamination measure. Under
    ``freshness="fresh"`` the legs actually used are all fresh by construction,
    so this stays non-zero (CPI 71%, FED 94%) simply because other legs of the
    ladder had not traded. Read it as "how much of the ladder was dark", and
    read ``n_fresh_legs`` for how much of it formed the number.
``n_fresh_legs``
    legs that actually traded that day and formed the snapshot
``is_bucket``
    contract-kind flag

Freshness
---------
``freshness="fresh"`` (the default) computes the implied distribution from
**only the legs that actually traded that day**, and emits no row for a day
that cannot muster ``min_fresh_legs`` of them. This is the same rule
:mod:`stg.panel.surprise` applies, and the two panels used to disagree: this
one passed ``build_daily``'s forward-filled ``close`` straight into
:func:`stg.events.implied.build_daily_implied_means`, so 74.5% of its rows were
distributions assembled from legs last traded on *different* days (34.7% with a
leg over a week stale, p99 = 137 days) — the incoherent cross-section
``build_daily``'s own docstring warns about, since an implied mean mixing prices
from different instants describes a market state that never existed.

Measured cost of the fresh rule at the ticker-day level: it keeps 23-59% of
days (CPI 4,972 -> 1,358; PAYROLLS 1,184 -> 695; CPICORE 2,562 -> 590). At
``cadence="event"`` the loss is much smaller, since resolution dates are days
the ladder is active anyway.

``freshness="filled"`` restores the old forward-filled behaviour, for the
AGCRN-style use that wants a value on every calendar day. Prefer carrying the
gap explicitly — a null row plus ``max_stale_days`` — over a silently stale
number, and forward-fill the *feature* downstream if the model needs it.
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
from stg.panel.registry import SPECS, series_filter_expr, target_universe
from stg.panel.surprise import MIN_FRESH_LEGS

_STAT_COLS = ["implied_mean", "implied_std", "implied_entropy",
              "implied_skew", "implied_kurtosis"]


# --------------------------------------------------------------------------
def _threshold_daily(canon: str, mk: pl.DataFrame, trades: pl.DataFrame,
                     freshness: str = "fresh",
                     min_fresh_legs: int = MIN_FRESH_LEGS) -> pl.DataFrame:
    if freshness not in ("fresh", "filled"):
        raise ValueError(f"unknown freshness {freshness!r}; "
                         "expected 'fresh' or 'filled'")
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

    # Staleness is measured over the *whole* ladder, fresh rule or not: it
    # describes the ladder the day offered, not the subset used.
    stale = (daily.group_by("event_ticker", "date")
             .agg(pl.col("stale_days").max().alias("max_stale_days")))

    if freshness == "fresh":
        used = daily.filter(pl.col("trade_count") > 0)
        n_fresh = (used.group_by("event_ticker", "date")
                   .agg(pl.len().alias("n_fresh_legs")))
        keep = n_fresh.filter(pl.col("n_fresh_legs") >= min_fresh_legs)
        used = used.join(keep.select("event_ticker", "date"),
                         on=["event_ticker", "date"], how="inner")
    else:  # "filled" — validated above
        used = daily
        n_fresh = (daily.group_by("event_ticker", "date")
                   .agg((pl.col("trade_count") > 0).sum().alias("n_fresh_legs")))
    if used.is_empty():
        return pl.DataFrame()

    stats = build_daily_implied_means(used, sub_mk, series_type=canon)
    return (stats
            .join(stale, on=["event_ticker", "date"], how="left")
            .join(n_fresh, on=["event_ticker", "date"], how="left")
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
        # every bucket leg here came from an actual trade that day
        rows.append(dict(event_ticker=ev, date=d, max_stale_days=0,
                         n_fresh_legs=int(g.height),
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
    clearance_days: int = 0,
    freshness: str = "fresh",
    min_fresh_legs: int = MIN_FRESH_LEGS,
    markets: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Node-feature rows.

    ``cadence="event"`` keeps only rows on macro-resolution dates (the snapshot
    grid); ``cadence="daily"`` keeps every day; ``cadence="weekly"`` resamples
    to Mondays. Every row describes the *nearest unresolved* event.

    ``clearance_days`` drops rows whose nearest event resolves within that many
    days. It previously read ``... if False else panel``, so it never applied at
    any setting — the parameter was documented, defaulted to 16, and did
    nothing. It works now, and the default is **0**, which is what was actually
    in force.

    0 is also the right default on the merits, so this is a repair rather than a
    behaviour change. A node feature here is "the market's current belief about
    the nearest unresolved event", and that belief is most informative in the
    days just before the print — excluding the last 16 days of every event's life
    would discard the part of the panel the thesis is about, and halve it (the
    median row sits 15 days from resolution). The constraint clearance was
    reaching for is a *label* constraint — a label window must not straddle a
    fold boundary — and that already exists where it belongs, as ``PURGE_DAYS``
    in :mod:`stg.splits`, applied by :func:`stg.models.train` and
    :mod:`stg.direction.folds`. Set it non-zero only for a deliberate experiment.
    """
    mk = markets if markets is not None else load_markets(is_only=True)
    tr_lf = scan_trades(is_only=True)
    tr_df = tr_lf.collect()
    names = series if series is not None else target_universe(min_events, mk)

    ev_meta = (mk.group_by("series_raw", "event_ticker")
               .agg(pl.col("close_time").min().alias("close_time")))

    frames: list[pl.DataFrame] = []
    for canon in names:
        spec = SPECS.get(canon)
        if spec is None or spec.kind == "categorical":
            continue
        daily = (_bucket_daily(canon, mk, tr_lf) if spec.kind == BUCKET
                 else _threshold_daily(canon, mk, tr_df, freshness=freshness,
                                       min_fresh_legs=min_fresh_legs))
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

        # Windows are measured in *days*, not rows. ``shift``/``rolling_sum``
        # count rows, and this frame has one row per day the series was active
        # -- so any gap (a weekend, a dark ladder, and far more of them now the
        # panel is fresh-only) silently stretched "5 days of momentum" and "7
        # days of volume" across however long the gap was.
        #
        # Volume uses the ``_by`` variants, which take the window width from the
        # date column. Momentum needs the *level* as of a date rather than an
        # aggregate over a window, so it is an as-of join against this frame's
        # own past: the last observation at or before ``date - momentum_days``.
        daily = daily.sort("date")
        past = (daily.select("date", "implied_mean")
                .rename({"implied_mean": "_mean_then"}))
        daily = (daily
                 .with_columns((pl.col("date") - pl.duration(days=momentum_days))
                               .alias("_ref_date"))
                 .sort("_ref_date")
                 .join_asof(past, left_on="_ref_date", right_on="date",
                            strategy="backward")
                 .sort("date")
                 .with_columns((pl.col("implied_mean") - pl.col("_mean_then"))
                               .alias("d_implied_mean"))
                 .drop("_ref_date", "_mean_then"))
        daily = daily.with_columns(
            pl.col("day_volume").rolling_sum_by("date", f"{activity_days}d")
            .alias("recent_volume"),
            pl.col("day_signed").rolling_sum_by("date", f"{activity_days}d")
            .alias("_signed_sum"),
        ).with_columns(
            (pl.col("_signed_sum") /
             pl.col("recent_volume").clip(lower_bound=1)).alias("net_flow"),
        ).drop("_signed_sum")
        frames.append(daily)

    if not frames:
        return pl.DataFrame()
    panel = pl.concat(frames, how="diagonal_relaxed").sort("series", "date")
    # Was ``... if False else panel``, so the documented clearance never
    # applied and nodes appeared on their own resolution day (days_to_close had
    # a 5th percentile of 0). ``clearance_days=0`` is now the way to ask for
    # that, rather than a literal that silently disabled the parameter.
    if clearance_days:
        panel = panel.filter(pl.col("days_to_close") >= clearance_days)

    if cadence == "event":
        from stg.panel.snapshots import macro_resolution_dates
        snaps = macro_resolution_dates(mk, names)
        panel = panel.filter(pl.col("date").is_in(snaps))
    elif cadence == "weekly":
        panel = panel.filter(pl.col("date").dt.weekday() == 1)
    keep = ["series", "date", "event_ticker", "days_to_close", "is_bucket",
            *_STAT_COLS, "implied_median", "d_implied_mean", "recent_volume",
            "net_flow", "max_stale_days", "n_fresh_legs", "n_submarkets",
            "resolved_value"]
    return panel.select([c for c in keep if c in panel.columns])
