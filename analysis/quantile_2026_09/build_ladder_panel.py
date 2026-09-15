#!/usr/bin/env python
"""Daily reconstruction-free ladder statistics for every macro event.

    venv/bin/python analysis/quantile_2026_09/build_ladder_panel.py

One row per (series, event, date): the quantile moments from
``stg.events.quantile`` alongside the integrated moments already in
``node_panel_event``, so the two can be compared on identical ladders.

The point of the comparison is that the quantile statistics are read from the
interior of the traded ladder and the integrated ones are not. If they agree,
``relations_findings.md`` item 1's coverage defect does not matter for that
statistic; where they disagree, the disagreement localises the damage.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.events.implied import (
    THRESHOLD, classify_contract, normalise_to_exclusive, parse_threshold,
    parse_threshold_from_subtitle,
)
from stg.events.quantile import quantile_moments
from stg.io.kalshi import KalshiOHLCV
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import SPECS, series_filter_expr, trigger_universe
from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("analysis/quantile_2026_09/out")


def daily_ladders(canon: str, mk: pl.DataFrame, tr: pl.LazyFrame) -> pl.DataFrame:
    sub = (mk.filter(series_filter_expr(canon) & pl.col("close_time").is_not_null())
           .select("ticker", "event_ticker", "yes_sub_title", "close_time")
           .unique(subset=["ticker"]))
    if sub.is_empty():
        return pl.DataFrame()
    kinds, strikes, convs = [], [], []
    for t, s in zip(sub["ticker"], sub["yes_sub_title"]):
        c = classify_contract(t, s)
        kinds.append(c)
        strikes.append(parse_threshold(t, s) if c == THRESHOLD else None)
        convs.append(parse_threshold_from_subtitle(s)[1] if c == THRESHOLD else None)
    sub = (sub.with_columns(pl.Series("kind", kinds),
                            pl.Series("strike", strikes, dtype=pl.Float64),
                            pl.Series("conv", convs, dtype=pl.Utf8))
           .filter((pl.col("kind") == THRESHOLD) & pl.col("strike").is_not_null()))
    if sub.is_empty():
        return pl.DataFrame()
    trades = tr.filter(pl.col("ticker").is_in(sub["ticker"].unique().to_list())).collect()
    if trades.is_empty():
        return pl.DataFrame()
    daily = (KalshiOHLCV.build_daily(trades, mk.filter(series_filter_expr(canon)))
             .join(sub.select("ticker", "strike", "conv"), on="ticker", how="inner")
             .filter((pl.col("trade_count") > 0) & pl.col("close").is_between(0, 100)))
    if daily.is_empty():
        return pl.DataFrame()

    rows = []
    for (ev, date), g in daily.group_by(["event_ticker", "date"]):
        thr = g["strike"].to_numpy().astype(float)
        conv = g["conv"].to_list()
        if any(c == "inclusive" for c in conv):
            thr = normalise_to_exclusive(thr, conv)
        p = g["close"].to_numpy().astype(float) / 100.0
        qm = quantile_moments(thr, p)
        rows.append(dict(series=canon, event_ticker=ev, date=date,
                         close_time=g["close_time"].min(),
                         n_traded=int(g.height), **qm))
    return pl.DataFrame(rows)


def main() -> None:
    mk, tr = load_markets(is_only=True), scan_trades(is_only=True)
    canons = [c for c in trigger_universe(5, mk) if SPECS[c].kind == THRESHOLD]
    parts = []
    for c in canons:
        d = daily_ladders(c, mk, tr)
        print(f"  {c:<14} event-days: {d.height}")
        if d.height:
            parts.append(d)
    q = pl.concat(parts, how="diagonal")
    assert_no_oos(q, time_col="close_time")

    node = pl.read_parquet(PANELS / "node_panel_event.parquet").select(
        "event_ticker", "date", "implied_mean", "implied_std", "implied_median",
        "implied_skew", "days_to_close", "resolved_value", "n_fresh_legs")
    q = q.join(node, on=["event_ticker", "date"], how="left")

    OUT.mkdir(parents=True, exist_ok=True)
    q.write_parquet(OUT / "ladder_panel.parquet")
    print(f"\nrows {q.height}   events {q['event_ticker'].n_unique()}   "
          f"series {q['series'].n_unique()}")

    print("\n=== how often is each statistic defined? ===")
    print("(a quantile is undefined when the traded ladder does not bracket it)\n")
    rows = []
    for col in ("q10", "q25", "q50", "q75", "q90", "iqr", "skew_q"):
        rows.append(dict(stat=col, defined=int(q[col].is_not_null().sum()),
                         frac=float(q[col].is_not_null().mean())))
    with pl.Config(float_precision=3):
        print(pl.DataFrame(rows))

    print("\n=== quantile vs integrated moments on identical ladders ===")
    j = q.drop_nulls(["q50", "implied_mean", "iqr", "implied_std"])
    j = j.with_columns([
        (pl.col("q50") - pl.col("implied_mean")).alias("loc_gap"),
        (pl.col("sigma_iqr") / pl.col("implied_std")).alias("width_ratio"),
    ])
    with pl.Config(tbl_rows=25, float_precision=3, tbl_width_chars=200):
        print(j.group_by("series").agg(
            pl.len().alias("n"),
            pl.col("loc_gap").median().alias("loc_gap_p50"),
            pl.col("width_ratio").median().alias("width_ratio_p50"),
            pl.col("width_ratio").quantile(0.1).alias("wr_p10"),
            pl.col("width_ratio").quantile(0.9).alias("wr_p90"),
        ).filter(pl.col("n") >= 20).sort("width_ratio_p50"))
    print("\nwidth_ratio = sigma_iqr / implied_std. Below 1 means the INTEGRATED")
    print("moment is wider than the ladder's own interquartile range implies --")
    print("which is what open-tail mass at spacing/2 would do.")
    print(f"\nwrote {OUT / 'ladder_panel.parquet'}")


if __name__ == "__main__":
    main()
