"""Chunk 1 diagnostics: node values (stg.sports.graph.node_values) on a daily grid.

    1. coverage: share of live nodes with a value, by kind and month
    2. wins fit: legs used, fitted sigma, and how well the normal fits the ladder (cents)
    3. end of season: the last implied expected wins before the season ends, against the
       interval the ladder's settlement implies (validation only: settlement is a final field)
    4. examples: a few nodes' daily values

    venv/bin/python -W ignore analysis/graph_2026_10/node_values_check.py > analysis/graph_2026_10/out/node_values_check.txt
    venv/bin/python -W ignore analysis/graph_2026_10/node_values_check.py --strict > analysis/graph_2026_10/out/node_values_check_strict.txt
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.asof import SportsData                                          # noqa: E402
from stg.sports.graph import _threshold, between_game_graph, node_values        # noqa: E402

RUNS = [("MLB", 2025), ("NBA", 2025), ("NFL", 2025)]
SEASON_END = {"MLB": dt.datetime(2025, 9, 28, tzinfo=dt.timezone.utc),
              "NBA": dt.datetime(2026, 4, 12, tzinfo=dt.timezone.utc),
              "NFL": dt.datetime(2026, 1, 4, tzinfo=dt.timezone.utc)}
OUT = Path(__file__).resolve().parent / "out"


def grid(g, league: str) -> pl.DataFrame:
    """Every live node at 16:00 UTC each day (before most games start)."""
    lo = g.nodes["known_at"].min().replace(hour=16, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
    hi = min(SEASON_END[league] + dt.timedelta(days=1), dt.datetime(2026, 6, 30, tzinfo=dt.timezone.utc))
    days = pl.datetime_range(lo, hi, "1d", time_zone="UTC", eager=True).alias("t")
    at = g.nodes.select("node", "known_at", "valid_to").join(pl.DataFrame(days), how="cross")
    return at.filter(pl.col("known_at") <= pl.col("t"),
                     pl.col("valid_to").is_null() | (pl.col("valid_to") > pl.col("t"))).select("node", "t")


def null_reasons(d, g, at: pl.DataFrame) -> pl.DataFrame:
    """Why legs have no valid mid: classify each (leg, t) by the raw book known at t."""
    legs = (at.join(g.nodes.select("node", "kind", "tickers"), on="node").explode("tickers")
            .rename({"tickers": "ticker"}))
    q = d.quotes_at(legs.select("ticker", "t").unique(), max_spread=100.0, max_age=dt.timedelta(days=3650))
    q = q.select("ticker", "t", "bid", "ask", "bid_at", "ask_at").unique(["ticker", "t"])
    x = legs.join(q, on=["ticker", "t"], how="left")
    age = pl.max_horizontal(pl.col("t") - pl.col("bid_at"), pl.col("t") - pl.col("ask_at"))
    reason = (pl.when(pl.col("bid").is_null() & pl.col("ask").is_null()).then(pl.lit("no quote yet"))
              .when((pl.col("bid").fill_null(0) <= 0) & (pl.col("ask").fill_null(100) <= 3)).then(pl.lit("one-sided, ask <= 3c"))
              .when((pl.col("ask").fill_null(100) >= 100) & (pl.col("bid").fill_null(0) >= 97)).then(pl.lit("one-sided, bid >= 97c"))
              .when((pl.col("bid").fill_null(0) <= 0) | (pl.col("ask").fill_null(100) >= 100)).then(pl.lit("one-sided, other"))
              .when(pl.col("ask") - pl.col("bid") > 10).then(pl.lit("spread > 10c"))
              .when(age > dt.timedelta(days=1)).then(pl.lit("stale > 1 day"))
              .otherwise(pl.lit("valid")))
    return (x.with_columns(reason.alias("reason"))
            .group_by("kind", "reason").len()
            .with_columns((pl.col("len") / pl.col("len").sum().over("kind")).round(3).alias("share"))
            .pivot(on="kind", index="reason", values="share").sort("reason"))


def ladder_fit_error(d, g, v: pl.DataFrame) -> pl.DataFrame:
    """RMSE (cents) between each wins ladder's leg mids and the fitted normal, per (node, t)."""
    w = v.filter(pl.col("kind") == "wins", pl.col("value").is_not_null())
    legs = (w.join(g.nodes.select("node", "tickers"), on="node").explode("tickers").rename({"tickers": "ticker"})
            .join(d.contracts.select("ticker", "strike", "strike_type"), on="ticker"))
    q = d.quotes_at(legs.select("ticker", "t").unique(), max_spread=10.0, max_age=dt.timedelta(days=1))
    legs = (legs.join(q.select("ticker", "t", "mid").unique(["ticker", "t"]), on=["ticker", "t"])
            .filter(pl.col("mid").is_not_null())
            .with_columns(_threshold(pl.col("strike"), pl.col("strike_type")).alias("k")))
    fit = norm.cdf((legs["value"].to_numpy() + 0.5 - legs["k"].to_numpy()) / legs["sigma"].to_numpy()) * 100
    legs = legs.with_columns(pl.Series("fit", fit))
    return legs.group_by("node", "t").agg(((pl.col("mid") - pl.col("fit")) ** 2).mean().sqrt().alias("rmse_c"))


def settled_interval(d, g) -> pl.DataFrame:
    """[lo, hi] for final wins implied by how each ladder settled (YES: W >= k)."""
    legs = (g.nodes.filter(pl.col("kind") == "wins").select("node", "team", "tickers").explode("tickers")
            .rename({"tickers": "ticker"})
            .join(d.contracts.select("ticker", "strike", "strike_type"), on="ticker")
            .join(d.finals.select("ticker", "result"), on="ticker")
            .join(d.contracts.select("ticker", "event_ticker", "open_time"), on="ticker")
            .filter(pl.col("open_time") == pl.col("open_time").max().over("node"))
            .with_columns(_threshold(pl.col("strike"), pl.col("strike_type")).alias("k")))
    return legs.group_by("node", "team").agg(
        pl.col("k").filter(pl.col("result") == "yes").max().alias("lo"),
        (pl.col("k").filter(pl.col("result") == "no").min() - 1).alias("hi"))


EXTREMES = "--strict" not in sys.argv     # node_values values one-sided extremes by default


def main() -> None:
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    d = SportsData()
    edges = pl.read_parquet("data/sports/derived/edges.parquet")
    OUT.mkdir(exist_ok=True)
    allv = []
    for lg, season in RUNS:
        g = between_game_graph(d, edges, lg, season)
        v = node_values(d, g, grid(g, lg), extremes=EXTREMES).with_columns(pl.lit(lg).alias("league"))
        allv.append(v)
        print(f"################ {lg} {season}: {g.nodes.height} nodes, {v.height:,} node-days"
              f"  (one-sided extremes valued: {EXTREMES})")

        print("\n1. coverage: share of live nodes with a value, by month")
        cov = (v.with_columns(pl.col("t").dt.strftime("%Y-%m").alias("month"))
               .group_by("month", "kind").agg(pl.col("value").is_not_null().mean().round(2).alias("c"))
               .pivot(on="kind", index="month", values="c").sort("month"))
        print(cov)
        print("\n1b. why legs have no value (share of leg-days, by kind)")
        print(null_reasons(d, g, grid(g, lg)))

        w = v.filter(pl.col("kind") == "wins")
        if w.height:
            err = ladder_fit_error(d, g, v)
            w = w.join(err, on=["node", "t"], how="left")
            print("\n2. wins fit: legs used, sigma (wins), RMSE of the normal vs the leg mids (cents)")
            print(w.filter(pl.col("value").is_not_null()).select(
                pl.len().alias("fits"),
                (pl.col("legs") == 1).sum().alias("node_days_with_1_leg_only"),
                pl.col("legs").median().alias("legs_med"),
                pl.col("sigma").quantile(0.1).round(2).alias("sigma_p10"),
                pl.col("sigma").median().round(2).alias("sigma_med"),
                pl.col("sigma").quantile(0.9).round(2).alias("sigma_p90"),
                pl.col("rmse_c").median().round(2).alias("rmse_med_c"),
                pl.col("rmse_c").quantile(0.9).round(2).alias("rmse_p90_c")))

            print("\n3. end of season: last implied expected wins vs the settled interval")
            last = (w.filter(pl.col("value").is_not_null(), pl.col("t") <= SEASON_END[lg])
                    .sort("t").group_by("node").agg(pl.col("t").last(), pl.col("value").last().alias("mu"),
                                                    pl.col("sigma").last()))
            iv = settled_interval(d, g)
            chk = iv.join(last, on="node", how="left").with_columns(
                ((pl.col("mu") >= pl.col("lo").fill_null(-1) - 1)
                 & (pl.col("mu") <= pl.col("hi").fill_null(999) + 1)).alias("inside_pm1"))
            print(chk.select(pl.len().alias("teams"), pl.col("mu").is_not_null().sum().alias("with_mu"),
                             pl.col("inside_pm1").sum().alias("mu_inside_interval_pm1"),
                             pl.col("t").min().alias("earliest_last_t")))
            print(chk.sort("team").select("team", "t", pl.col("mu").round(1), pl.col("sigma").round(2), "lo", "hi",
                                          "inside_pm1").head(8))

        print("\n4. examples (weekly)")
        ex = v.filter(pl.col("node").str.starts_with(sorted(g.nodes["team"].unique().to_list())[0] + "-"),
                      pl.col("t").dt.weekday() == 1)
        print(ex.with_columns(pl.col("value").round(2)).pivot(on="kind", index="t", values="value").sort("t")
              .head(40))
        print()
    pl.concat(allv, how="diagonal_relaxed").write_parquet(
        OUT / f"node_values_daily{'' if EXTREMES else '_strict'}.parquet")


if __name__ == "__main__":
    main()
