"""Is the in-season speed-up a seasonal cycle, or markets getting faster over calendar time?

b1_stability found winner contracts absorbing events faster late in each season. With one season of
full data per league that cannot separate a seasonal pattern from a calendar trend, so this uses
every game with a winner contract, including the winner-only ones (NBA 2024-25 playoffs, MLB 2025
regular season), and tracks by month:

    speed      mean move by +10 s / mean move by +60 s after events with |dWP| >= MIN_DWP (winner
               contract, signed the event's way, from REF s before)
    activity   winner-contract trades per in-play minute (both teams' contracts), and contracts traded
               per in-play minute

If speed drops back when a new season starts, it is seasonal; if it keeps rising across the season
break, it is a calendar trend, and if it moves with activity, more trading is the likely cause.
Within each month, speed is also shown for low- and high-activity games (split at that month's
median), to see whether activity explains the speed differences at a given time.

    venv/bin/python -W ignore analysis/graph_2026_10/b1_speed_trend.py NBA > analysis/graph_2026_10/out/b1_speed_trend_NBA.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import b1_response as R                                    # noqa: E402  (reads the league from argv)
from b1_stability import game_meta                         # noqa: E402
from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.loaders import scan_trades                 # noqa: E402
from stg.sports.prices import quotes_at, scan_quotes       # noqa: E402

LEAGUE = R.LEAGUE
MIN_DWP = 0.05


def main() -> None:
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    d = SportsData()
    meta = game_meta().filter(pl.col("type") != "spring training")
    w = (d.contracts.filter(pl.col("league") == LEAGUE, pl.col("kind") == "winner", pl.col("side").is_in(["home", "away"]),
                            pl.col("game_id").is_in(meta["game_id"].implode()))
         .select("ticker", "series_ticker", "side", "game_id"))
    res = d.results.filter(pl.col("league") == LEAGUE).select("game_id", "first_play", "feed_end")
    ev = R.events().filter(pl.col("dwp").abs() >= MIN_DWP, pl.col("game_id").is_in(w["game_id"].implode())).with_row_index("eid")
    x = ev.join(w, on="game_id").with_columns(
        (pl.when(pl.col("side") == "home").then(pl.col("dwp")).otherwise(-pl.col("dwp"))).sign().alias("sgn"))
    at = pl.concat([x.select("eid", "ticker", (pl.col("t") + pl.duration(seconds=o)).alias("t"), pl.lit(o).alias("o"))
                    for o in (-R.REF, 10, 60)])
    q = quotes_at(scan_quotes(w["series_ticker"].unique().to_list()), at.select("ticker", "t").unique(), max_spread=100.0)
    q = q.with_columns(R.value(q).alias("mid")).select("ticker", "t", "mid").unique(["ticker", "t"])
    m = at.join(q, on=["ticker", "t"], how="left").pivot(on="o", index=["eid", "ticker"], values="mid", aggregate_function="first")
    x = (x.join(m, on=["eid", "ticker"]).with_columns(
        (pl.col("sgn") * (pl.col("10") - pl.col(str(-R.REF)))).alias("m10"), (pl.col("sgn") * (pl.col("60") - pl.col(str(-R.REF)))).alias("m60"))
         .drop_nulls(["m10", "m60"]))

    # activity: winner trades and contracts per in-play minute, per game
    tr = (scan_trades(w["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(w["ticker"].implode()))
          .select("ticker", pl.col("created_time").cast(pl.Datetime("us", "UTC")).alias("t"), pl.col("count").cast(pl.Float64))
          .collect().join(w.select("ticker", "game_id"), on="ticker").join(res, on="game_id")
          .filter(pl.col("t") >= pl.col("first_play"), pl.col("t") <= pl.col("feed_end")))
    act = (tr.group_by("game_id").agg(pl.len().alias("trades"), pl.col("count").sum().alias("contracts"),
                                      ((pl.col("feed_end").first() - pl.col("first_play").first()).dt.total_seconds() / 60).alias("mins"))
           .with_columns((pl.col("trades") / pl.col("mins")).alias("trades_per_min"),
                         (pl.col("contracts") / pl.col("mins")).alias("contracts_per_min")))
    g = meta.join(act, on="game_id").with_columns(pl.col("start_utc").dt.strftime("%Y-%m").alias("month"))
    g = g.with_columns((pl.col("trades_per_min") > pl.col("trades_per_min").median().over("month")).alias("busy"))
    x = x.join(g.select("game_id", "month", "season", "type", "busy", "trades_per_min"), on="game_id")

    print(f"{LEAGUE}: {x['game_id'].n_unique()} games, {x.height:,} (event, winner contract) rows with |dWP| >= {MIN_DWP}")
    by_month = (x.group_by("month").agg(pl.col("season").first(), pl.col("type").mode().first().alias("main_type"),
                                        pl.col("game_id").n_unique().alias("games"), pl.len().alias("rows"),
                                        (pl.col("m10").mean() / pl.col("m60").mean()).round(2).alias("speed_10s_over_60s"),
                                        (pl.col("m10").filter(~pl.col("busy")).mean() / pl.col("m60").filter(~pl.col("busy")).mean())
                                        .round(2).alias("speed_quiet_games"),
                                        (pl.col("m10").filter(pl.col("busy")).mean() / pl.col("m60").filter(pl.col("busy")).mean())
                                        .round(2).alias("speed_busy_games"))
                .join(g.group_by("month").agg(pl.col("trades_per_min").median().round(1).alias("trades_per_min_med"),
                                              pl.col("contracts_per_min").median().round(0).alias("contracts_per_min_med")),
                      on="month").sort("month"))
    print("by month (speed = share of the +60 s move already there at +10 s; activity = medians over games)")
    print(by_month)
    xs = x.with_columns(pl.col("trades_per_min").qcut(5, labels=["q1 (quietest)", "q2", "q3", "q4", "q5 (busiest)"])
                        .cast(pl.String).alias("activity_quintile"))
    print("\nby activity quintile, all months pooled")
    print(xs.group_by("activity_quintile").agg(pl.col("game_id").n_unique().alias("games"),
                                               pl.col("trades_per_min").median().round(1).alias("trades_per_min"),
                                               (pl.col("m10").mean() / pl.col("m60").mean()).round(2).alias("speed_10s_over_60s"))
          .sort("activity_quintile"))


if __name__ == "__main__":
    main()
