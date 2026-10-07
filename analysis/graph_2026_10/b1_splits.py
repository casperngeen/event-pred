"""Step 4 of the remaining preliminaries: what data the in-game model has, and how to split it.

Per league:
1. Usable games: a result and play-by-play, a winner contract, before the holdout (markets closing
   on or after 2026-07-01 are never read). By game type and season, with how many also have spread
   / total ladders and both teams' futures at 1-min resolution; big events (|dWP| >= 0.08) per game.
2. Walk-forward folds by date, over the games the in-game graph can use: those with both spread and
   total ladders (spring training excluded). Games with only a winner contract (NBA 2024-25
   playoffs, MLB 2025 regular season) are listed but left out of the folds. Games sorted by first play; the first TRAIN_SHARE are training only
   (fold 0); the rest are cut into N_FOLDS contiguous test folds, with cuts moved to a day boundary
   so that no night is split between folds. Fold k trains on every game before it (expanding).
   Written to out/b1_folds_<league>.parquet (game_id, first_play, fold) for the training table.

    venv/bin/python -W ignore analysis/graph_2026_10/b1_splits.py NFL > analysis/graph_2026_10/out/b1_splits_NFL.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import b1_response as R                                    # noqa: E402  (reads the league from argv)
from b1_stability import game_meta                         # noqa: E402
from stg.sports.asof import SportsData                     # noqa: E402
from stg.splits import SPORTS_HOLDOUT_START                # noqa: E402

LEAGUE = R.LEAGUE
TRAIN_SHARE, N_FOLDS = 0.4, 5
OUT = Path(__file__).resolve().parent / "out"


def main() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    d = SportsData()
    con = d.contracts.filter(pl.col("league") == LEAGUE, pl.col("scope") == "game", pl.col("game_id").is_not_null())
    has = lambda kind: con.filter(pl.col("kind") == kind)["game_id"].unique().implode()   # noqa: E731
    res = d.results.filter(pl.col("league") == LEAGUE).select("game_id", "first_play", "known_at")
    assert res["known_at"].max() < SPORTS_HOLDOUT_START, "a result inside the holdout"
    g = (res.filter(pl.col("game_id").is_in(has("winner"))).join(game_meta(), on="game_id")
         .with_columns(pl.col("game_id").is_in(has("spread")).alias("spreads"),
                       pl.col("game_id").is_in(has("total")).alias("totals")))
    # both teams' futures with 1-min candles over the game
    fut = d.contracts.filter(pl.col("league") == LEAGUE, pl.col("scope") == "season",
                             pl.col("kind").is_in(["champ", "conference", "division", "playoffs"]), pl.col("team").is_not_null())
    have = {p.name for p in Path("data/sports/candles_futures_1m").iterdir()} if Path("data/sports/candles_futures_1m").exists() else set()
    fut_ok = fut.filter(pl.col("series_ticker").is_in(list(have)))
    ev = R.events()
    big = ev.filter(pl.col("dwp").abs() >= 0.08).group_by("game_id").len().rename({"len": "big_events"})
    g = g.join(big, on="game_id", how="left").with_columns(pl.col("big_events").fill_null(0))
    print(f"######## 1. {LEAGUE}: usable games (result + play-by-play + winner contract, before "
          f"{SPORTS_HOLDOUT_START.date()})")
    print(g.group_by("season", "type").agg(pl.len().alias("games"), pl.col("first_play").min().dt.date().alias("from"),
                                           pl.col("first_play").max().dt.date().alias("to"),
                                           pl.col("spreads").sum().alias("with_spreads"), pl.col("totals").sum().alias("with_totals"),
                                           pl.col("big_events").mean().round(1).alias("big_events_per_game"))
          .sort("season", "type"))
    print(f"futures with 1-min candles: {fut_ok['series_ticker'].n_unique()} series, {fut_ok.height} contracts "
          f"(see b2_futures_ingame for the per-game coverage)")

    out_of_graph = g.filter(~(pl.col("spreads") & pl.col("totals")) | (pl.col("type") == "spring training"))
    print(f"left out of the folds: {out_of_graph.height} games (no spread / total ladders, or spring training):",
          dict(out_of_graph.group_by("season", "type").len().sort("season", "type").select(
              pl.format("{} {}", "season", "type"), "len").iter_rows()))
    g = g.filter(pl.col("spreads"), pl.col("totals"), pl.col("type") != "spring training")
    g = g.sort("first_play").with_columns(pl.col("first_play").dt.convert_time_zone("America/New_York").dt.date().alias("day"))
    n = g.height
    cuts = [int(n * TRAIN_SHARE)] + [int(n * (TRAIN_SHARE + (1 - TRAIN_SHARE) * k / N_FOLDS)) for k in range(1, N_FOLDS)]
    days = g["day"].to_list()
    bounds = []
    for c in cuts:                       # move each cut forward to the first game of a new day
        while 0 < c < n and days[c] == days[c - 1]:
            c += 1
        bounds.append(c)
    fold = [0] * n
    for k, b in enumerate(bounds, start=1):
        for i in range(b, n):
            fold[i] = k
    g = g.with_columns(pl.Series("fold", fold))
    print(f"\n######## 2. walk-forward folds by date (fold 0 = training only; fold k trains on folds < k)")
    tab = (g.group_by("fold").agg(pl.len().alias("games"), pl.col("day").min().alias("from"), pl.col("day").max().alias("to"),
                                  pl.col("spreads").sum().alias("with_spreads"), pl.col("totals").sum().alias("with_totals"),
                                  (pl.col("type") != "regular").sum().alias("non_regular"),
                                  pl.col("big_events").sum().alias("big_events"))
           .sort("fold").with_columns(pl.col("games").cum_sum().shift(1).fill_null(0).alias("train_games_before")))
    print(tab)
    g.select("game_id", "first_play", "day", "fold", "type", "season", "spreads", "totals").write_parquet(OUT / f"b1_folds_{LEAGUE}.parquet")
    print(f"wrote out/b1_folds_{LEAGUE}.parquet")


if __name__ == "__main__":
    main()
