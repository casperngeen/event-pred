"""Step 3 of the remaining preliminaries: do the in-game patterns hold across seasons and game states?

Two quantities, split by game type (regular / playoffs / play-in / spring training), season, third
of the season (by date within the season), and the game's state:

1. Spread contracts vs the structural baseline (b1_baseline's saved minutes, test games only, so the
   fitted S_M / S_T never saw them): mean abs error, bias (market - baseline), and the share of the
   next minute's market change explained by the baseline's change (R^2 against no change), with the
   slope. State split: how close the game is (|margin|).
2. Winner contracts after events (b1_response's events): size of the move by +60 s per WP point of
   the event, speed (mean move by +10 s / by +60 s), and the part left after +60 s (to +600 s, t
   clustered by game). State split: how close the game was before the event (home win prob).

    venv/bin/python -W ignore analysis/graph_2026_10/b1_stability.py NFL > analysis/graph_2026_10/out/b1_stability_NFL.txt
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import b1_response as R                                    # noqa: E402  (reads the league from argv)
from b1_baseline import baseline                           # noqa: E402
from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.loaders import load_games                  # noqa: E402
from stg.sports.prices import quotes_at, scan_quotes       # noqa: E402

LEAGUE = R.LEAGUE
S = {"NFL": (13.0, 14.5), "NBA": (14.5, 19.5), "MLB": (4.0, 4.25)}[LEAGUE]
CLOSE = {"NFL": [3, 8, 14], "NBA": [5, 10, 20], "MLB": [1, 2, 4]}[LEAGUE]
OUT = Path(__file__).resolve().parent / "out"
TYPE_NAME = {"2": "regular", "3": "playoffs", "5": "play-in", "R": "regular", "S": "spring training",
             "F": "playoffs", "D": "playoffs", "L": "playoffs", "W": "playoffs"}


def game_meta() -> pl.DataFrame:
    g = load_games().filter(pl.col("league") == LEAGUE).select("game_id", "game_type", "start_utc").unique("game_id")
    season = (pl.when(pl.lit(LEAGUE) == "NBA").then(pl.when(pl.col("start_utc") < dt.datetime(2025, 8, 1, tzinfo=dt.timezone.utc))
                                                  .then(pl.lit("2024-25")).otherwise(pl.lit("2025-26")))
              .when(pl.lit(LEAGUE) == "NFL").then(pl.lit("2025"))
              .otherwise(pl.col("start_utc").dt.year().cast(pl.String)))
    g = g.with_columns(pl.col("game_type").replace_strict(TYPE_NAME, default="other").alias("type"), season.alias("season"))
    third = (pl.col("start_utc").rank("ordinal").over("season") * 3 / pl.len().over("season")).ceil().cast(pl.Int32)
    return g.with_columns(third.replace_strict({1: "early", 2: "middle", 3: "late"}, default="late").alias("third"))


def bucket(col: str, edges: list[float], unit: str) -> pl.Expr:
    lab = [f"<={edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]}"]
    return pl.col(col).cut(edges, labels=lab).cast(pl.String).alias(f"{col}_{unit}")


def spread_fit(meta: pl.DataFrame) -> None:
    x = pl.read_parquet(OUT / f"b1_baseline_{LEAGUE}.parquet").filter(~pl.col("train"), pl.col("kind") == "spread")
    x1 = x.drop_nulls(["mid_1", "p_home_1", "tau_1"]).with_columns(
        pl.col("p_home_1").alias("p_home"), pl.col("tau_1").alias("tau"), pl.col("total_now_1").alias("total_now"))
    x = x.join(x1.select("ticker", "t").with_columns(pl.Series("base_1", baseline(x1, *S))), on=["ticker", "t"], how="left")
    x = x.join(meta, on="game_id").with_columns(pl.col("margin").abs().alias("abs_margin"))
    x = x.with_columns(bucket("abs_margin", CLOSE, "runs" if LEAGUE == "MLB" else "pts"))
    print(f"######## 1. spread contracts vs the baseline (test games of b1_baseline; S_M = {S[0]})")
    for by in ("type", "season", "third", x.columns[-1]):
        rows = []
        for (key,), g in x.partition_by(by, as_dict=True).items():
            c = g.drop_nulls(["mid_1", "base_1"])
            dm, db = (c["mid_1"] - c["mid"]).to_numpy(), (c["base_1"] - c["base"]).to_numpy()
            rows.append({by: key, "games": g["game_id"].n_unique(), "minutes": g.height,
                         "mae": round(float((g["mid"] - g["base"]).abs().mean()), 2),
                         "bias": round(float((g["mid"] - g["base"]).mean()), 2),
                         "r2_next_min": round(1 - float(np.mean((dm - db) ** 2) / np.mean(dm ** 2)), 3) if len(dm) > 100 else None,
                         "slope": round(float(np.polyfit(db, dm, 1)[0]), 3) if len(dm) > 100 else None})
        print(f"--- by {by}")
        print(pl.DataFrame(rows).sort(by))


def winner_response(meta: pl.DataFrame) -> None:
    x, con, _ = R.event_contract_rows(SportsData())
    x = x.filter(pl.col("kind") == "winner")
    offs = [-R.REF, 10, 60, 600]
    at = pl.concat([x.select("eid", "ticker", (pl.col("t") + pl.duration(seconds=o)).alias("t"), pl.lit(o).alias("o"))
                    for o in offs])
    q = quotes_at(scan_quotes(con["series_ticker"].unique().to_list()), at.select("ticker", "t").unique(), max_spread=100.0)
    q = q.with_columns(R.value(q).alias("mid")).select("ticker", "t", "mid").unique(["ticker", "t"])
    w = at.join(q, on=["ticker", "t"], how="left").pivot(on="o", index=["eid", "ticker"], values="mid", aggregate_function="first")
    x = x.join(w, on=["eid", "ticker"]).with_columns(
        [(pl.col("sgn") * (pl.col(str(o)) - pl.col(str(-R.REF)))).alias(f"m{o}") for o in (10, 60, 600)])
    x = x.join(meta, on="game_id").with_columns(((pl.col("wp") - pl.col("dwp")) - 0.5).abs().alias("closeness"))
    x = x.with_columns(pl.col("closeness").cut([0.1, 0.25, 0.4], labels=["<=0.1", "0.1-0.25", "0.25-0.4", ">0.4"])
                       .cast(pl.String).alias("pre_wp_distance_from_50"))
    print("\n######## 2. winner contracts after events (|dWP| >= 0.03 or scoring)")
    for by in ("type", "season", "third", "pre_wp_distance_from_50"):
        rows = []
        for (key,), g in x.partition_by(by, as_dict=True).items():
            c = g.drop_nulls(["m10", "m60"])
            cc = g.drop_nulls(["m60", "m600"])
            left = (cc["m600"] - cc["m60"]).to_numpy()
            per = c.filter(pl.col("dwp").abs() >= 0.05)
            rows.append({by: key, "games": g["game_id"].n_unique(), "events": c.height,
                         "cents_per_wp_point_60s": round(float((per["m60"] / (100 * per["dwp"].abs())).mean()), 3) if per.height > 30 else None,
                         "speed_10s_over_60s": round(float(c["m10"].mean() / c["m60"].mean()), 2) if c.height > 30 else None,
                         "left_after_60s": round(float(left.mean()), 2) if len(left) > 30 else None,
                         "t_left": round(R.clustered_t(left, cc["game_id"].to_numpy()), 1) if len(left) > 30 else None})
        print(f"--- by {by}")
        print(pl.DataFrame(rows).sort(by))


def main() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    meta = game_meta()
    spread_fit(meta)
    winner_response(meta)


if __name__ == "__main__":
    main()
