"""Maker markouts on game-winner markets: does providing liquidity pay, and when is it picked off?

Every trade has a taker and a resting (maker) order on the other side. A taker buying YES at p means
the maker sold YES at p; the maker's markout at horizon tau is, in cents per contract,

    sold YES:    p - mid(t + tau)          bought YES:  mid(t + tau) - p

and held to settlement, mid is replaced by the result (100 / 0). Positive = the maker kept the
spread; its decay over tau is adverse selection (the taker knew something). Averages are weighted
by contracts. Net subtracts an assumed maker fee, MAKER_RATE x multiplier x p (1 - p) (multiplier
0.5 for KXMLBGAME, 1 otherwise, from the series API); the base rate is unverified, so net is
indicative only.

Sample: N_GAMES random regular-season games per league (2025 seasons), every winner-leg trade.
Phases: pre-game (before the first play), in-game (to the result), post (after the result).
Caveat: this is the average maker's fill; one's own fills depend on queue position.

    venv/bin/python -W ignore analysis/graph_2026_10/markouts.py > analysis/graph_2026_10/out/markouts.txt
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.features import REGULAR                    # noqa: E402
from stg.sports.loaders import scan_trades                 # noqa: E402

RUNS = [("MLB", 2025, "KXMLBGAME", 0.5), ("NBA", 2025, "KXNBAGAME", 1.0), ("NFL", 2025, "KXNFLGAME", 1.0)]
N_GAMES, SEED = 250, 0
TAUS = {"1m": dt.timedelta(minutes=1), "5m": dt.timedelta(minutes=5), "30m": dt.timedelta(minutes=30),
        "2h": dt.timedelta(hours=2)}
MAKER_RATE = 1.75          # cents x p (1 - p); unverified base rate (see reports/sports_data_sources.md)
OUT = Path(__file__).resolve().parent / "out"


def league_trades(d: SportsData, league: str, season: int, series: str, mult: float) -> pl.DataFrame:
    legs = (d.contracts.filter(pl.col("league") == league, pl.col("season") == season, pl.col("kind") == "winner",
                               pl.col("series_ticker") == series, pl.col("game_type").is_in(REGULAR[league]),
                               pl.col("game_id").is_not_null())
            .select("ticker", "game_id"))
    games = legs["game_id"].unique().sample(n=min(N_GAMES, legs["game_id"].n_unique()), seed=SEED)
    legs = legs.filter(pl.col("game_id").is_in(games.implode()))
    res = d.results.select("game_id", "first_play", pl.col("known_at").alias("result_at"))
    fin = d.finals.select("ticker", "result")
    tr = (scan_trades(series).filter(pl.col("ticker").is_in(legs["ticker"].implode())).collect()
          .join(legs, on="ticker").join(res, on="game_id", how="inner").join(fin, on="ticker", how="left")
          .filter(pl.col("result").is_in(["yes", "no"])))
    tr = tr.with_columns(
        pl.when(pl.col("created_time") < pl.col("first_play")).then(pl.lit("pre-game"))
        .when(pl.col("created_time") < pl.col("result_at")).then(pl.lit("in-game")).otherwise(pl.lit("post")).alias("phase"),
        pl.when(pl.col("taker_side") == "yes").then(-1.0).otherwise(1.0).alias("maker_side"),   # +1: maker bought YES
        ((pl.col("first_play") - pl.col("created_time")).dt.total_minutes() / 60).alias("hours_to_start"))
    at = pl.concat([tr.select("ticker", (pl.col("created_time") + dt_).alias("t")) for dt_ in TAUS.values()]).unique()
    q = d.quotes_at(at).select("ticker", "t", "mid").unique(["ticker", "t"])
    for name, dt_ in TAUS.items():
        tr = (tr.with_columns((pl.col("created_time") + dt_).alias("_t"))
              .join(q.rename({"t": "_t", "mid": f"mid_{name}"}), on=["ticker", "_t"], how="left").drop("_t")
              .with_columns((pl.col("maker_side") * (pl.col(f"mid_{name}") - pl.col("yes_price"))).alias(f"mo_{name}")))
    p = pl.col("yes_price") / 100
    return tr.with_columns(
        (pl.col("maker_side") * (pl.when(pl.col("result") == "yes").then(100.0).otherwise(0.0) - pl.col("yes_price")))
        .alias("mo_settle"),
        (MAKER_RATE * mult * p * (1 - p)).alias("maker_fee"),
        pl.lit(league).alias("league"))


def summary(tr: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    w = pl.col("count")
    wavg = lambda c: ((pl.col(c) * w).filter(pl.col(c).is_not_null()).sum()  # noqa: E731
                      / w.filter(pl.col(c).is_not_null()).sum()).round(2).alias(c)
    cols = [f"mo_{k}" for k in TAUS] + ["mo_settle"]
    return (tr.group_by(by).agg(pl.len().alias("trades"), w.sum().round(0).alias("contracts"),
                                *[wavg(c) for c in cols], wavg("maker_fee"),
                                (((pl.col("mo_settle") - pl.col("maker_fee")) * w).sum() / w.sum()).round(2)
                                .alias("net_settle"))
            .sort(by))


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(230)
    d = SportsData()
    tr = pl.concat([league_trades(d, *r) for r in RUNS], how="diagonal_relaxed")
    OUT.mkdir(exist_ok=True)
    print(__doc__.split("\n\n")[1], "\n")
    print(f"sample: {N_GAMES} games per league; {tr.height:,} trades, {tr['count'].sum():,.0f} contracts\n")
    print("1. maker markout by league and phase (cents per contract, contract-weighted)")
    print(summary(tr, ["league", "phase"]))
    pre = tr.filter(pl.col("phase") == "pre-game")
    print("\n2. pre-game, by hours to the first play")
    print(summary(pre.with_columns(pl.col("hours_to_start").cut([0.25, 1, 3, 12, 24]).alias("h_to_start")),
                  ["league", "h_to_start"]))
    print("\n3. pre-game, by price (the taker's YES price)")
    print(summary(pre.with_columns(pl.col("yes_price").cut([10, 30, 50, 70, 90]).alias("price")), ["league", "price"]))
    print("\n4. pre-game, by trade size (contracts)")
    print(summary(pre.with_columns(pl.col("count").cut([10, 100, 1000]).alias("size")), ["league", "size"]))


if __name__ == "__main__":
    main()
