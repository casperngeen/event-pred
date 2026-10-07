"""Step 2 of the remaining preliminaries: which contract moves first, and by how much does each lag?

A. Spread contracts vs the winner contract. For each spread trade during play (test games of
   b1_baseline), the structural baseline's price for that contract computed with the winner price
   from k seconds BEFORE the trade (score and clock always as of the trade). Compared in changes,
   so the baseline's level errors cancel: between consecutive trades of the same contract (at most
   GAP_S apart), the traded price change vs the baseline's change. The k with the highest R^2 says
   how stale spread prices are relative to the winner. A trade is shifted half a cent towards the
   mid (a YES taker paid the ask, a NO taker sold at the bid). Shown for all pairs and for pairs
   where the winner moved >= 3c between the two trades.
B. Total contracts vs scoring. The same in changes, for consecutive total trades spanning a score:
   the baseline with the score as of k seconds before each trade; the best k says how long totals
   take to absorb a score.
C. Team futures vs the game. On a 1-min grid in play, each futures contract's change over a minute
   regressed on the team's winner-price change over that minute and each of the 15 before
   (distributed lag). The cumulative share of the summed coefficients by lag says how many minutes
   futures take to follow the game. MLB uses 2025 games here (its futures data stops in 2025).

    venv/bin/python -W ignore analysis/graph_2026_10/b1_leadlag.py NFL > analysis/graph_2026_10/out/b1_leadlag_NFL.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import b1_baseline as B                                    # noqa: E402  (reads the league from argv)
from b2_futures_ingame import candles_1m, mids_at          # noqa: E402
from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.loaders import load_games, load_pbp, scan_trades   # noqa: E402

LEAGUE = B.LEAGUE
S = {"NFL": (13.0, 14.5), "NBA": (14.5, 19.5), "MLB": (4.0, 4.25)}[LEAGUE]   # fitted in b1_baseline (train games)
KS = [0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60]
MAX_TRADES = 300_000
GAP_S = 120
LAGS_MIN = 15
SEED = 0


def test_games(d: SportsData, con: pl.DataFrame) -> pl.DataFrame:
    """The same games and train/test split as b1_baseline."""
    with_ladders = con.filter(pl.col("kind") == "spread")["game_id"].unique().implode()
    g = (d.results.filter(pl.col("league") == LEAGUE).select("game_id", "first_play", "feed_end")
         .filter(pl.col("game_id").is_in(with_ladders)))
    g = g.sample(n=min(B.CFG["n_games"], g.height), seed=B.SEED).sort("first_play")
    return g.with_columns((pl.int_range(pl.len()) < int(g.height * B.TRAIN_SHARE)).alias("train")).filter(~pl.col("train"))


def state_at(pbp: pl.DataFrame, at: pl.DataFrame, tcol: str = "t") -> pl.DataFrame:
    return (at.sort(tcol).join_asof(pbp.rename({"t": "_ts"}), left_on=tcol, right_on="_ts", by="game_id",
                                    strategy="backward", check_sortedness=False)
            .with_columns(B.tau_left().alias("tau"), (pl.col("home_score") + pl.col("away_score")).alias("total_now"),
                          (pl.col("home_score") - pl.col("away_score")).alias("margin")).drop("_ts"))


def home_mid(con: pl.DataFrame, at: pl.DataFrame) -> pl.DataFrame:
    """at: (game_id, t) -> p_home, the home winner mid at t."""
    w = con.filter(pl.col("kind") == "winner", pl.col("side").is_in(["home", "away"]))
    a = at.select("game_id", "t").unique().join(w.select("game_id", "ticker", "side"), on="game_id")
    m = a.join(B.mids(con, a), on=["ticker", "t"], how="left").with_columns(
        pl.when(pl.col("side") == "home").then(pl.col("mid")).otherwise(100 - pl.col("mid")).alias("ph"))
    return m.group_by("game_id", "t").agg(pl.col("ph").mean().alias("p_home"))


def trades_in_play(con: pl.DataFrame, games: pl.DataFrame, kind: str) -> pl.DataFrame:
    c = con.filter(pl.col("kind") == kind, pl.col("game_id").is_in(games["game_id"].implode()))
    tr = (scan_trades(c["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(c["ticker"].implode()))
          .select("ticker", pl.col("created_time").cast(pl.Datetime("us", "UTC")).alias("t"),
                  pl.col("yes_price").cast(pl.Float64).alias("price"), "taker_side").collect())
    tr = (tr.join(c.select("ticker", "game_id", "kind", "side", "strike"), on="ticker")
          .join(games.select("game_id", "first_play", "feed_end"), on="game_id")
          .filter(pl.col("t") > pl.col("first_play"), pl.col("t") < pl.col("feed_end"))
          .with_columns((pl.col("price") + pl.when(pl.col("taker_side") == "yes").then(-0.5).otherwise(0.5)).alias("p_adj")))
    return tr.sample(n=min(MAX_TRADES, tr.height), seed=SEED)


def pair_r2(tr: pl.DataFrame, bases: dict[int, np.ndarray], flag: str | None = None) -> list[dict]:
    """R^2 of consecutive-trade price changes (same contract, <= GAP_S apart) on the baseline's
    changes, for each lag k. ``flag``: a column; keep only pairs where it changed between the trades
    by >= its threshold (handled by the caller as a boolean pair column ``keep``)."""
    df = tr.select("ticker", "t", "p_adj", *([flag] if flag else [])).with_columns(
        [pl.Series(f"b{k}", v) for k, v in bases.items()]).sort("ticker", "t")
    prev = lambda c: pl.col(c).shift(1).over("ticker")      # noqa: E731
    df = df.with_columns((pl.col("t") - prev("t")).dt.total_seconds().alias("gap"), (pl.col("p_adj") - prev("p_adj")).alias("dp"),
                         *[(pl.col(f"b{k}") - prev(f"b{k}")).alias(f"db{k}") for k in bases],
                         *([(pl.col(flag) - prev(flag)).abs().alias("dflag")] if flag else []))
    df = df.filter(pl.col("gap") <= GAP_S)
    if flag:
        df = df.filter(pl.col("dflag") >= (3 if flag == "w0" else 1))
    df = df.drop_nans().drop_nulls()
    out = []
    for k in bases:
        dp, db = df["dp"].to_numpy(), df[f"db{k}"].to_numpy()
        out.append(dict(k=k, pairs=len(dp), r2=round(1 - float(np.mean((dp - db) ** 2) / np.mean(dp ** 2)), 4),
                        corr=round(float(np.corrcoef(dp, db)[0, 1]), 4), slope=round(float(np.polyfit(db, dp, 1)[0]), 3)))
    return out


def lag_table(rows: list[dict]) -> pl.DataFrame:
    t = pl.DataFrame(rows)
    return t.with_columns(pl.col("r2").rank("min", descending=True).over("set").cast(pl.Int32).alias("rank")).sort("set", "k")


def part_a(con, games, pbp, L) -> None:
    tr = trades_in_play(con, games, "spread").filter(pl.col("side").is_in(["home", "away"]))
    tr = state_at(pbp, tr).join(L, on="game_id", how="left").drop_nulls("tau").with_row_index("rid")
    at = pl.concat([tr.select("rid", "game_id", (pl.col("t") - pl.duration(seconds=k)).alias("tk"), pl.lit(k).alias("k"))
                    for k in KS])
    hm = home_mid(con, at.rename({"tk": "t"}))
    at = at.join(hm.rename({"t": "tk"}), on=["game_id", "tk"], how="left").pivot(on="k", index="rid", values="p_home",
                                                                               aggregate_function="first")
    tr = tr.join(at, on="rid").drop_nulls([str(k) for k in KS]).with_columns(pl.col("0").alias("w0"))
    bases = {k: B.baseline(tr.with_columns(pl.col(str(k)).alias("p_home")), S[0], S[1]) for k in KS}
    rows = [dict(set="all pairs", **r) for r in pair_r2(tr, bases)]
    rows += [dict(set="winner moved >= 3c between the trades", **r) for r in pair_r2(tr, bases, flag="w0")]
    print(f"\n######## A. spread contracts: consecutive-trade price changes vs the baseline's changes using the "
          f"winner price k seconds earlier ({tr.height:,} trades)")
    print("    r2 against no change; rank 1 = the lag that fits best")
    print(lag_table(rows))


def part_b(con, games, pbp, L) -> None:
    tr = trades_in_play(con, games, "total").join(L, on="game_id", how="left").drop_nulls("L")
    tr = state_at(pbp, tr).drop_nulls("tau").with_row_index("rid").sort("rid")
    bases = {}
    for k in KS:
        sk = state_at(pbp, tr.select("rid", "game_id", (pl.col("t") - pl.duration(seconds=k)).alias("t")))
        x = (tr.drop("total_now").join(sk.select("rid", "total_now"), on="rid", how="left").sort("rid")
             .with_columns(pl.lit(50.0).alias("p_home")))
        bases[k] = B.baseline(x, S[0], S[1])
    rows = [dict(set="all pairs", **r) for r in pair_r2(tr, bases)]
    rows += [dict(set="a score between the trades", **r) for r in pair_r2(tr, bases, flag="total_now")]
    print(f"\n######## B. total contracts: consecutive-trade changes vs the baseline's changes using the score k "
          f"seconds earlier ({tr.height:,} trades)")
    print(lag_table(rows))


def part_c(d: SportsData, con, games) -> None:
    if LEAGUE == "MLB":
        res = d.results.filter(pl.col("league") == "MLB", pl.col("first_play").dt.year() == 2025)
        games = res.select("game_id", "first_play", "feed_end").sample(n=min(600, res.height), seed=SEED)
        con = d.contracts.filter(pl.col("league") == "MLB", pl.col("scope") == "game", pl.col("kind") == "winner",
                                 pl.col("game_id").is_in(games["game_id"].implode())).select(
            "ticker", "series_ticker", "kind", "side", "strike", "game_id")
    fut = d.contracts.filter(pl.col("league") == LEAGUE, pl.col("scope") == "season",
                             pl.col("kind").is_in(["champ", "conference", "division", "playoffs"]),
                             pl.col("team").is_not_null()).select("ticker", "series_ticker", "kind", "team")
    cd = candles_1m(fut["series_ticker"].unique().to_list())
    gt = load_games().filter(pl.col("league") == LEAGUE).select("game_id", "home", "away").unique("game_id")
    grid = (games.with_columns(pl.datetime_ranges(pl.col("first_play").dt.truncate("1m") - pl.duration(minutes=LAGS_MIN),
                                                  pl.col("feed_end").dt.truncate("1m"), "1m").alias("t"))
            .explode("t").select("game_id", "t"))
    hm = home_mid(con, grid)
    rows = []
    for side in ("home", "away"):
        g = grid.join(gt, on="game_id").select("game_id", "t", pl.col(side).alias("team")).join(fut, on="team")
        fm = mids_at(cd, g.select("ticker", "t", "game_id", "kind")).select("ticker", "t", "game_id", "kind", "mid")
        x = (fm.join(hm, on=["game_id", "t"], how="left")
             .with_columns((pl.col("p_home") if side == "home" else 100 - pl.col("p_home")).alias("p_team"))
             .sort("ticker", "game_id", "t"))
        x = x.with_columns((pl.col("mid") - pl.col("mid").shift(1)).over("ticker", "game_id").alias("dfut"),
                           (pl.col("p_team") - pl.col("p_team").shift(1)).over("ticker", "game_id").alias("dp0"))
        x = x.with_columns([pl.col("dp0").shift(j).over("ticker", "game_id").alias(f"dp{j}") for j in range(1, LAGS_MIN + 1)])
        rows.append(x)
    x = pl.concat(rows).drop_nulls(["dfut"] + [f"dp{j}" for j in range(LAGS_MIN + 1)])
    out = []
    for (kind,), g in x.partition_by("kind", as_dict=True).items():
        A = np.c_[[g[f"dp{j}"].to_numpy() for j in range(LAGS_MIN + 1)]].T
        coef = np.linalg.lstsq(np.c_[A, np.ones(len(A))], g["dfut"].to_numpy(), rcond=None)[0][:-1]
        cum = np.cumsum(coef) / coef.sum()
        out.append(dict(kind=kind, minutes=g.height, total_cents_per_point=round(float(coef.sum()), 4),
                        **{f"by_{j}m": round(float(cum[j]), 2) for j in (0, 1, 2, 5, 10, 15)}))
    print(f"\n######## C. team futures vs the game: share of the futures' response to a winner-price change "
          f"that has happened by j minutes after it (distributed lag, {LAGS_MIN} min)")
    print(pl.DataFrame(out).sort("kind") if out else "no futures minutes")


def main() -> None:
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    d = SportsData()
    con = (d.contracts.filter(pl.col("league") == LEAGUE, pl.col("scope") == "game",
                              pl.col("kind").is_in(["winner", "spread", "total"]), pl.col("game_id").is_not_null())
           .select("ticker", "series_ticker", "kind", "side", "strike", "game_id"))
    games = test_games(d, con)
    extra = ["half", "outs"] if LEAGUE == "MLB" else ["clock"]
    pbp = (load_pbp(LEAGUE).filter(~pl.col("t_outside_window"), pl.col("game_id").is_in(games["game_id"].implode()))
           .select("game_id", pl.col(B.CFG["state_t"]).alias("t"), "period", *extra, "home_score", "away_score")
           .drop_nulls("t").sort("t"))
    L = B.pregame_total(con, games)
    print(f"{LEAGUE}: {games.height} test games of b1_baseline; S_M = {S[0]}, S_T = {S[1]}")
    part_a(con, games, pbp, L)
    part_b(con, games, pbp, L)
    part_c(d, con, games)


if __name__ == "__main__":
    main()
