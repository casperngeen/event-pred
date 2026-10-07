"""Quick check: can a taker catch the futures move after a big in-game play? Priced only from trade prints.

Events: plays with |dWP| >= BIG_WP (win-prob change of the home team), for each season-long contract
(champion, conference, division, playoffs) of the two teams. The play is known at its end (NFL
nflverse play end, NBA possession time, MLB plate-appearance end) + DELAY_S (feed delay); direction =
+1 if the play helps the contract's team (buy YES), else -1 (buy NO).

    pre      last print at or before the play
    entry    the first print on our side after the play is known (YES-taker if buying YES, NO-taker if
             selling), within ENTRY_MAX_S: the price a taker paid then
    exits    volume-weighted price of all prints in [+5, +10] min, [+30, +40] min after the play is
             known, and [end + 30, end + 90] min (after the game)
    gross    direction x (exit - entry), cents per contract
    net      gross - taker fee at entry - HALF_GAP - taker fee at exit (FEE_RATE x p (1 - p); the
             futures fee multiplier is unverified)

Baseline: the same pairs at random in-game moments at least 5 min from a big play, random direction.
It loses about the half gap (enter at one side, exit near the middle); the play effect is the
difference. t clustered by game.

    venv/bin/python -W ignore analysis/graph_2026_10/b6_futures_taker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.loaders import load_games, load_pbp, scan_trades   # noqa: E402
from b5_backtest import FEE_RATE, clustered_t             # noqa: E402

KINDS = ["champ", "conference", "division", "playoffs"]
LEAGUES = ["NFL", "NBA", "MLB"]
BIG_WP, DELAY_S, ENTRY_MAX_S, HALF_GAP = 0.08, 10, 600, 1.0
EXITS = {"5m": (300, 600), "30m": (1800, 2400)}
POST = (1800, 5400)
N_PLACEBO, SEED = 3, 0                  # placebo moments per big play


def fee(c: str) -> pl.Expr:
    q = (pl.col(c) / 100).clip(0, 1)
    return FEE_RATE * q * (1 - q)


def plays(lg: str) -> pl.DataFrame:
    ev = load_pbp(lg).filter(~pl.col("t_outside_window"), pl.col("home_wp_post").is_not_null()).sort("game_id", "seq")
    t_known = pl.max_horizontal("t_event", "known_at") if lg == "MLB" else pl.col("t_event")
    ev = ev.with_columns((pl.col("home_wp_post") - pl.col("home_wp_post").shift(1).over("game_id")).alias("dwp"),
                         t_known.alias("t0"))
    return ev.filter(pl.col("dwp").abs() >= BIG_WP).select("game_id", "t0", "dwp")


def main() -> None:
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    d = SportsData()
    fut = d.contracts.filter(pl.col("scope") == "season", pl.col("kind").is_in(KINDS), pl.col("team").is_not_null(),
                             pl.col("league").is_in(LEAGUES)).select("ticker", "series_ticker", "league", "kind", "team")
    games = load_games().filter(pl.col("league").is_in(LEAGUES)).select("league", "game_id", "home", "away").unique("game_id")
    res = d.results.select("game_id", "first_play", "feed_end").join(games, on="game_id")
    tg = pl.concat([res.select("league", "game_id", "first_play", "feed_end", pl.col(s).alias("team"), pl.lit(s).alias("side"))
                    for s in ("home", "away")]).join(fut, on=["league", "team"])
    tr = (scan_trades(fut["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(tg["ticker"].unique().implode()))
          .select("ticker", "known_at", "yes_price", "taker_side", "count").collect().sort("ticker", "known_at"))
    span = tr.group_by("ticker").agg(pl.col("known_at").min().alias("f"), pl.col("known_at").max().alias("l"))
    tg = tg.join(span, on="ticker").filter(pl.col("first_play") >= pl.col("f"), pl.col("feed_end") <= pl.col("l"))
    print(f"futures prints: {tr.height:,} on {tr['ticker'].n_unique()} contracts; (contract, team game) pairs: {tg.height:,}")
    cum = tr.with_columns(pl.col("count").cum_sum().over("ticker").alias("cv"),
                          (pl.col("count") * pl.col("yes_price")).cum_sum().over("ticker").alias("cn"))
    yes = tr.filter(pl.col("taker_side") == "yes").select("ticker", pl.col("known_at").alias("ty"), pl.col("yes_price").alias("py"))
    no = tr.filter(pl.col("taker_side") == "no").select("ticker", pl.col("known_at").alias("tn"), pl.col("yes_price").alias("pn"))
    rng = np.random.default_rng(SEED)
    rows = []
    for lg in LEAGUES:
        big = plays(lg)
        pairs = tg.filter(pl.col("league") == lg)
        ev = big.join(pairs, on="game_id").with_columns(
            pl.when(pl.col("side") == "home").then(pl.col("dwp")).otherwise(-pl.col("dwp")).sign().cast(pl.Int64).alias("dir"),
            pl.lit("big play").alias("set"))
        n = ev.height * N_PLACEBO
        pc = ev.sample(n=n, with_replacement=True, seed=SEED).with_columns(
            (pl.col("first_play") + pl.duration(seconds=pl.Series(rng.uniform(0, 1, n))
                                                * (pl.col("feed_end") - pl.col("first_play")).dt.total_seconds())).alias("t0"),
            pl.Series("dir", rng.choice([-1, 1], n)), pl.lit("random").alias("set"))
        near = (pc.select("game_id", "t0").unique().join(big.select("game_id", pl.col("t0").alias("tb")), on="game_id")
                .filter((pl.col("t0") - pl.col("tb")).dt.total_seconds().abs() < 300).select("game_id", "t0").unique())
        pc = pc.join(near, on=["game_id", "t0"], how="anti")
        rows.append(pl.concat([ev, pc], how="diagonal_relaxed").select("league", "kind", "game_id", "ticker", "t0", "dir", "set",
                                                                        "feed_end"))
    r = pl.concat(rows).with_row_index("i").with_columns((pl.col("t0") + pl.duration(seconds=DELAY_S)).alias("tk"))
    r = r.sort("t0").join_asof(tr.select("ticker", "known_at", pl.col("yes_price").alias("pre")).sort("known_at"),
                               left_on="t0", right_on="known_at", by="ticker", strategy="backward").drop("known_at")
    r = (r.sort("tk").join_asof(yes.sort("ty"), left_on="tk", right_on="ty", by="ticker", strategy="forward")
         .join_asof(no.sort("tn"), left_on="tk", right_on="tn", by="ticker", strategy="forward"))
    r = r.with_columns(pl.when(pl.col("dir") == 1).then(pl.col("py")).otherwise(pl.col("pn")).alias("entry"),
                       pl.when(pl.col("dir") == 1).then(pl.col("ty")).otherwise(pl.col("tn")).alias("t_entry"))
    r = r.filter((pl.col("t_entry") - pl.col("tk")).dt.total_seconds() <= ENTRY_MAX_S)
    c = cum.select("ticker", "known_at", "cv", "cn").sort("known_at")
    windows = {k: (pl.col("tk") + pl.duration(seconds=a), pl.col("tk") + pl.duration(seconds=b)) for k, (a, b) in EXITS.items()}
    windows["after game"] = (pl.col("feed_end") + pl.duration(seconds=POST[0]), pl.col("feed_end") + pl.duration(seconds=POST[1]))
    for k, (a, b) in windows.items():
        for tag, e in (("a", a), ("b", b)):
            r = (r.with_columns(e.alias("_t")).sort("_t")
                 .join_asof(c, left_on="_t", right_on="known_at", by="ticker", strategy="backward")
                 .rename({"cv": f"cv_{tag}", "cn": f"cn_{tag}"}).drop("known_at", "_t"))
        r = r.with_columns(((pl.col("cn_b").fill_null(0) - pl.col("cn_a").fill_null(0))
                            / (pl.col("cv_b").fill_null(0) - pl.col("cv_a").fill_null(0))).alias(f"x_{k}")).drop("cv_a", "cn_a", "cv_b", "cn_b")
    r = r.with_columns((pl.col("dir") * (pl.col("entry") - pl.col("pre"))).alias("moved_before_entry"),
                       (pl.col("t_entry") - pl.col("tk")).dt.total_seconds().alias("entry_wait_s"))
    out = []
    for k in windows:
        x = r.filter(pl.col(f"x_{k}").is_finite()).with_columns((pl.col("dir") * (pl.col(f"x_{k}") - pl.col("entry"))).alias("g"))
        x = x.with_columns((pl.col("g") - fee("entry") - HALF_GAP - fee(f"x_{k}")).alias("net"))
        for (lg, kind, st), gg in x.group_by("league", "kind", "set"):
            if gg.height < 30:
                continue
            gm = gg["game_id"].to_numpy()
            out.append(dict(league=lg, kind=kind, set=st, exit=k, n=gg.height, games=gg["game_id"].n_unique(),
                            entry_med=gg["entry"].median(), wait_med_s=gg["entry_wait_s"].median(),
                            moved_before=round(gg["moved_before_entry"].mean(), 2),
                            gross=round(gg["g"].mean(), 2), t_gross=round(clustered_t(gg["g"].to_numpy(), gm), 1),
                            net=round(gg["net"].mean(), 2), t_net=round(clustered_t(gg["net"].to_numpy(), gm), 1)))
    o = pl.DataFrame(out).sort("league", "kind", "exit", "set")
    print("cents per contract; entry = first print on our side after the play is known (+10 s); moved_before = how far the "
          "price had already moved our way from the last pre-play print")
    print(o)
    o.write_parquet(Path(__file__).resolve().parent / "out" / "b6_futures_taker.parquet")


if __name__ == "__main__":
    main()
