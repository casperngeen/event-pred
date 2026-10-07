"""Can the 1-min futures candles be used together with the in-game data? A feasibility check.

Season-long contracts (champion, conference, division, playoffs) of the two teams in each game,
during that game (first play to the feed's end):

1. Coverage: contracts with 1-min candles, by league and kind.
2. Price availability in play: share of in-game minutes in which the contract has a candle (Kalshi
   writes a 1-min candle only in minutes with a trade or a quote change), and in which it has a
   usable price (two-sided, gap <= MAX_GAP, latest candle <= MAX_AGE old: no candle = no change).
3. Response to big plays (|dWP| >= BIG_WP, timed at the play's end): change in each team's futures
   mid from 1 min before the play to 1 / 5 / 15 min after, signed to the team (a play that helps
   the team should raise its futures). Slope on the team's dWP (cents per WP point), the share of
   changes with the right sign among those that moved, and the median |change| against the same
   contracts' changes over in-game minutes with no big play (placebo).
4. Tracking during the game. Each (contract, team game) sampled every STEP minutes from the first
   play to the end, plus once AFTER the end. Pooled slopes, cents per WP point:
       in-game   futures change over a step  on  the team's WP change over the same step
                 (and over the previous step: a catch-up term)
       result    futures change from the first play to AFTER the end  on  (result - WP at the
                 first play)
   If futures price the game as it unfolds, in-game ~ result (ratio ~1); if they wait for the
   final, in-game << result.

    venv/bin/python -W ignore analysis/graph_2026_10/b2_futures_ingame.py > analysis/graph_2026_10/out/b2_futures_ingame.txt
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.loaders import load_games, load_pbp, safe_path   # noqa: E402

KINDS = ["champ", "conference", "division", "playoffs"]
LEAGUES = ["NFL", "NBA", "MLB"]
MAX_GAP, MAX_AGE = 10.0, dt.timedelta(days=1)   # a candle is written only when the quote changes or a trade prints
BIG_WP = 0.08
HORIZONS = {"1m": 1, "5m": 5, "15m": 15}
N_PLACEBO, SEED = 4000, 0
STEP, AFTER = 15, 30                 # minutes


def candles_1m(series: list[str]) -> pl.DataFrame:
    files = [str(f) for s in series for f in safe_path("candles_futures_1m", s).glob("*.parquet")]
    if not files:
        return pl.DataFrame(schema={"ticker": pl.String, "ts": pl.Datetime("us", "UTC"), "bid": pl.Float64,
                                    "ask": pl.Float64, "volume": pl.Float64})
    return (pl.scan_parquet(files, missing_columns="insert", extra_columns="ignore")
            .select("ticker", pl.col("ts").cast(pl.Datetime("us", "UTC")),
                    pl.col("yes_bid_close").fill_null(0.0).alias("bid"),
                    pl.col("yes_ask_close").fill_null(100.0).alias("ask"), pl.col("volume").cast(pl.Float64))
            .unique(["ticker", "ts"]).collect().sort("ticker", "ts"))


def mids_at(cd: pl.DataFrame, at: pl.DataFrame) -> pl.DataFrame:
    """at: (ticker, t) -> latest candle at or before t: bid, ask, age, usable mid."""
    j = at.sort("t").join_asof(cd.select("ticker", "ts", "bid", "ask").sort("ts"), left_on="t", right_on="ts",
                                by="ticker", strategy="backward", check_sortedness=False)
    ok = ((pl.col("bid") > 0) & (pl.col("ask") < 100) & (pl.col("ask") >= pl.col("bid"))
          & (pl.col("ask") - pl.col("bid") <= MAX_GAP) & ((pl.col("t") - pl.col("ts")) <= MAX_AGE))
    return j.with_columns(pl.when(ok).then((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))


def main() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_cols(20)
    d = SportsData()
    fut = d.contracts.filter(pl.col("scope") == "season", pl.col("kind").is_in(KINDS), pl.col("team").is_not_null(),
                             pl.col("league").is_in(LEAGUES)).select("ticker", "series_ticker", "league", "kind", "team")
    cd = candles_1m(fut["series_ticker"].unique().to_list())
    have = cd.group_by("ticker").agg(pl.len().alias("candles"), (pl.col("volume") > 0).mean().alias("traded"),
                                     pl.col("ts").min().alias("first"), pl.col("ts").max().alias("last"))
    cov = fut.join(have, on="ticker", how="left")
    print("######## 1. coverage: season-long contracts with 1-min candles")
    print(cov.group_by("league", "kind").agg(pl.len().alias("contracts"), pl.col("candles").is_not_null().sum().alias("with_1m"),
                                            pl.col("candles").median().alias("candles_med"),
                                            pl.col("traded").mean().round(3).alias("share_minutes_traded"),
                                            pl.col("first").min().dt.date().alias("from"), pl.col("last").max().dt.date().alias("to"))
          .sort("league", "kind"))

    games = load_games().filter(pl.col("league").is_in(LEAGUES)).select("league", "game_id", "home", "away").unique("game_id")
    res = d.results.select("game_id", "first_play", "feed_end").join(games, on="game_id")
    team_games = pl.concat([res.select("league", "game_id", "first_play", "feed_end", pl.col(s).alias("team"),
                                       pl.lit(s).alias("side")) for s in ("home", "away")])
    tg = team_games.join(fut, on=["league", "team"])
    # restrict to games inside each contract's 1-min span
    tg = tg.join(have.select("ticker", "first", "last"), on="ticker").filter(
        pl.col("first_play") >= pl.col("first"), pl.col("feed_end") <= pl.col("last"))
    grid = (tg.with_columns(pl.datetime_ranges(pl.col("first_play").dt.truncate("1m") + pl.duration(minutes=1),
                                               pl.col("feed_end").dt.truncate("1m"), "1m").alias("t"))
            .explode("t").select("league", "kind", "ticker", "game_id", "t"))
    g = mids_at(cd, grid).with_columns((pl.col("ts") > pl.col("t") - pl.duration(minutes=1)).alias("candle_this_min"))
    print(f"\n######## 2. in play: {tg.height:,} (contract, team game) pairs, {g.height:,} contract-minutes")
    print(g.group_by("league", "kind").agg(pl.col("game_id").n_unique().alias("games"),
                                          pl.col("candle_this_min").mean().round(3).alias("minutes_with_candle"),
                                          pl.col("mid").is_not_null().mean().round(3).alias("minutes_usable_price"),
                                          (pl.col("ask") - pl.col("bid")).filter(pl.col("mid").is_not_null()).median()
                                          .alias("gap_med_cents"))
          .sort("league", "kind"))

    print(f"\n######## 3. response to big plays (|dWP| >= {BIG_WP}, at the play's end), signed to the team")
    rng = np.random.default_rng(SEED)
    for lg in ("NFL", "NBA"):
        ev = (load_pbp(lg).filter(~pl.col("t_outside_window")).sort("game_id", "seq")
              .with_columns(pl.col("home_wp_post").forward_fill().shift(1).over("game_id").alias("wp_pre"))
              .with_columns((pl.col("home_wp_post") - pl.col("wp_pre")).alias("dwp")))
        big = ev.filter(pl.col("dwp").abs() >= BIG_WP).select("game_id", pl.col("t_event").alias("t0"), "dwp")
        pairs = tg.filter(pl.col("league") == lg).select("game_id", "ticker", "kind", "side", "first_play", "feed_end")
        rows = big.join(pairs, on="game_id").with_columns(
            pl.when(pl.col("side") == "home").then(pl.col("dwp")).otherwise(-pl.col("dwp")).alias("dwp_team"),
            pl.lit("big").alias("set"))
        # placebo: random in-game minutes of the same pairs, at least 5 min from any big play
        pl_rows = pairs.sample(n=min(N_PLACEBO, pairs.height), with_replacement=True, seed=SEED).with_columns(
            (pl.col("first_play") + pl.duration(seconds=pl.Series(rng.uniform(0, 1, min(N_PLACEBO, pairs.height)))
                                                * (pl.col("feed_end") - pl.col("first_play")).dt.total_seconds()))
            .alias("t0"))
        near = pl_rows.join(big.select("game_id", pl.col("t0").alias("tb")), on="game_id", how="left").group_by(
            "game_id", "ticker", "t0").agg(((pl.col("t0") - pl.col("tb")).dt.total_seconds().abs().min()).alias("dist"))
        pl_rows = (pl_rows.join(near, on=["game_id", "ticker", "t0"], how="left")
                   .filter(pl.col("dist").is_null() | (pl.col("dist") > 300))
                   .with_columns(pl.lit(0.0).alias("dwp_team"), pl.lit(0.0).alias("dwp"), pl.lit("placebo").alias("set")))
        allr = pl.concat([rows.select("set", "kind", "ticker", "t0", "dwp_team"),
                          pl_rows.select("set", "kind", "ticker", "t0", "dwp_team")]).with_row_index("rid")
        at = pl.concat([allr.select("rid", "ticker", (pl.col("t0") - pl.duration(minutes=1)).alias("t"), pl.lit("pre").alias("k"))]
                       + [allr.select("rid", "ticker", (pl.col("t0") + pl.duration(minutes=m)).alias("t"), pl.lit(h).alias("k"))
                          for h, m in HORIZONS.items()])
        m = mids_at(cd, at.select("rid", "ticker", "t", "k")).pivot(on="k", index="rid", values="mid")
        r = allr.join(m, on="rid")
        out = []
        for h in HORIZONS:
            x = r.with_columns((pl.col(h) - pl.col("pre")).alias("chg")).drop_nulls("chg")
            for (kind,), gg in x.partition_by("kind", as_dict=True).items():
                b, p = gg.filter(pl.col("set") == "big"), gg.filter(pl.col("set") == "placebo")
                if b.height < 20:
                    continue
                xx, yy = b["dwp_team"].to_numpy() * 100, b["chg"].to_numpy()
                moved = b.filter(pl.col("chg") != 0)
                out.append(dict(kind=kind, horizon=h, n_big=b.height, slope=round(float(np.polyfit(xx, yy, 1)[0]), 3),
                                corr=round(float(np.corrcoef(xx, yy)[0, 1]), 3),
                                share_moved=round(moved.height / b.height, 3),
                                right_sign=round(float((np.sign(moved["chg"]) == np.sign(moved["dwp_team"])).mean()), 3)
                                if moved.height else None,
                                absmed_big=b["chg"].abs().median(), absmed_placebo=p["chg"].abs().median(),
                                n_placebo=p.height))
        print(f"--- {lg}: slope = cents of futures move per WP point; share_moved = changes != 0; right_sign among moved")
        print(pl.DataFrame(out).sort("kind", "horizon").select("kind", "horizon", "n_big", "slope", "corr", "share_moved",
                                                               "right_sign", "n_placebo"))
        # placebo: share of changes != 0 in quiet minutes (to compare with share_moved)
        for h in HORIZONS:
            x = r.filter(pl.col("set") == "placebo").with_columns((pl.col(h) - pl.col("pre")).alias("chg")).drop_nulls("chg")
            print(f"    placebo {h}: share of quiet-minute changes != 0, by kind:",
                  dict(x.group_by("kind").agg((pl.col("chg") != 0).mean().round(3)).sort("kind").iter_rows()))

    print(f"\n######## 4. tracking: {STEP}-min steps in play; result = futures change first play -> end + {AFTER} min")
    out = []
    for lg in LEAGUES:
        ev = (load_pbp(lg).filter(~pl.col("t_outside_window"), pl.col("home_wp_post").is_not_null())
              .select("game_id", pl.col("t_event").alias("t"), "home_wp_post").sort("t"))
        won = d.results.select("game_id", (pl.col("home_score") > pl.col("away_score")).cast(pl.Float64).alias("home_won"),
                               (pl.col("home_score") == pl.col("away_score")).alias("tie"))
        pairs = (tg.filter(pl.col("league") == lg).join(won, on="game_id").filter(~pl.col("tie"))
                 .with_row_index("pid"))
        ck = (pairs.with_columns(pl.datetime_ranges(pl.col("first_play"), pl.col("feed_end"), f"{STEP}m").alias("t"))
              .explode("t").with_columns(pl.int_range(pl.len()).over("pid").alias("k"))
              .select("pid", "game_id", "ticker", "side", "kind", "home_won", "k", "t"))
        wp = ck.sort("t").join_asof(ev, on="t", by="game_id", strategy="backward", check_sortedness=False)
        # before the feed's first WP (NFL / MLB start without one): the game's first WP
        wp = (wp.join_asof(ev.rename({"home_wp_post": "wp_next"}), on="t", by="game_id", strategy="forward",
                           check_sortedness=False)
              .with_columns(pl.coalesce("home_wp_post", "wp_next").alias("home_wp_post")).drop("wp_next"))
        mid = mids_at(cd, ck.select("pid", "k", "ticker", "t"))
        x = (wp.join(mid.select("pid", "k", "mid"), on=["pid", "k"])
             .with_columns(pl.when(pl.col("side") == "home").then(pl.col("home_wp_post"))
                           .otherwise(1 - pl.col("home_wp_post")).alias("wp_team"))
             .sort("pid", "k")
             .with_columns((pl.col("mid") - pl.col("mid").shift(1).over("pid")).alias("dfut"),
                           ((pl.col("wp_team") - pl.col("wp_team").shift(1).over("pid")) * 100).alias("dwp"))
             .with_columns(pl.col("dwp").shift(1).over("pid").alias("dwp_prev")))
        endm = mids_at(cd, pairs.select("pid", "ticker", (pl.col("feed_end") + pl.duration(minutes=AFTER)).alias("t")))
        first = x.filter(pl.col("k") == 0).select("pid", pl.col("mid").alias("mid0"), pl.col("wp_team").alias("wp0"))
        rr = (pairs.join(first, on="pid").join(endm.select("pid", pl.col("mid").alias("mid_end")), on="pid")
              .with_columns(pl.when(pl.col("side") == "home").then(pl.col("home_won")).otherwise(1 - pl.col("home_won"))
                            .alias("won_team"))
              .with_columns((pl.col("mid_end") - pl.col("mid0")).alias("dfut"),
                            ((pl.col("won_team") - pl.col("wp0")) * 100).alias("surprise")).drop_nulls(["dfut", "surprise"]))
        print(f"    {lg}: pairs {pairs.height}, steps with futures + WP changes "
              f"{x.drop_nulls(['dfut', 'dwp', 'dwp_prev']).height}, pairs with a result row {rr.height}, "
              f"WP null at checkpoints {x['wp_team'].is_null().mean():.3f}, mid null {x['mid'].is_null().mean():.3f}")
        for (kind,), gx in x.drop_nulls(["dfut", "dwp", "dwp_prev"]).partition_by("kind", as_dict=True).items():
            gr = rr.filter(pl.col("kind") == kind)
            if gx.height < 50 or gr.height < 20:
                continue
            A = np.c_[gx["dwp"].to_numpy(), gx["dwp_prev"].to_numpy(), np.ones(gx.height)]
            b_in, b_prev, _ = np.linalg.lstsq(A, gx["dfut"].to_numpy(), rcond=None)[0]
            b_res = float(np.polyfit(gr["surprise"].to_numpy(), gr["dfut"].to_numpy(), 1)[0])
            out.append(dict(league=lg, kind=kind, steps=gx.height, games=gr.height, in_game=round(float(b_in), 4),
                            catch_up=round(float(b_prev), 4), result=round(b_res, 4),
                            ratio=round(float(b_in) / b_res, 2) if b_res else None,
                            ratio_with_catch_up=round(float(b_in + b_prev) / b_res, 2) if b_res else None))
    print(pl.DataFrame(out).sort("league", "kind"))


if __name__ == "__main__":
    main()
