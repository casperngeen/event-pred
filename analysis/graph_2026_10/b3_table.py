"""Training table for the in-game graph model (track B), one league. Written to
data/sports/derived/ingame/<league>/:

    games.parquet       one row per game: fold (b1_folds), first play, feed end, home / away, the
                        pregame total L (for the baseline), the home team's ESPN id
    plays.parquet       one row per play in game order (the play-history memory reads this):
                        time (t_event, when the play's result is known: NFL play end, MLB plate-appearance
                        end, NBA possession end incl. free throws; known_at = feed time kept), play group, win-prob change dwp and level wp_post, margin / points
                        change, offense side, period, share of regulation left (tau), score after,
                        three league-specific situation fields s1-s3, stoppage and scoring flags.
                        The play unit: NFL a play (nflverse WP, ESPN when missing; s1-s3 = down,
                        distance, yard line); NBA a possession (b1_nba_events; ESPN WP; s1-s3 = 0);
                        MLB a plate appearance, timed at its last pitch's start (statsapi WP; s1-s3
                        = outs, balls, strikes before that pitch)
    steps.parquet       one row per game per STEP seconds of play (to the feed's end + the longest
                        horizon): the last play known (seq), plays in the step, game state (tau,
                        margin, total, offense side), the home winner mid, and recent activity
                        (winner trades in the previous ACT_S seconds)
    contract_steps/<game_id>.parquet
                        one row per contract per step, for contracts with any quote: bid, ask,
                        quote age, value (two-sided mid with gap <= 10c; a one-sided quote at an
                        extreme valued halfway to 0 / 100; for targets after the game's end, a
                        contract no longer quoted is worth 100 * y), live (two-sided, gap <= 10c, mid
                        3-97c, sides <= 5 min old), distance to the threshold; targets: bid / ask /
                        value at each horizon (+10 s, +30 s, +60 s, +120 s, +300 s, +600 s) (the same contract HORIZONS later on the grid),
                        drift = value(+600 s) - value(+60 s), and y = the contract's result (0 / 1)
                        trade prints (print_columns): last print and its age, signed flow and volume
                        over 60 s; per horizon the lowest / highest / volume-weighted traded price in
                        (t + h, t + h + 15 s] (plo_ / phi_ / pv_; the result after the end)
    contracts.parquet   static fields: kind, side, strike, game

Everything an input uses is known at the step (quotes as of t, plays with t_event <= t); targets are
later values and y. A live feed's delay is applied later, when batches are built (b4_export stores
play and step times; stg.models.ingame recomputes what is known for a given delay). The baseline is not stored: its inputs are, so it can be refitted per fold.

    venv/bin/python -W ignore analysis/graph_2026_10/b3_table.py NFL
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LEAGUE = sys.argv[1] if len(sys.argv) > 1 else "NFL"

import b1_baseline as B                                    # noqa: E402  (reads the league from argv)
import b1_response as R                                    # noqa: E402
from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.loaders import derived_dir, load_games, load_pbp, safe_path, scan_trades   # noqa: E402
from stg.sports.prices import QUOTE_SCHEMA, _trade_obs, quotes_at, scan_quotes   # noqa: E402
from stg.splits import SPORTS_HOLDOUT_START                # noqa: E402

STEP = 5                                   # seconds
HORIZONS = {"10s": 10, "30s": 30, "60s": 60, "120s": 120, "300s": 300, "600s": 600, "1800s": 1800}
ACT_S = 300
LIVE_GAP, LIVE_LO, LIVE_HI, LIVE_AGE_S = 10.0, 3.0, 97.0, 300
KINDS = ["winner", "spread", "total"]
FUT_KINDS = ["champ", "conference", "division", "playoffs"]   # each team's season-long contracts, as extra nodes
FUT_AGE_S = 86400       # a futures quote stays current up to a day (a 1-min candle is written only on a change)
PRINT_W_FUT_S = 120     # futures trade thinly: target window (t + h, t + h + 120 s]
OUT = derived_dir() / "ingame" / LEAGUE
FOLDS = Path(__file__).resolve().parent / "out" / f"b1_folds_{LEAGUE}.parquet"


def _finish_plays(ev: pl.DataFrame) -> pl.DataFrame:
    """Common columns for any league's play rows (needs game_id, seq, wp_post, scores, offense, tau)."""
    prev = lambda c: pl.col(c).shift(1).over("game_id")      # noqa: E731
    ev = ev.sort("game_id", "seq").with_columns(pl.col("wp_post").forward_fill().over("game_id"))
    ev = ev.with_columns(
        (pl.col("wp_post") - prev("wp_post")).alias("dwp"),
        ((pl.col("home_score") - pl.col("away_score")) - (prev("home_score") - prev("away_score"))).fill_null(0).alias("dmargin"),
        ((pl.col("home_score") + pl.col("away_score")) - (prev("home_score") + prev("away_score"))).fill_null(0).alias("dpoints"),
        (pl.col("home_score") - pl.col("away_score")).alias("margin"),
        (pl.col("home_score") + pl.col("away_score")).alias("total"))
    return ev.select("game_id", "seq", "t_event", "known_at", "play_group", "play_type", "dwp", "wp_post", "dmargin",
                     "dpoints", "offense", "period", "tau", "margin", "total", "s1", "s2", "s3", "is_stoppage", "is_scoring",
                     "home_id")


def plays_nfl(games: pl.DataFrame) -> pl.DataFrame:
    ev = (load_pbp("NFL").filter(~pl.col("t_outside_window"), pl.col("game_id").is_in(games["game_id"].implode()))
          .sort("game_id", "seq"))
    grp = pl.lit(None, pl.String)
    for name, pred in reversed(R.NFL_GROUPS):
        grp = pl.when(pred.fill_null(False)).then(pl.lit(name)).otherwise(grp)
    # the home team's ESPN id: the team whose scoring raises the home score
    hid = (ev.filter(pl.col("home_score").diff().over("game_id") > 0).group_by("game_id")
           .agg(pl.col("team_id").drop_nulls().mode().first().alias("home_id")))
    ev = ev.join(hid, on="game_id", how="left").with_columns(
        grp.alias("play_group"), pl.coalesce("nflverse_home_wp_post", "home_wp_post").alias("wp_post"),
        pl.when(pl.col("team_id").is_null()).then(None).when(pl.col("team_id") == pl.col("home_id")).then(pl.lit("home"))
        .otherwise(pl.lit("away")).alias("offense"),
        B.tau_left().alias("tau"), (pl.col("play_type") == "stoppage").alias("is_stoppage"),
        pl.col("down").alias("s1"), pl.col("distance").alias("s2"), pl.col("yard_line").alias("s3"))
    return _finish_plays(ev)


def plays_nba(games: pl.DataFrame) -> pl.DataFrame:
    p = (pl.read_parquet(Path(__file__).resolve().parent / "out" / "b1_nba_possessions.parquet")
         .filter(pl.col("game_id").is_in(games["game_id"].implode())).sort("game_id", "t", "end_seq"))
    # a possession enters at its end (t_end: after any free throws), when its full result is known
    p = p.with_columns(pl.int_range(pl.len()).over("game_id").alias("seq"), pl.max_horizontal("t", "t_end").alias("t_event"),
                       pl.col("t_end").alias("known_at"), pl.col("kind").alias("play_group"), pl.col("kind").alias("play_type"),
                       B.tau_left().alias("tau"), (pl.col("kind") == "period_end").alias("is_stoppage"),
                       ((pl.col("home_pts") + pl.col("away_pts")) > 0).alias("is_scoring"),
                       pl.lit(0).alias("s1"), pl.lit(0).alias("s2"), pl.lit(0).alias("s3"), pl.lit(None, pl.String).alias("home_id"))
    return _finish_plays(p)


PRINT_W_S = 15          # target window: trade prints in (t + h, t + h + PRINT_W_S]
FLOW_S = 60             # signed order flow / volume look-back

HR_DECIDED_S = 7       # MLB home run: pitch start -> ball out of the park, with a margin


def plays_mlb(games: pl.DataFrame) -> pl.DataFrame:
    grp = pl.lit("other")
    for name, pat in reversed(R.MLB_GROUPS):
        grp = pl.when(pl.col("pa_event").str.contains(pat)).then(pl.lit(name)).otherwise(grp)
    e = (load_pbp("MLB").filter(~pl.col("t_outside_window"), pl.col("pa_last_event").fill_null(False),
                                pl.col("game_id").is_in(games["game_id"].implode())).sort("game_id", "seq"))
    # a plate appearance enters when its result is logged (known_at = play end), not at the pitch start:
    # the outcome happens up to ~15 s after the pitch, so pitch-start timing would leak it. Exception: a
    # home run is decided when the ball leaves the park (~4-5 s), but is logged after the trot (median 25 s),
    # so it enters at pitch start + HR_DECIDED_S
    e = e.with_columns(pl.when(pl.col("pa_event") == "home_run")
                       .then(pl.min_horizontal(pl.col("t_event") + pl.duration(seconds=HR_DECIDED_S), "known_at"))
                       .otherwise(pl.max_horizontal("t_event", "known_at")).alias("t_event")).sort("game_id", "t_event", "seq")
    e = e.with_columns(grp.alias("play_group"), pl.col("pa_event").alias("play_type"), pl.col("home_wp_post").alias("wp_post"),
                       pl.when(pl.col("half") == "top").then(pl.lit("away")).otherwise(pl.lit("home")).alias("offense"),
                       B.tau_left().alias("tau"), pl.lit(False).alias("is_stoppage"),
                       pl.col("outs").alias("s1"), pl.col("balls").alias("s2"), pl.col("strikes").alias("s3"),
                       pl.lit(None, pl.String).alias("home_id"))
    return _finish_plays(e).with_columns((pl.col("dpoints") > 0).alias("is_scoring"))


def print_columns(cs: pl.DataFrame, pr: pl.DataFrame | None, g: dict, window_s: int = PRINT_W_S) -> pl.DataFrame:
    """Trade-print columns on the contract-step grid (cs: one row per ticker per step k, contiguous).

    A print at time tp is known from the first step at or after it: bin k = ceil((tp - first play) / STEP)
    (prints before the first play fall in bin 0). Inputs as of step k: last print price and age, signed
    flow (YES-taker minus NO-taker contracts) and volume over the last FLOW_S. Target window at step k:
    prints in bins k+1 .. k+window_s/STEP, i.e. in (t, t + window_s]: lowest, highest and
    volume-weighted price (w_lo / w_hi / w_vwap; shifted to each horizon by the caller)."""
    nb, nf = window_s // STEP, FLOW_S // STEP
    if pr is None or pr.height == 0:
        return cs.with_columns([pl.lit(None, pl.Float64).alias(c) for c in ("pr_px", "pr_age_s", "w_lo", "w_hi", "w_vwap")]
                               + [pl.lit(0.0).alias("flow_60s"), pl.lit(0.0).alias("vol_60s")])
    fp = g["first_play"]
    b = (pr.with_columns(((pl.col("tp") - fp).dt.total_microseconds() / (STEP * 1e6)).ceil().clip(0).cast(pl.Int64).alias("k"))
         .sort("tp").group_by("ticker", "k", maintain_order=True)
         .agg(pl.col("yes_price").last().alias("b_last"), pl.col("tp").last().alias("b_t"),
              pl.col("yes_price").min().alias("b_min"), pl.col("yes_price").max().alias("b_max"),
              pl.col("count").sum().alias("b_v"), (pl.col("count") * pl.col("yes_price")).sum().alias("b_n"),
              (pl.col("count") * pl.when(pl.col("is_yes")).then(1).otherwise(-1)).sum().alias("b_flow")))
    cs = cs.sort("ticker", "k").join(b, on=["ticker", "k"], how="left")
    o = lambda c, i: pl.col(c).shift(-i).over("ticker")   # noqa: E731
    wv = pl.sum_horizontal([o("b_v", i) for i in range(1, nb + 1)])
    cs = cs.with_columns(
        pl.col("b_last").forward_fill().over("ticker").alias("pr_px"),
        (pl.col("t") - pl.col("b_t").forward_fill().over("ticker")).dt.total_seconds().alias("pr_age_s"),
        pl.col("b_flow").fill_null(0).rolling_sum(nf, min_samples=1).over("ticker").cast(pl.Float64).alias("flow_60s"),
        pl.col("b_v").fill_null(0).rolling_sum(nf, min_samples=1).over("ticker").cast(pl.Float64).alias("vol_60s"),
        pl.min_horizontal([o("b_min", i) for i in range(1, nb + 1)]).alias("w_lo"),
        pl.max_horizontal([o("b_max", i) for i in range(1, nb + 1)]).alias("w_hi"),
        pl.when(wv > 0).then(pl.sum_horizontal([o("b_n", i) for i in range(1, nb + 1)]) / wv).alias("w_vwap"))
    return cs.drop("b_last", "b_t", "b_min", "b_max", "b_v", "b_n", "b_flow")


def futures_contracts(d: SportsData, games: pl.DataFrame) -> pl.DataFrame:
    """Per game: the home and away teams' season-long contracts (FUT_KINDS) open over the whole game
    (the soonest-expiring one per team and kind). side = which team of the game the contract is on."""
    fut = d.contracts.filter(pl.col("league") == LEAGUE, pl.col("scope") == "season", pl.col("kind").is_in(FUT_KINDS),
                             pl.col("team").is_not_null()).select("ticker", "series_ticker", "kind", "team", "open_time",
                                                                   "expiration_time")
    tg = pl.concat([games.select("game_id", "first_play", "feed_end", pl.col(s).alias("team"), pl.lit(s).alias("side"))
                    for s in ("home", "away")])
    return (tg.join(fut, on="team")
            .filter(pl.col("open_time") <= pl.col("first_play"), pl.col("expiration_time") >= pl.col("feed_end"))
            .sort("expiration_time").unique(["game_id", "side", "kind"], keep="first")
            .select("ticker", "series_ticker", "kind", "side", pl.lit(None, pl.Float64).alias("strike"), "game_id"))


def futures_quotes(series: list[str]) -> pl.LazyFrame:
    """Quote series of futures contracts, as stg.sports.prices.build_quotes but with the 1-min futures
    candles (closes known at the period end) merged with trade prints."""
    files = [str(f) for s_ in series for f in safe_path("candles_futures_1m", s_).glob("*.parquet")]
    parts = [_trade_obs(scan_trades(series))]
    if files:
        parts.append(pl.scan_parquet(files, missing_columns="insert", extra_columns="ignore")
                     .with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC"))).unique(["ticker", "ts"])
                     .filter(pl.col("ts") < SPORTS_HOLDOUT_START)
                     .select("ticker", pl.col("ts").alias("known_at"), pl.lit("candle").alias("src"),
                             pl.lit(None, pl.Float64).alias("price"), pl.lit(None, pl.Float64).alias("count"),
                             pl.lit(None, pl.String).alias("taker_side"),
                             pl.col("yes_bid_close").cast(pl.Float64).fill_null(0.0).alias("_bid"),
                             pl.col("yes_ask_close").cast(pl.Float64).fill_null(100.0).alias("_ask")))
    obs = pl.concat(parts, how="vertical_relaxed").sort("ticker", "known_at", "src")
    side_at = lambda c: pl.when(pl.col(c).is_not_null()).then(pl.col("known_at"))  # noqa: E731
    return (obs.with_columns(pl.col("_bid").forward_fill().over("ticker").alias("bid"),
                             side_at("_bid").forward_fill().over("ticker").alias("bid_at"),
                             pl.col("_ask").forward_fill().over("ticker").alias("ask"),
                             side_at("_ask").forward_fill().over("ticker").alias("ask_at"))
            .drop("_bid", "_ask").cast(QUOTE_SCHEMA))


def plays_table(games: pl.DataFrame) -> pl.DataFrame:
    return {"NFL": plays_nfl, "NBA": plays_nba, "MLB": plays_mlb}[LEAGUE](games)


def main() -> None:
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    d = SportsData()
    folds = pl.read_parquet(FOLDS).select("game_id", "fold")
    con = (d.contracts.filter(pl.col("league") == LEAGUE, pl.col("scope") == "game", pl.col("kind").is_in(KINDS),
                              pl.col("game_id").is_in(folds["game_id"].implode()))
           .select("ticker", "series_ticker", "kind", "side", "strike", "game_id"))
    con = con.filter((pl.col("kind") == "total") | pl.col("side").is_in(["home", "away"]))
    y = d.finals.select("ticker", pl.when(pl.col("result") == "yes").then(1.0).when(pl.col("result") == "no").then(0.0)
                        .otherwise(None).alias("y"))
    gm = load_games().filter(pl.col("league") == LEAGUE).select("game_id", "home", "away").unique("game_id")
    games = (d.results.filter(pl.col("league") == LEAGUE).select("game_id", "first_play", "feed_end")
             .join(folds, on="game_id").join(gm, on="game_id", how="left"))
    games = games.join(B.pregame_total(con, games), on="game_id", how="left")
    plays = plays_table(games)
    games = games.join(plays.group_by("game_id").agg(pl.col("home_id").first()), on="game_id", how="left")
    fut = futures_contracts(d, games)
    print(f"futures nodes: {fut.height} (game, team, kind) contracts over {fut['game_id'].n_unique()} games")
    con = pl.concat([con, fut], how="vertical_relaxed")
    # futures settle months later (some after the holdout start): no final-result target for them
    y = y.join(fut.select("ticker").unique(), on="ticker", how="anti")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contract_steps").mkdir(exist_ok=True)
    plays.drop("home_id").write_parquet(OUT / "plays.parquet")
    con.join(y, on="ticker", how="left").write_parquet(OUT / "contracts.parquet")
    games.write_parquet(OUT / "games.parquet")

    longest = max(HORIZONS.values())
    q_all = pl.concat([scan_quotes(con.filter(pl.col("kind").is_in(KINDS))["series_ticker"].unique().to_list()),
                       futures_quotes(fut["series_ticker"].unique().to_list()).collect().lazy()], how="vertical_relaxed")
    w = con.filter(pl.col("kind") == "winner")
    wtr = (scan_trades(w["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(w["ticker"].implode()))
           .select("ticker", pl.col("created_time").cast(pl.Datetime("us", "UTC")).alias("tt")).collect()
           .join(w.select("ticker", "game_id"), on="ticker"))
    # trade prints, compact: game contracts belong to one game (sorted by game, sliced per game without copies);
    # futures prints stay one small table and are cut to each game's window inside the loop
    def load_prints(c: pl.DataFrame) -> pl.DataFrame:
        return (scan_trades(c["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(c["ticker"].unique().implode()))
                .select("ticker", pl.col("created_time").cast(pl.Datetime("us", "UTC")).alias("tp"),
                        pl.col("yes_price").cast(pl.Float32), (pl.col("taker_side") == "yes").alias("is_yes"),
                        pl.col("count").cast(pl.Float32))
                .collect())
    gp = (load_prints(con.filter(pl.col("kind").is_in(KINDS))).join(con.select("ticker", "game_id"), on="ticker")
          .sort("game_id", "tp"))
    span = gp.with_row_index("i").group_by("game_id").agg(pl.col("i").min().alias("a"), pl.len().alias("n"))
    gp_slice = {r["game_id"]: (r["a"], r["n"]) for r in span.iter_rows(named=True)}
    fp_all = load_prints(fut).sort("tp")
    print(f"prints: {gp.height:,} on game contracts, {fp_all.height:,} on futures", flush=True)

    def prints_of(gid: str, g: dict, fut_tickers: pl.Series) -> pl.DataFrame:
        a, n = gp_slice.get(gid, (0, 0))
        f = fp_all.filter(pl.col("tp").is_between(g["first_play"] - dt.timedelta(days=1),
                                                  g["feed_end"] + dt.timedelta(seconds=longest + 3600)),
                          pl.col("ticker").is_in(fut_tickers.implode()))
        return pl.concat([gp.slice(a, n).drop("game_id"), f], how="vertical_relaxed")

    step_rows, n_cs, stats = [], 0, []
    for gi, g in enumerate(games.sort("first_play").iter_rows(named=True)):
        gid = g["game_id"]
        t = pl.datetime_range(g["first_play"], g["feed_end"] + dt.timedelta(seconds=longest), f"{STEP}s", eager=True,
                              time_unit="us", time_zone="UTC")
        steps = pl.DataFrame({"game_id": gid, "k": np.arange(len(t)), "t": t})
        # game state: the last play whose outcome had happened by t
        pg = plays.filter(pl.col("game_id") == gid).sort("t_event")
        st = steps.join_asof(pg.select("t_event", "seq", "tau", "margin", "total", "offense"), left_on="t", right_on="t_event",
                             strategy="backward").drop("t_event")
        newp = np.searchsorted(pg["t_event"].dt.epoch("us").to_numpy(), st["t"].dt.epoch("us").to_numpy(), side="right")
        st = st.with_columns(pl.Series("plays_known", newp)).with_columns(
            (pl.col("plays_known") - pl.col("plays_known").shift(1)).fill_null(pl.col("plays_known")).alias("new_plays"))
        tt = np.sort(wtr.filter(pl.col("game_id") == gid)["tt"].dt.epoch("us").to_numpy())
        tu = st["t"].dt.epoch("us").to_numpy()
        act = np.searchsorted(tt, tu, side="right") - np.searchsorted(tt, tu - ACT_S * 1_000_000, side="right")
        st = st.with_columns(pl.Series("winner_trades_5m", act), (pl.col("t") > g["feed_end"]).alias("after_end"))
        # quotes of every contract of the game at every step
        cg = con.filter(pl.col("game_id") == gid)
        at = steps.select("k", "t").join(cg.select("ticker"), how="cross")
        q = quotes_at(q_all, at.select("ticker", "t"), max_spread=100.0)
        is_fut = pl.col("ticker").is_in(cg.filter(pl.col("kind").is_in(FUT_KINDS))["ticker"].implode())
        q = q.with_columns(pl.when(is_fut).then(R.value(q, FUT_AGE_S)).otherwise(R.value(q)).alias("value"),
                           (pl.col("quote_age").dt.total_seconds()).alias("quote_age_s"))
        q = q.with_columns(((pl.col("bid") > 0) & (pl.col("ask") < 100) & (pl.col("ask") >= pl.col("bid"))
                            & (pl.col("ask") - pl.col("bid") <= LIVE_GAP)
                            & ((pl.col("bid") + pl.col("ask")) / 2).is_between(LIVE_LO, LIVE_HI)
                            & (pl.col("quote_age_s") <= pl.when(is_fut).then(FUT_AGE_S).otherwise(LIVE_AGE_S)))
                           .fill_null(False).alias("live"))
        cs = (at.join(q.select("ticker", "t", "bid", "ask", "quote_age_s", "value", "live"), on=["ticker", "t"], how="left")
              .join(y, on="ticker", how="left").sort("ticker", "k"))
        # after the game's end the result is known: a contract no longer quoted is worth 100 * y
        # (fills target rows only; steps after the end are never inputs)
        cs = cs.with_columns(pl.when((pl.col("t") > g["feed_end"]) & pl.col("value").is_null()).then(100 * pl.col("y"))
                             .otherwise(pl.col("value")).alias("value_for_targets"))
        fut_t = cg.filter(pl.col("kind").is_in(FUT_KINDS))["ticker"]
        pr_g = prints_of(gid, g, fut_t)
        cs = pl.concat([print_columns(cs.filter(~pl.col("ticker").is_in(fut_t.implode())), pr_g, g),
                        print_columns(cs.filter(pl.col("ticker").is_in(fut_t.implode())), pr_g, g, PRINT_W_FUT_S)],
                       how="vertical_relaxed").sort("ticker", "k")
        for name, sec in HORIZONS.items():
            sh = sec // STEP
            ended = (pl.col("t") + dt.timedelta(seconds=sec)) > g["feed_end"]
            cs = cs.with_columns([pl.col(c).shift(-sh).over("ticker").alias(f"{c}_{name}") for c in ("bid", "ask")]
                                 + [pl.col("value_for_targets").shift(-sh).over("ticker").alias(f"value_{name}")]
                                 + [pl.col(c).shift(-sh).over("ticker").alias(f"{o}_{name}")
                                    for c, o in (("w_lo", "plo"), ("w_hi", "phi"), ("w_vwap", "pv"))])
            # after the end with no prints: the result
            cs = cs.with_columns([pl.when(ended & pl.col(f"pv_{name}").is_null()).then(100 * pl.col("y"))
                                  .otherwise(pl.col(f"{o}_{name}")).alias(f"{o}_{name}") for o in ("plo", "phi", "pv")])
        cs = cs.with_columns((pl.col("value_600s") - pl.col("value_60s")).alias("drift_60_600"))
        cs = cs.drop("w_lo", "w_hi", "w_vwap")
        # distance to the threshold (spread: strike minus the contract team's margin; total: minus the total)
        cs = (cs.join(cg.select("ticker", "kind", "side", "strike"), on="ticker")
              .join(st.select("k", "margin", "total"), on="k", how="left")
              .with_columns(pl.when(pl.col("kind") == "spread")
                            .then(pl.col("strike") - pl.when(pl.col("side") == "home").then(1).otherwise(-1) * pl.col("margin"))
                            .when(pl.col("kind") == "total").then(pl.col("strike") - pl.col("total")).alias("dist"))
              .drop("margin", "total", "kind", "side", "strike")
              .filter(pl.col("bid").is_not_null() | pl.col("ask").is_not_null())
              .drop("value_for_targets"))
        # home winner mid per step (for the baseline and as a state input)
        ph = (cs.join(cg.filter(pl.col("kind") == "winner").select("ticker", "side"), on="ticker")
              .with_columns(pl.when(pl.col("live")).then((pl.col("bid") + pl.col("ask")) / 2).alias("mid"))
              .with_columns(pl.when(pl.col("side") == "home").then(pl.col("mid")).otherwise(100 - pl.col("mid")).alias("ph"))
              .group_by("k").agg(pl.col("ph").mean().alias("p_home")))
        st = st.join(ph, on="k", how="left").sort("k")
        step_rows.append(st)
        cs.drop("t").write_parquet(OUT / "contract_steps" / f"{gid}.parquet")
        n_cs += cs.height
        stats.append(dict(game_id=gid, steps=st.height, contract_steps=cs.height, live=int(cs["live"].sum())))
        if gi % 50 == 0:
            print(f"  {gi}/{games.height} games, {n_cs:,} contract-steps so far", flush=True)
    steps = pl.concat(step_rows, how="vertical_relaxed").join(folds, on="game_id")
    steps.write_parquet(OUT / "steps.parquet")
    (OUT / "meta.json").write_text(json.dumps(dict(league=LEAGUE, step_s=STEP, horizons_s=HORIZONS, act_s=ACT_S,
                                                   live=dict(gap=LIVE_GAP, lo=LIVE_LO, hi=LIVE_HI, age_s=LIVE_AGE_S),
                                                   built=dt.datetime.now(dt.timezone.utc).isoformat()), indent=1))
    s = pl.DataFrame(stats)
    print(f"\n{LEAGUE}: {games.height} games, {plays.height:,} plays, {steps.height:,} steps, {n_cs:,} contract-steps "
          f"({s['live'].sum() / n_cs:.3f} live) -> {OUT}")


if __name__ == "__main__":
    main()
