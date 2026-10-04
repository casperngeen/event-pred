"""As-of features (reports/sports_research_plan.md §9).

Every function takes a query frame with a time column ``t`` and returns one row
per query row, each feature group with its own ``<group>_known_at``: the
latest input time it used, always <= t (:func:`stg.sports.checks.assert_asof`
checks). Only rows with ``known_at <= t`` are ever joined, via backward as-of
joins on ``known_at``.

Groups:
    market_state    mid, spread, log-odds, quote age, trades and contracts in
                    the last hour, time since listing, time to scheduled start
    game_state      in-game: score, period, count / outs, down / distance /
                    field position, the source's live win probability
    standings       season-to-date W-L, win %, games remaining (fraction),
                    games behind the leader of the team's division or
                    conference (read from the futures events it is listed in)
    form            streak, last margin, last-10 record           (ablation)
    schedule        rest days, back-to-back (home/away: the contract's side) (ablation)
    strength        runs / points for and against per game       (hypothesis test)

This module must not read the listing's final fields (``loaders.FINAL_COLS``);
``tests/test_sports.py`` scans its source for them.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from stg.sports.asof import SportsData
from stg.sports.prices import MAX_AGE, MAX_SPREAD

SEASON_GAMES = {"MLB": 162, "NBA": 82, "NFL": 17}
REGULAR = {"MLB": ["R"], "NBA": ["2"], "NFL": ["2"]}


def _asof(left: pl.DataFrame, right: pl.DataFrame, by: str | list[str], t: str = "t") -> pl.DataFrame:
    left = left.with_row_index("_row").sort(t)
    return (left.join_asof(right.sort("known_at"), left_on=t, right_on="known_at", by=by,
                           strategy="backward").sort("_row").drop("_row"))


# --------------------------------------------------------------------------- market state

def market_state(data: SportsData, q: pl.DataFrame, max_spread: float = MAX_SPREAD,
                 max_age: dt.timedelta = MAX_AGE) -> pl.DataFrame:
    """q: ticker, t."""
    base = data.quotes_at(q.select("ticker", "t"), max_spread=max_spread, max_age=max_age)
    c = data.contracts.select("ticker", "series_ticker", "open_time", "sched_start")
    out = base.join(c, on="ticker", how="left").with_columns(
        ((pl.col("t") - pl.col("open_time")).dt.total_seconds() / 3600).alias("hours_listed"),
        ((pl.col("sched_start") - pl.col("t")).dt.total_seconds() / 3600).alias("hours_to_sched_start"))
    # trailing-hour activity, from the quote series' trade rows (known_at <= t)
    series = out["series_ticker"].drop_nulls().unique().to_list()
    tr = (data.scan_quotes(series).filter(pl.col("src") == "trade",
                                          pl.col("ticker").is_in(q["ticker"].unique().implode()))
          .select("ticker", "known_at", "count").collect())
    act = (q.select("ticker", "t").unique().join(tr, on="ticker")
           .filter(pl.col("known_at") <= pl.col("t"),
                   pl.col("known_at") > pl.col("t") - pl.duration(hours=1))
           .group_by("ticker", "t").agg(pl.len().alias("trades_1h"), pl.col("count").sum().alias("contracts_1h")))
    out = out.join(act, on=["ticker", "t"], how="left").with_columns(
        pl.col("trades_1h").fill_null(0), pl.col("contracts_1h").fill_null(0.0),
        pl.col("quote_known_at").alias("market_known_at"))
    return out.select("ticker", "t", "bid", "ask", "mid", "mid_lo", "spread", "quote_age", "two_sided",
                      "last_trade", "trades_1h", "contracts_1h", "hours_listed", "hours_to_sched_start",
                      "market_known_at")


# --------------------------------------------------------------------------- game state

GAME_COLS = {
    "MLB": ["period", "half", "home_score", "away_score", "balls", "strikes", "outs", "home_wp_post"],
    "NBA": ["period", "clock", "home_score", "away_score", "home_wp_post"],
    "NFL": ["period", "clock", "home_score", "away_score", "down", "distance", "yard_line", "home_wp_post"],
}


def game_state(data: SportsData, q: pl.DataFrame, league: str) -> pl.DataFrame:
    """q: game_id, t. The last play known at t (null before the first)."""
    ev = (data.pbp_all(league).filter(~pl.col("t_outside_window"))
          .select("game_id", "known_at", *GAME_COLS[league]))
    out = _asof(q.select("game_id", "t"), ev, by="game_id")
    return out.with_columns(
        (pl.col("home_score") - pl.col("away_score")).alias("home_margin"),
        (pl.col("home_score") + pl.col("away_score")).alias("points"),
        pl.col("known_at").alias("game_known_at")).drop("known_at")


# --------------------------------------------------------------------------- team-season tables

def team_games(data: SportsData) -> pl.DataFrame:
    """One row per (team, finished game), ``known_at`` = when the result was known.

    Games with a Kalshi market come from the play-by-play results; games Kalshi did not list
    (MLB before 2025-04-16) from the league's own schedule (``data.external_results``)."""
    r = data.results
    g = (data.contracts.filter(pl.col("kind") == "winner", pl.col("game_id").is_not_null())
         .select("league", "game_id", "home", "away", "game_type", "season", "sched_start")
         .unique(["league", "game_id"]))
    r = r.join(g, on=["league", "game_id"], how="inner")
    cols = ["league", "season", "game_id", "game_type", "sched_start", "home", "away", "home_score",
            "away_score", "first_play", "known_at"]
    ext = data.external_results.join(r.select("league", "game_id"), on=["league", "game_id"], how="anti")
    r = pl.concat([r.select(cols).with_columns(pl.col("season").cast(pl.Int32)),
                   ext.select(cols)], how="vertical_relaxed")
    side = lambda me, op, ms, os, home: r.select(  # noqa: E731
        "league", "season", "game_id", "game_type", "known_at", "first_play", "sched_start",
        pl.col(me).alias("team"), pl.col(op).alias("opp"), pl.lit(home).alias("home"),
        pl.col(ms).alias("pts_for"), pl.col(os).alias("pts_against"))
    tg = pl.concat([side("home", "away", "home_score", "away_score", True),
                    side("away", "home", "away_score", "home_score", False)])
    reg = pl.lit(False)
    for lg, types in REGULAR.items():
        reg = reg | ((pl.col("league") == lg) & pl.col("game_type").is_in(types))
    return tg.with_columns(
        reg.alias("regular"),
        (pl.col("pts_for") - pl.col("pts_against")).alias("margin"),
        pl.when(pl.col("pts_for") > pl.col("pts_against")).then(1.0)
        .when(pl.col("pts_for") < pl.col("pts_against")).then(0.0).otherwise(0.5).alias("win"),
    ).sort("league", "team", "known_at")


def _cumulative(tg: pl.DataFrame) -> pl.DataFrame:
    """Season-to-date regular-season stats after each game, keyed by its known_at."""
    reg = tg.filter(pl.col("regular")).sort("league", "season", "team", "known_at")
    k = ["league", "season", "team"]
    w = pl.col("win")
    streak_id = (w != w.shift(1)).fill_null(True).cast(pl.Int32).cum_sum().over(k)
    return reg.with_columns(
        pl.len().over(k).alias("_n"),
        pl.int_range(1, pl.len() + 1).over(k).alias("gp"),
        w.cum_sum().over(k).alias("wins"),
        pl.col("pts_for").cum_sum().over(k).alias("_pf"),
        pl.col("pts_against").cum_sum().over(k).alias("_pa"),
        w.rolling_mean(10, min_samples=1).over(k).alias("last10"),
        (pl.int_range(1, pl.len() + 1).over(k + [streak_id]) * (2 * w - 1)).alias("streak"),
        pl.col("margin").alias("last_margin"),
    ).with_columns(
        (pl.col("gp") - pl.col("wins")).alias("losses"),
        (pl.col("wins") / pl.col("gp")).alias("win_pct"),
        (pl.col("_pf") / pl.col("gp")).alias("pf_per_game"),
        (pl.col("_pa") / pl.col("gp")).alias("pa_per_game"),
    ).select(*k, "known_at", "gp", "wins", "losses", "win_pct", "last10", "streak", "last_margin",
             "pf_per_game", "pa_per_game")


def groups(data: SportsData) -> pl.DataFrame:
    """team -> the division (else conference) futures event it is listed in, per season.

    Read from the futures tickers, so no membership table is hard-coded; known
    at the futures' listing.
    """
    c = data.contracts.filter(pl.col("kind").is_in(["division", "conference"]), pl.col("team").is_not_null())
    c = c.with_columns(pl.when(pl.col("kind") == "division").then(0).otherwise(1).alias("_pri"))
    return (c.sort("_pri", "open_time").group_by("league", "season", "team", maintain_order=True)
            .agg(pl.col("event_ticker").first().alias("group"), pl.col("kind").first().alias("group_kind"),
                 pl.col("open_time").first().alias("group_known_at")))


def team_state(data: SportsData, q: pl.DataFrame) -> pl.DataFrame:
    """q: league, season, team, t. Standings, form, schedule and strength, all as of t."""
    q = q.with_columns(pl.col("season").cast(pl.Int32))
    tg = team_games(data)
    cum = _cumulative(tg)
    out = _asof(q.select("league", "season", "team", "t"), cum, by=["league", "season", "team"])
    out = out.with_columns(pl.col("known_at").alias("standings_known_at")).drop("known_at")
    for c in ("gp", "wins", "losses"):
        out = out.with_columns(pl.col(c).fill_null(0))
    n = pl.col("league").replace_strict(SEASON_GAMES, default=None)
    out = out.with_columns(((n - pl.col("gp")) / n).clip(0, 1).alias("frac_remaining"))

    # games behind the group leader, every team of the group read as of the same t
    gr = groups(data)
    qg = out.select("league", "season", "team", "t").join(gr, on=["league", "season", "team"], how="left")
    peers = (qg.filter(pl.col("group").is_not_null()).select("league", "season", "group", "t").unique()
             .join(gr.select("league", "season", "group", pl.col("team").alias("peer")),
                   on=["league", "season", "group"]))
    peers = _asof(peers.with_columns(pl.col("peer").alias("team")),
                  cum.select("league", "season", "team", "known_at", "wins", "losses"),
                  by=["league", "season", "team"])
    lead = (peers.with_columns((pl.col("wins").fill_null(0) - pl.col("losses").fill_null(0)).alias("_d"))
            .group_by("league", "season", "group", "t").agg(pl.col("_d").max().alias("_lead")))
    out = (out.join(qg.select("league", "season", "team", "t", "group", "group_kind"),
                    on=["league", "season", "team", "t"], how="left")
           .join(lead, on=["league", "season", "group", "t"], how="left")
           .with_columns(((pl.col("_lead") - (pl.col("wins") - pl.col("losses"))) / 2).alias("games_behind"))
           .drop("_lead"))

    # schedule: rest since the start of the team's last finished game (any season, any game type)
    last = tg.select("league", "team", "known_at", pl.col("first_play").alias("prev_start")).drop_nulls()
    prev = _asof(out.select("league", "team", "t"), last, by=["league", "team"])
    out = out.with_columns(
        ((pl.col("t") - prev["prev_start"]).dt.total_seconds() / 86400).alias("days_since_last_game"),
        prev["known_at"].alias("schedule_known_at"))
    return out.with_columns((pl.col("days_since_last_game") < 1.5).alias("back_to_back"))


# --------------------------------------------------------------------------- contender flag

def contention(data: SportsData, q: pl.DataFrame, kind: str = "playoffs") -> pl.DataFrame:
    """q: league, season, team, t -> the team's ``kind`` futures mid as of t (null if not listed)."""
    fut = data.contracts.filter(pl.col("kind") == kind, pl.col("team").is_not_null()).select(
        "league", "season", "team", "ticker")
    qq = q.join(fut, on=["league", "season", "team"], how="left")
    have = qq.filter(pl.col("ticker").is_not_null())
    if have.is_empty():
        return q.with_columns(pl.lit(None, pl.Float64).alias(f"{kind}_mid"))
    m = data.quotes_at(have.select("ticker", "t"), max_age=dt.timedelta(days=2), max_spread=10)
    have = have.with_columns(m["mid"].alias(f"{kind}_mid"), m["quote_known_at"].alias(f"{kind}_known_at"))
    return q.join(have.drop("ticker"), on=["league", "season", "team", "t"], how="left")
