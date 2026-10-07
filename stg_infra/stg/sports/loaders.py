"""Holdout-guarded loaders for the sports archive, each row stamped with ``known_at``.

Layout (``scripts/fetch_sports.py``, ``scripts/fetch_pbp.py``; run from
``event-pred/`` or set ``$KALSHI_SPORTS_DIR``)::

    data/sports/markets.parquet              listing, working block
    data/sports/trades/<series>/*.parquet    trades
    data/sports/candles/<series>/*.parquet   1-min / hourly bid-ask candles
    data/sports/pbp/games.parquet            Kalshi game event -> source game
    data/sports/pbp/events_<league>.parquet  one row per pitch / play
    data/sports/derived/                     what ``scripts/build_sports_layer.py`` writes

Nothing here reads ``_HOLDOUT_DO_NOT_USE/``: every path goes through
:func:`safe_path`, and every frame keyed on a market is checked against
``SPORTS_HOLDOUT_START``. The one exception is the final test, which unseals it on purpose
(``STG_SPORTS_UNSEAL=final-test``, see stg.splits): then everything reads the holdout folder only.

``known_at`` (reports/sports_research_plan.md §4):

    trade            created_time
    candle           end of its period (the API's end_period_ts)
    MLB pitch        its end time (t_end; non-pitch events: t_end if within a
                     minute of t, else t -- a few carry the game's end as t_end)
    NBA play         its wallclock
    NFL play         the next snap (the game's 'End of Game' event for the last play)
    game result      the league's end-of-game time (MLB: final pitch, checked against
                     statsapi's official end; NBA / NFL: ESPN's end-of-game event)
                     plus RESULT_LAG. Kalshi's close is not used (diagnostic only)
    listing, at open identity, titles, strike type, rules, expiration_time
    listing, final   status, result, close_time, settlement_ts, volume,
                     open_interest, last_price, expiration_value: settlement_ts

``t_event`` (play-by-play only): when the play's outcome happened on the field, for aligning
prices with plays. It is not when a feed would have delivered it (``known_at`` stays the strict
stamp for trading tests):

    MLB pitch        the pitch's start: the outcome is visible within seconds (a home run moves the
                     price 3-5 s after it), while the recorded end comes after the play finishes
                     (home runs: a median 26 s, after the trot); other events: known_at
    NBA play         its wallclock
    NFL play         the play's end (nflverse end_clock_time, matched to the ESPN row by its snap).
                     Without one: the snap plus the median snap-to-end time of the same play type
                     over plays in earlier games only (the snap itself before any exist). Stoppages
                     (timeouts, two-minute warning, ends of periods): their own timestamp

``close_time`` is final: Kalshi moves it to when the game actually ended. Use it
only through :class:`stg.sports.asof.SportsData`, which reveals it once it has
passed.
"""
from __future__ import annotations

import datetime as dt
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from stg.splits import SPORTS_HOLDOUT_DIR, SPORTS_HOLDOUT_START, assert_no_sports_holdout, sports_holdout_unsealed

UTC = dt.timezone.utc
LEAGUES = ("MLB", "NBA", "NFL")

# listing columns known when the market lists, and those only known at the end
LISTING_COLS = ("ticker", "event_ticker", "series_ticker", "market_type", "title", "yes_sub_title",
                "no_sub_title", "open_time", "expiration_time", "strike_type", "rules_primary")
FINAL_COLS = ("status", "result", "close_time", "settlement_ts", "volume", "open_interest",
              "last_price", "expiration_value")


def sports_dir() -> Path:
    """The sports root; while the holdout is unsealed (stg.splits), the holdout folder instead."""
    root = Path(os.environ.get("KALSHI_SPORTS_DIR", "data/sports"))
    return root / SPORTS_HOLDOUT_DIR if sports_holdout_unsealed() else root


def safe_path(*parts: str | Path) -> Path:
    """A path under the sports root, refusing anything inside the holdout folder (while unsealed:
    anything outside it)."""
    p = Path(os.path.normpath(sports_dir().joinpath(*map(str, parts))))     # no '..' around the check
    if (SPORTS_HOLDOUT_DIR in p.parts) != sports_holdout_unsealed():
        raise PermissionError(f"{p}: the sports holdout is read once, for the final test")
    return p


def _before_wall(col: str) -> pl.Expr:
    """Rows strictly before the holdout start; while unsealed, every row of the holdout's markets
    (their trades and candles from before the start are history known at the time)."""
    return pl.lit(True) if sports_holdout_unsealed() else pl.col(col) < SPORTS_HOLDOUT_START


def derived_dir() -> Path:
    return safe_path("derived")


# --------------------------------------------------------------------------- listing

@lru_cache(maxsize=1)
def _listing() -> pl.DataFrame:
    mk = pl.read_parquet(safe_path("markets.parquet")).unique("ticker", keep="last")
    assert_no_sports_holdout(mk, "close_time")
    return mk.with_columns(
        pl.col("open_time").alias("known_at"),
        pl.coalesce("settlement_ts", "close_time").alias("final_known_at"))


def load_listing() -> pl.DataFrame:
    """Listing fields known when each market lists, with ``known_at = open_time``."""
    return _listing().select(*LISTING_COLS, "known_at")


def load_listing_final() -> pl.DataFrame:
    """The end-of-life fields (result, close, volume, ...), known at settlement.

    For labels and the as-of layer only; feature code must not call this
    (``tests/test_sports.py`` checks).
    """
    return _listing().select("ticker", *FINAL_COLS, pl.col("final_known_at").alias("known_at"))


# --------------------------------------------------------------------------- trades / candles

def series_with(kind: str) -> list[str]:
    d = safe_path(kind)
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def scan_trades(series: str | list[str] | None = None) -> pl.LazyFrame:
    """Trades of the given series (all if None), deduplicated, ``known_at = created_time``."""
    names = [series] if isinstance(series, str) else (series or series_with("trades"))
    files = [str(f) for s in names for f in safe_path("trades", s).glob("*.parquet")]
    if not files:
        return pl.LazyFrame(schema={"trade_id": pl.String, "ticker": pl.String, "count": pl.Float64,
                                    "yes_price": pl.Float64, "no_price": pl.Float64,
                                    "taker_side": pl.String, "created_time": pl.Datetime("us", "UTC"),
                                    "known_at": pl.Datetime("us", "UTC")})
    return (pl.scan_parquet(files, missing_columns="insert", extra_columns="ignore")
            .select("trade_id", "ticker", "count", "yes_price", "no_price", "taker_side",
                    pl.col("created_time").cast(pl.Datetime("us", "UTC")))
            .unique("trade_id")
            .filter(_before_wall("created_time"))
            .with_columns(pl.col("created_time").alias("known_at")))


def scan_candles(series: str | list[str] | None = None) -> pl.LazyFrame | None:
    """Candles (bid/ask in cents), ``known_at`` = end of the period. None if not pulled yet."""
    names = [series] if isinstance(series, str) else (series or series_with("candles"))
    files = [str(f) for s in names for f in safe_path("candles", s).glob("*.parquet")]
    if not files:
        return None
    return (pl.scan_parquet(files, missing_columns="insert", extra_columns="ignore")
            .with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))
            .unique(["ticker", "ts"])
            .filter(_before_wall("ts"))
            .with_columns(pl.col("ts").alias("known_at")))


# --------------------------------------------------------------------------- games / pbp

@lru_cache(maxsize=1)
def load_games() -> pl.DataFrame:
    """One row per Kalshi game event matched to a source game (preseason dropped).

    ``key`` joins every game-level series of the same game (``event_ticker``
    after the first ``-``). ``start_utc`` is the source schedule's start, known
    in advance; the actual start comes from the play-by-play (``first_play``).
    """
    g = pl.read_parquet(safe_path("pbp", "games.parquet"))
    g = g.filter(pl.col("game_id").is_not_null(), ~pl.col("preseason"),
                 pl.col("holdout") if sports_holdout_unsealed() else ~pl.col("holdout"))
    assert_no_sports_holdout(g, "close_time")
    return g


@lru_cache(maxsize=3)
def load_pbp(league: str) -> pl.DataFrame:
    """Play-by-play of one league with ``known_at`` per the table above."""
    if league not in LEAGUES:
        raise ValueError(league)
    ev = pl.read_parquet(safe_path("pbp", f"events_{league}.parquet"))
    g = load_games()
    close = (g.filter(pl.col("league") == league, pl.col("series_ticker").str.ends_with("GAME"))
             .group_by("game_id").agg(pl.col("close_time").max().alias("game_close")))
    ev = ev.join(close, on="game_id", how="inner")          # working-block games only
    if league == "MLB":
        near = (pl.col("t_end") - pl.col("t")) <= pl.duration(minutes=1)
        known = pl.when(pl.col("is_pitch") | near).then(pl.col("t_end")).otherwise(pl.col("t"))
    elif league == "NBA":
        known = pl.col("t")
    else:
        # the last play has no next snap: the game's own end event (never Kalshi's close)
        end_evt = pl.col("t").filter(pl.col("event_type") == "End of Game").max().over("game_id")
        known = pl.coalesce(pl.col("t").shift(-1).over("game_id", order_by="seq"), end_evt, pl.col("t"))
    ev = ev.with_columns(known.alias("known_at"))
    if league == "NFL":
        ev = _nfl_play_ends(ev)
    else:
        ev = ev.with_columns(pl.lit(None, pl.String).alias("play_type"),
                             (pl.when(pl.col("is_pitch")).then(pl.col("t")).otherwise(pl.col("known_at"))
                              if league == "MLB" else pl.col("t")).alias("t_event"))
    return ev.sort("game_id", "seq")


NFL_STOPPAGES = ("Timeout", "Official Timeout", "Two-minute warning", "End Period", "End of Half",
                 "End of Regulation", "End of Game")
NFL_END_MATCH = dt.timedelta(seconds=3)     # ESPN wallclock vs nflverse snap: median 0 s


def _nflverse_plays() -> pl.DataFrame:
    """nflverse rows with a snap time: nflverse_key, snap, play end, play type, nflverse's home win
    probability after the play (its model is trained on earlier seasons)."""
    files = sorted(safe_path("pbp", "raw", "NFL").glob("nflverse_pbp_*.parquet"))
    cols = ["game_date", "away_team", "home_team", "time_of_day", "end_clock_time", "play_type", "home_wp_post"]
    nv = pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")
    ts_ = lambda c: pl.col(c).str.to_datetime(time_zone="UTC", strict=False).dt.cast_time_unit("us")  # noqa: E731
    return (nv.select(pl.format("{}_{}_{}", "game_date", "away_team", "home_team").alias("nflverse_key"),
                      ts_("time_of_day").alias("snap"), ts_("end_clock_time").alias("play_end"), "play_type",
                      pl.col("home_wp_post").alias("nflverse_home_wp_post"))
            .filter(pl.col("snap").is_not_null()).sort("snap"))


def _nfl_play_ends(ev: pl.DataFrame) -> pl.DataFrame:
    """Fill t_end with the play's end; add play_type, t_event (see the module docstring) and
    nflverse_home_wp_post (null on stoppages and unmatched rows; home_wp_post stays ESPN's)."""
    keys = load_games().filter(pl.col("league") == "NFL").select("game_id", "nflverse_key").unique("game_id")
    e = ev.drop("t_end").join(keys, on="game_id", how="left").sort("t")
    m = e.join_asof(_nflverse_plays(), left_on="t", right_on="snap", by="nflverse_key", strategy="nearest",
                    tolerance=NFL_END_MATCH, check_sortedness=False)
    stop = pl.col("event_type").is_in(NFL_STOPPAGES)
    m = m.with_columns(pl.when(stop).then(None).otherwise(pl.col("play_end")).alias("t_end"),
                       pl.when(stop).then(pl.lit("stoppage")).otherwise(pl.col("play_type")).alias("play_type"),
                       pl.when(stop).then(None).otherwise(pl.col("nflverse_home_wp_post")).alias("nflverse_home_wp_post"))
    # fallback: median snap-to-end of the same play type over earlier games' plays (expanding, as-of)
    dur = (m.filter(pl.col("t_end").is_not_null(), pl.col("t_end") > pl.col("t"))
           .with_columns((pl.col("t_end") - pl.col("t")).dt.total_milliseconds().alias("dur_ms"),
                         pl.col("t").min().over("game_id").alias("game_start"))
           .select("play_type", "game_start", "dur_ms"))
    starts = m.select("game_id", pl.col("t").min().over("game_id").alias("game_start")).unique("game_id")
    fb = []
    for (pt,), d in dur.partition_by("play_type", as_dict=True).items():
        d = d.sort("game_start")
        gs, ms = d["game_start"].to_numpy(), d["dur_ms"].to_numpy().astype(float)
        for gid, g0 in starts.iter_rows():
            k = int(np.searchsorted(gs, np.datetime64(g0.replace(tzinfo=None), "us"), side="left"))
            if k:                                   # plays of games that started before this one
                fb.append((gid, pt, float(np.median(ms[:k]))))
    fb = pl.DataFrame(fb, schema={"game_id": pl.String, "play_type": pl.String, "fallback_ms": pl.Float64},
                      orient="row")
    m = m.join(fb, on=["game_id", "play_type"], how="left")
    t_event = (pl.when(stop).then(pl.col("t"))
               .when(pl.col("t_end").is_not_null()).then(pl.col("t_end"))
               .otherwise(pl.col("t") + pl.duration(milliseconds=pl.col("fallback_ms").fill_null(0))))
    return m.with_columns(t_event.alias("t_event")).drop("nflverse_key", "snap", "play_end", "fallback_ms")


@lru_cache(maxsize=1)
def external_results() -> pl.DataFrame:
    """Finished games from the leagues' own schedules (scripts/fetch_mlb_schedule.py), for the
    games Kalshi did not list: MLB 2025-03-27 .. 04-15. Same columns as a team-game source:
    league, season, game_id, game_type, sched_start, home, away, home_score, away_score,
    first_play, known_at (the result's time: first pitch + duration + delays). Empty if the file
    is absent. Results known on or after the holdout start are never stored there."""
    p = safe_path("pbp", "mlb_schedule.parquet")
    if not p.exists():
        return pl.DataFrame(schema={"league": pl.String, "season": pl.Int32, "game_id": pl.String,
                                    "game_type": pl.String, "sched_start": pl.Datetime("us", "UTC"),
                                    "home": pl.String, "away": pl.String, "home_score": pl.Int64,
                                    "away_score": pl.Int64, "first_play": pl.Datetime("us", "UTC"),
                                    "known_at": pl.Datetime("us", "UTC")})
    r = (pl.read_parquet(p).filter(pl.col("home_score").is_not_null())
         .select("league", pl.col("season").cast(pl.Int32), "game_id", "game_type", "sched_start", "home", "away",
                 "home_score", "away_score", pl.col("first_pitch").alias("first_play"), "known_at"))
    assert_no_sports_holdout(r, "known_at")
    return r


END_EVENTS = ("End Game", "End of Game")    # ESPN's end-of-game events (NBA, NFL)
# A result is usable only once a live feed would have delivered it: the league's timestamp is when
# the game ended, our feed sees it a little later. Conservative for between-game work (hour-plus
# horizons); revisit per league for in-game work, from logged arrival times of a live feed.
RESULT_LAG = dt.timedelta(minutes=1)


def mlb_official_end() -> pl.DataFrame:
    """game_id -> statsapi's official end (first pitch + duration + delays) and final score, if
    fetched. The computed end runs late when a delay came before the first pitch, so it is only a
    fallback for an incomplete feed."""
    p = safe_path("pbp", "mlb_schedule.parquet")
    schema = {"game_id": pl.String, "official_end": pl.Datetime("us", "UTC"),
              "official_home": pl.Int64, "official_away": pl.Int64}
    if not p.exists():
        return pl.DataFrame(schema=schema)
    return (pl.read_parquet(p).filter(pl.col("known_at").is_not_null())
            .select("game_id", pl.col("known_at").alias("official_end"),
                    pl.col("home_score").alias("official_home"), pl.col("away_score").alias("official_away")))


def game_results() -> pl.DataFrame:
    """One row per game: final score, first play (actual start), and when the result was known.

    ``first_play`` is the first pitch (MLB) or first in-window play (NBA/NFL), known at itself.
    The result is known at the league's own end-of-game time plus RESULT_LAG:

        MLB  the end of the final pitch (statsapi feed). If the feed's final score differs from
             statsapi's official one, the feed stopped early (5 games of 3,429): then the
             official end (first pitch + duration + delays, scripts/fetch_mlb_schedule.py)
        NBA  ESPN's 'End Game' event, else the last play
        NFL  ESPN's 'End of Game' event, else the last snap

    Kalshi's close (``game_close``) is kept as a diagnostic only: Kalshi closes markets on its own
    schedule (the GB-DAL 40-40 tie closed 8 h before kickoff; 13 MLB markets of 2025-04-18 stayed
    open until 2026-03-25). ``feed_end`` is the league timestamp before the lag.
    """
    out = []
    for lg in LEAGUES:
        if not safe_path("pbp", f"events_{lg}.parquet").exists():      # a league with no games yet (holdout)
            continue
        ev = load_pbp(lg).filter(~pl.col("t_outside_window"))
        first = ev.filter(pl.col("is_pitch")) if lg == "MLB" else ev
        f = first.group_by("game_id").agg(pl.col("t").min().alias("first_play"))
        if lg == "MLB":
            end = ev.filter(pl.col("is_pitch")).group_by("game_id").agg(pl.col("t_end").max().alias("feed_end"))
        else:
            end = ev.group_by("game_id").agg(
                pl.coalesce(pl.col("t").filter(pl.col("event_type").is_in(END_EVENTS)).max(),
                            pl.col("t").max()).alias("feed_end"))
        last = (ev.sort("game_id", "seq").group_by("game_id")
                .agg(pl.col("home_score").last(), pl.col("away_score").last(), pl.col("game_close").first()))
        g = f.join(end, on="game_id").join(last, on="game_id").with_columns(pl.lit(lg).alias("league"))
        if lg == "MLB":
            incomplete = (pl.col("official_end").is_not_null()
                          & ((pl.col("home_score") != pl.col("official_home"))
                             | (pl.col("away_score") != pl.col("official_away"))))
            g = (g.join(mlb_official_end(), on="game_id", how="left")
                 .with_columns(pl.when(incomplete).then(pl.col("official_end")).otherwise(pl.col("feed_end"))
                               .alias("feed_end"),
                               pl.when(incomplete).then(pl.col("official_home")).otherwise(pl.col("home_score"))
                               .alias("home_score"),
                               pl.when(incomplete).then(pl.col("official_away")).otherwise(pl.col("away_score"))
                               .alias("away_score"))
                 .drop("official_end", "official_home", "official_away"))
        out.append(g.with_columns((pl.col("feed_end") + RESULT_LAG).alias("known_at"),
                                  pl.col("feed_end").alias("last_play_known")))
    return pl.concat(out, how="diagonal_relaxed")


def event_key(col: str = "event_ticker") -> pl.Expr:
    """``KXNBASPREAD-25NOV17CHIDEN`` -> ``25NOV17CHIDEN``, the key shared by one game's events."""
    return pl.col(col).str.splitn("-", 2).struct.field("field_1")


GAME_DATE = re.compile(r"^(\d{2}[A-Z]{3}\d{2})")
