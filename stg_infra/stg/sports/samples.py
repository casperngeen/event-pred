"""Training samples for the between-game models: one row per (graph node, night).

A night is one league's slate of regular-season games (first plays on one US-Eastern date).

    t_pre   the night's first first-play: the node's price before any of tonight's games
    t       the decision time: the first hourly candle close at least DECISION_DELAY after the
            night's last result (incl. RESULT_LAG), plus a minute. Futures quotes are hourly
            candles, so an earlier t would read a book from before the result and count moves that
            already happened as still to come (the rem_* labels)
    t + h   label times, one per horizon

Inputs (every one known by t; ``assert_asof`` checks the ``*_known_at`` columns against t):
    node        league, season, node, team, kind
    value       v_pre / v_t (node_values: mid in cents, or expected wins), with spread, one_sided,
                quote_age at each, and moved = v_t - v_pre (what tonight already did)
    standings   the team's state as of t_pre (features.team_state): gp, win_pct, games_behind,
                frac_remaining, streak, last10, rest
    shocks      z_own (the team's summed surprise tonight), played, x_rival (the summed z of every
                team linked to the node by its rival relation that played tonight), n_rival_games
    night       n_games on the slate
    live        the modelling set (decided 2026-10-03): single-leg contracts priced within
                LIVE_RANGE and two-sided both before the night and at t; wins nodes with a value
                at both. Dead contracts (longshots / near-certainties) barely move: their zeros are
                tick rounding, not information

Labels (after t, never inputs):
    total_<h>   v(t + h) - v_pre       the whole response to tonight (relation learning)
    rem_<h>     v(t + h) - v_t         what is left to happen after the decision (trading)
    label_end_<h> = t + h, for purged walk-forward splits; a label is null if the node closed or
    had no value at either end.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from stg.sports.asof import SportsData
from stg.sports.features import REGULAR, team_state
from stg.sports.graph import BetweenGameGraph, game_shocks, node_values

DECISION_DELAY = dt.timedelta(minutes=5)       # at least this long after the last result
CANDLE_READ = dt.timedelta(minutes=1)           # read the hourly close a minute after it lands
HORIZONS = {"1h": dt.timedelta(hours=1), "12h": dt.timedelta(hours=12), "24h": dt.timedelta(hours=24),
            "48h": dt.timedelta(hours=48)}


def decision_time(last_result: pl.Expr, decision_delay: dt.timedelta = DECISION_DELAY) -> pl.Expr:
    """The first hourly boundary at or after last_result + decision_delay, plus CANDLE_READ."""
    x = last_result + decision_delay
    hour = x.dt.truncate("1h")
    return pl.when(x == hour).then(hour).otherwise(hour + pl.duration(hours=1)) + CANDLE_READ
VALUE_COLS = ["value", "spread", "one_sided", "quote_age", "value_known_at"]
LIVE_RANGE = (3.0, 97.0)                        # cents


def nights(data: SportsData, league: str, season: int,
           decision_delay: dt.timedelta = DECISION_DELAY) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(nights, shocks): one row per night (t_pre, t, n_games), and the regular-season shocks."""
    sh = (game_shocks(data, league, season).filter(pl.col("game_type").is_in(REGULAR[league]))
          .with_columns(pl.col("first_play").dt.convert_time_zone("America/New_York").dt.date().alias("night")))
    n = (sh.group_by("night").agg(pl.col("first_play").min().alias("t_pre"),
                                   pl.col("known_at").max().alias("last_result"),
                                   pl.col("game_id").n_unique().alias("n_games"))
         .with_columns(decision_time(pl.col("last_result"), decision_delay).alias("t"))
         .sort("night"))
    return n, sh


def _values(data: SportsData, graph: BetweenGameGraph, at: pl.DataFrame, suffix: str) -> pl.DataFrame:
    v = node_values(data, graph, at).select("node", "t", *VALUE_COLS)
    return v.rename({c: f"{c}_{suffix}" for c in VALUE_COLS}).rename({"t": f"_t_{suffix}"})


def sample_table(data: SportsData, graph: BetweenGameGraph, horizons: dict = HORIZONS,
                 decision_delay: dt.timedelta = DECISION_DELAY) -> pl.DataFrame:
    lg, season = graph.league, graph.season
    nt, sh = nights(data, lg, season, decision_delay)
    nd = graph.nodes.select("node", "team", "kind", "known_at", "valid_to")
    rows = (nd.join(nt, how="cross")
            .filter(pl.col("known_at") <= pl.col("t_pre"),                       # listed before the night
                    pl.col("valid_to").is_null() | (pl.col("valid_to") > pl.col("t")))
            .drop("known_at"))

    # tonight's shocks: own team, and rivals through the node's rival relation
    zt = sh.group_by("night", "team").agg(pl.col("z").sum(), pl.col("known_at").max().alias("shock_known_at"))
    rows = (rows.join(zt.rename({"z": "z_own"}), on=["night", "team"], how="left")
            .with_columns(pl.col("z_own").is_not_null().alias("played"), pl.col("z_own").fill_null(0.0)))
    team_of = nd.select(pl.col("node").alias("src"), pl.col("team").alias("src_team"))
    links = (graph.relations.filter(pl.col("relation").str.starts_with("rival"))
             .join(team_of, on="src").select(pl.col("dst").alias("node"), "src_team"))
    ex = (links.join(rows.select("node", "night").unique(), on="node")
          .join(zt.rename({"team": "src_team"}), on=["night", "src_team"])
          .group_by("node", "night").agg(pl.col("z").sum().alias("x_rival"), pl.len().alias("n_rival_games"),
                                          pl.col("shock_known_at").max().alias("rival_known_at")))
    rows = (rows.join(ex, on=["node", "night"], how="left")
            .with_columns(pl.col("x_rival").fill_null(0.0), pl.col("n_rival_games").fill_null(0)))

    # node values before the night and at the decision; labels at t + h
    rows = (rows.join(_values(data, graph, rows.select("node", pl.col("t_pre").alias("t")), "pre"),
                      left_on=["node", "t_pre"], right_on=["node", "_t_pre"], how="left")
            .join(_values(data, graph, rows.select("node", "t"), "t"),
                  left_on=["node", "t"], right_on=["node", "_t_t"], how="left")
            .with_columns((pl.col("value_t") - pl.col("value_pre")).alias("moved")))
    for h, dh in horizons.items():
        at = rows.select("node", (pl.col("t") + dh).alias("t"))
        v = node_values(data, graph, at).select("node", pl.col("t").alias(f"label_end_{h}"),
                                                pl.col("value").alias(f"_v_{h}"))
        rows = (rows.with_columns((pl.col("t") + dh).alias(f"label_end_{h}"))
                .join(v.unique(["node", f"label_end_{h}"]), on=["node", f"label_end_{h}"], how="left")
                .with_columns(
                    pl.when(pl.col("valid_to").is_null() | (pl.col("valid_to") > pl.col(f"label_end_{h}")))
                    .then(pl.col(f"_v_{h}")).alias(f"_v_{h}"))
                .with_columns((pl.col(f"_v_{h}") - pl.col("value_pre")).alias(f"total_{h}"),
                              (pl.col(f"_v_{h}") - pl.col("value_t")).alias(f"rem_{h}"))
                .drop(f"_v_{h}"))

    # standings before the night
    q = rows.select("team", pl.col("t_pre").alias("t")).unique().with_columns(
        pl.lit(lg).alias("league"), pl.lit(season).alias("season"))
    st = team_state(data, q.select("league", "season", "team", "t")).select(
        "team", pl.col("t").alias("t_pre"), "gp", "win_pct", "games_behind", "frac_remaining", "streak", "last10",
        "days_since_last_game", "standings_known_at")
    rows = rows.join(st, on=["team", "t_pre"], how="left")

    lo, hi = LIVE_RANGE
    single_live = (pl.col("value_pre").is_between(lo, hi) & pl.col("value_t").is_between(lo, hi)
                   & ~pl.col("one_sided_pre") & ~pl.col("one_sided_t"))
    rows = rows.with_columns(
        pl.when(pl.col("kind") == "wins")
        .then(pl.col("value_pre").is_not_null() & pl.col("value_t").is_not_null())
        .otherwise(single_live).fill_null(False).alias("live"))

    return (rows.with_columns(pl.lit(lg).alias("league"), pl.lit(season).alias("season"))
            .drop("valid_to")
            .select("league", "season", "night", "node", "team", "kind", "t_pre", "last_result", "t", "n_games",
                    "live", "played", "z_own", "x_rival", "n_rival_games", "shock_known_at", "rival_known_at",
                    *[f"{c}_pre" for c in VALUE_COLS], *[f"{c}_t" for c in VALUE_COLS], "moved",
                    "gp", "win_pct", "games_behind", "frac_remaining", "streak", "last10", "days_since_last_game",
                    "standings_known_at",
                    *[c for h in horizons for c in (f"total_{h}", f"rem_{h}", f"label_end_{h}")])
            .sort("night", "node"))
