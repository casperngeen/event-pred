"""Next-game nodes: how tonight's results move the markets of the teams' upcoming games.

Every value is from the upcoming game's home side. A game's two markets give one ladder of
P(home margin >= k): the home winner leg is k = 1 (the away winner leg its complement), a home
spread leg "wins by over X" is k = floor(X) + 1, an away spread leg "wins by over Y" is
P(home margin <= -(floor(Y) + 1)) = 1 - P(home margin >= -floor(Y)). A normal fit through it
(graph.fit_wins_ladder) gives the implied home margin mu; the winner legs alone give p_home.

    game_values(data, at)          at: game_id, t -> p_home (cents), mu_home (points / runs), sigma,
                                   legs, spread / one_sided / quote_age of the winner legs, known_at
    nextgame_table(data, league, season)
        one row per (upcoming game G, night) where a team of G played its previous game that night:

        t_pre   the night's first first-play;  t   when every result of G's teams that night is
                known (+ DECISION_DELAY; game markets have 1-min candles, no hourly alignment)
        inputs  p / mu before the night and at t, moved_p / moved_mu, z_home / z_away (tonight's
                surprise of each side; 0 if it did not play), ms_home / ms_away (margin surprise:
                actual margin - the margin implied at that game's first play), days to G
        labels  p and mu at t + 1h, t + 12h (null past G's first play) and at the close (G's first
                play): total_* from t_pre, rem_* from t; y_home (G's result) with its known_at
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
from scipy.stats import norm

from stg.sports.asof import SportsData
from stg.sports.features import REGULAR
from stg.sports.graph import NODE_MAX_AGE, NODE_MAX_SPREAD, fit_wins_ladder, game_shocks

DECISION_DELAY = dt.timedelta(minutes=5)
MIN_SIGMA_GAMES = 50          # games behind the price-implied margin's spread
HORIZONS = {"1h": dt.timedelta(hours=1), "12h": dt.timedelta(hours=12)}


def _game_legs(data: SportsData, game_ids: list[str]) -> pl.DataFrame:
    """Winner and spread legs of these games, each with its threshold k on the home margin."""
    c = data.contracts.filter(pl.col("game_id").is_in(game_ids), pl.col("kind").is_in(["winner", "spread"]),
                              pl.col("team").is_not_null())
    home = pl.col("team") == pl.col("home")
    k_spread = pl.col("strike").floor() + 1
    return c.select(
        "game_id", "ticker", "kind", "home", "away", "open_time",
        pl.when(pl.col("kind") == "winner").then(pl.lit(1.0))
        .when(home).then(k_spread).otherwise(-k_spread + 1).alias("k"),
        # a leg on the away side gives the complement of P(home margin >= k)
        (~home).alias("flip"))


def game_values(data: SportsData, at: pl.DataFrame, max_spread: float = NODE_MAX_SPREAD,
                max_age: dt.timedelta = NODE_MAX_AGE) -> pl.DataFrame:
    legs = _game_legs(data, at["game_id"].unique().to_list())
    x = at.join(legs, on="game_id").filter(pl.col("open_time") <= pl.col("t"))
    q = data.quotes_at(x.select("ticker", "t").unique(), max_spread=max_spread, max_age=max_age)
    q = q.select("ticker", "t", "mid", "bid", "ask", "bid_at", "ask_at", "quote_known_at").unique(["ticker", "t"])
    x = (x.join(q, on=["ticker", "t"], how="left")
         .with_columns(pl.when(pl.col("flip")).then(100 - pl.col("mid")).otherwise(pl.col("mid")).alias("p_ge_k")))
    win = (x.filter(pl.col("kind") == "winner")
           .group_by("game_id", "t").agg(
               pl.col("p_ge_k").mean().alias("p_home"),
               (pl.col("ask") - pl.col("bid")).filter(pl.col("mid").is_not_null()).min().alias("spread"),
               pl.max_horizontal(pl.col("t") - pl.col("bid_at"), pl.col("t") - pl.col("ask_at"))
               .filter(pl.col("mid").is_not_null()).min().alias("quote_age"),
               pl.col("quote_known_at").filter(pl.col("mid").is_not_null()).max().alias("p_known_at")))
    rows = []
    lad = x.filter(pl.col("p_ge_k").is_not_null())
    for (gid, t), g in lad.group_by("game_id", "t"):
        g = g.group_by("k").agg(pl.col("p_ge_k").mean(), pl.col("quote_known_at").max())   # dedupe k
        mu, sigma = fit_wins_ladder(g["k"].to_numpy().astype(float), g["p_ge_k"].to_numpy() / 100)
        rows.append({"game_id": gid, "t": t, "mu_home": mu, "sigma": sigma, "legs": g.height,
                     "mu_known_at": g["quote_known_at"].max()})
    mu = pl.DataFrame(rows, schema={"game_id": pl.String, "t": at.schema["t"], "mu_home": pl.Float64,
                                    "sigma": pl.Float64, "legs": pl.Int64, "mu_known_at": pl.Datetime("us", "UTC")})
    return (at.join(win, on=["game_id", "t"], how="left").join(mu, on=["game_id", "t"], how="left")
            .with_columns(pl.col("mu_home").fill_nan(None), pl.col("sigma").fill_nan(None),
                          pl.when(pl.col("mu_home").is_null()).then(None).otherwise(pl.col("mu_known_at"))
                          .alias("mu_known_at")))


def night_shocks(data: SportsData, league: str, season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(per team per night: z, ms, ms_from_price, res_t; per night: t_pre). z is the result surprise;
    ms the margin surprise, actual margin minus the expected margin at the game's first play:

        from the ladder   the winner + spread ladder's implied margin (graph.fit_wins_ladder)
        from the price    where no ladder fits (MLB 2025 had run lines on few games):
                          sigma * Phi^-1(p_pre), sigma the spread of margins in the league-season's
                          games finished before this one started (at least MIN_SIGMA_GAMES)

    res_t is when the team's last result is known."""
    games = (data.contracts.filter(pl.col("league") == league, pl.col("kind") == "winner", pl.col("game_id").is_not_null())
             .group_by("game_id").agg(pl.col("home").first()))
    res = data.results.select("game_id", "home_score", "away_score")
    sh = (game_shocks(data, league, season).filter(pl.col("game_type").is_in(REGULAR[league]))
          .with_columns(pl.col("first_play").dt.convert_time_zone("America/New_York").dt.date().alias("night")))
    pre = game_values(data, sh.select("game_id", pl.col("first_play").alias("t")).unique())
    sh = (sh.join(pre.select("game_id", pl.col("t").alias("first_play"), pl.col("mu_home").alias("mu_pre_game")),
                  on=["game_id", "first_play"], how="left")
          .join(games, on="game_id", how="left").join(res, on="game_id", how="left")
          .with_columns(pl.when(pl.col("team") == pl.col("home"))
                        .then(pl.col("home_score") - pl.col("away_score"))
                        .otherwise(pl.col("away_score") - pl.col("home_score")).alias("margin"),
                        pl.when(pl.col("team") == pl.col("home")).then(pl.col("mu_pre_game"))
                        .otherwise(-pl.col("mu_pre_game")).alias("mu_team"))
          )
    # the spread of margins so far (home side, one row per game), from games finished before each start
    hm = (sh.filter(pl.col("team") == pl.col("home")).select("game_id", "known_at", "first_play", "margin")
          .unique("game_id").drop_nulls("margin").sort("known_at")
          .with_columns(pl.int_range(1, pl.len() + 1).alias("n"), pl.col("margin").cum_sum().alias("s1"),
                        (pl.col("margin") ** 2).cum_sum().alias("s2")))
    hm = hm.with_columns(((pl.col("s2") - pl.col("s1") ** 2 / pl.col("n")) / (pl.col("n") - 1)).sqrt().alias("sigma"))
    sig = (sh.select("game_id", "first_play").unique("game_id").sort("first_play")
           .join_asof(hm.select(pl.col("known_at").alias("k"), "n", "sigma").sort("k"), left_on="first_play",
                      right_on="k", strategy="backward", allow_exact_matches=False)
           .with_columns(pl.when(pl.col("n") >= MIN_SIGMA_GAMES).then(pl.col("sigma")).alias("sigma"))
           .select("game_id", "sigma"))
    sh = (sh.join(sig, on="game_id", how="left")
          .with_columns(pl.Series("_q", norm.ppf(sh["p_pre"].clip(0.01, 0.99).to_numpy())).alias("_q"))
          .with_columns(pl.col("mu_team").is_null().alias("ms_from_price"),
                        pl.coalesce(pl.col("mu_team"), pl.col("sigma") * pl.col("_q")).alias("mu_team"))
          .with_columns((pl.col("margin") - pl.col("mu_team")).alias("ms")).drop("_q"))
    tonight = sh.group_by("night", "team").agg(pl.col("z").sum(), pl.col("ms").sum(),
                                               pl.col("ms").is_null().any().alias("ms_missing"),
                                               pl.col("ms_from_price").any(),
                                               pl.col("known_at").max().alias("res_t"))
    tonight = tonight.with_columns(pl.when(pl.col("ms_missing")).then(None).otherwise(pl.col("ms")).alias("ms")).drop("ms_missing")
    nights = sh.group_by("night").agg(pl.col("first_play").min().alias("t_pre"))
    return tonight, nights


def nextgame_table(data: SportsData, league: str, season: int, horizons: dict = HORIZONS,
                   decision_delay: dt.timedelta = DECISION_DELAY) -> pl.DataFrame:
    games = (data.contracts.filter(pl.col("league") == league, pl.col("season") == season, pl.col("kind") == "winner",
                                   pl.col("game_type").is_in(REGULAR[league]), pl.col("game_id").is_not_null())
             .group_by("game_id").agg(pl.col("home").first(), pl.col("away").first(),
                                      pl.col("sched_start").first(), pl.col("open_time").min().alias("listed")))
    res = data.results.select("game_id", "first_play", "home_score", "away_score", pl.col("known_at").alias("res_known"))
    games = games.join(res, on="game_id", how="left")

    tonight, nights = night_shocks(data, league, season)

    # each team's next game after each night it played
    tg = pl.concat([games.select("game_id", "sched_start", pl.col("home").alias("team")),
                    games.select("game_id", "sched_start", pl.col("away").alias("team"))])
    played = tonight.join(nights, on="night").select("night", "team", "t_pre", "res_t")
    nxt = (played.join(tg, on="team").filter(pl.col("sched_start") > pl.col("res_t"))
           .sort("sched_start").group_by("night", "team").agg(pl.col("game_id").first()))
    rows = nxt.select("night", "game_id").unique().join(games, on="game_id").join(nights, on="night")
    for side in ("home", "away"):
        rows = (rows.join(tonight.rename({"team": side, "z": f"z_{side}", "ms": f"ms_{side}", "res_t": f"res_{side}"}),
                          on=["night", side], how="left")
                .with_columns(pl.col(f"z_{side}").is_not_null().alias(f"played_{side}"),
                              pl.col(f"z_{side}").fill_null(0.0)))
    rows = (rows.with_columns((pl.max_horizontal("res_home", "res_away") + decision_delay).alias("t"))
            .filter(pl.col("listed") <= pl.col("t_pre"),            # trading before the night: a move to measure
                    pl.col("sched_start") > pl.col("t"))
            .with_columns(((pl.col("sched_start") - pl.col("t")).dt.total_hours() / 24).alias("days_to_game")))

    def vals(t_col: str, sfx: str) -> pl.DataFrame:
        v = game_values(data, rows.select("game_id", pl.col(t_col).alias("t")).unique())
        return v.select("game_id", pl.col("t").alias(t_col), pl.col("p_home").alias(f"p_{sfx}"),
                        pl.col("mu_home").alias(f"mu_{sfx}"), pl.col("spread").alias(f"spread_{sfx}"),
                        pl.col("p_known_at").alias(f"p_{sfx}_known_at"), pl.col("mu_known_at").alias(f"mu_{sfx}_known_at"))

    rows = (rows.join(vals("t_pre", "pre"), on=["game_id", "t_pre"], how="left")
            .join(vals("t", "t"), on=["game_id", "t"], how="left")
            .with_columns((pl.col("p_t") - pl.col("p_pre")).alias("moved_p"),
                          (pl.col("mu_t") - pl.col("mu_pre")).alias("moved_mu"),
                          pl.col("first_play").fill_null(pl.col("sched_start")).alias("t_close")))
    labels = {h: pl.col("t") + dh for h, dh in horizons.items()}
    labels["close"] = pl.col("t_close")
    for h, e in labels.items():
        rows = rows.with_columns(e.alias(f"label_end_{h}"))
        v = vals(f"label_end_{h}", h).select("game_id", f"label_end_{h}", f"p_{h}", f"mu_{h}")
        past = pl.col(f"label_end_{h}") > pl.col("t_close")
        rows = (rows.join(v.unique(["game_id", f"label_end_{h}"]), on=["game_id", f"label_end_{h}"], how="left")
                .with_columns(*[pl.when(past).then(None).otherwise(pl.col(f"{x}_{h}")).alias(f"{x}_{h}")
                                for x in ("p", "mu")])
                .with_columns((pl.col(f"p_{h}") - pl.col("p_pre")).alias(f"total_p_{h}"),
                              (pl.col(f"p_{h}") - pl.col("p_t")).alias(f"rem_p_{h}"),
                              (pl.col(f"mu_{h}") - pl.col("mu_pre")).alias(f"total_mu_{h}"),
                              (pl.col(f"mu_{h}") - pl.col("mu_t")).alias(f"rem_mu_{h}"))
                .drop(f"p_{h}", f"mu_{h}"))
    rows = rows.with_columns(
        pl.when(pl.col("home_score") > pl.col("away_score")).then(1.0)
        .when(pl.col("home_score") < pl.col("away_score")).then(0.0)
        .when(pl.col("home_score").is_not_null()).then(0.5).alias("y_home"),
        pl.col("res_known").alias("label_end_y"))
    return (rows.with_columns(pl.lit(league).alias("league"), pl.lit(season).alias("season"))
            .select("league", "season", "night", "game_id", "home", "away", "t_pre", "t", "t_close", "days_to_game",
                    "played_home", "played_away", "z_home", "z_away", "ms_home", "ms_away",
                    "p_pre", "mu_pre", "spread_pre", "p_pre_known_at", "mu_pre_known_at",
                    "p_t", "mu_t", "spread_t", "p_t_known_at", "mu_t_known_at", "moved_p", "moved_mu",
                    *[c for h in labels for c in (f"total_p_{h}", f"rem_p_{h}", f"total_mu_{h}", f"rem_mu_{h}",
                                                  f"label_end_{h}")],
                    "y_home", "label_end_y")
            .sort("night", "game_id"))
