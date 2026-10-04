"""The edge table (reports/sports_research_plan.md §8), at contract level.

Columns: ``src``, ``dst``, ``edge_type``, ``group``, ``sign_prior`` (+1 / -1 /
null = unknown), ``strength`` (exact / learned), ``valid_from``, ``valid_to``,
``known_at``, ``provenance``. Always cause -> effect, as in ``agcrn.py``'s
``[i, j]`` = influence of i on j.

    exact           complement (-), ladder (+ along a team's rungs, - across
                    the two teams' lowest rungs), part_of_whole (+: a half /
                    first-5 / team total into the full game; a playoff game
                    into its series)
    competition     result_own (+: a team's game winner into its futures),
                    result_opp (-: into the opponent's futures), rival (-:
                    legs of one exclusive futures event), hierarchy (+:
                    division -> playoffs -> conference -> champ, series ->
                    conference / champ, win total -> playoffs)
    participation   plays_in (a game winner leg -> ``team:<league>:<code>``),
                    same_slate (winner legs of one league's games on one date)
    statistical     comovement (:func:`comovement_edges`, trailing window)

``known_at`` is when both ends are listed (the later ``open_time``);
``valid_from`` the same, ``valid_to`` when the first end closes. ``valid_to``
comes from the final ``close_time``: test an edge as active at t with
``valid_from <= t < valid_to``, which only needs to know whether the close has
happened by t. Not yet built: bracket edges (need the playoff bracket), and
player-of / co-attention / semantic edges.

Team-level graphs come from :func:`team_kind_graph`.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from stg.sports.asof import SportsData
from stg.sports.outcomes import outcome_table

EDGE_SCHEMA = {"src": pl.String, "dst": pl.String, "edge_type": pl.String, "group": pl.String,
               "sign_prior": pl.Int8, "strength": pl.String, "valid_from": pl.Datetime("us", "UTC"),
               "valid_to": pl.Datetime("us", "UTC"), "known_at": pl.Datetime("us", "UTC"),
               "provenance": pl.String}
GROUP = {"complement": "exact", "ladder": "exact", "part_of_whole": "exact", "result_own": "competition",
         "result_opp": "competition", "rival": "competition", "hierarchy": "competition",
         "plays_in": "participation", "same_slate": "participation", "comovement": "statistical"}
FUTURES = ("champ", "conference", "division", "playoffs", "wins", "series")
HIER = [("division", "playoffs"), ("playoffs", "conference"), ("conference", "champ"),
        ("division", "conference"), ("series", "conference"), ("series", "champ"), ("wins", "playoffs")]
PARTS = {"winner_1h": "winner", "winner_2h": "winner", "winner_f5": "winner", "team_total": "total"}


def conference_map(contracts: pl.DataFrame) -> pl.DataFrame:
    """league, season, team -> conf (AL / NL, EAST / WEST, AFC / NFC) and
    ``conf_known_at``, read from the conference futures the team is listed in.

    For season s: the majority over the listings of seasons <= s, since Kalshi
    occasionally lists a team in both conferences (DAL in KXNBAEAST-26); ties
    give no row. ``conf_known_at`` is the latest listing counted, so nothing
    listed later is used. The series names change between seasons (KXAFC,
    KXNFLAFCCHAMP), hence the label. No final fields are read.
    """
    c = (contracts.filter(pl.col("kind") == "conference", pl.col("team").is_not_null(),
                          pl.col("season").is_not_null())
         .with_columns(pl.col("series_ticker").str.extract(r"(AFC|NFC|EAST|WEST|AL|NL)").alias("conf"))
         .filter(pl.col("conf").is_not_null())
         .select("league", "season", "team", "conf", "event_ticker", "open_time"))
    seasons = c.select("league", pl.col("season").alias("as_of_season")).unique()
    upto = (c.join(seasons, on="league").filter(pl.col("season") <= pl.col("as_of_season"))
            .group_by("league", "as_of_season", "team", "conf")
            .agg(pl.col("event_ticker").n_unique().alias("n"), pl.col("open_time").max().alias("conf_known_at")))
    k = ["league", "as_of_season", "team"]
    # every listing counted decides the majority, so the latest of them all dates it
    upto = upto.with_columns(pl.col("conf_known_at").max().over(k))
    top = upto.filter(pl.col("n") == pl.col("n").max().over(k))
    return (top.filter(pl.len().over(k) == 1)
            .select("league", pl.col("as_of_season").alias("season"), "team", "conf", "conf_known_at"))


def _pairs(a: pl.DataFrame, b: pl.DataFrame, on: list[str], etype: str, sign: int | None, strength: str,
           prov: str, both: bool = False) -> pl.DataFrame:
    x = a.join(b, on=on, suffix="_d").filter(pl.col("ticker") != pl.col("ticker_d"))
    e = x.select(pl.col("ticker").alias("src"), pl.col("ticker_d").alias("dst"),
                 pl.max_horizontal("open_time", "open_time_d").alias("known_at"),
                 pl.min_horizontal("close_time", "close_time_d").alias("valid_to"))
    if both:
        e = pl.concat([e, e.select(pl.col("dst").alias("src"), pl.col("src").alias("dst"), "known_at", "valid_to")])
    return e.with_columns(pl.lit(etype).alias("edge_type"), pl.lit(GROUP[etype]).alias("group"),
                          pl.lit(sign, pl.Int8).alias("sign_prior"), pl.lit(strength).alias("strength"),
                          pl.col("known_at").alias("valid_from"), pl.lit(prov).alias("provenance"))


def edge_table(data: SportsData) -> pl.DataFrame:
    oc = outcome_table(data.contracts, with_labels=False)
    close = data.finals.select("ticker", "close_time")
    n = (oc.join(data.contracts.select("ticker", "open_time", "sched_start", "opp", "event_ticker"),
                 on=["ticker", "event_ticker"])
         .join(close, on="ticker", how="left"))
    cols = ["ticker", "open_time", "close_time"]
    out = []

    # exact ------------------------------------------------------------------
    pair = n.filter(pl.col("outcome_type") == "pair")
    out.append(_pairs(pair.select("outcome_id", *cols), pair.select("outcome_id", *cols), ["outcome_id"],
                      "complement", -1, "exact", "event: two exclusive legs"))
    lad = n.filter(pl.col("outcome_type") == "ladder", pl.col("strike").is_not_null()).sort("strike")
    lad = lad.with_columns(pl.col("ticker").shift(-1).over("outcome_id", "team").alias("_next"))
    nxt = lad.filter(pl.col("_next").is_not_null()).select("ticker", "_next", "open_time", "close_time")
    nxt2 = nxt.join(lad.select(pl.col("ticker").alias("_next"), pl.col("open_time").alias("open_time_d"),
                               pl.col("close_time").alias("close_time_d")), on="_next")
    e = nxt2.select(pl.col("ticker").alias("src"), pl.col("_next").alias("dst"),
                    pl.max_horizontal("open_time", "open_time_d").alias("known_at"),
                    pl.min_horizontal("close_time", "close_time_d").alias("valid_to"))
    e = pl.concat([e, e.select(pl.col("dst").alias("src"), pl.col("src").alias("dst"), "known_at", "valid_to")])
    out.append(e.with_columns(pl.lit("ladder").alias("edge_type"), pl.lit("exact").alias("group"),
                              pl.lit(1, pl.Int8).alias("sign_prior"), pl.lit("exact").alias("strength"),
                              pl.col("known_at").alias("valid_from"), pl.lit("adjacent strikes").alias("provenance")))
    low = (lad.filter(pl.col("team").is_not_null()).group_by("outcome_id", "team")
           .agg(pl.col("ticker").first(), pl.col("open_time").first(), pl.col("close_time").first()))
    out.append(_pairs(low, low, ["outcome_id"], "ladder", -1, "exact", "two teams' lowest spread rungs"))
    for part, whole in PARTS.items():
        a = n.filter(pl.col("kind") == part).select("game_id", "team", "strike", *cols)
        b = n.filter(pl.col("kind") == whole).select("game_id", "team", "strike", *cols)
        if whole == "winner":
            out.append(_pairs(a.drop("strike"), b.drop("strike"), ["game_id", "team"], "part_of_whole", 1,
                              "learned", f"{part} -> {whole}"))
        else:   # a team over x implies the game total over every y <= x
            x = a.drop("team").join(b.drop("team"), on="game_id", suffix="_d").filter(
                pl.col("strike_d") <= pl.col("strike"))
            out.append(_pairs(x.select(*cols, pl.format("{}|{}", "ticker", "ticker_d").alias("_k")),
                              x.select(pl.col("ticker_d").alias("ticker"), pl.col("open_time_d").alias("open_time"),
                                       pl.col("close_time_d").alias("close_time"),
                                       pl.format("{}|{}", "ticker", "ticker_d").alias("_k")),
                              ["_k"], "part_of_whole", 1, "exact", f"{part} over x -> {whole} over y <= x"))
    # playoff games -> their series (same team, game scheduled while the series is listed)
    gw = n.filter(pl.col("kind") == "winner", pl.col("team").is_not_null())
    ser = n.filter(pl.col("kind") == "series", pl.col("team").is_not_null())
    x = gw.join(ser, on=["league", "team"], suffix="_d").filter(
        pl.col("sched_start") >= pl.col("open_time_d"), pl.col("sched_start") <= pl.col("close_time_d"))
    out.append(_pairs(x.select("ticker", "open_time", "close_time", pl.col("ticker_d").alias("_k")),
                      x.select(pl.col("ticker_d").alias("ticker"), pl.col("open_time_d").alias("open_time"),
                               pl.col("close_time_d").alias("close_time"), pl.col("ticker_d").alias("_k")).unique(),
                      ["_k"], "part_of_whole", 1, "learned", "playoff game -> series"))

    # competition ------------------------------------------------------------
    fut = n.filter(pl.col("kind").is_in(FUTURES), pl.col("team").is_not_null()).select(
        "league", "season", "team", "kind", *cols)
    g = gw.select("league", "season", "team", "opp", *cols)
    fut_r = fut.filter(pl.col("kind") != "series")          # games -> series are part_of_whole
    own = g.join(fut_r, on=["league", "season", "team"], suffix="_d").filter(
        pl.col("open_time") < pl.col("close_time_d"), pl.col("open_time_d") < pl.col("close_time"))
    opp = g.join(fut_r, left_on=["league", "season", "opp"], right_on=["league", "season", "team"], suffix="_d").filter(
        pl.col("open_time") < pl.col("close_time_d"), pl.col("open_time_d") < pl.col("close_time"))
    for x, et, sgn, prov in ((own, "result_own", 1, "team's game -> its futures"),
                             (opp, "result_opp", -1, "team's game -> opponent's futures")):
        out.append(x.select(pl.col("ticker").alias("src"), pl.col("ticker_d").alias("dst"),
                            pl.max_horizontal("open_time", "open_time_d").alias("known_at"),
                            pl.min_horizontal("close_time", "close_time_d").alias("valid_to"))
                   .with_columns(pl.lit(et).alias("edge_type"), pl.lit("competition").alias("group"),
                                 pl.lit(sgn, pl.Int8).alias("sign_prior"), pl.lit("learned").alias("strength"),
                                 pl.col("known_at").alias("valid_from"), pl.lit(prov).alias("provenance")))
    excl = n.filter(pl.col("kind").is_in(["champ", "conference", "division"]))
    out.append(_pairs(excl.select("event_ticker", *cols), excl.select("event_ticker", *cols), ["event_ticker"],
                      "rival", -1, "learned", "legs of one futures event"))
    # the playoffs event is league-wide but not exclusive: only teams of one conference compete for its spots
    po = (n.filter(pl.col("kind") == "playoffs", pl.col("team").is_not_null())
          .join(conference_map(data.contracts), on=["league", "season", "team"], how="inner")
          # the pair is known once both legs are listed and both teams' conferences are
          .with_columns(pl.max_horizontal("open_time", "conf_known_at").alias("open_time"))
          .select("event_ticker", "conf", *cols))
    out.append(_pairs(po, po, ["event_ticker", "conf"], "rival", -1, "learned",
                      "playoffs: same conference, competing for its spots"))
    for lo, hi in HIER:
        a = fut.filter(pl.col("kind") == lo).drop("kind")
        b = fut.filter(pl.col("kind") == hi).drop("kind")
        out.append(_pairs(a, b, ["league", "season", "team"], "hierarchy", 1, "learned", f"{lo} -> {hi}"))

    # participation ------------------------------------------------------------
    out.append(gw.select(pl.col("ticker").alias("src"),
                         pl.format("team:{}:{}", "league", "team").alias("dst"),
                         pl.col("open_time").alias("known_at"), pl.col("close_time").alias("valid_to"))
               .with_columns(pl.lit("plays_in").alias("edge_type"), pl.lit("participation").alias("group"),
                             pl.lit(None, pl.Int8).alias("sign_prior"), pl.lit("exact").alias("strength"),
                             pl.col("known_at").alias("valid_from"), pl.lit("ticker").alias("provenance")))
    refw = gw.filter(pl.col("orient") == 1).with_columns(
        pl.col("sched_start").dt.convert_time_zone("America/New_York").dt.date().alias("slate"))
    out.append(_pairs(refw.select("league", "slate", *cols), refw.select("league", "slate", *cols),
                      ["league", "slate"], "same_slate", None, "learned", "schedule: same league, same date"))

    e = pl.concat([x.select(list(EDGE_SCHEMA)).cast(EDGE_SCHEMA) for x in out])
    return e.unique(["src", "dst", "edge_type", "provenance"]).sort("edge_type", "src", "dst")


def active(edges: pl.DataFrame, t: dt.datetime) -> pl.DataFrame:
    """Edges known and live at t."""
    tt = pl.lit(t, dtype=pl.Datetime("us", "UTC"))
    return edges.filter(pl.col("known_at") <= tt, pl.col("valid_from") <= tt,
                        pl.col("valid_to").is_null() | (pl.col("valid_to") > tt))


def comovement_edges(data: SportsData, tickers: list[str], end: dt.datetime,
                     window: dt.timedelta = dt.timedelta(days=30), every: str = "1h",
                     min_obs: int = 50, min_abs: float = 0.2) -> pl.DataFrame:
    """Correlation of mid log-odds changes over [end - window, end], known at ``end``."""
    grid = pl.datetime_range(end - window, end, every, time_zone="UTC", eager=True).alias("t")
    at = pl.DataFrame({"ticker": tickers}).join(pl.DataFrame(grid), how="cross")
    q = data.quotes_at(at, max_age=dt.timedelta(hours=6))
    wide = (q.select("ticker", "t", "mid_lo").pivot(on="ticker", index="t", values="mid_lo").sort("t"))
    names = [c for c in wide.columns if c != "t"]
    d = np.diff(wide.select(names).to_numpy().astype(float), axis=0)
    rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            ok = np.isfinite(d[:, i]) & np.isfinite(d[:, j]) & ((d[:, i] != 0) | (d[:, j] != 0))
            if ok.sum() < min_obs:
                continue
            r = np.corrcoef(d[ok, i], d[ok, j])[0, 1]
            if np.isfinite(r) and abs(r) >= min_abs:
                rows += [(a, b, r), (b, a, r)]
    e = pl.DataFrame(rows, schema=["src", "dst", "rho"], orient="row")
    return e.with_columns(
        pl.lit("comovement").alias("edge_type"), pl.lit("statistical").alias("group"),
        pl.col("rho").sign().cast(pl.Int8).alias("sign_prior"), pl.lit("learned").alias("strength"),
        pl.lit(end - window, dtype=pl.Datetime("us", "UTC")).alias("valid_from"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("valid_to"),
        pl.lit(end, dtype=pl.Datetime("us", "UTC")).alias("known_at"),
        pl.lit(f"corr of {every} mid log-odds changes, {window.days}d").alias("provenance"))


def team_kind_graph(edges: pl.DataFrame, contracts: pl.DataFrame, edge_types=("rival", "hierarchy"),
                    league: str | None = None, season: int | None = None) -> pl.DataFrame:
    """Collapse contract edges onto ``<team>-<kind>`` nodes (the between-game model's nodes)."""
    node = contracts.filter(pl.col("kind").is_in(FUTURES), pl.col("team").is_not_null())
    if league:
        node = node.filter(pl.col("league") == league)
    if season:
        node = node.filter(pl.col("season") == season)
    node = node.select("ticker", pl.format("{}-{}", "team", "kind").alias("node"))
    e = (edges.filter(pl.col("edge_type").is_in(edge_types))
         .join(node.rename({"ticker": "src", "node": "src_node"}), on="src")
         .join(node.rename({"ticker": "dst", "node": "dst_node"}), on="dst"))
    return (e.group_by("src_node", "dst_node", "edge_type")
            .agg(pl.col("sign_prior").first(), pl.col("known_at").min(), pl.len().alias("n_contract_edges")))
