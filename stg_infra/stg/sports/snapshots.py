"""Pre-game snapshots: each game outcome's price 24 h, 1 h and 0 h before the start.

The anchor is the actual start (the first pitch / first play), per the plan;
``anchor="scheduled"`` uses the schedule instead. The actual start is only
known once it happens, so the 24 h / 1 h rows use a time chosen with a little
hindsight (how late the game started); the 0 h row does not.

One row per (outcome, rung, offset). Pairs are merged onto their reference leg
(:mod:`stg.sports.outcomes`): ``mid`` is the mean of the reference leg's mid
and 100 minus the other leg's, when both are valid (else whichever is), and
the executable prices are the better of the two routes (buy YES on the
reference at its ask, or buy NO through the other leg's YES).

Labels: ``y`` (did the reference leg settle YES) and ``label_end`` (its
settlement), from the as-of finals, so a build with a cutoff has them only for
outcomes settled by then.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from stg.sports.asof import SportsData
from stg.sports.outcomes import outcome_table

OFFSETS = (dt.timedelta(hours=24), dt.timedelta(hours=1), dt.timedelta(0))
CORE_KINDS = ("winner", "spread", "total", "team_total", "winner_1h", "spread_1h", "total_1h",
              "winner_f5", "spread_f5", "total_f5")


def pregame_snapshots(data: SportsData, offsets=OFFSETS, kinds=CORE_KINDS, anchor: str = "actual",
                      leagues=("MLB", "NBA", "NFL")) -> pl.DataFrame:
    oc = outcome_table(data.contracts, with_labels=False).filter(
        pl.col("scope") == "game", pl.col("kind").is_in(kinds), pl.col("league").is_in(leagues),
        pl.col("game_id").is_not_null())
    starts = data.results.select("league", "game_id", "first_play").drop_nulls()
    sched = data.contracts.select("ticker", "sched_start")
    oc = oc.join(starts, on=["league", "game_id"], how="inner").join(sched, on="ticker", how="left")
    oc = oc.with_columns((pl.col("first_play") if anchor == "actual" else pl.col("sched_start")).alias("anchor"))

    # reference legs (pairs: the ref leg; everything else: each leg) and their complements
    ref = oc.filter(pl.col("orient").is_null() | (pl.col("orient") == 1))
    comp = oc.filter(pl.col("orient") == -1).select("outcome_id", pl.col("ticker").alias("comp_ticker"))
    ref = ref.join(comp, on="outcome_id", how="left")
    snaps = pl.concat([ref.with_columns(pl.lit(off.total_seconds() / 3600).alias("offset_h"),
                                        (pl.col("anchor") - pl.lit(off)).alias("t")) for off in offsets])
    snaps = snaps.filter(pl.col("t").is_not_null())
    if data.cutoff is not None:
        snaps = snaps.filter(pl.col("t") <= pl.lit(data.cutoff, dtype=pl.Datetime("us", "UTC")))

    qr = data.quotes_at(snaps.select("ticker", "t"))
    has_c = snaps.filter(pl.col("comp_ticker").is_not_null())
    qc = data.quotes_at(has_c.select(pl.col("comp_ticker").alias("ticker"), "t"))
    snaps = snaps.with_columns(qr.select(
        pl.col("bid").alias("ref_bid"), pl.col("ask").alias("ref_ask"), pl.col("mid").alias("ref_mid"),
        pl.col("quote_known_at").alias("ref_known_at")))
    comp_cols = has_c.select("ticker", "t").with_columns(qc.select(
        pl.col("bid").alias("comp_bid"), pl.col("ask").alias("comp_ask"), pl.col("mid").alias("comp_mid"),
        pl.col("quote_known_at").alias("comp_known_at")))
    snaps = snaps.join(comp_cols.unique(["ticker", "t"]), on=["ticker", "t"], how="left")
    snaps = snaps.with_columns(
        pl.when(pl.col("ref_mid").is_not_null() & pl.col("comp_mid").is_not_null())
        .then((pl.col("ref_mid") + 100 - pl.col("comp_mid")) / 2)
        .otherwise(pl.coalesce("ref_mid", 100 - pl.col("comp_mid"))).alias("mid"),
        pl.min_horizontal("ref_ask", 100 - pl.col("comp_bid")).alias("buy_yes"),
        pl.max_horizontal("ref_bid", 100 - pl.col("comp_ask")).alias("sell_yes"),
        pl.max_horizontal("ref_known_at", "comp_known_at").alias("quote_known_at"))

    f = data.finals.select("ticker", pl.col("result"), pl.col("known_at").alias("label_end"))
    snaps = snaps.join(f, on="ticker", how="left").with_columns(
        pl.when(pl.col("result") == "yes").then(1.0).when(pl.col("result") == "no").then(0.0).alias("y")
    ).drop("result")
    return snaps.select(
        "league", "season", "game_id", "outcome_id", "outcome_type", "kind", "ticker", "comp_ticker",
        "team", "side", "strike", "rung", "offset_h", "anchor", "t", "mid", "ref_mid", "comp_mid",
        "ref_bid", "ref_ask", "buy_yes", "sell_yes", "quote_known_at", "y", "label_end",
    ).sort("league", "game_id", "outcome_id", "ticker", "offset_h")
