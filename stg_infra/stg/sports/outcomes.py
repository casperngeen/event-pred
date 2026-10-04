"""The outcome table: which markets are one outcome, and how each settled.

Rule 4 of the plan: count each outcome once. An outcome is one Kalshi event
(``outcome_id = event_ticker``), typed from its structure, not its result:

    pair         two exclusive legs (a winner, a playoff series): complements,
                 merged onto the reference leg (the home team's, else the
                 first alphabetically); ``orient`` = +1 on it, -1 on the other,
                 so the merged price is mean(p_ref, 100 - p_other)
    exclusive    three or more exclusive legs (a half with a tie leg, a
                 championship, a division)
    ladder       strikes on one quantity (spread, total, team total, win
                 total): one outcome, ``rung`` orders the strikes
    multi        several legs can settle YES (playoff qualifiers)
    single       everything else, one leg per outcome as far as we know (props)

Labels (``y``, ``expiration_value``) come from the listing's final fields and
carry ``label_known_at`` = settlement; they are for labels only.
"""
from __future__ import annotations

import polars as pl

from stg.sports.contracts import contracts
from stg.sports.loaders import load_listing_final

EXCLUSIVE = ("winner", "winner_1h", "winner_2h", "winner_f5", "series", "champ", "conference", "division")
LADDER = ("spread", "spread_1h", "spread_2h", "spread_f5", "total", "total_1h", "total_2h", "total_f5",
          "team_total", "wins")
MULTI = ("playoffs",)


def outcome_table(c: pl.DataFrame | None = None, with_labels: bool = True) -> pl.DataFrame:
    c = contracts() if c is None else c
    n = pl.len().over("event_ticker")
    typ = (pl.when(pl.col("kind").is_in(EXCLUSIVE) & (n == 2)).then(pl.lit("pair"))
           .when(pl.col("kind").is_in(EXCLUSIVE)).then(pl.lit("exclusive"))
           .when(pl.col("kind").is_in(LADDER)).then(pl.lit("ladder"))
           .when(pl.col("kind").is_in(MULTI)).then(pl.lit("multi"))
           .otherwise(pl.lit("single")))
    out = c.select("ticker", "event_ticker", "league", "kind", "scope", "team", "side", "strike",
                   "game_id", "season").with_columns(
        pl.col("event_ticker").alias("outcome_id"), typ.alias("outcome_type"), n.alias("n_legs"))
    # pair: the reference leg is the home team's, else the first ticker
    ref = (out.filter(pl.col("outcome_type") == "pair")
           .sort(pl.col("side") != "home", "ticker")
           .group_by("outcome_id", maintain_order=True).agg(pl.col("ticker").first().alias("ref_ticker")))
    out = out.join(ref, on="outcome_id", how="left").with_columns(
        pl.when(pl.col("ref_ticker").is_null()).then(None)
        .when(pl.col("ticker") == pl.col("ref_ticker")).then(1).otherwise(-1).cast(pl.Int8).alias("orient"),
        # ladders: signed strike so one ordering covers both teams' spreads (margin of the home side)
        pl.when(pl.col("outcome_type") == "ladder").then(
            pl.when(pl.col("side") == "away").then(-pl.col("strike")).otherwise(pl.col("strike"))
        ).alias("ladder_x"))
    out = out.with_columns(pl.col("ladder_x").rank("dense").over("outcome_id", "team").cast(pl.Int32).alias("rung"))
    if with_labels:
        f = load_listing_final().select(
            "ticker", pl.col("result"), "expiration_value", pl.col("known_at").alias("label_known_at"))
        out = out.join(f, on="ticker", how="left").with_columns(
            pl.when(pl.col("result") == "yes").then(1.0).when(pl.col("result") == "no").then(0.0)
            .otherwise(None).alias("y"))
    return out


def game_weights(out: pl.DataFrame) -> pl.DataFrame:
    """Rule 4: weight 1 per game, split equally over its outcomes (season markets: 1 per outcome)."""
    o = out.unique("outcome_id").with_columns(pl.coalesce("game_id", "outcome_id").alias("_g"))
    return o.select("outcome_id", (1.0 / pl.len().over("_g")).alias("weight"))
