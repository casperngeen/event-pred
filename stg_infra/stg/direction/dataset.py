"""Stage-2 dataset: one row per (trigger event -> next target event) match.

The AGCRN study (``reports/agcrn_study.md``) failed on a *target/horizon
mismatch*, not on architecture: the label was a magnitude regression of a
weeks-ahead belief revision, measured after the effect had decayed. Its
post-mortem prescribes the replacement — predict at the **dormant horizon**
(trigger resolution -> the target's 3rd subsequent trade, median ~0.5 h) and
as a **direction classifier**, because every magnitude-weighted test in the
in-sample record is null while sign/rank statistics survive
(``research_log.md`` §1-2).

This module builds the panel that task needs. It is the same grid Stage-1
searched (``stg.structure.estimator``) and the same response window
(``stg.panel.targets.response_panel``), reshaped from "one Spearman per pair"
into "one labelled observation per matched event pair", so that a learner can
be fit on the past and scored on the future.

Schema (one row per matched pair)::

    trigger target side trigger_event target_event
    t0                  trigger resolution time -- the walk-forward clock
    surprise implied_std implied_entropy coverage
    z_surprise          surprise / implied_std (unit-free, no fold leakage)
    gap_days days_to_close p0
    response y          target's dormant move, and its sign (the label)
    same_release

``y`` is +1/-1; zero-response rows are dropped (a flat 3rd print carries no
direction). Everything is in-sample only -- inputs come from the IS surprise
panel and ``scan_trades(is_only=True)``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import is_same_release, universe
from stg.panel.surprise import usable_triggers
from stg.panel.targets import representative_tickers, response_panel

FEDDECISION_SIDES = ("hike", "cut")

# carried from the surprise panel onto each matched pair
_TRIGGER_COLS = ("close_time", "implied_std", "implied_entropy", "coverage")


def build_pair_panel(
    surprise_panel: pl.DataFrame,
    *,
    horizon: str = "dormant",
    targets: Optional[list[str]] = None,
    min_n: int = 10,
    min_pair_n: int = 10,
    markets: Optional[pl.DataFrame] = None,
    trades: Optional[pl.LazyFrame] = None,
) -> pl.DataFrame:
    """Long panel of labelled (trigger event, target event) observations.

    ``min_n`` gates which series count as triggers (as Stage-1 does);
    ``min_pair_n`` drops ordered pairs with too few matches to be estimable in
    any fold.
    """
    mk = markets if markets is not None else load_markets(is_only=True)
    tr = trades if trades is not None else scan_trades(is_only=True)
    triggers = usable_triggers(surprise_panel, min_n)
    tgts = targets if targets is not None else universe(5, mk)

    rep_cache: dict[tuple[str, str], pl.DataFrame] = {}

    def reps(t: str, side: str) -> pl.DataFrame:
        key = (t, side)
        if key not in rep_cache:
            rep_cache[key] = representative_tickers(t, side, markets=mk, trades=tr)
        return rep_cache[key]

    frames: list[pl.DataFrame] = []
    for trig in triggers:
        s_panel = surprise_panel.filter(pl.col("series") == trig)
        meta = s_panel.select("event_ticker", *_TRIGGER_COLS).rename(
            {"event_ticker": "trigger_event", "close_time": "t0"})
        for tgt in tgts:
            if tgt == trig:
                continue
            sides = FEDDECISION_SIDES if tgt == "FEDDECISION" else ("any",)
            for side in sides:
                if reps(tgt, side).height == 0:
                    continue
                rp = response_panel(s_panel, tgt, side, horizon=horizon,
                                    markets=mk, trades=tr)
                if rp.height < min_pair_n:
                    continue
                frames.append(rp.join(meta, on="trigger_event", how="left"))

    if not frames:
        return pl.DataFrame()
    panel = pl.concat(frames, how="vertical")
    return finalise(panel)


def finalise(panel: pl.DataFrame) -> pl.DataFrame:
    """Add the label, the fold-safe scaled surprise, and pair keys."""
    return (
        panel
        .with_columns(
            pair=pl.col("trigger") + "->" + pl.col("target") + "/" + pl.col("side"),
            y=pl.col("response").sign().cast(pl.Int8),
            # unit-free surprise: forecast error in units of the market's own
            # implied sd. Uses only information available at t0, so it is safe
            # to compute once, outside the walk-forward loop.
            z_surprise=pl.when(pl.col("implied_std") > 0)
            .then(pl.col("surprise") / pl.col("implied_std"))
            .otherwise(None),
            same_release=pl.struct("trigger", "target").map_elements(
                lambda r: is_same_release(r["trigger"], r["target"]),
                return_dtype=pl.Boolean),
        )
        .filter(pl.col("y") != 0, pl.col("z_surprise").is_finite())
        .sort("t0")
    )


def pair_counts(panel: pl.DataFrame) -> pl.DataFrame:
    """Rows and label balance per ordered pair — the sample-size audit."""
    return (panel.group_by("pair")
            .agg(pl.len().alias("n"),
                 (pl.col("y") > 0).mean().alias("up_rate"),
                 pl.col("t0").min().alias("first"),
                 pl.col("t0").max().alias("last"))
            .sort("n", descending=True))
