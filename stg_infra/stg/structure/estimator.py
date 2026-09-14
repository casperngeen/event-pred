"""Stage 1: direct estimation of the cross-market influence adjacency.

This is the CA report's actual contribution (§3.2: "no existing work has
characterised whether these cross-market belief updates are *structured*"),
estimated directly rather than learned from a forecasting loss — so every edge
carries a point estimate, an FDR-corrected p-value and a permutation check, and
can be reported as a finding and confirmed out of sample.

Promotes ``analysis/exploratory_2026_08/structure_discovery.py`` §1–2.

    estimate_adjacency(surprise_panel) -> edge table
        trigger target side n n_targets rho p_asymptotic p_permutation
        bh_rank bh_crit survives same_release

Selection runs on ``p_permutation`` by default, not ``p_asymptotic``. The
asymptotic p is a t-approximation on ranks that are ~98% tied (``response`` is
a difference of integer cents), so its distributional assumption is not met;
the permutation p is exact by construction with the same statistic. Selecting
on the asymptotic p reported **8** survivors where the permutation p supports
**3** — see ``stg/structure/stats.py`` and the ``select_on`` argument. Both
p-values are always reported so the gap stays visible.
"""

from __future__ import annotations

import zlib
from typing import Optional

import numpy as np
import polars as pl

from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import is_same_release, target_universe
from stg.panel.surprise import usable_triggers
from stg.panel.targets import response_panel, target_frames
from stg.structure.stats import (
    benjamini_hochberg, bh_critical, permutation_p, spearman, spearman_p,
)

FEDDECISION_SIDES = ("hike", "cut")


def estimate_adjacency(
    surprise_panel: pl.DataFrame,
    *,
    targets: Optional[list[str]] = None,
    horizon: str = "dormant",
    min_n: int = 10,
    q: float = 0.10,
    n_perm: int = 2000,
    select_on: str = "permutation",
    seed: int = 0,
    markets: Optional[pl.DataFrame] = None,
    trades: Optional[pl.LazyFrame] = None,
) -> pl.DataFrame:
    """Estimate the cross-market influence adjacency.

    ``select_on`` picks which p-value BH controls:

    ``"permutation"`` (default)
        Exact, tie-robust, and computed for **every** pair rather than only the
        survivors — BH needs the whole vector, so the old "permute the
        survivors and near-misses" shortcut could not support selection.
    ``"asymptotic"``
        The t-approximation. Reproduces the pre-fix behaviour; keep it only for
        comparing against the published table.

    Each pair's permutation null is drawn from its own generator, seeded from
    ``seed`` and the pair's identity, so a p-value does not depend on how many
    pairs were evaluated before it. The pair is folded into the seed with
    ``crc32``, not ``hash()`` -- Python randomises string hashing per process,
    so ``hash()`` would have made every p-value differ between runs.
    """
    if select_on not in ("permutation", "asymptotic"):
        raise ValueError(f"unknown select_on {select_on!r}")
    mk = markets if markets is not None else load_markets(is_only=True)
    tr = trades if trades is not None else scan_trades(is_only=True)
    triggers = usable_triggers(surprise_panel, min_n)
    tgts = targets if targets is not None else target_universe(5, mk)

    # cache the (target, side) -> representative tickers + their trades once.
    # These depend only on the target, so building them inside the trigger loop
    # made the sweep rebuild INXU's 15,289-ticker frame 21 times over.
    frame_cache: dict[tuple[str, str], tuple[pl.DataFrame, pl.DataFrame]] = {}

    def reps(t: str, side: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        key = (t, side)
        if key not in frame_cache:
            frame_cache[key] = target_frames(t, side, markets=mk, trades=tr)
        return frame_cache[key]

    rows: list[dict] = []
    pair_arrays: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for trig in triggers:
        s_panel = surprise_panel.filter(pl.col("series") == trig)
        for tgt in tgts:
            if tgt == trig:
                continue
            sides = FEDDECISION_SIDES if tgt == "FEDDECISION" else ("any",)
            for side in sides:
                fr = reps(tgt, side)
                if fr[0].height == 0:
                    continue
                rp = response_panel(s_panel, tgt, side, horizon=horizon,
                                    markets=mk, trades=tr, frames=fr)
                if rp.height < min_n:
                    continue
                S = rp["surprise"].to_numpy().astype(float)
                R = rp["response"].to_numpy().astype(float)
                rho = spearman(S, R)
                p = spearman_p(rho, len(S))
                if not np.isfinite(rho) or not np.isfinite(p):
                    continue
                key = (trig, tgt, side)
                pair_arrays[key] = (S, R)
                rows.append(dict(
                    trigger=trig, target=tgt, side=side, n=len(S),
                    n_targets=rp["target_event"].n_unique(),
                    rho=rho, p_asymptotic=p,
                    same_release=is_same_release(trig, tgt),
                ))

    if not rows:
        return pl.DataFrame()
    g = pl.DataFrame(rows)

    # Permutation p for *every* pair: BH controls whichever p it is given, so
    # a partial vector cannot be selected on. Per-pair seeding keeps each value
    # independent of grid order and of how many pairs preceded it.
    perm = []
    for r in g.iter_rows(named=True):
        key = (r["trigger"], r["target"], r["side"])
        S, R = pair_arrays[key]
        rng = np.random.default_rng([seed, *(zlib.crc32(k.encode()) for k in key)])
        perm.append(permutation_p(S, R, n_perm, rng=rng))
    g = g.with_columns(pl.Series("p_permutation", perm))

    sel = "p_permutation" if select_on == "permutation" else "p_asymptotic"
    g = g.sort(sel, "trigger", "target", "side")
    pvals = g[sel].to_numpy()
    return g.with_columns(
        pl.Series("bh_rank", np.arange(1, g.height + 1)),
        pl.Series("bh_crit", bh_critical(pvals, q)),
        pl.Series("survives", benjamini_hochberg(pvals, q)),
        pl.lit(select_on).alias("selected_on"),
    )


def adjacency_matrix(edges: pl.DataFrame, nodes: list[str],
                     survivors_only: bool = True) -> pl.DataFrame:
    """Wide N x N signed-rho adjacency (trigger = row, target = column).

    FEDDECISION hike/cut are collapsed to a single FEDDECISION column by the
    edge with the larger |rho|.
    """
    e = edges.filter(pl.col("survives")) if survivors_only else edges
    e = e.with_columns(pl.col("target").alias("target_node"))
    e = (e.sort(pl.col("rho").abs(), descending=True)
         .group_by("trigger", "target_node").first())
    m = {n: {c: 0.0 for c in nodes} for n in nodes}
    for r in e.iter_rows(named=True):
        if r["trigger"] in m and r["target_node"] in m[r["trigger"]]:
            m[r["trigger"]][r["target_node"]] = r["rho"]
    return pl.DataFrame(
        [{"trigger": n, **m[n]} for n in nodes]
    )
