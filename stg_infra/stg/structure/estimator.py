"""Stage 1: direct estimation of the cross-market influence adjacency.

This is the CA report's actual contribution (§3.2: "no existing work has
characterised whether these cross-market belief updates are *structured*"),
estimated directly rather than learned from a forecasting loss — so every edge
carries a point estimate, an FDR-corrected p-value and a permutation check, and
can be reported as a finding and confirmed out of sample.

Promotes ``analysis/exploratory_2026_08/structure_discovery.py`` §1–2.

    estimate_adjacency(surprise_panel) -> edge table
        trigger target side n n_targets rho p_asymptotic bh_rank bh_crit
        survives p_permutation same_release
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import is_same_release, universe
from stg.panel.surprise import usable_triggers
from stg.panel.targets import representative_tickers, response_panel
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
    markets: Optional[pl.DataFrame] = None,
    trades: Optional[pl.LazyFrame] = None,
) -> pl.DataFrame:
    mk = markets if markets is not None else load_markets(is_only=True)
    tr = trades if trades is not None else scan_trades(is_only=True)
    triggers = usable_triggers(surprise_panel, min_n)
    tgts = targets if targets is not None else universe(5, mk)

    # cache the (target, side) -> representative tickers + their trades once
    rep_cache: dict[tuple[str, str], pl.DataFrame] = {}

    def reps(t: str, side: str) -> pl.DataFrame:
        key = (t, side)
        if key not in rep_cache:
            rep_cache[key] = representative_tickers(t, side, markets=mk, trades=tr)
        return rep_cache[key]

    rows: list[dict] = []
    pair_arrays: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for trig in triggers:
        s_panel = surprise_panel.filter(pl.col("series") == trig)
        for tgt in tgts:
            if tgt == trig:
                continue
            sides = FEDDECISION_SIDES if tgt == "FEDDECISION" else ("any",)
            for side in sides:
                if reps(tgt, side).height == 0:
                    continue
                rp = response_panel(s_panel, tgt, side, horizon=horizon,
                                    markets=mk, trades=tr)
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
    g = pl.DataFrame(rows).sort("p_asymptotic")
    pvals = g["p_asymptotic"].to_numpy()
    survives = benjamini_hochberg(pvals, q)
    g = g.with_columns(
        pl.Series("bh_rank", np.arange(1, g.height + 1)),
        pl.Series("bh_crit", bh_critical(pvals, q)),
        pl.Series("survives", survives),
    )

    # permutation check — only where it matters (survivors + near-misses)
    perm = []
    for r in g.iter_rows(named=True):
        key = (r["trigger"], r["target"], r["side"])
        if r["survives"] or r["p_asymptotic"] < 0.05:
            S, R = pair_arrays[key]
            perm.append(permutation_p(S, R, n_perm))
        else:
            perm.append(float("nan"))
    return g.with_columns(pl.Series("p_permutation", perm))


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
