"""Mediation / multi-hop test — the part a pairwise sweep cannot do.

A pairwise sweep returns a *list* of edges. The graph question is whether
A -> C survives once B is controlled for. If partial(A, C | B) collapses
toward zero, the apparent A -> C edge is routed A -> B -> C and the structure
has genuine multi-hop depth. If it is unchanged, the edges are independent
bilateral effects and message passing across hops has nothing to operate on.

Promotes ``structure_discovery.py`` §3. As of the August run every partial was
unchanged or slightly stronger (n = 14–25, low power) — recorded as *no
evidence of* multi-hop structure, not evidence of its absence.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from stg.panel._io import load_markets, scan_trades
from stg.panel.targets import response_panel
from stg.structure.stats import partial_spearman, spearman


def test_mediation(
    surprise_panel: pl.DataFrame,
    a: str, b: str, c: str,
    side: str = "any",
    *,
    horizon: str = "dormant",
    min_n: int = 10,
    markets: Optional[pl.DataFrame] = None,
    trades: Optional[pl.LazyFrame] = None,
) -> Optional[dict]:
    """Partial Spearman of A->C controlling B, on events where both A and B
    surprises map to the same next C contract."""
    mk = markets if markets is not None else load_markets(is_only=True)
    tr = trades if trades is not None else scan_trades(is_only=True)

    ra = response_panel(surprise_panel.filter(pl.col("series") == a), c, side,
                        horizon=horizon, markets=mk, trades=tr)
    rb = response_panel(surprise_panel.filter(pl.col("series") == b), c, side,
                        horizon=horizon, markets=mk, trades=tr)
    if ra.height == 0 or rb.height == 0:
        return None
    j = ra.join(rb.select("target_event", pl.col("surprise").alias("surprise_b")),
                on="target_event", how="inner")
    if j.height < min_n:
        return None
    SA = j["surprise"].to_numpy().astype(float)
    SB = j["surprise_b"].to_numpy().astype(float)
    RC = j["response"].to_numpy().astype(float)
    r0 = spearman(SA, RC)
    r1 = partial_spearman(SA, SB, RC)
    return dict(a=a, b=b, c=c, side=side, n=j.height,
                rho=r0, partial=r1, change=r1 - r0)


def mediation_grid(
    surprise_panel: pl.DataFrame,
    triples: list[tuple[str, str, str, str]],
    **kw,
) -> pl.DataFrame:
    out = [r for t in triples if (r := test_mediation(surprise_panel, *t, **kw))]
    return pl.DataFrame(out) if out else pl.DataFrame()
