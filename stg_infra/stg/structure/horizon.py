"""Term structure of diffusion — edges re-estimated as a function of how far
the target is from its own resolution.

Turns a single per-edge coefficient into a curve: does a surprise propagate
only to near-dated target contracts, or all the way along the curve?
(research_summary.md §6.5, Phase 1.5 — the temporal axis of the structure.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from stg.panel._io import load_markets, scan_trades
from stg.panel.targets import response_panel
from stg.structure.stats import spearman, spearman_p

DEFAULT_BUCKETS = ((0, 7), (7, 21), (21, 60))


def estimate_by_horizon(
    surprise_panel: pl.DataFrame,
    edges: list[tuple[str, str, str]],
    *,
    response_horizon: str = "dormant",
    dtc_buckets: tuple[tuple[int, int], ...] = DEFAULT_BUCKETS,
    min_n: int = 8,
    markets: Optional[pl.DataFrame] = None,
    trades: Optional[pl.LazyFrame] = None,
) -> pl.DataFrame:
    """For each edge, Spearman(surprise, response) within each target
    days-to-close bucket. ``edges`` is a list of (trigger, target, side)."""
    mk = markets if markets is not None else load_markets(is_only=True)
    tr = trades if trades is not None else scan_trades(is_only=True)

    rows: list[dict] = []
    for trig, tgt, side in edges:
        rp = response_panel(surprise_panel.filter(pl.col("series") == trig),
                            tgt, side, horizon=response_horizon,
                            markets=mk, trades=tr)
        if rp.height == 0:
            continue
        for lo, hi in dtc_buckets:
            b = rp.filter((pl.col("days_to_close") >= lo)
                          & (pl.col("days_to_close") < hi))
            if b.height < min_n:
                rows.append(dict(trigger=trig, target=tgt, side=side,
                                 dtc_lo=lo, dtc_hi=hi, n=b.height,
                                 rho=float("nan"), p=float("nan")))
                continue
            S = b["surprise"].to_numpy().astype(float)
            R = b["response"].to_numpy().astype(float)
            rho = spearman(S, R)
            rows.append(dict(trigger=trig, target=tgt, side=side,
                             dtc_lo=lo, dtc_hi=hi, n=b.height,
                             rho=rho, p=spearman_p(rho, b.height)))
    return pl.DataFrame(rows) if rows else pl.DataFrame()
