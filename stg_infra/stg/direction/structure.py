"""Per-fold structure: the Stage-1 adjacency re-estimated on training rows only.

Stage-1's published edge table (``artifacts/adjacency_is.parquet``) is fitted
on the whole in-sample block. Feeding those rho's to a predictor scored on the
same block would leak: the edge weight already saw the outcomes it is used to
predict, and BH selection saw them too. So every fold re-runs the estimator on
its own training rows, and a rung that consumes structure consumes only the
structure that fold could have known.

The gap between "edges refit per fold" and "the published 8" is itself
reportable: it measures how stable the recovered structure is over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from stg.structure.stats import benjamini_hochberg, spearman, spearman_p


@dataclass(frozen=True)
class FoldStructure:
    """Edge weights known to a fold. ``rho[pair]`` is the train-only Spearman."""
    rho: dict[str, float] = field(default_factory=dict)
    p: dict[str, float] = field(default_factory=dict)
    n: dict[str, int] = field(default_factory=dict)
    survives: frozenset[str] = frozenset()

    def weight(self, pair: str, survivors_only: bool = False) -> float:
        if survivors_only and pair not in self.survives:
            return 0.0
        r = self.rho.get(pair, 0.0)
        return r if np.isfinite(r) else 0.0

    def signal(self, pairs, z, survivors_only: bool = False) -> np.ndarray:
        """rho_train * z_surprise, elementwise — the edge-weighted surprise."""
        w = np.array([self.weight(p, survivors_only) for p in pairs], float)
        return w * np.asarray(z, float)

    def covered(self, pairs, gate: str) -> np.ndarray:
        """Which rows the fold's structure claims an edge for, at ``gate``.

        ``all`` every row; ``p05`` nominally significant train edge;
        ``bh`` BH-FDR survivor in that fold — the strictest and the one
        Stage-1 reports.
        """
        if gate == "all":
            return np.ones(len(pairs), bool)
        if gate == "bh":
            return np.array([pr in self.survives for pr in pairs])
        if gate == "p05":
            return np.array([self.p.get(pr, 1.0) < 0.05 for pr in pairs])
        raise ValueError(f"unknown gate {gate!r}")


def fit_structure(train: pl.DataFrame, *, min_n: int = 10, q: float = 0.10) -> FoldStructure:
    """Estimate per-pair signed rank association on ``train`` rows only."""
    rho: dict[str, float] = {}
    pmap: dict[str, float] = {}
    n: dict[str, int] = {}
    pvals: list[float] = []
    keys: list[str] = []
    for (pair,), g in train.group_by("pair", maintain_order=True):
        if g.height < min_n:
            continue
        s = g["surprise"].to_numpy().astype(float)
        r = g["response"].to_numpy().astype(float)
        rh = spearman(s, r)
        p = spearman_p(rh, g.height)
        if not np.isfinite(rh) or not np.isfinite(p):
            continue
        rho[pair] = rh
        pmap[pair] = p
        n[pair] = g.height
        keys.append(pair)
        pvals.append(p)
    surv = frozenset()
    if pvals:
        mask = benjamini_hochberg(np.array(pvals), q)
        surv = frozenset(k for k, m in zip(keys, mask) if m)
    return FoldStructure(rho=rho, p=pmap, n=n, survives=surv)
