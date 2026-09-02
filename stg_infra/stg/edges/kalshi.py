"""Kalshi edge strategies — series-level directed influence.

Two edge types, both directed:

``SurpriseInfluenceEdges``
    Loaded from the Stage-1 validated adjacency artifact
    (``artifacts/adjacency_is.parquet``). Weight = signed rho-hat; metadata
    carries the FDR verdict, n, and same-release flag. This is the estimated
    structure used as a prior by the STG model.

``SameReleaseEdges``
    Deterministic. Connects series that resolve from the same official print
    (BLS CPI -> all CPI ladders; BLS Employment Situation -> PAYROLLS + U3).
    Kept as a *separate* edge type so identification work (research_summary.md
    §5 / update_2026_08.md §7) can condition on the common-signal channel.

The contract-level and event-super-node strategies from before the series-level
refactor (``KalshiTickerNodes``/``KalshiEventNodes`` and their edges, plus the
sentence-transformers semantic-topic edges) were removed — see git history at
tag ``pre-series-refactor`` / branch ``main``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import polars as pl

from stg.core import EdgeState, GraphSnapshot
from stg.panel.registry import same_release_groups

DEFAULT_ADJACENCY = Path("artifacts/adjacency_is.parquet")


class SurpriseInfluenceEdges:
    """Directed edges from the directly-estimated adjacency.

    Parameters
    ----------
    adjacency : path to the edge table written by
        ``stg.structure.estimate_adjacency`` (parquet).
    survivors_only : keep only BH-FDR survivors (default) or every searched pair.
    min_abs_rho : drop edges weaker than this.
    """

    def __init__(self, adjacency: str | Path = DEFAULT_ADJACENCY,
                 survivors_only: bool = True, min_abs_rho: float = 0.0) -> None:
        path = Path(adjacency)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run scripts/run_structure_estimation.py first"
            )
        e = pl.read_parquet(path)
        if survivors_only and "survives" in e.columns:
            e = e.filter(pl.col("survives"))
        e = e.filter(pl.col("rho").abs() >= min_abs_rho)
        # collapse FEDDECISION hike/cut to one edge (largest |rho|)
        e = (e.with_columns(pl.col("target").alias("tgt"))
             .sort(pl.col("rho").abs(), descending=True)
             .group_by("trigger", "tgt").first())
        self._edges = e

    def build_edges(self, snapshot: GraphSnapshot, data: pl.DataFrame,
                    **kwargs: Any) -> List[EdgeState]:
        present = set(snapshot.node_ids)
        out: List[EdgeState] = []
        for r in self._edges.iter_rows(named=True):
            s, t = r["trigger"], r["tgt"]
            if s in present and t in present:
                out.append(EdgeState(s, t, float(r["rho"]), metadata={
                    "edge_type": "surprise_influence",
                    "n": r.get("n"),
                    "q_survives": r.get("survives"),
                    "same_release": r.get("same_release"),
                    "side": r.get("side"),
                }))
        return out


class SameReleaseEdges:
    """Bidirectional edges between series resolving from the same official print."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight
        self._groups = same_release_groups()

    def build_edges(self, snapshot: GraphSnapshot, data: pl.DataFrame,
                    **kwargs: Any) -> List[EdgeState]:
        present = set(snapshot.node_ids)
        out: List[EdgeState] = []
        for group, members in self._groups.items():
            active = [m for m in members if m in present]
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    for a, b in ((active[i], active[j]), (active[j], active[i])):
                        out.append(EdgeState(a, b, self.weight, metadata={
                            "edge_type": "same_release", "release": group,
                        }))
        return out
