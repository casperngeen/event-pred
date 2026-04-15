"""Built-in edge construction strategies — pure Polars."""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, List

import numpy as np
import polars as pl

from stg.core import EdgeState, GraphSnapshot


class CosineSimilarityEdges:
    """Connect nodes whose features exceed a cosine-similarity threshold."""

    def __init__(self, threshold: float = 0.5, symmetric: bool = True) -> None:
        self.threshold = threshold
        self.symmetric = symmetric

    def build_edges(self, snapshot: GraphSnapshot, data: pl.DataFrame, **kwargs: Any) -> List[EdgeState]:
        nodes = snapshot.node_ids
        if len(nodes) < 2:
            return []
        X = snapshot.feature_matrix()
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        sim = (X / norms) @ (X / norms).T
        edges: List[EdgeState] = []
        for i in range(len(nodes)):
            j_start = 0 if self.symmetric else i + 1
            for j in range(j_start, len(nodes)):
                if i != j and sim[i, j] >= self.threshold:
                    edges.append(EdgeState(nodes[i], nodes[j], float(sim[i, j])))
        return edges


class KNNEdges:
    """Connect each node to its K nearest neighbours in feature space."""

    def __init__(self, k: int = 5, metric: str = "euclidean") -> None:
        self.k = k
        self.metric = metric

    def build_edges(self, snapshot: GraphSnapshot, data: pl.DataFrame, **kwargs: Any) -> List[EdgeState]:
        nodes = snapshot.node_ids
        if len(nodes) < 2:
            return []
        X = snapshot.feature_matrix()
        k = min(self.k, len(nodes) - 1)
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-10, norms)
            dist = 1.0 - (X / norms) @ (X / norms).T
        else:
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dist, np.inf)
        edges: List[EdgeState] = []
        for i in range(len(nodes)):
            for j in np.argsort(dist[i])[:k]:
                edges.append(EdgeState(nodes[i], nodes[int(j)], float(1.0 / (1.0 + dist[i, j]))))
        return edges


class SharedAttributeEdges:
    """Connect nodes sharing a common value in ``group_col``."""

    def __init__(self, group_col: str, node_col: str, weight: float = 1.0) -> None:
        self.group_col = group_col
        self.node_col = node_col
        self.weight = weight

    def build_edges(self, snapshot: GraphSnapshot, data: pl.DataFrame, **kwargs: Any) -> List[EdgeState]:
        node_set = set(snapshot.node_ids)
        groups = (
            data.select([self.group_col, self.node_col])
            .unique()
            .group_by(self.group_col)
            .agg(pl.col(self.node_col).alias("members"))
        )
        edges: List[EdgeState] = []
        for row in groups.iter_rows(named=True):
            members = [m for m in row["members"] if m in node_set]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    edges.append(EdgeState(members[i], members[j], self.weight,
                                           metadata={"edge_type": "shared_attr"}))
                    edges.append(EdgeState(members[j], members[i], self.weight,
                                           metadata={"edge_type": "shared_attr"}))
        return edges


class ExplicitEdges:
    """Edges from an explicit callable: ``(snapshot, data, **kw) -> [(src, tgt, weight), ...]``."""

    def __init__(self, edge_fn: Callable[..., List[Any]]) -> None:
        self._fn = edge_fn

    def build_edges(self, snapshot: GraphSnapshot, data: pl.DataFrame, **kwargs: Any) -> List[EdgeState]:
        return [EdgeState(s, t, w) for s, t, w in self._fn(snapshot, data, **kwargs)]


class CompositeEdges:
    """Merge edges from multiple strategies; duplicate (src, tgt) weights are summed."""

    def __init__(self, strategies: List[Any]) -> None:
        self.strategies = strategies

    def build_edges(self, snapshot: GraphSnapshot, data: pl.DataFrame, **kwargs: Any) -> List[EdgeState]:
        merged: Dict[tuple, EdgeState] = {}
        for strat in self.strategies:
            for e in strat.build_edges(snapshot, data, **kwargs):
                key = (e.source, e.target)
                if key in merged:
                    merged[key] = EdgeState(
                        e.source, e.target, merged[key].weight + e.weight,
                        e.features, {**merged[key].metadata, **e.metadata},
                    )
                else:
                    merged[key] = e
        return list(merged.values())
