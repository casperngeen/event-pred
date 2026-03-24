"""
Core spatial-temporal graph data model.

Zero dependency on the data layer (Polars).  Operates purely on numpy
arrays and networkx graphs so downstream ML consumers never import Polars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterator, List, Optional, Tuple, Union, overload
from datetime import datetime

import networkx as nx
import numpy as np


@dataclass
class NodeState:
    """Immutable snapshot of a single node at one point in time."""

    node_id: Hashable
    features: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.features, np.ndarray):
            self.features = np.asarray(self.features, dtype=np.float64)


@dataclass
class EdgeState:
    """Immutable snapshot of a single edge at one point in time."""

    source: Hashable
    target: Hashable
    weight: float = 1.0
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.features is not None and not isinstance(self.features, np.ndarray):
            self.features = np.asarray(self.features, dtype=np.float64)


@dataclass
class GraphSnapshot:
    """A single time-slice of the spatial-temporal graph."""

    timestamp: datetime
    _graph: nx.DiGraph = field(default_factory=nx.DiGraph, repr=False)

    # -- nodes --------------------------------------------------------------
    def add_node(self, state: NodeState) -> None:
        self._graph.add_node(
            state.node_id, features=state.features, metadata=state.metadata,
        )

    def get_node(self, node_id: Hashable) -> Optional[NodeState]:
        if node_id not in self._graph:
            return None
        d = self._graph.nodes[node_id]
        return NodeState(node_id, d["features"], d.get("metadata", {}))

    @property
    def node_ids(self) -> List[Hashable]:
        return list(self._graph.nodes)

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    # -- edges --------------------------------------------------------------
    def add_edge(self, state: EdgeState) -> None:
        self._graph.add_edge(
            state.source, state.target,
            weight=state.weight, features=state.features, metadata=state.metadata,
        )

    def get_edge(self, source: Hashable, target: Hashable) -> Optional[EdgeState]:
        if not self._graph.has_edge(source, target):
            return None
        d = self._graph.edges[source, target]
        return EdgeState(
            source, target, d.get("weight", 1.0),
            d.get("features"), d.get("metadata", {}),
        )

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def edges(self) -> List[Tuple[Hashable, Hashable]]:
        return list(self._graph.edges)

    # -- matrix export ------------------------------------------------------
    def adjacency_matrix(self, weight_attr: str = "weight") -> np.ndarray:
        return nx.to_numpy_array(
            self._graph, nodelist=self.node_ids, weight=weight_attr,
        )

    def feature_matrix(self) -> np.ndarray:
        """(N, F) node feature matrix."""
        return np.vstack(
            [self._graph.nodes[n]["features"] for n in self.node_ids]
        )

    def edge_feature_matrix(self) -> Optional[np.ndarray]:
        """(E, Fe) edge feature matrix, or None if edges lack features."""
        feats: List[np.ndarray] = []
        for u, v in self._graph.edges:
            ef = self._graph.edges[u, v].get("features")
            if ef is None:
                return None
            feats.append(ef)
        return np.vstack(feats) if feats else None

    @property
    def nx_graph(self) -> nx.DiGraph:
        return self._graph

    def copy(self) -> GraphSnapshot:
        gs = GraphSnapshot(timestamp=self.timestamp)
        gs._graph = self._graph.copy()
        return gs


class SpatioTemporalGraph:
    """Ordered sequence of ``GraphSnapshot`` objects indexed by time."""

    def __init__(self, snapshots: Optional[List[GraphSnapshot]] = None) -> None:
        self._snapshots: List[GraphSnapshot] = snapshots or []

    # -- construction -------------------------------------------------------
    def add_snapshot(self, snapshot: GraphSnapshot) -> None:
        self._snapshots.append(snapshot)

    def sort(self, key: Any = None) -> None:
        self._snapshots.sort(key=key or (lambda s: s.timestamp))

    # -- access -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._snapshots)

    @overload
    def __getitem__(self, idx: int) -> GraphSnapshot: ...
    @overload
    def __getitem__(self, idx: slice) -> SpatioTemporalGraph: ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[GraphSnapshot, SpatioTemporalGraph]:
        if isinstance(idx, slice):
            return SpatioTemporalGraph(self._snapshots[idx])
        return self._snapshots[idx]

    def __iter__(self) -> Iterator[GraphSnapshot]:
        return iter(self._snapshots)

    @property
    def timestamps(self) -> List[Any]:
        return [s.timestamp for s in self._snapshots]

    @property
    def snapshots(self) -> List[GraphSnapshot]:
        return list(self._snapshots)

    # -- temporal queries ---------------------------------------------------
    def window(self, start: datetime, end: datetime) -> SpatioTemporalGraph:
        """Sub-graph covering ``[start, end]`` inclusive."""
        return SpatioTemporalGraph(
            [s for s in self._snapshots if start <= s.timestamp <= end]
        )

    def latest(self) -> Optional[GraphSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    # -- bulk export --------------------------------------------------------
    def feature_tensor(self) -> np.ndarray:
        """(T, N, F) — requires consistent node sets across snapshots."""
        return np.stack([s.feature_matrix() for s in self._snapshots])

    def adjacency_tensor(self) -> np.ndarray:
        """(T, N, N)."""
        return np.stack([s.adjacency_matrix() for s in self._snapshots])

    # -- node identity across time ------------------------------------------
    def all_node_ids(self) -> List[Hashable]:
        """Union of all node IDs present in any snapshot, in first-seen order."""
        seen: Dict[Hashable, None] = {}
        for snap in self._snapshots:
            for nid in snap.node_ids:
                seen.setdefault(nid, None)
        return list(seen)

    # -- per-node time series -----------------------------------------------
    def feature_trajectory(self, node_id: Hashable) -> Tuple[np.ndarray, np.ndarray]:
        """Feature trajectory for one node across all snapshots.

        Returns
        -------
        times : np.ndarray, shape (T,)
            POSIX timestamps (float64) for each snapshot.
        features : np.ndarray, shape (T, F)
            Node features at each snapshot; NaN where the node was absent.
        """
        times: List[float] = []
        rows: List[Optional[np.ndarray]] = []
        feat_dim: Optional[int] = None

        for snap in self._snapshots:
            ts = snap.timestamp
            times.append(ts.timestamp() if isinstance(ts, datetime) else float(ts))
            ns = snap.get_node(node_id)
            if ns is not None:
                if feat_dim is None:
                    feat_dim = len(ns.features)
                rows.append(ns.features.copy())
            else:
                rows.append(None)

        T = len(times)
        F = feat_dim or 0
        features = np.full((T, F), np.nan, dtype=np.float64)
        for i, row in enumerate(rows):
            if row is not None:
                features[i] = row

        return np.array(times, dtype=np.float64), features

    # -- padded bulk export -------------------------------------------------
    def feature_tensor_padded(self) -> Tuple[np.ndarray, np.ndarray]:
        """(T, N, F) feature tensor over the union node set, with NaN for absent nodes.

        Returns
        -------
        tensor : np.ndarray, shape (T, N, F)
        mask   : np.ndarray bool, shape (T, N) — True where node was observed.

        Node ordering matches ``all_node_ids()``.
        """
        node_ids = self.all_node_ids()
        node_idx: Dict[Hashable, int] = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)
        F = 0
        for snap in self._snapshots:
            if snap.num_nodes > 0:
                F = snap.feature_matrix().shape[1]
                break

        T = len(self._snapshots)
        tensor = np.full((T, N, F), np.nan, dtype=np.float64)
        mask = np.zeros((T, N), dtype=bool)

        for t, snap in enumerate(self._snapshots):
            for nid in snap.node_ids:
                i = node_idx[nid]
                ns = snap.get_node(nid)
                if ns is not None:
                    tensor[t, i] = ns.features
                    mask[t, i] = True

        return tensor, mask

    # -- graph Laplacians ---------------------------------------------------
    def laplacian_tensor(self, normalized: bool = False) -> np.ndarray:
        """(T, N, N) graph Laplacian for each snapshot.

        Parameters
        ----------
        normalized : bool
            If False (default): L = D - A (combinatorial Laplacian).
            If True: symmetric normalized L = I - D^{-1/2} A D^{-1/2}.

        Node ordering per snapshot matches ``GraphSnapshot.node_ids``.
        """
        laps: List[np.ndarray] = []
        for snap in self._snapshots:
            A = snap.adjacency_matrix()
            d = A.sum(axis=1)  # out-degree vector
            if normalized:
                with np.errstate(divide="ignore", invalid="ignore"):
                    d_inv_sqrt = np.where(d > 0, d ** -0.5, 0.0)
                D_inv_sqrt = np.diag(d_inv_sqrt)
                L = np.eye(len(d)) - D_inv_sqrt @ A @ D_inv_sqrt
            else:
                L = np.diag(d) - A
            laps.append(L)
        return np.stack(laps).astype(np.float64)

    # -- summary ------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        if not self._snapshots:
            return {"num_snapshots": 0}
        nc = [s.num_nodes for s in self._snapshots]
        ec = [s.num_edges for s in self._snapshots]
        return {
            "num_snapshots": len(self._snapshots),
            "time_range": (self._snapshots[0].timestamp, self._snapshots[-1].timestamp),
            "nodes_min": min(nc), "nodes_max": max(nc),
            "edges_min": min(ec), "edges_max": max(ec),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"SpatioTemporalGraph(snapshots={s['num_snapshots']}, "
            f"time_range={s.get('time_range', 'N/A')})"
        )
