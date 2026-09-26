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
class SparseSnapshotEdges:
    """One snapshot's edges of ONE edge type, in GLOBAL node-index space.
    ``edge_index`` is (2, E) int64 [row_global, col_global]; ``weight`` is
    (E,) float64; ``features`` is (E, Fe) float64 or None if this edge
    type never carries features anywhere in the graph. See
    SpatioTemporalGraph.sparse_edges_by_type() for why this exists instead
    of a dense (N,N) matrix."""

    edge_index: np.ndarray
    weight: np.ndarray
    features: Optional[np.ndarray] = None


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

    def adjacency_tensor_padded(self, weight_attr: str = "weight") -> np.ndarray:
        """(T, N, N) adjacency tensor over the UNION node set, using the
        EXACT SAME node ordering as ``feature_tensor_padded()``
        (``all_node_ids()``) so the two can be combined directly — index
        ``i`` refers to the same physical node in both tensors.

        This is deliberately NOT the same as ``adjacency_tensor()``, which
        orders each snapshot by its OWN local node list (a different,
        varying order and size at every timestep). That makes
        ``adjacency_tensor()`` unusable together with
        ``feature_tensor_padded()`` — index ``i`` would mean a different
        physical node in each. This method exists specifically to fix that
        for any consumer that needs both feature and adjacency tensors
        aligned (e.g. a torch bridge feeding a GAT layer).

        Nodes absent from a given snapshot get all-zero rows/columns at
        their global index for that timestep — an ordinary "no edge" value,
        indistinguishable here from a genuine zero-weight edge. Use
        ``feature_tensor_padded()``'s mask to tell the two apart if that
        distinction matters downstream.

        ``networkx.to_numpy_array`` was NOT used with an inflated nodelist
        here — it raises ``NetworkXError`` if the nodelist contains nodes
        absent from that specific graph, which is the normal case for
        every snapshot given how sparse this data is. Instead, each
        snapshot's small LOCAL adjacency matrix is computed as-is (correct,
        unchanged), then scattered into the right global-index positions of
        the full-size padded tensor directly.
        """
        node_ids = self.all_node_ids()
        node_idx: Dict[Hashable, int] = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)
        T = len(self._snapshots)
        tensor = np.zeros((T, N, N), dtype=np.float64)

        for t, snap in enumerate(self._snapshots):
            local_ids = snap.node_ids
            if not local_ids:
                continue
            local_adj = snap.adjacency_matrix(weight_attr=weight_attr)
            global_indices = [node_idx[nid] for nid in local_ids]
            row_idx, col_idx = np.ix_(global_indices, global_indices)
            tensor[t][row_idx, col_idx] = local_adj

        return tensor

    def adjacency_tensor_padded_by_type(self, weight_attr: str = "weight") -> Dict[str, np.ndarray]:
        """Like ``adjacency_tensor_padded()``, but split into one (T, N, N)
        tensor PER EDGE TYPE (keyed by each edge's ``metadata["edge_type"]``)
        instead of one tensor mixing every edge type together at the same
        weight.

        Necessary because this project deliberately builds structurally
        different edge types in one graph — MECE hyperedge spokes
        (symmetric, no edge features) and ladder chain edges (directed,
        carrying historical violation/gap features) — which a model needs
        to attend over SEPARATELY (a two-channel design), not as if they
        were one undifferentiated relation. ``adjacency_tensor_padded()``
        alone can't support that, since it collapses every edge type into
        a single combined value and a model reading it has no way to tell
        a MECE spoke from a ladder edge.

        Returns
        -------
        dict mapping edge_type string -> (T, N, N) tensor, aligned to the
        same global node ordering as ``feature_tensor_padded()`` /
        ``adjacency_tensor_padded()``. Every returned tensor has the same
        shape regardless of whether that edge type happens to be absent
        from any particular snapshot — discovered up front by scanning all
        snapshots once, so a type missing from snapshot 3 but present in
        snapshot 7 still gets a consistent all-zero slice at t=3, not a
        missing key. Edges with no ``edge_type`` in their metadata fall
        under ``"unknown"``.
        """
        node_ids = self.all_node_ids()
        node_idx: Dict[Hashable, int] = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)
        T = len(self._snapshots)

        edge_types: set = set()
        for snap in self._snapshots:
            for _u, _v, ed in snap.nx_graph.edges(data=True):
                edge_types.add(ed.get("metadata", {}).get("edge_type", "unknown"))

        tensors: Dict[str, np.ndarray] = {
            et: np.zeros((T, N, N), dtype=np.float64) for et in edge_types
        }

        for t, snap in enumerate(self._snapshots):
            for u, v, ed in snap.nx_graph.edges(data=True):
                et = ed.get("metadata", {}).get("edge_type", "unknown")
                w = ed.get(weight_attr, 1.0)
                i, j = node_idx[u], node_idx[v]
                tensors[et][t, i, j] = w

        return tensors

    def edge_feature_tensor_padded(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """(T, N, N, Fe) edge-feature tensor, aligned to the same global node
        ordering as ``feature_tensor_padded()`` / ``adjacency_tensor_padded()``.

        Same rationale and scatter technique as ``adjacency_tensor_padded()``,
        generalised from a single weight per edge to an arbitrary feature
        vector per edge (e.g. the historical violation/gap features
        ``KalshiLadderChainEdges`` attaches).

        Returns
        -------
        tensor : np.ndarray or None, shape (T, N, N, Fe) — None if no edge in
            any snapshot carries features (nothing to build a tensor from).
        has_features : np.ndarray bool, shape (T, N, N) — True where a real
            edge feature vector was scattered in; distinguishes a genuine
            all-zero feature vector from "no edge feature present here".
        """
        node_ids = self.all_node_ids()
        node_idx: Dict[Hashable, int] = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)
        T = len(self._snapshots)

        Fe: Optional[int] = None
        for snap in self._snapshots:
            # NOT using snap.edge_feature_matrix() here -- it returns None for
            # the ENTIRE snapshot if even one edge lacks a features value,
            # which breaks exactly when a graph mixes edge types with and
            # without features (e.g. MECE hyperedges, which never carry
            # features, alongside ladder edges, which optionally do).
            # Scanning directly finds the first REAL feature vector without
            # being tripped up by other featureless edges alongside it.
            for _u, _v, ed in snap.nx_graph.edges(data=True):
                ef = ed.get("features")
                if ef is not None:
                    Fe = len(ef)
                    break
            if Fe is not None:
                break

        has_features = np.zeros((T, N, N), dtype=bool)
        if Fe is None:
            return None, has_features

        tensor = np.zeros((T, N, N, Fe), dtype=np.float64)
        for t, snap in enumerate(self._snapshots):
            for u, v in snap.edges:
                ed = snap.nx_graph.edges[u, v]
                ef = ed.get("features")
                if ef is None:
                    continue
                i, j = node_idx[u], node_idx[v]
                tensor[t, i, j] = ef
                has_features[t, i, j] = True

        return tensor, has_features

    # -- sparse padded export (for large T/N where dense (T,N,N) is infeasible) --
    def sparse_edges_by_type(self, weight_attr: str = "weight") -> Dict[str, List["SparseSnapshotEdges"]]:
        """Per-edge-type, per-snapshot SPARSE edge lists in GLOBAL node-index
        space (same indexing as feature_tensor_padded()/all_node_ids()) --
        the memory-safe counterpart to adjacency_tensor_padded_by_type() /
        edge_feature_tensor_padded(), which allocate one dense (T,N,N)
        tensor per edge type.

        WHY THIS EXISTS: adjacency_tensor_padded_by_type() is fine at the
        single-day scale this project was first verified at (N~788,
        T~10 -- a few MB), but at multi-month scale N (the union of every
        node ever seen across the WHOLE requested range) can run into the
        tens of thousands and T into the thousands. (T, N, N) at that
        scale is TERABYTES, not gigabytes -- confirmed directly, not
        assumed: a real 5-month run OOM-killed the machine running it.
        Every downstream consumer (TwoChannelSpatialAttention,
        MeceOutputHead, LadderOutputHead) only ever used a snapshot's
        small ACTIVE-node submatrix anyway (a few hundred nodes, matching
        this project's own edges_min=54/edges_max=161 per-snapshot
        counts from the very first real-data run) -- the dense (N,N)
        padding around that submatrix was always wasted allocation, this
        just stops paying for it.

        Memory here is O(total real edges across every snapshot), which
        this project's own data already shows is small (hundreds per
        snapshot) regardless of how large the global node universe N
        gets -- exactly the same "restrict to what's actually active"
        principle already applied when spatial_attention.py moved from a
        dense implementation to GATv2Conv over real edges only.

        Returns
        -------
        dict mapping edge_type -> list of length T, each entry a
        SparseSnapshotEdges (edge_index GLOBAL, weight, and features if
        this edge type ever carries any -- see KalshiLadderChainEdges).
        Every returned list has length T regardless of whether that edge
        type happens to be absent from a particular snapshot (an empty
        (2,0) edge_index there, not a missing entry) -- same "consistent
        shape even when absent" contract adjacency_tensor_padded_by_type()
        already used.
        """
        node_ids = self.all_node_ids()
        node_idx: Dict[Hashable, int] = {nid: i for i, nid in enumerate(node_ids)}

        edge_types: set = set()
        feature_dim_by_type: Dict[str, int] = {}
        for snap in self._snapshots:
            for _u, _v, ed in snap.nx_graph.edges(data=True):
                et = ed.get("metadata", {}).get("edge_type", "unknown")
                edge_types.add(et)
                ef = ed.get("features")
                if ef is not None and et not in feature_dim_by_type:
                    feature_dim_by_type[et] = len(ef)

        result: Dict[str, List[SparseSnapshotEdges]] = {et: [] for et in edge_types}

        for snap in self._snapshots:
            per_type: Dict[str, Tuple[List[int], List[int], List[float], List[Optional[np.ndarray]]]] = {
                et: ([], [], [], []) for et in edge_types
            }
            for u, v, ed in snap.nx_graph.edges(data=True):
                et = ed.get("metadata", {}).get("edge_type", "unknown")
                rows, cols, weights, feats = per_type[et]
                rows.append(node_idx[u])
                cols.append(node_idx[v])
                weights.append(ed.get(weight_attr, 1.0))
                feats.append(ed.get("features"))

            for et in edge_types:
                rows, cols, weights, feats = per_type[et]
                if rows:
                    edge_index = np.array([rows, cols], dtype=np.int64)
                    weight_arr = np.array(weights, dtype=np.float64)
                else:
                    edge_index = np.zeros((2, 0), dtype=np.int64)
                    weight_arr = np.zeros((0,), dtype=np.float64)

                features_arr: Optional[np.ndarray] = None
                if et in feature_dim_by_type:
                    fe = feature_dim_by_type[et]
                    if rows:
                        features_arr = np.stack(
                            [f if f is not None else np.zeros(fe) for f in feats]
                        ).astype(np.float64)
                    else:
                        features_arr = np.zeros((0, fe), dtype=np.float64)

                result[et].append(SparseSnapshotEdges(edge_index=edge_index, weight=weight_arr, features=features_arr))

        return result

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