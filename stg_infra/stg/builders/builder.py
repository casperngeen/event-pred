"""GraphBuilder — orchestrates strategies into a SpatioTemporalGraph.

The builder slices all auxiliary DataFrames to the same time window as the
primary data, so node/edge strategies always receive temporally aligned data.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import polars as pl

from stg.core import GraphSnapshot, SpatioTemporalGraph
from stg.strategies.protocols import (
    EdgeStrategy, FeatureStrategy, LabelStrategy,
    NodeStrategy, PostProcessStrategy, TemporalStrategy,
)

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Declarative builder — wire strategies via fluent API, then call .build().

    Auxiliary DataFrames (e.g. trades) are automatically sliced to each time
    window using the ``window_start`` / ``window_end`` emitted by the temporal
    strategy.  Configure which time column to filter on per auxiliary table
    via ``auxiliary_time_cols``.

    >>> stg = (
    ...     GraphBuilder()
    ...     .with_temporal(FixedWindowTemporal(every="1h"))
    ...     .with_nodes(KalshiTradeAugmentedNodes())
    ...     .with_edges(KalshiEventEdges())
    ...     .with_auxiliary_time_col("trades", "created_time")
    ...     .build(markets_df, auxiliary={"trades": trades_df})
    ... )
    """

    def __init__(self) -> None:
        self._temporal: Optional[TemporalStrategy] = None
        self._nodes: List[NodeStrategy] = []
        self._edges: List[EdgeStrategy] = []
        self._features: Optional[FeatureStrategy] = None
        self._post_processors: List[PostProcessStrategy] = []
        self._label: Optional[LabelStrategy] = None
        self._extra_kwargs: Dict[str, Any] = {}
        self._aux_time_cols: Dict[str, str] = {}
        self._static_aux_names: set = set()

    def with_temporal(self, s: TemporalStrategy) -> GraphBuilder:
        self._temporal = s; return self

    def with_nodes(self, s: NodeStrategy) -> GraphBuilder:
        """Register a node strategy. Multiple calls are supported — strategies
        run in registration order, each receiving the snapshot-so-far via
        ``kwargs["snapshot"]`` so later strategies can read already-built nodes."""
        self._nodes.append(s); return self

    def with_edges(self, s: EdgeStrategy) -> GraphBuilder:
        """Register an edge strategy. Multiple calls are supported — all node
        strategies run first, then all edge strategies run in order."""
        self._edges.append(s); return self

    def with_features(self, s: FeatureStrategy) -> GraphBuilder:
        self._features = s; return self

    def with_post_process(self, s: PostProcessStrategy) -> GraphBuilder:
        self._post_processors.append(s); return self

    def with_labels(self, s: LabelStrategy) -> GraphBuilder:
        self._label = s; return self

    def with_kwargs(self, **kwargs: Any) -> GraphBuilder:
        self._extra_kwargs.update(kwargs); return self

    def with_static_auxiliary(self, *names: str) -> GraphBuilder:
        """Mark auxiliary DataFrames that are reference tables and should NOT be
        sliced per window (e.g. a markets metadata table)."""
        self._static_aux_names.update(names)
        return self

    def with_auxiliary_time_col(self, aux_name: str, time_col: str) -> GraphBuilder:
        """Specify which time column to use when slicing an auxiliary DataFrame.

        If not set, defaults to ``"created_time"`` for that auxiliary table.
        """
        self._aux_time_cols[aux_name] = time_col
        return self

    def _slice_auxiliary(
        self,
        auxiliary: Dict[str, pl.DataFrame],
        window_start: Any,
        window_end: Any,
    ) -> Dict[str, pl.DataFrame]:
        """Filter each auxiliary DataFrame to [window_start, window_end)."""
        sliced: Dict[str, pl.DataFrame] = {}
        for name, df in auxiliary.items():
            if name in self._static_aux_names:
                sliced[name] = df
                continue
            time_col = self._aux_time_cols.get(name, "created_time")
            if time_col not in df.columns:
                # Can't slice — pass through unfiltered with a warning
                logger.warning(
                    "Auxiliary '%s' has no column '%s', passing unsliced", name, time_col,
                )
                sliced[name] = df
                continue
            sliced[name] = df.filter(
                (pl.col(time_col) >= window_start) & (pl.col(time_col) < window_end)
            )
        return sliced

    def build(
        self,
        data: pl.DataFrame,
        auxiliary: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> SpatioTemporalGraph:
        if self._temporal is None:
            raise ValueError("TemporalStrategy required — call .with_temporal()")
        if not self._nodes:
            raise ValueError("At least one NodeStrategy required — call .with_nodes()")
        if not self._edges:
            raise ValueError("At least one EdgeStrategy required — call .with_edges()")

        auxiliary = auxiliary or {}
        base_kw: Dict[str, Any] = {**self._extra_kwargs}

        t0 = time.perf_counter()
        slices = self._temporal.slice(data, **base_kw)
        n_windows = len(slices)
        logger.info("Temporal slicing produced %d windows (%.1fs)", n_windows, time.perf_counter() - t0)

        snapshots: List[GraphSnapshot] = []
        labels_over_time: Dict[Any, Dict[Any, Any]] = {}

        # Stage timing accumulators
        t_nodes = t_edges = t_feats = t_post = t_labels = 0.0
        last_log_time = time.perf_counter()
        log_interval = 10.0  # seconds between progress updates

        for i, window in enumerate(slices):
            ts = window["timestamp"]
            wd: pl.DataFrame = window["data"]
            snap = GraphSnapshot(timestamp=ts)

            window_start = window.get("window_start")
            window_end = window.get("window_end")

            if auxiliary and window_start is not None and window_end is not None:
                windowed_aux = self._slice_auxiliary(auxiliary, window_start, window_end)
            else:
                windowed_aux = auxiliary

            kw = {
                **base_kw,
                "auxiliary": windowed_aux,
                "timestamp": ts,
                "window_start": window_start,
                "window_end": window_end,
            }

            # Nodes
            t = time.perf_counter()
            for node_strategy in self._nodes:
                for nid in node_strategy.identify_nodes(wd, **kw):
                    snap.add_node(node_strategy.build_node_state(nid, wd, **kw))
            t_nodes += time.perf_counter() - t

            # Edges
            t = time.perf_counter()
            for edge_strategy in self._edges:
                for e in edge_strategy.build_edges(snap, wd, **kw):
                    snap.add_edge(e)
            t_edges += time.perf_counter() - t

            # Feature transforms
            t = time.perf_counter()
            if self._features is not None:
                self._apply_feature_transforms(snap)
            t_feats += time.perf_counter() - t

            # Post-processing
            t = time.perf_counter()
            for pp in self._post_processors:
                snap = pp.process(snap, **kw)
            t_post += time.perf_counter() - t

            # Labels
            t = time.perf_counter()
            if self._label is not None:
                labels_over_time[ts] = self._label.extract_labels(snap.node_ids, wd, **kw)
            t_labels += time.perf_counter() - t

            snapshots.append(snap)
            logger.debug("t=%s  nodes=%d  edges=%d", ts, snap.num_nodes, snap.num_edges)

            now = time.perf_counter()
            if now - last_log_time >= log_interval or (i + 1) == n_windows:
                last_log_time = now
                pct = 100 * (i + 1) / n_windows
                elapsed = t_nodes + t_edges + t_feats + t_post + t_labels
                logger.info(
                    "[%5.1f%%] %d/%d windows | nodes %.1fs  edges %.1fs  feats %.1fs  post %.1fs  labels %.1fs | total %.1fs",
                    pct, i + 1, n_windows,
                    t_nodes, t_edges, t_feats, t_post, t_labels, elapsed,
                )

        logger.info(
            "Build complete — %d snapshots in %.1fs",
            len(snapshots), t_nodes + t_edges + t_feats + t_post + t_labels,
        )

        stg = SpatioTemporalGraph(snapshots)
        stg._labels = labels_over_time  # type: ignore[attr-defined]
        return stg

    def _apply_feature_transforms(self, snapshot: GraphSnapshot) -> None:
        assert self._features is not None
        nids = snapshot.node_ids
        if nids:
            X = snapshot.feature_matrix()  # (N, F) — single batch transform
            X_t = self._features.transform_node_features(X)
            for i, nid in enumerate(nids):
                snapshot._graph.nodes[nid]["features"] = X_t[i]
        for u, v in snapshot.edges:
            ed = snapshot._graph.edges[u, v]
            ef = ed.get("features")
            if ef is not None:
                ed["features"] = self._features.transform_edge_features(
                    ef.reshape(1, -1)
                ).squeeze(0)
