"""GraphBuilder — orchestrates strategies into a SpatioTemporalGraph.

The builder slices all auxiliary DataFrames to the same time window as the
primary data, so node/edge strategies always receive temporally aligned data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import polars as pl

from stg_infra.stg.core import GraphSnapshot, SpatioTemporalGraph
from stg_infra.stg.strategies.protocols import (
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

        slices = self._temporal.slice(data, **base_kw)
        logger.info("Temporal slicing produced %d windows", len(slices))

        snapshots: List[GraphSnapshot] = []
        labels_over_time: Dict[Any, Dict[Any, Any]] = {}

        for window in slices:
            ts = window["timestamp"]
            wd: pl.DataFrame = window["data"]
            snap = GraphSnapshot(timestamp=ts)

            # Slice auxiliary data to this window's time range
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

            # All node strategies run first, in registration order.
            for node_strategy in self._nodes:
                for nid in node_strategy.identify_nodes(wd, **kw):
                    snap.add_node(node_strategy.build_node_state(nid, wd, **kw))

            # All edge strategies run after every node is present, in registration order.
            for edge_strategy in self._edges:
                for e in edge_strategy.build_edges(snap, wd, **kw):
                    snap.add_edge(e)

            if self._features is not None:
                self._apply_feature_transforms(snap)

            for pp in self._post_processors:
                snap = pp.process(snap, **kw)

            if self._label is not None:
                labels_over_time[ts] = self._label.extract_labels(snap.node_ids, wd, **kw)

            snapshots.append(snap)
            logger.debug("t=%s  nodes=%d  edges=%d", ts, snap.num_nodes, snap.num_edges)

        stg = SpatioTemporalGraph(snapshots)
        stg._labels = labels_over_time  # type: ignore[attr-defined]
        return stg

    def _apply_feature_transforms(self, snapshot: GraphSnapshot) -> None:
        assert self._features is not None
        for nid in snapshot.node_ids:
            nd = snapshot._graph.nodes[nid]
            nd["features"] = self._features.transform_node_features(
                nd["features"].reshape(1, -1)
            ).squeeze(0)
        for u, v in snapshot.edges:
            ed = snapshot._graph.edges[u, v]
            ef = ed.get("features")
            if ef is not None:
                ed["features"] = self._features.transform_edge_features(
                    ef.reshape(1, -1)
                ).squeeze(0)
