"""Strategy protocols — abstract contracts for pluggable behaviours."""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Protocol, runtime_checkable

import numpy as np
import polars as pl

from stg.core import EdgeState, GraphSnapshot, NodeState


@runtime_checkable
class NodeStrategy(Protocol):
    def identify_nodes(self, data: pl.DataFrame, **kwargs: Any) -> List[Hashable]: ...
    def build_node_state(self, node_id: Hashable, data: pl.DataFrame, **kwargs: Any) -> NodeState: ...


@runtime_checkable
class EdgeStrategy(Protocol):
    def build_edges(self, snapshot: GraphSnapshot, data: pl.DataFrame, **kwargs: Any) -> List[EdgeState]: ...


@runtime_checkable
class TemporalStrategy(Protocol):
    def slice(self, data: pl.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]: ...


@runtime_checkable
class FeatureStrategy(Protocol):
    def transform_node_features(self, f: np.ndarray) -> np.ndarray: ...
    def transform_edge_features(self, f: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class PostProcessStrategy(Protocol):
    def process(self, snapshot: GraphSnapshot, **kwargs: Any) -> GraphSnapshot: ...


@runtime_checkable
class LabelStrategy(Protocol):
    def extract_labels(self, node_ids: List[Hashable], data: pl.DataFrame, **kwargs: Any) -> Dict[Hashable, Any]: ...
