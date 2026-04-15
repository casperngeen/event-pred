"""Built-in node construction strategies — pure Polars."""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, List, Optional

import numpy as np
import polars as pl

from stg.core import NodeState


class ColumnNodeStrategy:
    """Each unique value in ``node_col`` becomes a node with features from ``feature_cols``."""

    def __init__(
        self,
        node_col: str,
        feature_cols: List[str],
        agg: str = "mean",
        metadata_cols: Optional[List[str]] = None,
    ) -> None:
        self.node_col = node_col
        self.feature_cols = feature_cols
        self.agg = agg
        self.metadata_cols = metadata_cols or []

    def identify_nodes(self, data: pl.DataFrame, **kwargs: Any) -> List[Hashable]:
        return data[self.node_col].unique().sort().to_list()

    def build_node_state(self, node_id: Hashable, data: pl.DataFrame, **kwargs: Any) -> NodeState:
        subset = data.filter(pl.col(self.node_col) == node_id)
        available = [c for c in self.feature_cols if c in subset.columns]

        if self.agg == "last":
            row = subset.select(available).tail(1)
        elif self.agg == "first":
            row = subset.select(available).head(1)
        elif self.agg == "sum":
            row = subset.select([pl.col(c).sum() for c in available])
        elif self.agg == "max":
            row = subset.select([pl.col(c).max() for c in available])
        elif self.agg == "min":
            row = subset.select([pl.col(c).min() for c in available])
        else:
            row = subset.select([pl.col(c).mean() for c in available])

        features = np.nan_to_num(row.to_numpy().flatten().astype(np.float64), nan=0.0)

        meta: Dict[str, Any] = {}
        if self.metadata_cols:
            last = subset.tail(1)
            for col in self.metadata_cols:
                if col in last.columns:
                    meta[col] = last[col][0]

        return NodeState(node_id=node_id, features=features, metadata=meta)


class CustomNodeStrategy:
    """Fully custom strategy via callables."""

    def __init__(
        self,
        identify_fn: Callable[..., List[Hashable]],
        build_fn: Callable[..., NodeState],
    ) -> None:
        self._identify = identify_fn
        self._build = build_fn

    def identify_nodes(self, data: pl.DataFrame, **kwargs: Any) -> List[Hashable]:
        return self._identify(data, **kwargs)

    def build_node_state(self, node_id: Hashable, data: pl.DataFrame, **kwargs: Any) -> NodeState:
        return self._build(node_id, data, **kwargs)
