"""Kalshi node strategy — series-level market belief.

A node is a **macro series' current market-implied belief about its nearest
unresolved event** (see ``reports/graph_definition.md`` (FYP root)). It persists across
snapshots; the event it points at rolls forward as prints resolve.

The primary ``data`` handed to :class:`stg.builders.GraphBuilder` is the
node-feature panel from :func:`stg.panel.build_node_panel` — one row per
(series, date). This strategy simply reads the rows for the current snapshot.

Feature vector (11-D, order fixed)::

    0  implied_mean        6  d_implied_mean   (belief momentum)
    1  implied_std         7  days_to_close
    2  implied_entropy     8  recent_volume
    3  implied_skew        9  net_flow
    4  implied_kurtosis   10  is_bucket
    5  max_stale_days

Standardisation is left to a downstream FeatureStrategy so the split boundary
(train-fold only) is respected.
"""

from __future__ import annotations

from typing import Any, Hashable, List

import numpy as np
import polars as pl

from stg.core import NodeState

FEATURE_ORDER = (
    "implied_mean", "implied_std", "implied_entropy", "implied_skew",
    "implied_kurtosis", "max_stale_days", "d_implied_mean", "days_to_close",
    "recent_volume", "net_flow", "is_bucket",
)


class SeriesBeliefNodes:
    """One node per canonical macro series active in the snapshot.

    A row with a null ``implied_mean`` (ladder too thin that day to recover a
    distribution) is treated as the node being absent.
    """

    N_FEATURES = len(FEATURE_ORDER)

    def __init__(self, series_col: str = "series",
                 require_belief: bool = True) -> None:
        self.series_col = series_col
        self.require_belief = require_belief

    def identify_nodes(self, data: pl.DataFrame, **kwargs: Any) -> List[Hashable]:
        df = data
        if self.require_belief and "implied_mean" in df.columns:
            df = df.filter(pl.col("implied_mean").is_not_null())
        return df[self.series_col].unique().sort().to_list()

    def build_node_state(self, node_id: Hashable, data: pl.DataFrame,
                         **kwargs: Any) -> NodeState:
        row = data.filter(pl.col(self.series_col) == node_id).tail(1)
        feats = np.zeros(self.N_FEATURES, dtype=np.float64)
        meta: dict[str, Any] = {"node_type": "series", "series": node_id}
        if row.height:
            r = row.row(0, named=True)
            for i, col in enumerate(FEATURE_ORDER):
                v = r.get(col)
                feats[i] = float(v) if v is not None else 0.0
            meta["event_ticker"] = r.get("event_ticker")
            meta["days_to_close"] = r.get("days_to_close")
        return NodeState(node_id=node_id, features=np.nan_to_num(feats), metadata=meta)
