"""Kalshi prediction-market adapter"""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional

import numpy as np
import polars as pl

from stg_infra.stg.core import EdgeState, GraphSnapshot, NodeState
from stg_infra.stg.strategies.protocols import NodeStrategy, EdgeStrategy, LabelStrategy

class KalshiMarketNodes(NodeStrategy):
    """Each market ticker → node.  8-D default feature vector."""

    FEATURE_COLS = [
        "yes_bid", "yes_ask", "no_bid", "no_ask",
        "last_price", "volume", "volume_24h", "open_interest",
    ]

    def __init__(self, node_col: str = "ticker", feature_cols: Optional[List[str]] = None,
                 agg: str = "last") -> None:
        self.node_col = node_col
        self.feature_cols = feature_cols or self.FEATURE_COLS
        self.agg = agg

    def identify_nodes(self, data: pl.DataFrame, **kwargs: Any) -> List[Hashable]:
        return data[self.node_col].unique().sort().to_list()

    def build_node_state(self, node_id: Hashable, data: pl.DataFrame, **kwargs: Any) -> NodeState:
        subset = data.filter(pl.col(self.node_col) == node_id)
        available = [c for c in self.feature_cols if c in subset.columns]
        if self.agg == "last":
            row = subset.select(available).tail(1)
        elif self.agg == "first":
            row = subset.select(available).head(1)
        else:
            row = subset.select([pl.col(c).mean() for c in available])
        feats = np.nan_to_num(row.to_numpy().flatten().astype(np.float64), nan=0.0)
        last = subset.tail(1)
        meta: Dict[str, Any] = {}
        for col in ["event_ticker", "title", "status", "market_type"]:
            if col in last.columns:
                meta[col] = last[col][0]
        return NodeState(node_id, feats, meta)

class KalshiTradeAugmentedNodes(KalshiMarketNodes):
    """Extends market nodes with rich trade-derived features.

    8 base market features + 17 trade features = 25-D default vector.

    Trade features (in order):
        --- Volume & counts ---
        0.  trade_count          total contracts traded (sum of count)
        1.  num_trades           number of individual trades

        --- Price dynamics ---
        2.  mean_yes_price       average yes_price
        3.  vwap                 volume-weighted average yes_price
        4.  price_return         last trade price - first trade price
        5.  price_range          max yes_price - min yes_price
        6.  price_std            std dev of yes_price (realised volatility)

        --- Order flow imbalance ---
        7.  buy_ratio            fraction of trades where taker_side == "yes"
        8.  net_flow             buy_volume - sell_volume (signed contracts)
        9.  weighted_imbalance   net_flow / total_volume (normalised -1 to +1)

        --- Trade arrival intensity ---
        10. mean_inter_arrival   mean seconds between consecutive trades
        11. std_inter_arrival    std dev of inter-arrival times (burstiness)
        12. q1_trade_frac        fraction of trades in 1st quarter of window
        13. q2_trade_frac        fraction of trades in 2nd quarter
        14. q3_trade_frac        fraction of trades in 3rd quarter
        15. q4_trade_frac        fraction of trades in 4th quarter
        16. trade_acceleration   (q4 count - q1 count) / total trades
    """

    N_TRADE_FEATURES = 17

    def build_node_state(self, node_id: Hashable, data: pl.DataFrame, **kwargs: Any) -> NodeState:
        base = super().build_node_state(node_id, data, **kwargs)
        trades: Optional[pl.DataFrame] = kwargs.get("auxiliary", {}).get("trades")

        if trades is None or trades.is_empty():
            extra = np.zeros(self.N_TRADE_FEATURES, dtype=np.float64)
            return NodeState(node_id, np.concatenate([base.features, extra]), base.metadata)

        t_sub = trades.filter(pl.col("ticker") == node_id)
        if t_sub.is_empty():
            extra = np.zeros(self.N_TRADE_FEATURES, dtype=np.float64)
            return NodeState(node_id, np.concatenate([base.features, extra]), base.metadata)

        # Sort by time for temporal calculations
        if "created_time" in t_sub.columns:
            t_sub = t_sub.sort("created_time")

        extra = self._compute_trade_features(t_sub)
        return NodeState(node_id, np.concatenate([base.features, extra]), base.metadata)

    @staticmethod
    def _compute_trade_features(t_sub: pl.DataFrame) -> np.ndarray:
        """Compute all 17 trade features from a single market's windowed trades."""

        # --- Volume & counts ---
        trade_count = t_sub["count"].sum()
        num_trades = t_sub.height

        # --- Price dynamics ---
        yes_prices = t_sub["yes_price"].cast(pl.Float64)
        counts = t_sub["count"].cast(pl.Float64)

        mean_yes = yes_prices.mean()
        if not isinstance(mean_yes, float):
            raise ValueError(f"Unexpected mean type: {type(mean_yes)}")
        
        total_volume = counts.sum()
        vwap = float((yes_prices * counts).sum()) / float(max(total_volume, 1.0))
        price_return = float(yes_prices[-1]) - float(yes_prices[0]) if num_trades >= 2 else 0.0
        price_range = float(yes_prices.max() - yes_prices.min()) if num_trades >= 2 else 0.0  # type: ignore[operator]
        price_std = float(yes_prices.std()) if num_trades >= 2 else 0.0  # type: ignore[arg-type]

        # --- Order flow imbalance ---
        is_buy = t_sub["taker_side"] == "yes"
        buy_ratio = is_buy.mean()
        if not isinstance(buy_ratio, float):
            raise ValueError(f"Unexpected mean type: {type(mean_yes)}")

        buy_volume = float(t_sub.filter(is_buy)["count"].sum())
        sell_volume = float(t_sub.filter(~is_buy)["count"].sum())
        net_flow = float(buy_volume - sell_volume)
        weighted_imbalance = net_flow / max(float(total_volume), 1.0)

        # --- Trade arrival intensity ---
        mean_ia = 0.0
        std_ia = 0.0
        if "created_time" in t_sub.columns and num_trades >= 2:
            timestamps = t_sub["created_time"]
            diffs = timestamps.diff().drop_nulls()
            if diffs.len() > 0:
                # Convert durations to seconds
                secs = diffs.dt.total_seconds()
                mean_ia = float(secs.mean())  # type: ignore[arg-type]
                std_ia = float(secs.std()) if diffs.len() >= 2 else 0.0  # type: ignore[arg-type]

        # Quarter-of-window activity profile
        q1_frac = 0.0; q2_frac = 0.0; q3_frac = 0.0; q4_frac = 0.0
        accel = 0.0
        if "created_time" in t_sub.columns and num_trades >= 1:
            timestamps = t_sub["created_time"]
            t_min = timestamps.min()
            t_max = timestamps.max()
            if t_min != t_max:
                # Normalise timestamps to [0, 1] within the window
                span_us = (t_max - t_min).total_seconds() * 1e6  # type: ignore[union-attr]
                if span_us > 0:
                    offsets = (timestamps - t_min).dt.total_microseconds().cast(pl.Float64) / span_us
                    q1_count = offsets.filter(offsets < 0.25).len()
                    q2_count = offsets.filter((offsets >= 0.25) & (offsets < 0.5)).len()
                    q3_count = offsets.filter((offsets >= 0.5) & (offsets < 0.75)).len()
                    q4_count = offsets.filter(offsets >= 0.75).len()
                    q1_frac = q1_count / num_trades
                    q2_frac = q2_count / num_trades
                    q3_frac = q3_count / num_trades
                    q4_frac = q4_count / num_trades
                    accel = (q4_count - q1_count) / num_trades

        features = np.array([
            # Volume & counts
            float(trade_count), float(num_trades),
            # Price dynamics
            float(mean_yes), float(vwap), float(price_return),
            float(price_range), float(price_std),
            # Order flow imbalance
            float(buy_ratio), float(net_flow), float(weighted_imbalance),
            # Trade arrival intensity
            float(mean_ia), float(std_ia),
            float(q1_frac), float(q2_frac), float(q3_frac), float(q4_frac),
            float(accel),
        ], dtype=np.float64)

        return np.nan_to_num(features, nan=0.0)

class KalshiEventEdges(EdgeStrategy):
    """Connect markets belonging to the same Kalshi event."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def build_edges(self, snapshot: GraphSnapshot, data: pl.DataFrame, **kwargs: Any) -> List[EdgeState]:
        groups: Dict[str, List[Hashable]] = {}
        for nid in snapshot.node_ids:
            node = snapshot.get_node(nid)
            if node is None:
                continue
            evt = node.metadata.get("event_ticker", "")
            if evt:
                groups.setdefault(evt, []).append(nid)
        edges: List[EdgeState] = []
        for members in groups.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    edges.append(EdgeState(members[i], members[j], self.weight,
                                           metadata={"edge_type": "same_event"}))
                    edges.append(EdgeState(members[j], members[i], self.weight,
                                           metadata={"edge_type": "same_event"}))
        return edges


class KalshiPriceCorrelationEdges(EdgeStrategy):
    """Connect markets with correlated price movements."""

    def __init__(self, threshold: float = 0.3, price_col: str = "last_price") -> None:
        self.threshold = threshold
        self.price_col = price_col

    def build_edges(self, snapshot: GraphSnapshot, data: pl.DataFrame, **kwargs: Any) -> List[EdgeState]:
        nodes = snapshot.node_ids
        if len(nodes) < 2:
            return []
        series: Dict[Hashable, np.ndarray] = {}
        for nid in nodes:
            vals = data.filter(pl.col("ticker") == nid).select(
                pl.col(self.price_col).cast(pl.Float64)
            ).to_numpy().flatten()
            if len(vals) >= 2:
                series[nid] = vals
        ids = list(series.keys())
        if len(ids) < 2:
            return []
        min_len = min(len(v) for v in series.values())
        mat = np.vstack([series[k][:min_len] for k in ids])
        corr = np.nan_to_num(np.corrcoef(mat))
        edges: List[EdgeState] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                c = abs(corr[i, j])
                if c >= self.threshold:
                    edges.append(EdgeState(ids[i], ids[j], float(c),
                                           metadata={"edge_type": "price_correlation"}))
                    edges.append(EdgeState(ids[j], ids[i], float(c),
                                           metadata={"edge_type": "price_correlation"}))
        return edges


class KalshiOutcomeLabels(LabelStrategy):
    """Binary outcome labels: yes→1, no→0, else→-1."""

    def __init__(self, node_col: str = "ticker", result_col: str = "result") -> None:
        self.node_col = node_col
        self.result_col = result_col

    def extract_labels(self, node_ids: List[Hashable], data: pl.DataFrame, **kwargs: Any) -> Dict[Hashable, int]:
        labels: Dict[Hashable, int] = {}
        for nid in node_ids:
            sub = data.filter(pl.col(self.node_col) == nid)
            if sub.is_empty() or self.result_col not in sub.columns:
                labels[nid] = -1
                continue
            val = sub[self.result_col][-1]
            labels[nid] = 1 if val == "yes" else (0 if val == "no" else -1)
        return labels
