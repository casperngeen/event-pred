"""
Example: Building a Spatio-Temporal Graph from Kalshi data (pure Polars).
"""

import logging
import sys
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from stg.io.loaders import DatasetLoader
from stg.edges.kalshi import KalshiEventEdges
from stg.labels.kalshi import KalshiPriceChangeLabels
from stg.nodes.kalshi import KalshiTickerNodes
from stg.builders.builder import GraphBuilder
from stg.analysis import TemporalMetrics, NodeEvolution, GraphDiff
from stg.edges.strategies import CompositeEdges, KNNEdges
from stg.temporal.strategies import FixedWindowTemporal
from stg.strategies.post_process import AddSelfLoops, Symmetrise, TopKEdges, NormaliseEdgeWeights
from stg.strategies.features import LogTransformFeatures, StandardScaleFeatures, ChainFeatures


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ============================================================================
# 1. LOAD DATA
# ============================================================================
# For your real multi-file Kalshi data:
markets = DatasetLoader("data/markets_*.parquet", sort_by="created_time").load()
trades  = DatasetLoader("data/trades_*.parquet", sort_by="created_time").load()
#
# Lazy scan (only reads what's needed from disk):
#   lf = DatasetLoader("data/markets_*.parquet",
#                      columns=["ticker", "event_ticker", "yes_bid", ...],
#                      filters={"status": ["active", "finalized"]}).scan()
#   markets = lf.collect()

# --- Synthetic data ---
# rng = np.random.RandomState(42)
# base = datetime(2025, 1, 1, tzinfo=timezone.utc)
# events = [f"EVT-{i}" for i in range(14)]
# rows = []
# for si in range(24):
#     t = base + timedelta(hours=si)
#     for m in range(50):
#         yb = int(rng.randint(5, 95)); sp = int(rng.randint(1, 10))
#         rows.append({
#             "ticker": f"MKT-{m:03d}", "event_ticker": events[m % len(events)],
#             "market_type": "binary", "title": f"Event {events[m % len(events)]} #{m}",
#             "status": str(rng.choice(["active", "finalized"])),
#             "yes_bid": yb, "yes_ask": yb + sp, "no_bid": 100 - yb - sp, "no_ask": 100 - yb,
#             "last_price": int(yb + rng.randint(-2, 3)),
#             "volume": int(rng.randint(100, 50000)),
#             "volume_24h": int(rng.randint(50, 10000)),
#             "open_interest": int(rng.randint(10, 5000)),
#             "result": str(rng.choice(["yes", "no", ""])),
#             "created_time": t, "_fetched_at": t + timedelta(minutes=5),
#         })

# trade_rows = []
# for i in range(5000):
#     yp = int(rng.randint(5, 95))
#     trade_rows.append({
#         "trade_id": f"TRD-{i:06d}", "ticker": f"MKT-{rng.randint(0, 50):03d}",
#         "count": int(rng.randint(1, 20)), "yes_price": yp, "no_price": 100 - yp,
#         "taker_side": str(rng.choice(["yes", "no"])),
#         "created_time": base + timedelta(minutes=int(rng.randint(0, 1440))),
#     })

# markets_df = pl.DataFrame(rows).with_columns(
#     pl.col("created_time").cast(pl.Datetime("us", "UTC")),
#     pl.col("_fetched_at").cast(pl.Datetime("us", "UTC")),
# )
# trades_df = pl.DataFrame(trade_rows).with_columns(
#     pl.col("created_time").cast(pl.Datetime("us", "UTC")),
# )

# print(f"Markets: {markets_df.shape}, Trades: {trades_df.shape}\n")

# ============================================================================
# 2. BUILD
# ============================================================================
stg = (
    GraphBuilder()
    .with_temporal(FixedWindowTemporal(time_col="created_time", every="2h"))
    .with_nodes(KalshiTickerNodes())
    .with_edges(CompositeEdges([
        KalshiEventEdges(weight=2.0),
        KNNEdges(k=3, metric="cosine"),
    ]))
    .with_features(ChainFeatures([LogTransformFeatures(), StandardScaleFeatures()]))
    .with_post_process(Symmetrise())
    .with_post_process(AddSelfLoops())
    .with_post_process(TopKEdges(k=8))
    .with_post_process(NormaliseEdgeWeights())
    .with_labels(KalshiPriceChangeLabels(trades))
    .build(trades, auxiliary={"markets": markets})
)

print(f"Built: {stg}")
print(f"Summary: {stg.summary()}\n")

# ============================================================================
# 3. ANALYSE
# ============================================================================
print(TemporalMetrics.compute(stg))

# Access a snapshot and its nodes — stg[0] returns GraphSnapshot
first_snapshot = stg[0]
sample_node = first_snapshot.node_ids[0]
print(f"\nNode evolution for {sample_node}:")
print(NodeEvolution.track(stg, sample_node).head(5))

print("\nStructural evolution:")
print(GraphDiff.evolution_summary(stg))

# ============================================================================
# 4. EXPORT TENSORS
# ============================================================================
# feat = stg.feature_tensor()    # (T, N, F) numpy
# adj  = stg.adjacency_tensor()  # (T, N, N) numpy
# print(f"Feature tensor: {feat.shape}, Adjacency tensor: {adj.shape}")

print("\nDone!")
