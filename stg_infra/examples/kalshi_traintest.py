"""
Train/Test Split Example: Odd months = Train, Even months = Test
"""

import logging
import sys
import os

import polars as pl

from stg_infra.stg.io.loaders import DatasetLoader
from stg_infra.stg.edges.kalshi import KalshiEventEdges
from stg_infra.stg.labels.kalshi import KalshiPriceChangeLabels
from stg_infra.stg.nodes.kalshi import KalshiTickerNodes
from stg_infra.stg.builders.builder import GraphBuilder
from stg_infra.stg.analysis import TemporalMetrics
from stg_infra.stg.edges.strategies import CompositeEdges, KNNEdges
from stg_infra.stg.temporal.strategies import FixedWindowTemporal
from stg_infra.stg.strategies.post_process import AddSelfLoops, Symmetrise, TopKEdges, NormaliseEdgeWeights
from stg_infra.stg.strategies.features import LogTransformFeatures, StandardScaleFeatures, ChainFeatures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ============================================================================
# 1. LOAD DATA 
# ============================================================================
markets_train_loader = DatasetLoader("data/markets/markets_kalshi_odd/markets_2025-01.parquet", file_format="parquet")
trades_train_loader  = DatasetLoader("data/trades/trades_kalshi_odd/trades_2025-01.parquet", file_format="parquet")

markets_test_loader  = DatasetLoader("data/markets/markets_kalshi_even/markets_2025-02.parquet", file_format="parquet")
trades_test_loader   = DatasetLoader("data/trades/trades_kalshi_even/trades_2025-02.parquet", file_format="parquet")

markets_train = markets_train_loader.load()
trades_train  = trades_train_loader.load()
markets_test  = markets_test_loader.load()
trades_test   = trades_test_loader.load()

# ============================================================================
# 2. BUILD TRAIN GRAPH (Jan 2025)
# ============================================================================
stg_train = (
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
    .with_labels(KalshiPriceChangeLabels(trades_train))
    .build(trades_train, auxiliary={"markets": markets_train})
)

print("\nTrain (Jan 2025) summary:")
print(stg_train.summary())
print("Train metrics:", TemporalMetrics.compute(stg_train))

# ============================================================================
# 3. BUILD TEST GRAPH (Feb 2025)
# ============================================================================
stg_test = (
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
    .with_labels(KalshiPriceChangeLabels(trades_test))
    .build(trades_test, auxiliary={"markets": markets_test})
)

print("\nTest (Feb 2025) summary:")
print(stg_test.summary())
print("Test metrics:", TemporalMetrics.compute(stg_test))
