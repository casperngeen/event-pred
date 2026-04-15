"""
Train/Test Split Example: Jan 2025 = Train, Feb 2025 = Test
Process all data for each month together
"""

import logging
import sys
import os

import polars as pl

from stg.edges.kalshi import KalshiEventEdges
from stg.labels.kalshi import KalshiPriceChangeLabels
from stg.nodes.kalshi import KalshiTickerNodes
from stg.builders.builder import GraphBuilder
from stg.analysis import TemporalMetrics
from stg.edges.strategies import CompositeEdges, KNNEdges
from stg.temporal.strategies import FixedWindowTemporal
from stg.strategies.post_process import AddSelfLoops, Symmetrise, TopKEdges, NormaliseEdgeWeights
from stg.strategies.features import LogTransformFeatures, StandardScaleFeatures, ChainFeatures

from scripts.data_split_odd_even_months import SplitByOddEvenMonths

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ============================================================================
# 1. STREAM DATA AND FILTER BY MONTH
# ============================================================================
markets_splitter = SplitByOddEvenMonths("data/markets/markets_kalshi/*.parquet", file_format="parquet", time_col="created_time", batch_size=5)
trades_splitter  = SplitByOddEvenMonths("data/trades/trades_kalshi/*.parquet", file_format="parquet", time_col="created_time", batch_size=5)

def collect_month(splitter, year: int, month: int, which: str) -> pl.DataFrame:
    """Collect all batches for a given year-month from either train or test side."""
    parts = []
    for train_chunk, test_chunk in splitter.stream_split():
        chunk = train_chunk if which == "train" else test_chunk
        filtered = chunk.filter(
            (pl.col("created_time").dt.year() == year) &
            (pl.col("created_time").dt.month() == month)
        )
        if not filtered.is_empty():
            parts.append(filtered)
    return pl.concat(parts) if parts else pl.DataFrame()

# Collect all Jan 2025 data for train
markets_jan = collect_month(markets_splitter, 2025, 1, "train")
trades_jan  = collect_month(trades_splitter, 2025, 1, "train")

# Collect all Feb 2025 data for test
markets_feb = collect_month(markets_splitter, 2025, 2, "test")
trades_feb  = collect_month(trades_splitter, 2025, 2, "test")


# ============================================================================
# 2. BUILD TRAIN GRAPH (Jan 2025)
# ============================================================================
print("\nBuilding Train Graph (Jan 2025)...")
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
    .with_labels(KalshiPriceChangeLabels(trades_jan))
    .build(trades_jan, auxiliary={"markets": markets_jan})
)

print("Train summary:")
print(stg_train.summary())
print("Train metrics:", TemporalMetrics.compute(stg_train))


# ============================================================================
# 3. BUILD TEST GRAPH (Feb 2025)
# ============================================================================
print("\nBuilding Test Graph (Feb 2025)...")
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
    .with_labels(KalshiPriceChangeLabels(trades_feb))
    .build(trades_feb, auxiliary={"markets": markets_feb})
)

print("Test summary:")
print(stg_test.summary())
print("Test metrics:", TemporalMetrics.compute(stg_test))
