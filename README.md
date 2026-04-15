# `stg` — Spatio-Temporal Graph Infrastructure

Strategy-pattern framework for modelling event prediction markets as spatio-temporal graphs.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from stg.builders.builder import GraphBuilder
from stg.io.loaders import DatasetLoader
from stg.adapters.kalshi import KalshiMarketNodes, KalshiEventEdges
from stg.edges.strategies import KNNEdges, CompositeEdges
from stg.temporal.strategies import FixedWindowTemporal
from stg.strategies.post_process import AddSelfLoops, Symmetrise

# load data from google drive in a data/ folder
markets = DatasetLoader("data/markets_*.parquet", sort_by="created_time").load()
trades  = DatasetLoader("data/trades_*.parquet").load()

stg = (
    GraphBuilder()
    .with_temporal(FixedWindowTemporal(every="1h"))
    .with_nodes(KalshiMarketNodes())
    .with_edges(CompositeEdges([KalshiEventEdges(weight=2.0), KNNEdges(k=5)]))
    .with_post_process(Symmetrise())
    .with_post_process(AddSelfLoops())
    .build(markets, auxiliary={"trades": trades})
)

feat = stg.feature_tensor()    # (T, N, F) numpy
adj  = stg.adjacency_tensor()  # (T, N, N) numpy
```

## Large Datasets

```python
# Lazy scan — predicates and column selection pushed to Parquet I/O
lf = DatasetLoader(
    paths="data/markets_*.parquet",
    columns=["ticker", "event_ticker", "yes_bid", "yes_ask", "last_price", "volume", "status", "created_time"],
    filters={"status": ["active", "finalized"]},
).scan()
markets = lf.collect()

# Out-of-core incremental building
from stg import IncrementalGraphBuilder
inc = IncrementalGraphBuilder(builder)
for chunk in DatasetLoader("data/markets_*.parquet").load_iter(files_per_batch=20):
    inc.ingest(chunk)
stg = inc.finalise()
```
