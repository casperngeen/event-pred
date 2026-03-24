"""Test suite for stg — pure Polars."""

import os, sys, tempfile, json, pickle
from datetime import datetime, timedelta, timezone
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stg_infra.stg.core import NodeState, EdgeState, GraphSnapshot, SpatioTemporalGraph
from stg_infra.stg.builders.builder import GraphBuilder
from stg_infra.stg.nodes.strategies import ColumnNodeStrategy, CustomNodeStrategy
from stg_infra.stg.edges.strategies import CosineSimilarityEdges, KNNEdges, SharedAttributeEdges, CompositeEdges, ExplicitEdges
from stg_infra.stg.temporal.strategies import FixedWindowTemporal, SnapshotTemporal, SlidingWindowTemporal, EventDrivenTemporal
from stg_infra.stg.strategies.post_process import AddSelfLoops, Symmetrise, PruneIsolated, TopKEdges, NormaliseEdgeWeights
from stg_infra.stg.strategies.features import StandardScaleFeatures, MinMaxScaleFeatures, LogTransformFeatures, ChainFeatures
from stg_infra.stg.adapters.kalshi import KalshiMarketNodes, KalshiTradeAugmentedNodes, KalshiEventEdges, KalshiOutcomeLabels
from stg_infra.stg.io.loaders import DatasetLoader, IncrementalGraphBuilder, GraphSerialiser
from stg_infra.stg.analysis import TemporalMetrics, NodeEvolution, GraphDiff


def make_market_data(n_markets=20, n_snapshots=5, seed=42) -> pl.DataFrame:
    rng = np.random.RandomState(seed)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    events = [f"EVT-{i}" for i in range(n_markets // 4 + 1)]
    rows = []
    for si in range(n_snapshots):
        t = base + timedelta(hours=si)
        for m in range(n_markets):
            yb = int(rng.randint(5, 95)); sp = int(rng.randint(1, 10))
            rows.append({
                "ticker": f"MKT-{m:03d}", "event_ticker": events[m % len(events)],
                "market_type": "binary", "title": f"Outcome {m}",
                "yes_sub_title": "Yes", "no_sub_title": "No",
                "status": str(rng.choice(["active", "finalized"])),
                "yes_bid": yb, "yes_ask": yb + sp, "no_bid": 100 - yb - sp, "no_ask": 100 - yb,
                "last_price": int(yb + rng.randint(-2, 3)),
                "volume": int(rng.randint(100, 50000)),
                "volume_24h": int(rng.randint(50, 10000)),
                "open_interest": int(rng.randint(10, 5000)),
                "result": str(rng.choice(["yes", "no", ""])),
                "created_time": t, "_fetched_at": t + timedelta(minutes=5),
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("created_time").cast(pl.Datetime("us", "UTC")),
        pl.col("_fetched_at").cast(pl.Datetime("us", "UTC")),
    )


def make_trade_data(n_trades=100, n_markets=20, seed=42) -> pl.DataFrame:
    rng = np.random.RandomState(seed)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n_trades):
        yp = int(rng.randint(5, 95))
        rows.append({
            "trade_id": f"TRD-{i:06d}", "ticker": f"MKT-{rng.randint(0, n_markets):03d}",
            "count": int(rng.randint(1, 20)), "yes_price": yp, "no_price": 100 - yp,
            "taker_side": str(rng.choice(["yes", "no"])),
            "created_time": base + timedelta(minutes=int(rng.randint(0, 300))),
            "_fetched_at": base + timedelta(minutes=int(rng.randint(300, 600))),
        })
    return pl.DataFrame(rows).with_columns(
        pl.col("created_time").cast(pl.Datetime("us", "UTC")),
        pl.col("_fetched_at").cast(pl.Datetime("us", "UTC")),
    )


passed = 0; failed = 0; errors = []
def check(name, condition):
    global passed, failed
    if condition: passed += 1; print(f"  ✓ {name}")
    else: failed += 1; errors.append(name); print(f"  ✗ {name}")


print("=" * 70)
print("STG TEST SUITE — pure Polars")
print("=" * 70)

# --- 1. Core ---------------------------------------------------------------
print("\n[1] Core data model")
ns = NodeState("A", [1.0, 2.0, 3.0]); check("NodeState ndarray", ns.features.shape == (3,))
es = EdgeState("A", "B", 0.7, [1, 2]); check("EdgeState", es.weight == 0.7)
snap = GraphSnapshot(timestamp=0)
snap.add_node(NodeState("A", [1.0, 2.0])); snap.add_node(NodeState("B", [3.0, 4.0]))
snap.add_edge(EdgeState("A", "B", 0.5))
check("nodes=2", snap.num_nodes == 2); check("edges=1", snap.num_edges == 1)
check("get_node", snap.get_node("A").features[0] == 1.0)
check("get_edge", snap.get_edge("A", "B").weight == 0.5)
check("feature_matrix", snap.feature_matrix().shape == (2, 2))
check("adjacency", snap.adjacency_matrix()[0, 1] == 0.5)

stg = SpatioTemporalGraph()
for t in range(5):
    s = GraphSnapshot(timestamp=t); s.add_node(NodeState("X", [float(t)])); stg.add_snapshot(s)
check("STG len", len(stg) == 5); check("STG slice", len(stg[1:3]) == 2)
check("STG window", len(stg.window(1, 3)) == 3); check("STG latest", stg.latest().timestamp == 4)

# Test __getitem__ int returns GraphSnapshot with .node_ids
first: GraphSnapshot = stg[0]
check("stg[0].node_ids", first.node_ids == ["X"])

# --- 2. Temporal -----------------------------------------------------------
print("\n[2] Temporal strategies")
df_t = make_market_data(10, 5)
slices = FixedWindowTemporal(time_col="created_time", every="1h").slice(df_t)
check("FixedWindow", len(slices) > 0); check("slice is pl.DataFrame", isinstance(slices[0]["data"], pl.DataFrame))
check("SnapshotTemporal", len(SnapshotTemporal(time_col="_fetched_at").slice(df_t)) > 0)
check("SlidingWindow", len(SlidingWindowTemporal(time_col="created_time", window_size="2h", stride="1h").slice(df_t)) >= len(slices))
check("EventDriven", len(EventDrivenTemporal(time_col="created_time", event_col="status").slice(df_t)) > 0)

# --- 3. Nodes --------------------------------------------------------------
print("\n[3] Node strategies")
col_node = ColumnNodeStrategy("ticker", ["yes_bid", "yes_ask", "volume"], "mean", ["event_ticker"])
window_data = slices[0]["data"]
node_ids = col_node.identify_nodes(window_data)
check("identifies nodes", len(node_ids) > 0)
ns1 = col_node.build_node_state(node_ids[0], window_data)
check("3D features", ns1.features.shape == (3,)); check("metadata", "event_ticker" in ns1.metadata)

# --- 4. Edges --------------------------------------------------------------
print("\n[4] Edge strategies")
test_snap = GraphSnapshot(timestamp=0)
for nid in node_ids[:6]: test_snap.add_node(col_node.build_node_state(nid, window_data))
cos_edges = CosineSimilarityEdges(0.3).build_edges(test_snap, window_data)
check("Cosine edges", len(cos_edges) > 0)
check("KNN edges", len(KNNEdges(k=2).build_edges(test_snap, window_data)) > 0)
check("SharedAttribute", isinstance(SharedAttributeEdges("event_ticker", "ticker").build_edges(test_snap, window_data), list))
check("Composite", len(CompositeEdges([CosineSimilarityEdges(0.3), KNNEdges(k=2)]).build_edges(test_snap, window_data)) > 0)
exs = GraphSnapshot(timestamp=0); exs.add_node(NodeState("MKT-000", [1.0])); exs.add_node(NodeState("MKT-001", [2.0]))
check("Explicit", ExplicitEdges(lambda s, d, **kw: [("MKT-000", "MKT-001", 0.9)]).build_edges(exs, window_data)[0].weight == 0.9)

# --- 5. Post-processing ----------------------------------------------------
print("\n[5] Post-processing")
snap_pp = test_snap.copy()
for e in cos_edges[:10]: snap_pp.add_edge(e)
check("SelfLoops", any(u == v for u, v in AddSelfLoops().process(snap_pp.copy()).edges))
check("Symmetrise", all(Symmetrise().process(snap_pp.copy())._graph.has_edge(v, u) for u, v in list(snap_pp.edges)[:3]))
sn = NormaliseEdgeWeights().process(snap_pp.copy())
for n in sn.node_ids:
    out = list(sn._graph.out_edges(n, data=True))
    if out: check("NormaliseWeights", abs(sum(e[2].get("weight", 0) for e in out) - 1.0) < 1e-6); break
st = TopKEdges(k=1).process(snap_pp.copy())
for n in st.node_ids:
    if st._graph.out_degree(n) > 0: check("TopK", st._graph.out_degree(n) <= 1); break

# --- 6. Features ------------------------------------------------------------
print("\n[6] Feature transforms")
r = LogTransformFeatures().transform_node_features(np.array([[100.0, -5.0, 0.0]]))
check("Log+", r[0,0] > 0); check("Log-", r[0,1] < 0); check("Log0", r[0,2] == 0.0)
check("Chain", ChainFeatures([LogTransformFeatures(), MinMaxScaleFeatures()]).transform_node_features(np.random.randn(10,4)).shape == (10,4))

# --- 7. Kalshi adapter -----------------------------------------------------
print("\n[7] Kalshi adapter")
markets_df = make_market_data(20, 3); trades_df = make_trade_data(200, 20)
kn = KalshiMarketNodes(); k_ids = kn.identify_nodes(markets_df)
check("20 markets", len(k_ids) == 20)
ks = kn.build_node_state(k_ids[0], markets_df); check("8D", ks.features.shape == (8,))
check("meta", "event_ticker" in ks.metadata)
check("TradeAugmented 12D", KalshiTradeAugmentedNodes().build_node_state(k_ids[0], markets_df, auxiliary={"trades": trades_df}).features.shape == (12,))
k_snap = GraphSnapshot(timestamp=0)
for nid in k_ids[:10]: k_snap.add_node(kn.build_node_state(nid, markets_df))
check("EventEdges", len(KalshiEventEdges().build_edges(k_snap, markets_df)) > 0)
labels = KalshiOutcomeLabels().extract_labels(k_ids[:5], markets_df)
check("Labels", len(labels) == 5 and all(isinstance(v, int) for v in labels.values()))

# --- 8. Full builder -------------------------------------------------------
print("\n[8] Full GraphBuilder")
builder = (
    GraphBuilder()
    .with_temporal(FixedWindowTemporal(time_col="created_time", every="1h"))
    .with_nodes(KalshiMarketNodes())
    .with_edges(KalshiEventEdges())
    .with_post_process(AddSelfLoops()).with_post_process(Symmetrise())
    .with_labels(KalshiOutcomeLabels())
)
full_stg = builder.build(markets_df)
check("STG", isinstance(full_stg, SpatioTemporalGraph) and len(full_stg) > 0)
check("all nodes", all(s.num_nodes > 0 for s in full_stg))
check("all edges", all(s.num_edges > 0 for s in full_stg))
check("labels", hasattr(full_stg, "_labels") and len(full_stg._labels) > 0)
# Verify __getitem__ typing
snap0: GraphSnapshot = full_stg[0]
check("stg[0].node_ids accessible", len(snap0.node_ids) > 0)
print(f"  → {full_stg.summary()}")

# --- 9. Incremental --------------------------------------------------------
print("\n[9] Incremental builder")
inc = IncrementalGraphBuilder(
    GraphBuilder().with_temporal(FixedWindowTemporal(time_col="created_time", every="1h"))
    .with_nodes(ColumnNodeStrategy("ticker", ["yes_bid", "volume"]))
    .with_edges(CosineSimilarityEdges(0.5)),
)
inc.ingest(markets_df.head(markets_df.height // 2))
inc.ingest(markets_df.tail(markets_df.height - markets_df.height // 2))
check("Incremental", len(inc.finalise()) > 0)

# --- 10. Serialisation -----------------------------------------------------
print("\n[10] Serialisation")
with tempfile.TemporaryDirectory() as tmpdir:
    GraphSerialiser.to_pickle(full_stg, os.path.join(tmpdir, "t.pkl"))
    check("Pickle", len(GraphSerialiser.from_pickle(os.path.join(tmpdir, "t.pkl"))) == len(full_stg))
    paths = GraphSerialiser.to_numpy(full_stg, os.path.join(tmpdir, "np"))
    check("Numpy", "timestamps" in paths and "node_ids" in paths)
    GraphSerialiser.to_edge_list(full_stg, os.path.join(tmpdir, "e.csv"))
    el = pl.read_csv(os.path.join(tmpdir, "e.csv"))
    check("Edge CSV", el.height > 0 and set(el.columns) == {"timestamp", "source", "target", "weight"})

# --- 11. Analysis ----------------------------------------------------------
print("\n[11] Analysis")
check("Metrics", "density" in TemporalMetrics.compute(full_stg).columns)
check("NodeEvolution", NodeEvolution.track(full_stg, k_ids[0]).height > 0)
if len(full_stg) >= 2:
    check("Diff", "nodes_added" in GraphDiff.diff(full_stg[0], full_stg[1]))
    check("EvolutionSummary", GraphDiff.evolution_summary(full_stg).height == len(full_stg) - 1)

# --- 12. Multi-file loader -------------------------------------------------
print("\n[12] DatasetLoader")
with tempfile.TemporaryDirectory() as tmpdir:
    cs = markets_df.height // 5
    for i in range(5):
        markets_df.slice(i * cs, cs).write_csv(os.path.join(tmpdir, f"m_{i:04d}.csv"))
    loader = DatasetLoader(os.path.join(tmpdir, "m_*.csv"), sort_by="ticker")
    check("5 files", loader.metadata()["num_files"] == 5)
    full = loader.load(); check(f"rows={full.height}", full.height == cs * 5)
    lf = loader.scan(); check("LazyFrame", isinstance(lf, pl.LazyFrame))
    check("scan==load", lf.collect().height == full.height)
    bc = 0; rc = 0
    for b in loader.load_iter(files_per_batch=2): bc += 1; rc += b.height
    check("iter batches=3", bc == 3); check("iter rows", rc == cs * 5)
    check("filtered", DatasetLoader(os.path.join(tmpdir, "m_*.csv"), filters={"status": "active"}).load().height <= full.height)
    check("columns", set(DatasetLoader(os.path.join(tmpdir, "m_*.csv"), columns=["ticker", "yes_bid"]).load().columns) == {"ticker", "yes_bid"})

# --- 13. Parquet reading ---------------------------------------------------
print("\n[13] Native Parquet")
mkt_file = "/mnt/user-data/uploads/markets_710000_720000.parquet"
trd_file = "/mnt/user-data/uploads/trades_30270000_30280000.parquet"
if os.path.exists(mkt_file):
    mkt = pl.read_parquet(mkt_file)
    check(f"Markets {mkt.shape}", mkt.height > 0 and "ticker" in mkt.columns)
    trd = pl.read_parquet(trd_file)
    check(f"Trades {trd.shape}", trd.height > 0 and "ticker" in trd.columns)
    lf = pl.scan_parquet(mkt_file)
    check("Lazy scan", isinstance(lf, pl.LazyFrame))
    filt = lf.filter(pl.col("status") == "active").select(["ticker", "yes_bid"]).collect()
    check("Pushdown", filt.width == 2)
    real_stg = (
        GraphBuilder()
        .with_temporal(FixedWindowTemporal(time_col="_fetched_at", every="1h"))
        .with_nodes(KalshiMarketNodes())
        .with_edges(KalshiEventEdges())
        .with_post_process(AddSelfLoops())
        .build(mkt)
    )
    check(f"Real STG: {real_stg.summary()}", len(real_stg) > 0)
else:
    print("  (parquet files not found)")

print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
if errors: print(f"FAILED: {errors}")
print("=" * 70)
sys.exit(0 if failed == 0 else 1)
