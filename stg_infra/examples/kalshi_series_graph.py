"""Build the series-level spatio-temporal graph end to end.

    cd event-pred
    venv/bin/python scripts/build_panels.py
    venv/bin/python scripts/run_structure_estimation.py
    venv/bin/python stg_infra/examples/kalshi_series_graph.py

Node  = a macro series' market-implied belief about its nearest unresolved event
Edge  = directed influence (from the Stage-1 validated adjacency) + same-release
Snap  = one per macro-resolution date
Label = 3-snapshot-ahead change in the node's implied mean
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "stg_infra")

from stg.builders import GraphBuilder
from stg.temporal.strategies import MacroResolutionTemporal
from stg.nodes.kalshi import SeriesBeliefNodes
from stg.edges.kalshi import SameReleaseEdges, SurpriseInfluenceEdges
from stg.labels.kalshi import ImpliedMeanChangeLabels
from stg.panel import snapshot_dates

NODE_PANEL = Path("artifacts/panels/node_panel_event.parquet")


def main() -> None:
    node_panel = pl.read_parquet(NODE_PANEL)
    dates = snapshot_dates("event")

    builder = (
        GraphBuilder()
        .with_temporal(MacroResolutionTemporal(snapshot_dates=dates))
        .with_nodes(SeriesBeliefNodes())
        .with_labels(ImpliedMeanChangeLabels(node_panel, horizon=3))
    )
    adj = Path("artifacts/adjacency_is.parquet")
    if adj.exists():
        builder = builder.with_edges(SurpriseInfluenceEdges(adj))
    builder = builder.with_edges(SameReleaseEdges())

    stg = builder.build(node_panel)
    print(stg.summary())
    non_empty = [s for s in stg if s.num_nodes]
    print(f"non-empty snapshots: {len(non_empty)}")
    if non_empty:
        s = non_empty[len(non_empty) // 2]
        print(f"\nmid snapshot {s.timestamp.date()}: "
              f"{s.num_nodes} nodes, {s.num_edges} edges")
        print("  nodes:", ", ".join(sorted(s.node_ids)))
        print("  edges:", [(u, v, round(s.get_edge(u, v).weight, 2))
                           for u, v in s.edges][:8])


if __name__ == "__main__":
    main()
