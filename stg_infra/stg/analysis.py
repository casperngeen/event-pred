"""Analysis and visualisation utilities."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import polars as pl

from stg.core import GraphSnapshot, SpatioTemporalGraph

logger = logging.getLogger(__name__)


class TemporalMetrics:
    @staticmethod
    def compute(stg: SpatioTemporalGraph) -> pl.DataFrame:
        records: List[Dict[str, Any]] = []
        for snap in stg:
            g = snap.nx_graph
            n = snap.num_nodes
            degrees = [d for _, d in g.degree()]
            ug = g.to_undirected()
            records.append({
                "timestamp": str(snap.timestamp),
                "num_nodes": n,
                "num_edges": snap.num_edges,
                "density": round(nx.density(g), 6) if n > 1 else 0.0,
                "avg_degree": round(float(np.mean(degrees)), 4) if degrees else 0.0,
                "num_components": nx.number_weakly_connected_components(g) if n > 0 else 0,
                "avg_clustering": round(nx.average_clustering(ug), 6) if n > 0 else 0.0,
            })
        return pl.DataFrame(records)


class NodeEvolution:
    @staticmethod
    def track(stg: SpatioTemporalGraph, node_id: Any) -> pl.DataFrame:
        records: List[Dict[str, Any]] = []
        for snap in stg:
            ns = snap.get_node(node_id)
            if ns is None:
                continue
            row: Dict[str, Any] = {"timestamp": str(snap.timestamp)}
            for i, v in enumerate(ns.features):
                row[f"f_{i}"] = float(v)
            row.update({k: str(v) for k, v in ns.metadata.items()})
            records.append(row)
        return pl.DataFrame(records)


class GraphDiff:
    @staticmethod
    def diff(a: GraphSnapshot, b: GraphSnapshot) -> Dict[str, Any]:
        na, nb = set(a.node_ids), set(b.node_ids)
        ea, eb = set(a.edges), set(b.edges)
        return {
            "timestamp_a": a.timestamp, "timestamp_b": b.timestamp,
            "nodes_added": nb - na, "nodes_removed": na - nb, "nodes_stable": na & nb,
            "edges_added": eb - ea, "edges_removed": ea - eb, "edges_stable": ea & eb,
        }

    @staticmethod
    def evolution_summary(stg: SpatioTemporalGraph) -> pl.DataFrame:
        records: List[Dict[str, Any]] = []
        for i in range(len(stg) - 1):
            d = GraphDiff.diff(stg[i], stg[i + 1])
            records.append({
                "from": str(d["timestamp_a"]), "to": str(d["timestamp_b"]),
                "nodes_added": len(d["nodes_added"]), "nodes_removed": len(d["nodes_removed"]),
                "nodes_stable": len(d["nodes_stable"]),
                "edges_added": len(d["edges_added"]), "edges_removed": len(d["edges_removed"]),
                "edges_stable": len(d["edges_stable"]),
            })
        return pl.DataFrame(records)


class GraphVisualiser:
    @staticmethod
    def plot_snapshot(
        snapshot: GraphSnapshot,
        figsize: Tuple[int, int] = (12, 8),
        node_color_feature: int = 0,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        g = snapshot.nx_graph
        pos = nx.spring_layout(g, seed=42, k=1.5 / max(1, np.sqrt(snapshot.num_nodes)))
        colours = []
        for n in g.nodes:
            f = g.nodes[n].get("features")
            if f is not None and len(f) > node_color_feature:
                colours.append(float(f[node_color_feature]))
            else:
                colours.append(0.0)
        widths = [g.edges[e].get("weight", 0.5) for e in g.edges]
        mx = max(widths) if widths else 1
        widths = [w / mx * 3 for w in widths]
        fig, ax = plt.subplots(figsize=figsize)
        drawn = nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colours, cmap=plt.get_cmap("viridis"), node_size=80, alpha=0.85)
        nx.draw_networkx_edges(g, pos, ax=ax, width=widths, alpha=0.4, arrows=True, arrowsize=8)
        plt.colorbar(drawn, ax=ax, label=f"Feature {node_color_feature}")
        ax.set_title(title or f"Snapshot @ {snapshot.timestamp}")
        ax.axis("off"); plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
