"""Built-in post-processing strategies."""

from __future__ import annotations

from typing import Any

from stg_infra.stg.core import EdgeState, GraphSnapshot


class AddSelfLoops:
    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight
    def process(self, snapshot: GraphSnapshot, **kwargs: Any) -> GraphSnapshot:
        for nid in snapshot.node_ids:
            if not snapshot._graph.has_edge(nid, nid):
                snapshot.add_edge(EdgeState(nid, nid, self.weight))
        return snapshot


class Symmetrise:
    def process(self, snapshot: GraphSnapshot, **kwargs: Any) -> GraphSnapshot:
        to_add = []
        for u, v in list(snapshot.edges):
            if not snapshot._graph.has_edge(v, u):
                d = snapshot._graph.edges[u, v]
                to_add.append(EdgeState(v, u, d.get("weight", 1.0), d.get("features"), d.get("metadata", {})))
        for e in to_add:
            snapshot.add_edge(e)
        return snapshot


class PruneIsolated:
    def process(self, snapshot: GraphSnapshot, **kwargs: Any) -> GraphSnapshot:
        snapshot._graph.remove_nodes_from(
            [n for n in snapshot.node_ids if snapshot._graph.degree(n) == 0]
        )
        return snapshot


class TopKEdges:
    def __init__(self, k: int = 10) -> None:
        self.k = k
    def process(self, snapshot: GraphSnapshot, **kwargs: Any) -> GraphSnapshot:
        to_remove = []
        for node in snapshot.node_ids:
            out = list(snapshot._graph.out_edges(node, data=True))
            if len(out) > self.k:
                out.sort(key=lambda e: e[2].get("weight", 0), reverse=True)
                to_remove.extend((node, v) for _, v, _ in out[self.k:])
        snapshot._graph.remove_edges_from(to_remove)
        return snapshot


class NormaliseEdgeWeights:
    def process(self, snapshot: GraphSnapshot, **kwargs: Any) -> GraphSnapshot:
        for node in snapshot.node_ids:
            out = list(snapshot._graph.out_edges(node, data=True))
            if not out:
                continue
            total = sum(e[2].get("weight", 1.0) for e in out)
            if total == 0:
                continue
            for _, v, data in out:
                data["weight"] = data.get("weight", 1.0) / total
        return snapshot
