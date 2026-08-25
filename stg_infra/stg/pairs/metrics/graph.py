from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict


def _normalize_edge(u, v):
    return tuple(sorted((u, v)))


def edge_turnover(edges_t: set[tuple], edges_prev: set[tuple]) -> float:
    """
    1 - Jaccard similarity between consecutive edge sets.
    """
    if len(edges_t) == 0 and len(edges_prev) == 0:
        return 0.0
    inter = len(edges_t.intersection(edges_prev))
    union = len(edges_t.union(edges_prev))
    if union == 0:
        return 0.0
    return float(1.0 - inter / union)


def edge_persistence_lengths(edge_sets_by_time: list[set[tuple]]) -> dict[tuple, list[int]]:
    """
    For each edge, list contiguous run lengths over time.
    """
    runs = defaultdict(list)
    active_run = defaultdict(int)

    all_edges = set().union(*edge_sets_by_time) if edge_sets_by_time else set()

    for edges in edge_sets_by_time:
        # increment runs for edges present
        for e in all_edges:
            if e in edges:
                active_run[e] += 1
            else:
                if active_run[e] > 0:
                    runs[e].append(active_run[e])
                    active_run[e] = 0

    # flush final runs
    for e in all_edges:
        if active_run[e] > 0:
            runs[e].append(active_run[e])

    return dict(runs)


def average_edge_persistence(edge_sets_by_time: list[set[tuple]]) -> float:
    runs = edge_persistence_lengths(edge_sets_by_time)
    vals = [l for _, arr in runs.items() for l in arr]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def homophily_ratio(edges: set[tuple], node_to_category: dict) -> float:
    if len(edges) == 0:
        return float("nan")
    same = 0
    valid = 0
    for u, v in edges:
        cu = node_to_category.get(u)
        cv = node_to_category.get(v)
        if cu is None or cv is None:
            continue
        valid += 1
        if cu == cv:
            same += 1
    if valid == 0:
        return float("nan")
    return float(same / valid)


def graph_metrics_over_time(
    dated_edge_sets: list[tuple[pd.Timestamp, set[tuple]]],
    node_to_category: dict | None = None,
) -> pd.DataFrame:
    """
    Input:
      dated_edge_sets: [(date, {(u,v),...}), ...] sorted by date
    Output DataFrame columns:
      date, num_edges, turnover, homophily
    """
    rows = []
    prev = None
    for dt, edges_raw in dated_edge_sets:
        edges = {_normalize_edge(u, v) for (u, v) in edges_raw}
        turnover = float("nan") if prev is None else edge_turnover(edges, prev)
        homo = float("nan")
        if node_to_category is not None:
            homo = homophily_ratio(edges, node_to_category)
        rows.append(
            {
                "date": pd.Timestamp(dt),
                "num_edges": int(len(edges)),
                "turnover": turnover,
                "homophily": homo,
            }
        )
        prev = edges

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out
