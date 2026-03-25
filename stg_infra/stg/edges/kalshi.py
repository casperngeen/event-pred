"""Kalshi-specific edge strategies."""

from __future__ import annotations

from typing import Any, Dict, Hashable, List

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

from stg_infra.stg.core import EdgeState, GraphSnapshot


class KalshiEventEdges:
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


class KalshiPriceCorrelationEdges:
    """Connect markets with correlated price movements."""

    def __init__(self, threshold: float = 0.7, price_col: str = "yes_price") -> None:
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


class KalshiMarketToEventEdges:
    """Bidirectional edges connecting each market node to its event super-node.

    Requires both ``KalshiTradeBasedNodes`` and ``KalshiEventNodes`` to have
    run via ``.with_nodes()``, and must be registered via ``.with_edges()``.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def build_edges(self, snapshot: GraphSnapshot, data: pl.DataFrame, **kwargs: Any) -> List[EdgeState]:
        edges: List[EdgeState] = []
        for nid in snapshot.node_ids:
            node = snapshot.get_node(nid)
            if node is None or node.metadata.get("node_type") == "event":
                continue
            evt = node.metadata.get("event_ticker", "")
            if not evt:
                continue
            super_id: Hashable = f"EVENT:{evt}"
            if super_id not in snapshot.node_ids:
                continue
            edges.append(EdgeState(nid, super_id, self.weight,
                                   metadata={"edge_type": "market_to_event"}))
            edges.append(EdgeState(super_id, nid, self.weight,
                                   metadata={"edge_type": "event_to_market"}))
        return edges


class KalshiSemanticTopicEdges:
    """Connect event super-nodes whose titles are semantically similar.

    Embeddings are computed **once at construction time** from the markets
    table, so ``build_edges`` at each snapshot is just a cosine similarity
    lookup — no repeated model calls.

    Uses ``sentence-transformers`` (``pip install sentence-transformers``).
    The default model ``all-MiniLM-L6-v2`` is 80 MB, fast, and captures
    economic / political / sports topic boundaries well.

    Parameters
    ----------
    markets_df : pl.DataFrame
        The full markets table.  One embedding is computed per unique
        ``event_ticker`` using the first ``title`` found for that event.
    threshold : float
        Cosine similarity in ``[0, 1]`` above which an edge is created.
    weight_from_similarity : bool
        If True the edge weight equals the cosine similarity score.
        If False a fixed weight of 1.0 is used.
    model_name : str
        Any ``sentence-transformers`` model name.
    event_ticker_col : str
        Column name for the event ticker in the markets table.
    title_col : str
        Column name for the human-readable event title.
    """

    def __init__(
        self,
        markets_df: pl.DataFrame,
        threshold: float = 0.5,
        weight_from_similarity: bool = True,
        model_name: str = "all-MiniLM-L6-v2",
        event_ticker_col: str = "event_ticker",
        title_col: str = "title",
    ) -> None:
        self.threshold = threshold
        self.weight_from_similarity = weight_from_similarity

        event_titles: Dict[str, str] = {}
        for row in (
            markets_df
            .select([event_ticker_col, title_col])
            .unique(subset=[event_ticker_col], keep="first")
            .iter_rows()
        ):
            et, title = row
            if et and title:
                event_titles[str(et)] = str(title)

        if not event_titles:
            self._embeddings: Dict[str, np.ndarray] = {}
            return

        model = SentenceTransformer(model_name)
        event_list = list(event_titles.keys())
        title_list = [event_titles[e] for e in event_list]
        vecs = model.encode(title_list, normalize_embeddings=True, show_progress_bar=False)

        self._embeddings = {et: vecs[i] for i, et in enumerate(event_list)}

    def build_edges(
        self, snapshot: GraphSnapshot, data: pl.DataFrame, **kwargs: Any
    ) -> List[EdgeState]:
        if not self._embeddings:
            return []

        event_nodes: List[tuple] = []
        for nid in snapshot.node_ids:
            node = snapshot.get_node(nid)
            if node is None or node.metadata.get("node_type") != "event":
                continue
            evt = str(node.metadata.get("event_ticker", ""))
            if evt and evt in self._embeddings:
                event_nodes.append((nid, evt))

        edges: List[EdgeState] = []
        for i in range(len(event_nodes)):
            nid_i, evt_i = event_nodes[i]
            vec_i = self._embeddings[evt_i]
            for j in range(i + 1, len(event_nodes)):
                nid_j, evt_j = event_nodes[j]
                vec_j = self._embeddings[evt_j]
                sim = float(np.dot(vec_i, vec_j))
                if sim >= self.threshold:
                    w = sim if self.weight_from_similarity else 1.0
                    edges.append(EdgeState(
                        nid_i, nid_j, w,
                        metadata={"edge_type": "semantic_topic", "similarity": sim},
                    ))
                    edges.append(EdgeState(
                        nid_j, nid_i, w,
                        metadata={"edge_type": "semantic_topic", "similarity": sim},
                    ))
        return edges
