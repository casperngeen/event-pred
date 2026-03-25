"""Kalshi-specific label strategies."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Hashable, List

import polars as pl

from stg_infra.stg.util import parse_duration_td


class KalshiOutcomeLabels:
    """Binary outcome labels: yes→1, no→0, else→-1."""

    def __init__(self, node_col: str = "ticker", result_col: str = "result") -> None:
        self.node_col = node_col
        self.result_col = result_col

    def extract_labels(self, node_ids: List[Hashable], data: pl.DataFrame, **kwargs: Any) -> Dict[Hashable, int]:
        labels: Dict[Hashable, int] = {}
        for nid in node_ids:
            sub = data.filter(pl.col(self.node_col) == nid)
            if sub.is_empty() or self.result_col not in sub.columns:
                labels[nid] = -1
                continue
            val = sub[self.result_col][-1]
            labels[nid] = 1 if val == "yes" else (0 if val == "no" else -1)
        return labels


class KalshiPriceChangeLabels:
    """Per-node yes_price change over a forward-looking horizon.

    At each snapshot the label is the change in yes_price between the last
    trade in the current window and the last trade in a future horizon window.
    This turns the task into a price-movement regression (or direction classification).

    Parameters
    ----------
    all_trades : pl.DataFrame
        The **full, unsliced** trades DataFrame.  Must be passed at
        construction time because the builder only supplies the current
        window's data to the label strategy.
    horizon : str
        Polars-style duration string for the lookahead (e.g. ``"1h"``,
        ``"30m"``).  The future window is ``[window_end, window_end+horizon)``.
    mode : str
        ``"raw"``       — absolute price change in cents  (float, regression)
        ``"pct"``       — percentage change relative to current price (float)
        ``"direction"`` — discretised: +1 (up), -1 (down), 0 (flat)
    flat_threshold : float
        Only used when ``mode="direction"``.  Changes whose absolute value is
        below this (in cents) are labelled 0.  Default 1 cent.
    node_col : str
        Column name for the market ticker.
    """

    def __init__(
        self,
        all_trades: pl.DataFrame,
        horizon: str = "1h",
        mode: str = "raw",
        flat_threshold: float = 1.0,
        node_col: str = "ticker",
    ) -> None:
        if mode not in ("raw", "pct", "direction"):
            raise ValueError(f"mode must be 'raw', 'pct', or 'direction', got {mode!r}")
        self.all_trades = all_trades.sort("created_time") if "created_time" in all_trades.columns else all_trades
        self.horizon_td = parse_duration_td(horizon)
        self.mode = mode
        self.flat_threshold = flat_threshold
        self.node_col = node_col

    def extract_labels(
        self, node_ids: List[Hashable], data: pl.DataFrame, **kwargs: Any
    ) -> Dict[Hashable, Any]:
        window_end = kwargs.get("window_end")
        if window_end is None:
            if "created_time" in data.columns and not data.is_empty():
                window_end = data["created_time"].max()
            else:
                return {nid: float("nan") for nid in node_ids}

        assert isinstance(window_end, datetime)
        horizon_end = window_end + self.horizon_td

        labels: Dict[Hashable, Any] = {}
        for nid in node_ids:
            cur = data.filter(pl.col(self.node_col) == nid)
            if cur.is_empty():
                labels[nid] = float("nan")
                continue
            cur_price = float(cur["yes_price"][-1])

            fut = self.all_trades.filter(
                (pl.col(self.node_col) == nid)
                & (pl.col("created_time") >= window_end)
                & (pl.col("created_time") < horizon_end)
            )
            if fut.is_empty():
                labels[nid] = float("nan")
                continue
            fut_price = float(fut["yes_price"][-1])

            change = fut_price - cur_price

            if self.mode == "raw":
                labels[nid] = float(change)
            elif self.mode == "pct":
                labels[nid] = float(change / cur_price * 100.0) if cur_price != 0 else float("nan")
            else:
                if abs(change) < self.flat_threshold:
                    labels[nid] = 0
                else:
                    labels[nid] = 1 if change > 0 else -1

        return labels
