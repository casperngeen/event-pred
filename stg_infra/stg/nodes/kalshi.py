"""Kalshi-specific node strategies."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Hashable, List, Optional

import numpy as np
import polars as pl

from stg_infra.stg.core import NodeState


def _seconds_to_close(window_end: datetime, close_time: datetime) -> float:
    """Seconds from window_end to close_time, clamped to >= 0."""
    if hasattr(window_end, "timestamp") and hasattr(close_time, "timestamp"):
        delta = close_time.timestamp() - window_end.timestamp()
        return max(delta, 0.0)
    return 0.0


class KalshiTradeBasedNodes:
    """Node features reconstructed from the Kalshi trade data.

    ``yes_price`` is used as the canonical market probability throughout.
    YES-side trades (``taker_side == "yes"``) represent buying pressure;
    NO-side trades represent selling pressure on YES.

    Feature vector (10-D):
        --- Price (yes_price as canonical probability) ---
        0.  last_yes_price      yes_price of the last trade in the window
        1.  yes_vwap            volume-weighted average yes_price across all trades
        2.  price_return        last_yes_price – first_yes_price
        3.  price_std           std dev of yes_price  (realised volatility)

        --- Volume ---
        4.  window_volume       total contracts traded in this window
        5.  cum_volume          cumulative contracts (falls back to window_volume)

        --- Order flow ---
        6.  net_flow            (buy_vol – sell_vol) / total_vol  (normalised, signed)
        7.  buy_ratio           YES-side volume / total_vol  (volume-weighted)

        --- Activity ---
        8.  trade_intensity     trades per second in window
        9.  time_to_close       seconds from window_end to market close_time  (0 if past)

    Requires
    --------
    ``auxiliary["markets"]`` — the markets table, used to look up
    ``close_time`` and metadata columns (``event_ticker``, ``title``,
    ``status``, ``market_type``).  Only the last-seen row per ticker is
    used so the stale order-book values are intentionally ignored.
    """

    N_FEATURES = 10

    def __init__(self, node_col: str = "ticker") -> None:
        self.node_col = node_col

    def identify_nodes(self, data: pl.DataFrame, **kwargs: Any) -> List[Hashable]:
        return data[self.node_col].unique().sort().to_list()

    def build_node_state(self, node_id: Hashable, data: pl.DataFrame, **kwargs: Any) -> NodeState:
        window_end: Optional[datetime] = kwargs.get("window_end")
        auxiliary: Dict[str, pl.DataFrame] = kwargs.get("auxiliary") or {}
        markets: Optional[pl.DataFrame] = auxiliary.get("markets")

        t = data.filter(pl.col(self.node_col) == node_id)
        feats = np.zeros(self.N_FEATURES, dtype=np.float64)
        meta: Dict[str, Any] = {}

        close_time: Optional[datetime] = None
        if markets is not None and not markets.is_empty():
            mkt = markets.filter(pl.col("ticker") == node_id)
            if not mkt.is_empty():
                row = mkt.tail(1)
                for col in ("event_ticker", "title", "status", "market_type"):
                    if col in row.columns:
                        meta[col] = row[col][0]
                if "close_time" in row.columns:
                    ct = row["close_time"][0]
                    if ct is not None:
                        close_time = ct

        if t.is_empty():
            if window_end is not None and close_time is not None:
                feats[9] = _seconds_to_close(window_end, close_time)
            return NodeState(node_id, feats, meta)

        if "created_time" in t.columns:
            t = t.sort("created_time")

        yes_prices = t["yes_price"].cast(pl.Float64)
        counts = t["count"].cast(pl.Float64)
        n = t.height
        total_vol = float(counts.sum())

        is_buy = t["taker_side"] == "yes"
        t_yes = t.filter(is_buy)
        t_no = t.filter(~is_buy)
        buy_vol = float(t_yes["count"].sum()) if not t_yes.is_empty() else 0.0
        sell_vol = float(t_no["count"].sum()) if not t_no.is_empty() else 0.0

        feats[0] = float(yes_prices[-1])
        feats[1] = float((yes_prices * counts).sum()) / max(total_vol, 1.0)
        feats[2] = float(yes_prices[-1]) - float(yes_prices[0]) if n >= 2 else 0.0
        feats[3] = float(yes_prices.std()) if n >= 2 else 0.0  # type: ignore[arg-type]
        feats[4] = total_vol
        feats[5] = total_vol
        feats[6] = (buy_vol - sell_vol) / max(total_vol, 1.0)
        feats[7] = buy_vol / max(total_vol, 1.0)
        if "created_time" in t.columns and n >= 2:
            span = (t["created_time"][-1] - t["created_time"][0]).total_seconds()
            feats[8] = n / max(span, 1.0)
        if window_end is not None and close_time is not None:
            feats[9] = _seconds_to_close(window_end, close_time)

        return NodeState(node_id, np.nan_to_num(feats, nan=0.0), meta)


class KalshiEventNodes:
    """Event-level super-nodes constructed directly from trade data.

    One node per unique ``event_ticker`` active in the current time window.
    Features are aggregated across every market belonging to the event —
    completely independent of ``KalshiTradeBasedNodes``.

    Feature vector (7-D):
        0.  total_volume        total contracts traded across all event markets
        1.  event_vwap          volume-weighted yes_price across all trades
        2.  buy_ratio           YES-side volume / total_vol  (volume-weighted)
        3.  net_flow            (buy_vol - sell_vol) / total_vol  (normalised)
        4.  num_markets         number of active markets for this event
        5.  min_time_to_close   seconds to the earliest-resolving market  (0 if past)
        6.  max_time_to_close   seconds to the latest-resolving market    (0 if past)

    Requires ``auxiliary["markets"]`` to map ``ticker → event_ticker`` and to
    provide ``close_time``.

    Super-node IDs use the prefix ``"EVENT:"`` (e.g. ``"EVENT:CPI-21JUN"``) so
    they never collide with market ticker strings.
    """

    _PREFIX = "EVENT:"
    N_FEATURES = 7

    def __init__(self, node_col: str = "ticker") -> None:
        self.node_col = node_col

    def identify_nodes(self, data: pl.DataFrame, **kwargs: Any) -> List[Hashable]:
        markets: Optional[pl.DataFrame] = (kwargs.get("auxiliary") or {}).get("markets")
        if markets is None or data.is_empty():
            return []
        active = data[self.node_col].unique()
        event_tickers = (
            markets.filter(pl.col(self.node_col).is_in(active))
            .select("event_ticker")
            .unique()
            ["event_ticker"]
            .drop_nulls()
            .to_list()
        )
        return [f"{self._PREFIX}{et}" for et in event_tickers]

    def build_node_state(self, node_id: Hashable, data: pl.DataFrame, **kwargs: Any) -> NodeState:
        evt = str(node_id)[len(self._PREFIX):]
        markets: Optional[pl.DataFrame] = (kwargs.get("auxiliary") or {}).get("markets")
        window_end = kwargs.get("window_end")
        meta: Dict[str, Any] = {"node_type": "event", "event_ticker": evt}
        feats = np.zeros(self.N_FEATURES, dtype=np.float64)

        if markets is None:
            return NodeState(node_id, feats, meta)

        event_mkts = markets.filter(pl.col("event_ticker") == evt)
        event_tickers_list = event_mkts[self.node_col].unique().to_list()
        feats[4] = float(len(event_tickers_list))

        t = data.filter(pl.col(self.node_col).is_in(event_tickers_list))
        if not t.is_empty():
            if "created_time" in t.columns:
                t = t.sort("created_time")
            counts = t["count"].cast(pl.Float64)
            total_vol = float(counts.sum())
            feats[0] = total_vol

            yes_prices = t["yes_price"].cast(pl.Float64)
            feats[1] = float((yes_prices * counts).sum()) / max(total_vol, 1.0)

            is_buy = t["taker_side"] == "yes"
            buy_vol = float(t.filter(is_buy)["count"].sum())
            sell_vol = float(t.filter(~is_buy)["count"].sum())
            feats[2] = buy_vol / max(total_vol, 1.0)
            feats[3] = (buy_vol - sell_vol) / max(total_vol, 1.0)

        if window_end is not None and "close_time" in event_mkts.columns:
            close_times = event_mkts["close_time"].drop_nulls()
            if close_times.len() > 0:
                feats[5] = _seconds_to_close(window_end, close_times.min())  # type: ignore[arg-type]
                feats[6] = _seconds_to_close(window_end, close_times.max())  # type: ignore[arg-type]

        return NodeState(node_id, np.nan_to_num(feats, nan=0.0), meta)
