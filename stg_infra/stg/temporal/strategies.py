"""Built-in temporal slicing strategies — pure Polars."""

from __future__ import annotations

from datetime import timedelta, datetime
from typing import Any, Dict, List

import numpy as np
import polars as pl

from stg.util import parse_duration_td

class FixedWindowTemporal:
    """Fixed-width time windows via Polars ``group_by_dynamic``."""

    def __init__(self, time_col: str = "created_time", every: str = "1h") -> None:
        self.time_col = time_col
        self.every = every

    def slice(self, data: pl.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        if data.is_empty():
            return []
        every_dur = parse_duration_td(self.every)
        result: List[Dict[str, Any]] = []
        for key, group_df in data.sort(self.time_col).group_by_dynamic(self.time_col, every=self.every):
            if not group_df.is_empty():
                ws = key[0]
                assert isinstance(ws, datetime)
                result.append({
                    "timestamp": ws,
                    "data": group_df,
                    "window_start": ws,
                    "window_end": ws + every_dur,
                })
        return result


class SnapshotTemporal:
    """One snapshot per unique value in a column."""

    def __init__(self, time_col: str = "_fetched_at") -> None:
        self.time_col = time_col

    def slice(self, data: pl.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for key, group_df in data.sort(self.time_col).group_by(self.time_col, maintain_order=True):
            ts = key[0]
            col = group_df[self.time_col]
            result.append({
                "timestamp": ts,
                "data": group_df,
                "window_start": col.min(),
                "window_end": col.max(),
            })
        return result


class SlidingWindowTemporal:
    """Overlapping sliding windows via ``group_by_dynamic`` with period > every."""

    def __init__(self, time_col: str = "created_time", window_size: str = "2h", stride: str = "1h") -> None:
        self.time_col = time_col
        self.window_size = window_size
        self.stride = stride

    def slice(self, data: pl.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        if data.is_empty():
            return []
        period_dur = parse_duration_td(self.window_size)
        result: List[Dict[str, Any]] = []
        for key, group_df in data.sort(self.time_col).group_by_dynamic(
            self.time_col, every=self.stride, period=self.window_size,
        ):
            if not group_df.is_empty():
                ws = key[0]
                assert isinstance(ws, datetime)
                result.append({
                    "timestamp": ws,
                    "data": group_df,
                    "window_start": ws,
                    "window_end": ws + period_dur,
                })
        return result


class EventDrivenTemporal:
    """Snapshot at each significant change-point in the data stream.

    Three modes are supported via the ``mode`` parameter:

    ``"column_change"`` *(default for non-trade data)*
        Original behaviour: one snapshot per contiguous run of the same value
        in ``event_col`` (e.g. market ``status`` transitions).

    ``"price_threshold"`` *(recommended for trade data)*
        Opens a new snapshot whenever the taker price moves ``>= threshold``
        cents from the price at the start of the current segment.  Taker
        price is ``yes_price`` when ``taker_side == "yes"`` and ``no_price``
        otherwise, so it correctly reflects the direction of each trade.

    ``"trade_count"``
        Opens a new snapshot every ``n_trades`` rows regardless of price.
        Useful for markets with very regular, high-frequency activity.

    Parameters
    ----------
    time_col : str
        Timestamp column used to sort the data and derive window bounds.
    mode : str
        One of ``"column_change"``, ``"price_threshold"``, ``"trade_count"``.
    event_col : str
        Column to watch for changes.  Only used when ``mode="column_change"``.
    threshold : float
        Minimum taker-price move in cents to trigger a new segment.
        Only used when ``mode="price_threshold"``.
    n_trades : int
        Number of trades per segment.  Only used when ``mode="trade_count"``.
    yes_price_col : str
        Column name for the YES-side price.
    no_price_col : str
        Column name for the NO-side price.
    taker_col : str
        Column name for the taker side indicator (``"yes"`` / ``"no"``).
    """

    def __init__(
        self,
        time_col: str = "created_time",
        mode: str = "price_threshold",
        # column_change params
        event_col: str = "status",
        # price_threshold params
        threshold: float = 2.0,
        yes_price_col: str = "yes_price",
        no_price_col: str = "no_price",
        taker_col: str = "taker_side",
        # trade_count params
        n_trades: int = 50,
    ) -> None:
        if mode not in ("column_change", "price_threshold", "trade_count"):
            raise ValueError(
                f"mode must be 'column_change', 'price_threshold', or 'trade_count', got {mode!r}"
            )
        self.time_col = time_col
        self.mode = mode
        self.event_col = event_col
        self.threshold = threshold
        self.yes_price_col = yes_price_col
        self.no_price_col = no_price_col
        self.taker_col = taker_col
        self.n_trades = n_trades

    def slice(self, data: pl.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        if data.is_empty():
            return []
        df = data.sort(self.time_col)

        if self.mode == "column_change":
            return self._slice_column_change(df)
        elif self.mode == "price_threshold":
            return self._slice_price_threshold(df)
        else:
            return self._slice_trade_count(df)

    # ------------------------------------------------------------------
    # mode implementations
    # ------------------------------------------------------------------

    def _slice_column_change(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
        tagged = df.with_columns(
            (pl.col(self.event_col) != pl.col(self.event_col).shift(1))
            .fill_null(True)
            .cum_sum()
            .alias("_cg")
        )
        result: List[Dict[str, Any]] = []
        for _, group_df in tagged.group_by("_cg", maintain_order=True):
            clean = group_df.drop("_cg")
            col = clean[self.time_col]
            result.append({
                "timestamp": col[0],
                "data": clean,
                "window_start": col.min(),
                "window_end": col.max(),
                "event_value": group_df[self.event_col][0],
            })
        return result

    def _slice_price_threshold(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
        # Compute taker price per row: yes_price for YES takers, no_price for NO takers
        df = df.with_columns(
            pl.when(pl.col(self.taker_col) == "yes")
            .then(pl.col(self.yes_price_col))
            .otherwise(pl.col(self.no_price_col))
            .cast(pl.Float64)
            .alias("_tp")
        )
        prices = df["_tp"].to_numpy()

        # Stateful walk: open a new segment when price drifts >= threshold from anchor
        group_ids = np.empty(len(prices), dtype=np.int32)
        g = 0
        anchor = prices[0]
        for i, p in enumerate(prices):
            if abs(p - anchor) >= self.threshold:
                g += 1
                anchor = p
            group_ids[i] = g

        df = df.with_columns(pl.Series("_cg", group_ids)).drop("_tp")
        return self._groups_to_windows(df)

    def _slice_trade_count(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
        group_ids = (np.arange(len(df)) // self.n_trades).astype(np.int32)
        df = df.with_columns(pl.Series("_cg", group_ids))
        return self._groups_to_windows(df)

    def _groups_to_windows(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for _, group_df in df.group_by("_cg", maintain_order=True):
            clean = group_df.drop("_cg")
            col = clean[self.time_col]
            result.append({
                "timestamp": col[0],
                "data": clean,
                "window_start": col.min(),
                "window_end": col.max(),
            })
        return result