"""Built-in temporal slicing strategies — pure Polars."""

from __future__ import annotations

from datetime import timedelta, datetime
from typing import Any, Dict, List

import polars as pl

class FixedWindowTemporal:
    """Fixed-width time windows via Polars ``group_by_dynamic``."""

    def __init__(self, time_col: str = "created_time", every: str = "1h") -> None:
        self.time_col = time_col
        self.every = every

    def slice(self, data: pl.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        if data.is_empty():
            return []
        every_dur = _parse_duration(self.every)
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
        period_dur = _parse_duration(self.window_size)
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
    """Snapshot at each change-point in an event column."""

    def __init__(self, time_col: str = "created_time", event_col: str = "status") -> None:
        self.time_col = time_col
        self.event_col = event_col

    def slice(self, data: pl.DataFrame, **kwargs: Any) -> List[Dict[str, Any]]:
        if data.is_empty():
            return []
        df = data.sort(self.time_col)
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


def _parse_duration(s: str) -> timedelta:
    """Convert a Polars-style duration string ('1h', '30m', '1d') to timedelta."""
    s = s.strip()
    units = {
        "us": "microseconds", "ms": "milliseconds", "s": "seconds",
        "m": "minutes", "h": "hours", "d": "days", "w": "weeks",
    }
    for suffix in sorted(units, key=len, reverse=True):
        if s.endswith(suffix):
            return timedelta(**{units[suffix]: int(s[: -len(suffix)])})
    raise ValueError(f"Cannot parse duration: {s!r}")
