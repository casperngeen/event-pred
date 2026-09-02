"""Kalshi label strategy — k-step-ahead change in a node's implied belief.

The label for node B at snapshot t is the change in B's ``implied_mean`` between
t and the k-th following snapshot (or k calendar days, per ``unit``). Sign is
exposed via ``direction`` for directional-accuracy scoring.

The full node panel is passed at construction because the builder only hands
each label call the current window's slice.

Replaces the pre-refactor ``KalshiPriceChangeLabels`` / ``KalshiOutcomeLabels``
(contract-level).
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Hashable, List

import numpy as np
import polars as pl


class ImpliedMeanChangeLabels:
    """Forward change in ``implied_mean`` per series node.

    Parameters
    ----------
    node_panel : the full (series, date) feature panel.
    horizon : k. With ``unit="snapshots"`` the label looks k rows ahead in that
        series' own snapshot sequence; with ``unit="days"`` it looks for the
        first row on/after ``date + k days``.
    unit : ``"snapshots"`` or ``"days"``.
    """

    def __init__(self, node_panel: pl.DataFrame, horizon: int = 3,
                 unit: str = "snapshots", series_col: str = "series") -> None:
        if unit not in ("snapshots", "days"):
            raise ValueError("unit must be 'snapshots' or 'days'")
        self.horizon = horizon
        self.unit = unit
        self.series_col = series_col
        self._by_series: Dict[str, pl.DataFrame] = {
            str(k[0]): v.sort("date")
            for k, v in node_panel.partition_by(series_col, as_dict=True).items()
        }

    def _future_mean(self, series: str, when: dt.date) -> float | None:
        df = self._by_series.get(str(series))
        if df is None:
            return None
        idx = df["date"].search_sorted(when)
        if self.unit == "snapshots":
            j = int(idx) + self.horizon
        else:
            target = when + dt.timedelta(days=self.horizon)
            j = int(df["date"].search_sorted(target))
        if j >= df.height:
            return None
        v = df["implied_mean"][j]
        return float(v) if v is not None else None

    def extract_labels(self, node_ids: List[Hashable], data: pl.DataFrame,
                       **kwargs: Any) -> Dict[Hashable, Any]:
        ts = kwargs.get("timestamp") or (
            data["date"].max() if "date" in data.columns and data.height else None)
        when = ts.date() if isinstance(ts, dt.datetime) else ts
        out: Dict[Hashable, Any] = {}
        for nid in node_ids:
            row = data.filter(pl.col(self.series_col) == nid).tail(1)
            cur = float(row["implied_mean"][0]) if row.height and row["implied_mean"][0] is not None else None
            fut = self._future_mean(nid, when) if when is not None else None
            if cur is None or fut is None:
                out[nid] = {"delta": float("nan"), "direction": 0}
            else:
                out[nid] = {"delta": fut - cur,
                            "direction": int(np.sign(fut - cur)) if (fut - cur) else 0}
        return out
