"""The snapshot grid for the spatio-temporal graph.

``cadence="event"`` — one snapshot per date on which some macro event in the
universe resolves. This is the mechanism-matched grid: a surprise is revealed
when a print resolves, and that is when beliefs about other series can move.

``cadence="weekly"`` — every Monday, for a uniform-timestep robustness check.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import polars as pl

from stg.panel._io import load_markets
from stg.panel.registry import series_filter_expr, universe


def macro_resolution_dates(
    markets: Optional[pl.DataFrame] = None,
    series: Optional[list[str]] = None,
    *,
    min_events: int = 5,
) -> list[dt.date]:
    """Sorted unique dates on which a universe event resolves (finalized)."""
    mk = markets if markets is not None else load_markets(is_only=True)
    names = series if series is not None else universe(min_events, mk)
    sel = pl.any_horizontal([series_filter_expr(n) for n in names])
    dates = (mk.filter(sel & pl.col("close_time").is_not_null())
             .select(pl.col("close_time").dt.date().alias("d"))
             .unique().sort("d")["d"].to_list())
    return dates


def weekly_dates(
    start: dt.date = dt.date(2022, 1, 3),
    end: Optional[dt.date] = None,
) -> list[dt.date]:
    from stg.splits import OOS_START
    end = end or (OOS_START.date() - dt.timedelta(days=1))
    out, d = [], start
    while d <= end:
        out.append(d)
        d += dt.timedelta(days=7)
    return out


def snapshot_dates(cadence: str = "event", **kw) -> list[dt.date]:
    if cadence == "event":
        return macro_resolution_dates(**kw)
    if cadence == "weekly":
        return weekly_dates()
    raise ValueError(f"unknown cadence {cadence!r}")
