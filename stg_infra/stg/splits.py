"""Single source of truth for the in-sample / out-of-sample boundary.

Import from here rather than hardcoding dates. Every ad-hoc script that writes
its own ``IS_CUT = datetime(2026, 1, 1)`` is one typo away from silently
training on the holdout.

Boundaries
----------
``OOS_START`` (2026-01-01) is the hard wall. It is deliberately **not** chosen
by any data-driven optimisation: it is the date the original archive stopped,
fixed months before any of this analysis existed. That is what makes it immune
to "did you pick this cut because it flattered the result" — an event-density
optimised boundary could never make that claim. Do not move it.

``TRAIN_VAL_SPLIT`` (2025-09-15) *is* data-driven, and legitimately so — both
sides get looked at repeatedly during development, so choosing it by event
density leaks nothing. It was picked to land the long-running core series near
80/20; see ``data_prep_plan.md`` §3.

``PURGE_DAYS`` covers the [-3, +14] event-study window plus slack, so a trigger
resolving near a boundary cannot have its label window straddle the split.

Usage
-----
    from stg.splits import OOS_START, filter_is, assert_no_oos

    trades = filter_is(trades)            # drop anything at/after the wall
    assert_no_oos(trades)                 # or fail loudly
"""
from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from typing import TypeVar

    Frame = TypeVar("Frame", pl.DataFrame, pl.LazyFrame)

UTC = dt.timezone.utc

# --- the wall: nothing exploratory may cross this -------------------------
OOS_START = dt.datetime(2026, 1, 1, tzinfo=UTC)

# --- within-IS train/validation boundary ----------------------------------
TRAIN_VAL_SPLIT = dt.datetime(2025, 9, 15, tzinfo=UTC)

# --- burn-in: excluded from estimation (ZIRP regime, ~86K trades all year) -
BURN_IN_END = dt.datetime(2022, 1, 1, tzinfo=UTC)

# --- purge band around each boundary (event-study window + slack) ---------
PURGE_DAYS = 21

SPLITS = {
    "burn_in": (None, BURN_IN_END),
    "train": (BURN_IN_END, TRAIN_VAL_SPLIT),
    "val": (TRAIN_VAL_SPLIT, OOS_START),
    "test": (OOS_START, None),
}


def _time_col(frame) -> str:
    cols = frame.collect_schema().names() if isinstance(frame, pl.LazyFrame) else frame.columns
    for c in ("created_time", "date", "close_time", "resolution_date"):
        if c in cols:
            return c
    raise ValueError(f"no recognised time column in {cols}")


def filter_is(frame, time_col: str | None = None, purge: bool = False):
    """Restrict a frame to in-sample rows (strictly before ``OOS_START``).

    Set ``purge=True`` to also drop the ``PURGE_DAYS`` band immediately before
    the wall, for label windows that would otherwise reach across it.
    """
    col = time_col or _time_col(frame)
    cutoff = OOS_START - dt.timedelta(days=PURGE_DAYS) if purge else OOS_START
    return frame.filter(pl.col(col) < cutoff)


def filter_split(frame, split: str, time_col: str | None = None, purge: bool = False):
    """Restrict a frame to one of ``burn_in`` / ``train`` / ``val`` / ``test``."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    lo, hi = SPLITS[split]
    col = time_col or _time_col(frame)
    out = frame
    if lo is not None:
        out = out.filter(pl.col(col) >= (lo + dt.timedelta(days=PURGE_DAYS) if purge else lo))
    if hi is not None:
        out = out.filter(pl.col(col) < (hi - dt.timedelta(days=PURGE_DAYS) if purge else hi))
    return out


def assert_no_oos(frame, time_col: str | None = None) -> None:
    """Raise if any row falls at or after the OOS wall.

    Cheap insurance to drop at the top of any exploratory script.
    """
    col = time_col or _time_col(frame)
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    n = lf.filter(pl.col(col) >= OOS_START).select(pl.len()).collect().item()
    if n:
        raise AssertionError(
            f"{n} rows at/after OOS_START ({OOS_START.date()}) in column {col!r}. "
            "Exploratory analysis must not touch the holdout — see stg/splits.py."
        )
