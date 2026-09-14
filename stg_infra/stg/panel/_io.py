"""Shared, in-sample-guarded loaders for the raw Kalshi archive.

Every ad-hoc script under ``analysis/exploratory_2026_08/`` re-implemented this
block with its own ``IS_CUT = datetime(2026, 1, 1)``. It lives here once now, and
routes the wall through :mod:`stg.splits` so it cannot drift.

Default layout (run from ``event-pred/``)::

    data/markets/*.parquet
    data/trades/*.parquet

Override the root with ``$KALSHI_ARCHIVE_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from stg.splits import OOS_START, assert_no_oos

ARCHIVE_DIR = Path(os.environ.get("KALSHI_ARCHIVE_DIR", "data"))

_MARKET_COLS = (
    "ticker", "event_ticker", "market_type", "title", "yes_sub_title",
    "no_sub_title", "status", "result", "open_time", "close_time",
    "volume", "open_interest", "_fetched_at",
)
_TRADE_COLS = (
    "trade_id", "ticker", "count", "yes_price", "no_price", "taker_side",
    "created_time",
)


def markets_glob() -> str:
    return str(ARCHIVE_DIR / "markets" / "*.parquet")


def trades_glob() -> str:
    return str(ARCHIVE_DIR / "trades" / "*.parquet")


def load_markets(is_only: bool = True) -> pl.DataFrame:
    """One metadata row per ticker (latest fetch), optionally IS-only.

    ``is_only`` filters on ``close_time`` — a market that *closes* in 2026 is
    out of sample even if it opened earlier.
    """
    mk = (
        pl.scan_parquet(markets_glob())
        .select(_MARKET_COLS)
        .sort("_fetched_at", descending=True)
        .unique(subset=["ticker"], keep="first")
    )
    if is_only:
        mk = mk.filter(pl.col("close_time") < OOS_START)
    out = mk.with_columns(
        pl.col("ticker").str.replace(r"^KX", "").str.split("-").list.first()
        .alias("series_raw")
    ).collect()
    if is_only:
        assert_no_oos(out, time_col="close_time")
    return out


def scan_trades(is_only: bool = True) -> pl.LazyFrame:
    """Lazy trades scan, IS-only by default. Collect after filtering to tickers."""
    tr = pl.scan_parquet(trades_glob()).select(_TRADE_COLS)
    if is_only:
        tr = tr.filter(pl.col("created_time") < OOS_START)
    return tr


# --------------------------------------------------------------------------
# true settlement values
# --------------------------------------------------------------------------
#
# ``data/markets/*.parquet`` carries ``result`` (yes/no) per strike but not the
# value that was actually printed, so ``implied.resolved_value()`` has to infer
# it from the ladder: midpoint between the highest YES and the lowest NO strike.
# That is only ever accurate to ``spacing / 2`` — 0.05pp for CPI, against a
# median |surprise| of 0.074pp — and much worse on one-sided ladders, where the
# open tail is truncated at half a strike (measured errors up to 0.35 on CPI,
# 1.65 on GDP, 23,000 on JOBLESSCLAIMS).
#
# The raw ``/historical/markets`` re-pull *does* carry it, as
# ``expiration_value``. It covers 843 in-sample events — 690 of the 799 rows in
# the surprise panel (WTI 368/368, CPI 50/50, U3 46/46, CPICORE 41/41; the
# CPI subcomponents and WTIW are absent and keep the ladder fallback).
#
# See ``data/MANIFEST_new_pulls.md``: this file is structural/settlement
# metadata, not price history, and its ``result``/``expiration_value`` fields
# must not be used on 2026 events. ``is_only=True`` enforces that here.

SETTLEMENT_PATH = ARCHIVE_DIR / "markets_api_pull" / "markets_api_pull_raw.jsonl"

_SETTLEMENT_STRIP = str.maketrans("", "", ",$%")


def _parse_settlement(raw) -> "float | None":
    """``"0.3%"``/``"224,000"``/``"$68.10"`` -> float; categorical -> None.

    FEDDECISION settles to prose ("Hike 25bps", "No hike or cut"), which is not
    a point on a numeric ladder and is correctly dropped — those series are
    ``kind="categorical"`` in the registry and never carry a surprise.
    """
    if raw is None:
        return None
    s = str(raw).translate(_SETTLEMENT_STRIP).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_settlement_values(is_only: bool = True) -> pl.DataFrame:
    """``event_ticker -> resolved_value_true`` from the raw re-pull.

    Returns an empty frame (not an error) when the file is absent, so the
    surprise panel degrades to the ladder-inferred value rather than failing.

    Columns: ``event_ticker, resolved_value_true``.
    """
    import json

    empty = pl.DataFrame(schema={"event_ticker": pl.Utf8,
                                 "resolved_value_true": pl.Float64})
    if not SETTLEMENT_PATH.exists():
        return empty

    vals: dict[str, set[float]] = {}
    closes: dict[str, str] = {}
    with SETTLEMENT_PATH.open() as fh:
        for line in fh:
            rec = json.loads(line)
            v = _parse_settlement(rec.get("expiration_value"))
            if v is None:
                continue
            ev = rec["event_ticker"]
            vals.setdefault(ev, set()).add(v)
            ct = rec.get("close_time")
            if ct and (ev not in closes or ct < closes[ev]):
                closes[ev] = ct

    # Formatting differs across a ladder's rows ("0.0" vs "0.00") but the
    # parsed number must not. A genuine conflict means the pull is mixing two
    # settlements under one event key, which would silently corrupt a surprise.
    conflicts = {e: sorted(s) for e, s in vals.items() if len(s) > 1}
    if conflicts:
        raise AssertionError(
            f"{len(conflicts)} events carry conflicting expiration_value: "
            f"{dict(list(conflicts.items())[:5])}"
        )

    rows = [{"event_ticker": e, "resolved_value_true": next(iter(s)),
             "_close": closes.get(e)} for e, s in vals.items()]
    if not rows:
        return empty
    out = pl.DataFrame(rows).with_columns(
        pl.col("_close").str.to_datetime(time_unit="us", time_zone="UTC",
                                         strict=False)
    )
    if is_only:
        out = out.filter(pl.col("_close").is_not_null()
                         & (pl.col("_close") < OOS_START))
    return out.drop("_close").sort("event_ticker")
