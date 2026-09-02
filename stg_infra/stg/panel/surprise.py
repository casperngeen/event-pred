"""Per-event market surprise: ``resolved_value - implied_mean``.

Promotes ``analysis/exploratory_2026_08/liquid_window.py::pdf_surprise`` (the
cumulative-threshold path) and ``bucket_surprise.py::build_bucket_surprise``
(the WTI-style bucket path) into one durable function keyed by canonical series
name, dispatching on the contract kind recorded in :mod:`stg.panel.registry`.

The surprise is the market's *forecast error* on its central estimate, in the
units of the underlying (pp of CPI, $/bbl of WTI, thousands of claims). The
implied mean is read from the strike ladder on the **last pre-resolution day
that carried a real cross-section** (``MIN_FRESH_LEGS`` legs that actually
traded that day — never forward-filled quotes; see research_log.md §7).

Output schema (one row per usable event)::

    series close_time snap_date
    implied_mean implied_std implied_entropy implied_skew implied_kurtosis
    resolved_value surprise
    n_legs n_ladder coverage ladder_mass is_bucket

``series`` is the canonical node name. ``coverage = n_legs / n_ladder`` is the
fraction of the ladder that formed the cross-section; ``ladder_mass`` is the
(pre-normalisation) sum of leg probabilities — the coherence-violation metric
(research_summary.md §5), ~1.0 for a coherent ladder.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from stg.io.kalshi import KalshiOHLCV
from stg.events.implied import (
    BUCKET, THRESHOLD,
    classify_contract, parse_bucket, parse_threshold,
    parse_threshold_from_subtitle, recover_pdf, pdf_implied_stats,
    resolved_value, infer_spacing, normalise_to_exclusive,
)
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import SPECS, series_filter_expr, universe

MIN_FRESH_LEGS = 3     # need a real cross-section to call a day's quotes a distribution
MIN_MASS = 0.5         # bucket path: normalised mass must not be dominated by gaps
MIN_N_DEFAULT = 10     # events needed before a series counts as a usable trigger

_SCHEMA = [
    "series", "event_ticker", "close_time", "snap_date",
    "implied_mean", "implied_std", "implied_entropy", "implied_skew",
    "implied_kurtosis", "resolved_value", "surprise",
    "n_legs", "n_ladder", "coverage", "ladder_mass", "is_bucket",
]


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema={c: (pl.Utf8 if c in ("series", "event_ticker") else
                                    pl.Boolean if c == "is_bucket" else
                                    pl.Datetime if c == "close_time" else
                                    pl.Date if c == "snap_date" else
                                    pl.Int32 if c in ("n_legs", "n_ladder") else
                                    pl.Float64) for c in _SCHEMA})


# --------------------------------------------------------------------------
# threshold (cumulative "Above X") path
# --------------------------------------------------------------------------
def _threshold_surprise(canon: str, mk: pl.DataFrame, tr: pl.LazyFrame) -> pl.DataFrame:
    sub_mk = mk.filter(series_filter_expr(canon))
    tickers = sub_mk["ticker"].unique().to_list()
    trades = tr.filter(pl.col("ticker").is_in(tickers)).collect()
    if trades.is_empty():
        return _empty()

    daily = KalshiOHLCV.build_daily(trades, sub_mk).join(
        sub_mk.select("ticker", "yes_sub_title").unique(subset=["ticker"]),
        on="ticker", how="left",
    )
    daily = daily.with_columns(
        pl.struct("ticker", "yes_sub_title").map_elements(
            lambda r: parse_threshold(r["ticker"], r["yes_sub_title"]),
            return_dtype=pl.Float64).alias("threshold"),
        pl.col("yes_sub_title").map_elements(
            lambda s: parse_threshold_from_subtitle(s)[1],
            return_dtype=pl.Utf8).alias("conv"),
    ).filter(pl.col("threshold").is_not_null())

    fresh = daily.filter(pl.col("trade_count") > 0)
    rows: list[dict] = []
    for ev in fresh["event_ticker"].unique().to_list():
        ev_all = daily.filter(pl.col("event_ticker") == ev)
        n_ladder = ev_all["threshold"].n_unique()
        ev_fresh = fresh.filter(pl.col("event_ticker") == ev)
        close = ev_fresh["close_time"].min()
        if close is None:
            continue
        cand = (ev_fresh.group_by("date").agg(pl.len().alias("legs"))
                .filter(pl.col("legs") >= MIN_FRESH_LEGS).sort("date"))
        if cand.height == 0:
            continue
        day = cand["date"][-1]
        lad = ev_fresh.filter(pl.col("date") == day).sort("threshold")
        thr = lad["threshold"].to_numpy().astype(float)
        prb = lad["close"].to_numpy().astype(float) / 100.0
        conv = lad["conv"].to_list()
        if any(c == "inclusive" for c in conv):
            thr = normalise_to_exclusive(thr, conv)
        order = np.argsort(thr)
        thr, prb = thr[order], prb[order]
        if thr.size < MIN_FRESH_LEGS:
            continue
        mass = float(np.clip(np.concatenate([[1 - prb[0]], prb[:-1] - prb[1:],
                                             [prb[-1]]]), 0, None).sum())
        mids, probs = recover_pdf(thr, prb)
        st = pdf_implied_stats(mids, probs)
        rv = resolved_value(sub_mk.filter(pl.col("event_ticker") == ev))
        if rv is None:
            continue
        rows.append(dict(
            series=canon, event_ticker=ev, close_time=close, snap_date=day,
            implied_mean=st["mean"], implied_std=st["std"],
            implied_entropy=st["entropy"], implied_skew=st["skew"],
            implied_kurtosis=st["kurtosis"], resolved_value=float(rv),
            surprise=float(rv) - st["mean"],
            n_legs=int(thr.size), n_ladder=int(n_ladder),
            coverage=thr.size / max(n_ladder, 1), ladder_mass=mass,
            is_bucket=False,
        ))
    return pl.DataFrame(rows, schema_overrides={c: pl.Int32 for c in ("n_legs", "n_ladder")}).select(_SCHEMA) if rows else _empty()


# --------------------------------------------------------------------------
# bucket (P(a <= X <= b), already a pmf) path
# --------------------------------------------------------------------------
def _bucket_surprise(canon: str, mk: pl.DataFrame, tr: pl.LazyFrame) -> pl.DataFrame:
    m = mk.filter(series_filter_expr(canon)).select(
        "ticker", "event_ticker", "yes_sub_title", "result", "close_time")
    cls, lo, hi, thr = [], [], [], []
    for t, s in zip(m["ticker"], m["yes_sub_title"]):
        c = classify_contract(t, s)
        cls.append(c)
        b = parse_bucket(t, s) if c == BUCKET else None
        lo.append(b[0] if b else None)
        hi.append(b[1] if b else None)
        thr.append(parse_threshold(t, s) if c == THRESHOLD else None)
    m = m.with_columns(
        pl.Series("cls", cls), pl.Series("lo", lo, dtype=pl.Float64),
        pl.Series("hi", hi, dtype=pl.Float64), pl.Series("thr", thr, dtype=pl.Float64))

    tickers = m["ticker"].unique().to_list()
    trades = (tr.filter(pl.col("ticker").is_in(tickers))
              .select("ticker", "yes_price", "created_time")
              .with_columns(pl.col("created_time").dt.date().alias("d"))
              .sort("ticker", "created_time")
              .group_by("ticker", "d")
              .agg(pl.col("yes_price").last().alias("px"))
              .collect())
    px_hist: dict[str, list[tuple]] = {}
    for r in trades.iter_rows(named=True):
        px_hist.setdefault(r["ticker"], []).append((r["d"], r["px"]))

    rows: list[dict] = []
    for ev in m["event_ticker"].unique().to_list():
        sub = m.filter(pl.col("event_ticker") == ev)
        bk = sub.filter(pl.col("cls") == BUCKET).drop_nulls(["lo", "hi"])
        if bk.height < MIN_FRESH_LEGS:
            continue
        close = sub["close_time"].min()
        width = float(np.median(bk["hi"].to_numpy() - bk["lo"].to_numpy())) + 0.01

        day_px: dict = {}
        for r in bk.iter_rows(named=True):
            for d, p in px_hist.get(r["ticker"], []):
                day_px.setdefault(d, {})[r["ticker"]] = p
        if not day_px:
            continue
        best = max(day_px, key=lambda d: len(day_px[d]))
        if len(day_px[best]) < MIN_FRESH_LEGS:
            continue
        quotes = day_px[best]

        mids, probs = [], []
        for r in bk.iter_rows(named=True):
            if r["ticker"] in quotes:
                mids.append((r["lo"] + r["hi"]) / 2.0)
                probs.append(quotes[r["ticker"]] / 100.0)
        for r in sub.filter(pl.col("cls") == THRESHOLD).drop_nulls("thr").iter_rows(named=True):
            p = quotes.get(r["ticker"])
            if p is None:
                continue
            above = "above" in (r["yes_sub_title"] or "").lower()
            mids.append(r["thr"] + width / 2.0 if above else r["thr"] - width / 2.0)
            probs.append(p / 100.0)
        mids = np.array(mids, float)
        probs = np.array(probs, float)
        mass = float(probs.sum())
        if mids.size < MIN_FRESH_LEGS or mass < MIN_MASS:
            continue
        pmf = probs / mass
        order = np.argsort(mids)
        st = pdf_implied_stats(mids[order], pmf[order])

        won = sub.filter(pl.col("result") == "yes")
        if won.height == 0:
            continue
        w = won.row(0, named=True)
        if w["cls"] == BUCKET and w["lo"] is not None:
            rv = (w["lo"] + w["hi"]) / 2.0
        elif w["thr"] is not None:
            above = "above" in (w["yes_sub_title"] or "").lower()
            rv = w["thr"] + width / 2.0 if above else w["thr"] - width / 2.0
        else:
            continue
        rows.append(dict(
            series=canon, event_ticker=ev, close_time=close, snap_date=best,
            implied_mean=st["mean"], implied_std=st["std"],
            implied_entropy=st["entropy"], implied_skew=st["skew"],
            implied_kurtosis=st["kurtosis"], resolved_value=float(rv),
            surprise=float(rv) - st["mean"],
            n_legs=int(mids.size), n_ladder=int(bk.height),
            coverage=mids.size / max(bk.height, 1), ladder_mass=mass,
            is_bucket=True,
        ))
    return pl.DataFrame(rows, schema_overrides={c: pl.Int32 for c in ("n_legs", "n_ladder")}).select(_SCHEMA) if rows else _empty()


# --------------------------------------------------------------------------
# public
# --------------------------------------------------------------------------
def build_surprise_panel(
    series: Optional[list[str]] = None,
    *,
    min_events: int = 5,
    markets: Optional[pl.DataFrame] = None,
    trades: Optional[pl.LazyFrame] = None,
) -> pl.DataFrame:
    """Surprise rows for every canonical series (or the given subset).

    Dispatches per series on ``registry.SPECS[canon].kind``. Categorical series
    (FEDDECISION, RATECUT) yield nothing here — they are targets only.
    """
    mk = markets if markets is not None else load_markets(is_only=True)
    tr = trades if trades is not None else scan_trades(is_only=True)
    names = series if series is not None else universe(min_events, mk)

    frames: list[pl.DataFrame] = []
    for canon in names:
        spec = SPECS.get(canon)
        if spec is None or spec.kind == "categorical":
            continue
        fn = _bucket_surprise if spec.kind == BUCKET else _threshold_surprise
        frames.append(fn(canon, mk, tr))
    out = pl.concat([f for f in frames if f.height], how="vertical") if any(
        f.height for f in frames) else _empty()
    return out.sort("series", "close_time")


def usable_triggers(panel: pl.DataFrame, min_n: int = MIN_N_DEFAULT) -> list[str]:
    """Series in the panel with >= min_n events carrying a finite surprise."""
    ok = (panel.filter(pl.col("surprise").is_finite())
          .group_by("series").agg(pl.len().alias("n"))
          .filter(pl.col("n") >= min_n)["series"].to_list())
    return sorted(ok)
