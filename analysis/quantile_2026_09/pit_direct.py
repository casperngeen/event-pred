#!/usr/bin/env python
"""Item 1, re-tested without any reconstruction.

    venv/bin/python analysis/quantile_2026_09/pit_direct.py
    (needs build_ladder_panel.py)

``relations_findings.md`` item 1 found the realised value's position in the
market's implied distribution far from uniform -- mean PIT 0.65-0.91 for the
macro ladders, read as "the market is biased low". That reading was then
withdrawn, because ``settlement_trade.py`` priced the same bias with real
traded strikes and found the 50c contract accurate (50.9c paid, 48% realised).
The document's standing verdict is: *read item 1 as a finding about the
reconstruction, not about the market, until this is resolved.*

This resolves it. The PIT is computed by interpolating ``P(X > resolved)``
directly off the traded ladder -- no ``recover_pdf``, no open tails, no
assumption about strikes that never traded. If the non-uniformity survives,
item 1 is about the market. If it collapses toward uniform, item 1 was about
``recover_pdf`` and the price test was right.

Censoring is reported rather than hidden: when the outcome falls outside the
traded strike range the PIT is only known as an inequality, and that frequency
is itself the coverage defect measured in outcome terms.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.events.implied import (
    THRESHOLD, classify_contract, normalise_to_exclusive, parse_threshold,
    parse_threshold_from_subtitle,
)
from stg.events.quantile import ladder_pit
from stg.io.kalshi import KalshiOHLCV
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import SPECS, series_filter_expr, trigger_universe
from stg.panel.surprise import MIN_FRESH_LEGS
from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("analysis/quantile_2026_09/out")


def ks_uniform(u: np.ndarray) -> tuple[float, float]:
    """Exact-ish two-sided KS against U(0,1) with the asymptotic p-value."""
    u = np.sort(np.asarray(u, dtype=float))
    n = len(u)
    if n < 5:
        return float("nan"), float("nan")
    i = np.arange(1, n + 1)
    d = max(np.max(i / n - u), np.max(u - (i - 1) / n))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    j = np.arange(1, 101)
    p = 2 * np.sum((-1) ** (j - 1) * np.exp(-2 * j ** 2 * lam ** 2))
    return float(d), float(min(max(p, 0.0), 1.0))


def main() -> None:
    mk, tr = load_markets(is_only=True), scan_trades(is_only=True)
    node = pl.read_parquet(PANELS / "node_panel_event.parquet")
    truth = {r["event_ticker"]: r["resolved_value"]
             for r in node.group_by("event_ticker")
             .agg(pl.col("resolved_value").first()).iter_rows(named=True)
             if r["resolved_value"] is not None}

    rows = []
    for canon in [c for c in trigger_universe(5, mk) if SPECS[c].kind == THRESHOLD]:
        sub = (mk.filter(series_filter_expr(canon) & pl.col("close_time").is_not_null())
               .select("ticker", "event_ticker", "yes_sub_title", "close_time")
               .unique(subset=["ticker"]))
        if sub.is_empty():
            continue
        kinds, strikes, convs = [], [], []
        for t, s in zip(sub["ticker"], sub["yes_sub_title"]):
            c = classify_contract(t, s)
            kinds.append(c)
            strikes.append(parse_threshold(t, s) if c == THRESHOLD else None)
            convs.append(parse_threshold_from_subtitle(s)[1] if c == THRESHOLD else None)
        sub = (sub.with_columns(pl.Series("kind", kinds),
                                pl.Series("strike", strikes, dtype=pl.Float64),
                                pl.Series("conv", convs, dtype=pl.Utf8))
               .filter((pl.col("kind") == THRESHOLD) & pl.col("strike").is_not_null()))
        if sub.is_empty():
            continue
        trades = tr.filter(pl.col("ticker").is_in(sub["ticker"].unique().to_list())).collect()
        if trades.is_empty():
            continue
        daily = (KalshiOHLCV.build_daily(trades, mk.filter(series_filter_expr(canon)))
                 .join(sub.select("ticker", "strike", "conv"), on="ticker", how="inner")
                 .filter((pl.col("trade_count") > 0) & pl.col("close").is_between(0, 100)))
        if daily.is_empty():
            continue
        for ev in daily["event_ticker"].unique().to_list():
            rv = truth.get(ev)
            if rv is None:
                continue
            g_ev = daily.filter(pl.col("event_ticker") == ev)
            cand = (g_ev.group_by("date").agg(pl.len().alias("legs"))
                    .filter(pl.col("legs") >= MIN_FRESH_LEGS).sort("date"))
            if cand.height == 0:
                continue
            g = g_ev.filter(pl.col("date") == cand["date"][-1])
            thr = g["strike"].to_numpy().astype(float)
            conv = g["conv"].to_list()
            if any(c == "inclusive" for c in conv):
                thr = normalise_to_exclusive(thr, conv)
            p = g["close"].to_numpy().astype(float) / 100.0
            pit, censor = ladder_pit(thr, p, float(rv))
            if pit is None:
                continue
            rows.append(dict(series=canon, event_ticker=ev,
                             close_time=g["close_time"].min(),
                             pit=pit, censor=censor or "none",
                             n_legs=int(g.height),
                             span_lo=float(np.min(thr)), span_hi=float(np.max(thr)),
                             resolved=float(rv)))
    d = pl.DataFrame(rows)
    assert_no_oos(d, time_col="close_time")

    print("=== coverage, in outcome terms ===")
    print("How often did the print land OUTSIDE the range of strikes that")
    print("traded? That is the coverage defect, measured where it bites.\n")
    with pl.Config(tbl_rows=25, float_precision=3, tbl_width_chars=200):
        print(d.group_by("series").agg(
            pl.len().alias("events"),
            (pl.col("censor") == "none").mean().alias("uncensored"),
            (pl.col("censor") == "left").mean().alias("below_ladder"),
            (pl.col("censor") == "right").mean().alias("above_ladder"),
        ).filter(pl.col("events") >= 8).sort("uncensored"))
    print(f"\noverall uncensored: {(d['censor'] == 'none').mean():.3f} "
          f"of {d.height} events")

    print("\n=== the re-test: is the LADDER PIT uniform? ===")
    print("Uncensored events only. Item 1's recovered-pdf figures are quoted")
    print("alongside for the series it reported.\n")
    item1 = {"WTI": (0.462, 0.217), "CPI": (0.722, 0.387), "U3": (0.648, 0.247),
             "PAYROLLS": (0.552, 0.143), "CPICORE": (0.716, 0.438),
             "CPIYOY": (0.668, 0.365), "CPICOREYOY": (0.745, 0.412),
             "PCECORE": (0.791, 0.503), "FED": (0.906, 0.710)}
    u = d.filter(pl.col("censor") == "none")
    rows = []
    for s in sorted(u["series"].unique().to_list()):
        x = u.filter(pl.col("series") == s)["pit"].to_numpy()
        if len(x) < 8:
            continue
        ks, p = ks_uniform(x)
        prev = item1.get(s)
        rows.append(dict(series=s, n=len(x), mean_pit=float(x.mean()),
                         tail20=float(((x < 0.1) | (x > 0.9)).mean()),
                         ks_D=ks, ks_p=p,
                         item1_mean=prev[0] if prev else None,
                         item1_D=prev[1] if prev else None))
    t = pl.DataFrame(rows).sort("n", descending=True)
    with pl.Config(tbl_rows=25, float_precision=3, tbl_width_chars=220):
        print(t)
    print("\ntail20 is the share in the outer decile pair; 0.20 under calibration.")

    allu = u["pit"].to_numpy()
    ks, p = ks_uniform(allu)
    print(f"\npooled: n = {len(allu)}   mean PIT {allu.mean():.3f}   "
          f"KS D = {ks:.3f}   p = {p:.4f}")
    print(f"share above 0.5: {(allu > 0.5).mean():.3f}   "
          f"(0.50 under calibration)")

    OUT.mkdir(parents=True, exist_ok=True)
    d.write_parquet(OUT / "pit_direct.parquet")
    print(f"\nwrote {OUT / 'pit_direct.parquet'}")


if __name__ == "__main__":
    main()
