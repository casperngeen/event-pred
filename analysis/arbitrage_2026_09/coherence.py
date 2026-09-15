#!/usr/bin/env python
"""Direction 3: is each ladder internally arbitrage-free?

    venv/bin/python analysis/arbitrage_2026_09/coherence.py

A threshold ladder prices ``P(X > K)`` for a grid of K. Two constraints follow
from nothing but the contract definitions:

* **Monotonicity.** ``K1 < K2  =>  P(X > K1) >= P(X > K2)``. A violation is a
  locked arbitrage: buy the low strike, sell the high one, and the low leg pays
  whenever the high one does, so the pair can never lose at settlement and the
  price difference is banked up front.
* **Non-negative density.** The first difference across adjacent strikes is the
  probability mass in that interval, so a monotonicity violation is exactly a
  negative implied density -- the digital-ladder form of a negative butterfly.

Bucket ladders (WTI) carry a third: the buckets **partition** the outcome space,
so their prices must sum to 100c.

Why this is worth doing at all
------------------------------
Every other result in this project is bottlenecked on ``n`` -- median 9 target
events per ordered pair. A coherence violation is not an estimate. One ladder
and arithmetic settle it, so the sample-size problem that has shaped the whole
project does not apply here.

The threat is staleness, not significance
-----------------------------------------
Prices are last trades, not quotes. Two legs last traded six hours apart can
look incoherent while never having been simultaneously mispriced. So every
violation is reported at three synchronicity tiers -- any time on the day,
within 60 minutes, within 5 minutes -- and only the tight tier is evidence of
something a trader could have taken.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.events.implied import (
    BUCKET, THRESHOLD, classify_contract, normalise_to_exclusive, parse_bucket,
    parse_threshold, parse_threshold_from_subtitle,
)
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import SPECS, series_filter_expr, trigger_universe
from stg.splits import assert_no_oos

OUT = Path("analysis/arbitrage_2026_09/out")
FEE_RATE, CONTRACTS = 0.07, 100
TIERS = [("any time same day", 10 ** 9), ("within 60 min", 3600), ("within 5 min", 300)]


def fee_cents(p):
    p = np.asarray(p, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def last_trade_per_day(canon: str, mk: pl.DataFrame, tr: pl.LazyFrame) -> pl.DataFrame:
    """One row per (ticker, date): last trade price and its timestamp."""
    sub = (mk.filter(series_filter_expr(canon) & pl.col("close_time").is_not_null())
           .select("ticker", "event_ticker", "yes_sub_title", "result")
           .unique(subset=["ticker"]))
    if sub.is_empty():
        return pl.DataFrame()
    kinds, strikes, los, his, convs = [], [], [], [], []
    for t, s in zip(sub["ticker"], sub["yes_sub_title"]):
        c = classify_contract(t, s)
        kinds.append(c)
        strikes.append(parse_threshold(t, s) if c == THRESHOLD else None)
        b = parse_bucket(t, s) if c == BUCKET else None
        los.append(b[0] if b else None)
        his.append(b[1] if b else None)
        convs.append(parse_threshold_from_subtitle(s)[1] if c == THRESHOLD else None)
    sub = sub.with_columns(
        pl.Series("kind", kinds), pl.Series("strike", strikes, dtype=pl.Float64),
        pl.Series("lo", los, dtype=pl.Float64), pl.Series("hi", his, dtype=pl.Float64),
        pl.Series("conv", convs, dtype=pl.Utf8))
    tk = sub["ticker"].unique().to_list()
    t = (tr.filter(pl.col("ticker").is_in(tk))
         .select("ticker", "yes_price", "created_time")
         .with_columns(pl.col("created_time").dt.date().alias("date"))
         .sort("ticker", "created_time")
         .group_by("ticker", "date")
         .agg(pl.col("yes_price").last().alias("price"),
              pl.col("created_time").last().alias("ts"),
              pl.len().alias("n_trades"))
         .collect())
    if t.is_empty():
        return pl.DataFrame()
    return t.join(sub, on="ticker", how="inner").with_columns(pl.lit(canon).alias("series"))


def main() -> None:
    mk, tr = load_markets(is_only=True), scan_trades(is_only=True)
    canons = [c for c in trigger_universe(5, mk) if SPECS[c].kind != "categorical"]
    parts = [d for c in canons if (d := last_trade_per_day(c, mk, tr)).height]
    d = pl.concat(parts, how="diagonal")
    d = d.with_columns(pl.col("ts").dt.epoch("s").alias("sec"))
    assert_no_oos(d.with_columns(pl.col("ts").alias("t")), time_col="t")
    print(f"ticker-days: {d.height}   events: {d['event_ticker'].n_unique()}   "
          f"series: {d['series'].n_unique()}\n")

    # ------------------------------------------------- monotonicity (threshold)
    th = d.filter((pl.col("kind") == THRESHOLD) & pl.col("strike").is_not_null()
                  & pl.col("price").is_between(1, 99))
    rows = []
    for (ev, date), g in th.group_by(["event_ticker", "date"]):
        if g.height < 2:
            continue
        thr = g["strike"].to_numpy().astype(float)
        conv = g["conv"].to_list()
        if any(c == "inclusive" for c in conv):
            thr = normalise_to_exclusive(thr, conv)
        order = np.argsort(thr)
        thr = thr[order]
        px = g["price"].to_numpy().astype(float)[order]
        sec = g["sec"].to_numpy()[order]
        ser = g["series"].to_list()[0]
        for i in range(len(thr) - 1):
            if thr[i + 1] <= thr[i]:
                continue
            rows.append(dict(series=ser, event_ticker=ev, date=date,
                             k_lo=float(thr[i]), k_hi=float(thr[i + 1]),
                             p_lo=float(px[i]), p_hi=float(px[i + 1]),
                             gap=float(px[i] - px[i + 1]),
                             dt_sec=int(abs(sec[i] - sec[i + 1]))))
    adj = pl.DataFrame(rows)
    print("=== 1. monotonicity on adjacent strikes: P(X>K) must fall as K rises ===")
    print("`gap` = p(low strike) - p(high strike); negative is an arbitrage.\n")
    out = []
    for label, tol in TIERS:
        s = adj.filter(pl.col("dt_sec") <= tol)
        if s.height < 50:
            continue
        v = s.filter(pl.col("gap") < 0)
        out.append(dict(tier=label, adjacent_pairs=s.height,
                        events=s["event_ticker"].n_unique(),
                        violations=v.height,
                        viol_rate=v.height / s.height,
                        mean_size_c=float(-v["gap"].mean()) if v.height else 0.0,
                        max_size_c=float(-v["gap"].min()) if v.height else 0.0,
                        gt_2c=int((v["gap"] < -2).sum()) if v.height else 0))
    with pl.Config(float_precision=4, tbl_width_chars=210):
        print(pl.DataFrame(out))

    if adj.height:
        print("\nviolation rate by series (within 60 min):")
        s60 = adj.filter(pl.col("dt_sec") <= 3600)
        with pl.Config(tbl_rows=25, float_precision=4):
            print(s60.group_by("series").agg(
                pl.len().alias("pairs"),
                (pl.col("gap") < 0).sum().alias("viol"),
                (pl.col("gap") < 0).mean().alias("rate"),
                pl.col("gap").filter(pl.col("gap") < 0).mean().alias("mean_gap"),
            ).filter(pl.col("pairs") >= 30).sort("rate", descending=True))

    # ------------------------------------------------------- the arb, priced
    print("\n=== 2. what the violations are worth after costs ===")
    print("Buy the low strike, sell the high one. The low leg pays whenever the")
    print("high one does, so the pair cannot lose at settlement: profit is the")
    print("price gap, banked at entry, minus two crossings.\n")
    rows = []
    for label, tol in TIERS:
        v = adj.filter((pl.col("dt_sec") <= tol) & (pl.col("gap") < 0))
        if v.height < 5:
            continue
        p_buy = v["p_lo"].to_numpy()
        p_sell = v["p_hi"].to_numpy()
        gross = p_sell - p_buy                      # = -gap > 0
        cost = fee_cents(p_buy) + fee_cents(100.0 - p_sell) + 1.0
        rows.append(dict(tier=label, n=v.height, events=v["event_ticker"].n_unique(),
                         gross=float(gross.mean()), cost=float(cost.mean()),
                         net=float((gross - cost).mean()),
                         frac_net_pos=float(((gross - cost) > 0).mean())))
    with pl.Config(float_precision=3, tbl_width_chars=210):
        print(pl.DataFrame(rows))
    print("cost = fee on both legs + 1c of crossing; no exit fee (settlement).")

    # ------------------------------------------------------ bucket sum-to-one
    bk = d.filter((pl.col("kind") == BUCKET) & pl.col("price").is_between(1, 99))
    if bk.height:
        print("\n=== 3. bucket ladders must sum to 100c (they partition outcomes) ===")
        g = (bk.group_by("event_ticker", "date")
             .agg(pl.col("price").sum().alias("mass"), pl.len().alias("legs"),
                  (pl.col("sec").max() - pl.col("sec").min()).alias("span"),
                  pl.col("series").first()))
        out = []
        for label, tol in TIERS:
            s = g.filter((pl.col("span") <= tol) & (pl.col("legs") >= 4))
            if s.height < 20:
                continue
            m = s["mass"].to_numpy()
            out.append(dict(tier=label, ladder_days=s.height,
                            events=s["event_ticker"].n_unique(),
                            median_mass=float(np.median(m)),
                            p10=float(np.percentile(m, 10)),
                            p90=float(np.percentile(m, 90)),
                            frac_over_105=float((m > 105).mean()),
                            frac_under_95=float((m < 95).mean())))
        with pl.Config(float_precision=2, tbl_width_chars=210):
            print(pl.DataFrame(out))
        print("\nNote: an incomplete ladder (legs that did not trade) biases the sum")
        print("DOWN, so a sum above 100 cannot be explained by missing legs.")

    OUT.mkdir(parents=True, exist_ok=True)
    adj.write_parquet(OUT / "coherence_adjacent.parquet")
    print(f"\nwrote {OUT / 'coherence_adjacent.parquet'}")


if __name__ == "__main__":
    main()
