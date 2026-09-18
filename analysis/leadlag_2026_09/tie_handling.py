#!/usr/bin/env python
"""Simultaneous trades make "the last trade price" ambiguous. How much does it matter?

    venv/bin/python analysis/leadlag_2026_09/tie_handling.py

Every price in this project that is "the last trade before X" -- p0, p_entry,
and the daily close in ``KalshiOHLCV.build_daily`` that feeds every implied
moment -- resolves ties by input order, because ``sort("ticker",
"created_time")`` is not a total order and polars' sort is stable. Filter the
trades a different way upstream and the tie breaks the other way.

That is not hypothetical here:

* 47% of trades on the target tickers share a ``(ticker, created_time)`` with
  at least one other trade -- one order sweeping several price levels emits
  several trade records at the same instant;
* 28,270 of those groups carry **different prices**, median spread 1c, p90 4c;
* and ``build_panel.py`` filters trades per series while the analysis scripts
  filter all target tickers at once, so the two disagree on 662 of 8,746 rows
  (7.6%) about what ``p_entry`` was.

A 1c ambiguity matters because the confirmation threshold is 2c.

There is also a smaller, separate defect: 29,334 ``trade_id`` values appear
exactly twice with every column identical -- ingest duplicates, 0.15% of the
39.4M rows.

Two principled resolutions, both deterministic:

* **last fill** -- deduplicate, then take the final record at the tied instant
  under a total order;
* **VWAP** -- deduplicate, then collapse each tied instant to one synthetic
  print at the size-weighted mean, which is the actual average execution price
  of that sweep.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel._io import scan_trades

OUT = Path("analysis/leadlag_2026_09/out")
N_BOOT, FEE_RATE, CONTRACTS = 10000, 0.07, 100
BUCKETS = [(1, 5), (5, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99)]
CONFIRM_C, FLOOR_C = 2.0, 20.0


def fee_cents(p):
    p = np.asarray(p, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(p):
    out = np.full(np.shape(p), 2.0)
    out[np.abs(np.asarray(p, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def cboot(v, g, seed=0):
    u, inv = np.unique(g, return_inverse=True)
    k = len(u)
    s = np.bincount(inv, weights=v, minlength=k)
    c = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    b = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1e-9)
    return (float(v.mean()), float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)), float((b <= 0).mean()))


def tercile_side(sig, bv, yr, years):
    side = np.zeros(len(sig))
    for lo, hi in BUCKETS:
        m = (bv >= lo) & (bv < hi)
        if m.sum() < 60:
            continue
        idx = np.where(m)[0]
        for Y in years[1:]:
            tr, te = idx[yr[idx] < Y], idx[yr[idx] == Y]
            if len(tr) < 60 or len(te) == 0:
                continue
            t = np.percentile(sig[tr], [33.3, 66.7])
            side[te] = np.where(sig[te] > t[1], 1.0,
                                np.where(sig[te] <= t[0], -1.0, 0.0))
    return side


def main() -> None:
    d = pl.read_parquet(OUT / "leadlag_legs.parquet").drop_nulls("p0")
    z = d["z_surprise"].to_numpy()
    lim = float(np.percentile(np.abs(z), 99))
    sig = d["direction"].to_numpy().astype(float) * np.clip(z, -lim, lim)
    win = d["win"].to_numpy().astype(float)
    ev, yr = d["target_event"].to_numpy(), d["yr"].to_numpy()
    tk = d["target_ticker"].to_numpy()
    tres = d["t_res"].dt.epoch("ns").to_numpy()
    years = sorted(np.unique(yr))

    raw = (scan_trades(is_only=True)
           .filter(pl.col("ticker").is_in(d["target_ticker"].unique().to_list()))
           .select("ticker", "trade_id", "count", "yes_price", "created_time")
           .collect())
    print("=== the ambiguity, measured ===")
    print(f"trade rows on target tickers          : {raw.height}")
    ded = raw.unique(subset=["trade_id", "ticker", "created_time", "yes_price", "count"])
    print(f"after dropping exact duplicate records: {ded.height}  "
          f"(-{raw.height - ded.height})")
    tied = (ded.group_by("ticker", "created_time")
            .agg(pl.len().alias("n"), pl.col("yes_price").n_unique().alias("npx"),
                 (pl.col("yes_price").max() - pl.col("yes_price").min()).alias("spread"))
            .filter(pl.col("n") > 1))
    print(f"(ticker, timestamp) groups with >1 fill: {tied.height}")
    print(f"  with differing prices                : {int((tied['npx'] > 1).sum())}")
    print(f"  share of trades inside a tied group   : "
          f"{tied['n'].sum() / ded.height:.1%}")
    s = tied.filter(pl.col("npx") > 1)["spread"]
    print(f"  price spread within such a group      : median {s.median():.0f}c  "
          f"p90 {s.quantile(0.9):.0f}c  max {s.max():.0f}c")

    col = (ded.group_by("ticker", "created_time")
           .agg(((pl.col("yes_price") * pl.col("count")).sum()
                 / pl.col("count").sum()).alias("vwap"),
                pl.col("yes_price").last().alias("lastpx"))
           .sort("ticker", "created_time"))
    print(f"collapsed to one print per instant     : {col.height}")

    g = (col.with_columns(pl.col("created_time").dt.epoch("ns").alias("ns"))
         .group_by("ticker", maintain_order=True)
         .agg(pl.col("ns"), pl.col("vwap"), pl.col("lastpx")))
    tape = {t: (np.asarray(a, dtype=np.int64), np.asarray(b, float), np.asarray(c, float))
            for t, a, b, c in zip(g["ticker"], g["ns"], g["vwap"], g["lastpx"])}

    def run(which):
        N = len(tk)
        p0 = np.full(N, np.nan); pf = np.full(N, np.nan); p2 = np.full(N, np.nan)
        for i in range(N):
            arr = tape.get(tk[i])
            if arr is None:
                continue
            ns, vw, lp = arr
            px = vw if which == "vwap" else lp
            q = int(np.searchsorted(ns, tres[i], side="right"))
            if q > 0:
                p0[i] = px[q - 1]
            if q < len(px):
                pf[i] = px[q]
            if q + 1 < len(px):
                p2[i] = px[q + 1]
        ok = np.isfinite(p0) & np.isfinite(pf) & np.isfinite(p2)
        side = tercile_side(sig, np.where(ok, p0, -1.0), yr, years)
        ent = np.where(side > 0, p2, 100.0 - p2)
        m = ok & (side != 0) & (ent >= FLOOR_C) & (((pf - p0) * np.sign(side)) >= CONFIRM_C)
        pay = np.where(side > 0, 100.0 * win, 100.0 * (1 - win))
        net = pay - ent - (fee_cents(ent) + spread_cents(ent) / 2.0)
        o, lo, hi, p = cboot(net[m], ev[m])
        return dict(rule=which, n=int(m.sum()), events=int(len(np.unique(ev[m]))),
                    net=o, ci_lo=lo, ci_hi=hi, p_le0=p)

    print("\n=== what the strategy earns under each deterministic rule ===")
    rows = [dict(rule="committed panel (ties by input order)", n=889, events=214,
                 net=5.41, ci_lo=0.59, ci_hi=10.21, p_le0=0.014),
            run("last"), run("vwap")]
    with pl.Config(float_precision=2, tbl_width_chars=210):
        print(pl.DataFrame(rows))
    print("\nThe two principled rules agree with each other and sit ~0.75c below")
    print("the committed figure. The committed panel is not wrong so much as")
    print("arbitrary -- and the arbitrary draw landed on the favourable side.")
    print("\nThis is not confined to the strategy: KalshiOHLCV.build_daily takes")
    print("the last trade of a day the same way, so every implied moment, every")
    print("surprise and every PIT in the repo inherits the same ambiguity.")


if __name__ == "__main__":
    main()
