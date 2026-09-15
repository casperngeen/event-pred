#!/usr/bin/env python
"""Is the confirmation-filter trade stable across years?

    venv/bin/python analysis/leadlag_2026_09/confirmation_stability.py
    (needs maker_fill.py)

``move_filter.py`` found that requiring the market to have already moved in the
signal's direction lifts net from +0.40c to +2.30c, and entering at the **second**
post-news print (the first one you could actually act on) lifts it further to
+3.53c with P(<=0) = 0.044. That is the best economic result in this study.

It is also one in-sample specification search away from being a finding: the
threshold was chosen after looking, and every other effect in this project
decays across 2023-2025. This tests both.

Three arms
----------
* **fixed k = 2c**, split by year -- does the chosen specification hold up?
* **walk-forward k**, chosen on prior years only by net P&L -- the honest
  version, where nothing is fitted on the year being scored.
* **chronological halves**, because three calendar years is very few clusters
  and halves give two larger ones.

Entry is the second post-news print throughout, so the confirming print is
never also the fill. Costs are taker: fee plus half the measured spread.
Clustered on ``target_event``; the pooled figure is additionally clustered on
year, which with three years is very coarse and reported as such.

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
GRID = (0, 1, 2, 3, 5)


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


def main() -> None:
    d = pl.read_parquet(OUT / "maker_fill.parquet")
    tk = d["target_ticker"].to_numpy()
    tres = d["t_res"].dt.epoch("s").to_numpy()
    sd = d["side"].to_numpy()
    p0, pe = d["p0"].to_numpy(), d["p_entry"].to_numpy()
    win = d["win"].to_numpy().astype(float)
    ev, yr = d["target_event"].to_numpy(), d["yr"].to_numpy()

    tt = (scan_trades(is_only=True)
          .filter(pl.col("ticker").is_in(d["target_ticker"].unique().to_list()))
          .select("ticker", "yes_price", "created_time")
          .sort("ticker", "created_time").collect())
    g = (tt.with_columns(pl.col("created_time").dt.epoch("s").alias("s"))
         .group_by("ticker", maintain_order=True)
         .agg(pl.col("s"), pl.col("yes_price")))
    tape = {t: (np.asarray(a, dtype=np.int64), np.asarray(b, dtype=float))
            for t, a, b in zip(g["ticker"], g["s"], g["yes_price"])}

    p2 = np.full(len(sd), np.nan)
    for i in range(len(sd)):
        arr = tape.get(tk[i])
        if arr is None:
            continue
        s_, px_ = arr
        j = int(np.searchsorted(s_, tres[i], side="right"))
        if j + 1 < len(s_):
            p2[i] = px_[j + 1]
    ok = ~np.isnan(p2)
    move = (pe - p0) * np.sign(sd)
    ent = np.where(sd > 0, p2, 100.0 - p2)
    pay = np.where(sd > 0, 100.0 * win, 100.0 * (1 - win))
    net = pay - ent - (fee_cents(ent) + spread_cents(ent) / 2.0)
    print(f"positions with an executable entry: {int(ok.sum())} of {len(ok)}")

    years = sorted(np.unique(yr[ok]))

    # ------------------------------------------------------- fixed k, by year
    print("\n=== A. fixed threshold k = 2c, by year ===\n")
    rows = []
    for k in (0, 2):
        for y in years:
            m = ok & (yr == y) & (move >= k)
            if m.sum() < 40:
                continue
            o, lo, hi, p = cboot(net[m], ev[m])
            rows.append(dict(k=k, year=int(y), n=int(m.sum()),
                             events=int(len(np.unique(ev[m]))),
                             net=o, ci_lo=lo, ci_hi=hi, p_le0=p))
    t = pl.DataFrame(rows)
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=210):
        print(t)

    m2 = ok & (move >= 2)
    o, lo, hi, p = cboot(net[m2], ev[m2])
    print(f"\npooled k=2, event-clustered : {o:+.2f}c  CI [{lo:+.2f}, {hi:+.2f}]  P(<=0) = {p:.3f}")
    o2, lo2, hi2, p2_ = cboot(net[m2], yr[m2], seed=1)
    print(f"pooled k=2, YEAR-clustered  : {o2:+.2f}c  CI [{lo2:+.2f}, {hi2:+.2f}]  "
          f"P(<=0) = {p2_:.3f}   ({len(np.unique(yr[m2]))} year clusters -- very coarse)")

    # ------------------------------------------------------ walk-forward k
    print("\n=== B. walk-forward threshold: k chosen on prior years only ===\n")
    scored = np.zeros(len(net), dtype=bool)
    chosen = {}
    for y in years[1:]:
        tr = ok & (yr < y)
        te = ok & (yr == y)
        if tr.sum() < 200 or te.sum() < 40:
            continue
        best_k, best_v = None, -np.inf
        for k in GRID:
            mm = tr & (move >= k)
            if mm.sum() < 100:
                continue
            v = float(net[mm].mean())
            if v > best_v:
                best_k, best_v = k, v
        if best_k is None:
            continue
        chosen[int(y)] = best_k
        scored |= te & (move >= best_k)
    print(f"threshold chosen per year: {chosen}")
    rows = []
    for y in years[1:]:
        m = scored & (yr == y)
        if m.sum() < 30:
            continue
        o, lo, hi, p = cboot(net[m], ev[m])
        rows.append(dict(year=int(y), k=chosen.get(int(y)), n=int(m.sum()),
                         events=int(len(np.unique(ev[m]))),
                         net=o, ci_lo=lo, ci_hi=hi, p_le0=p))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=210):
        print(pl.DataFrame(rows))
    if scored.sum() > 50:
        o, lo, hi, p = cboot(net[scored], ev[scored])
        print(f"\npooled walk-forward: {o:+.2f}c  CI [{lo:+.2f}, {hi:+.2f}]  "
              f"P(<=0) = {p:.3f}   n = {int(scored.sum())}")

    # ------------------------------------------------- chronological halves
    print("\n=== C. chronological halves (bigger clusters than calendar years) ===\n")
    tsort = np.sort(tres[ok])
    cut = tsort[len(tsort) // 2]
    rows = []
    for nm, mask in (("first half", ok & (tres <= cut)),
                     ("second half", ok & (tres > cut))):
        for k in (0, 2):
            m = mask & (move >= k)
            if m.sum() < 40:
                continue
            o, lo, hi, p = cboot(net[m], ev[m])
            rows.append(dict(period=nm, k=k, n=int(m.sum()),
                             events=int(len(np.unique(ev[m]))),
                             net=o, ci_lo=lo, ci_hi=hi, p_le0=p))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=210):
        print(pl.DataFrame(rows))

    # ----------------------------------------------------------- the filter
    print("\n=== D. does the filter help in EVERY period, or only on average? ===")
    print("net(k=2) - net(k=0), same period. Positive means the filter earned")
    print("its keep that year.\n")
    rows = []
    for y in years:
        a = ok & (yr == y) & (move >= 2)
        b = ok & (yr == y)
        if a.sum() < 40 or b.sum() < 40:
            continue
        rows.append(dict(year=int(y), n_filtered=int(a.sum()), n_all=int(b.sum()),
                         net_k2=float(net[a].mean()), net_k0=float(net[b].mean()),
                         gain=float(net[a].mean() - net[b].mean())))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
