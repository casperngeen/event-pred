#!/usr/bin/env python
"""Why does the thin-minus-liquid spread swing 0.94c -> 7.43c across k?

    venv/bin/python analysis/leadlag_2026_09/liquidity_spread_diag.py

`liquidity_gates.py` §4 reported the spread at four confirmation windows and
it was wildly unstable. Two things were wrong with how that table was read.

1. **No CI was ever put on the spread itself.** Two cells each with a ~+/-7c
   interval were differenced and the difference reported as a point estimate.
   Comparing two overlapping CIs is not a test of their difference, so §4
   could not distinguish "the gradient moved" from "nothing moved".

2. **Three things change at once when k changes**, and §4 attributed all of
   it to the gate:
     (a) SELECTION -- which legs clear `move >= 2c`, since the confirming
         price is a median-of-k and k changes it;
     (b) PRICING   -- the fill moves from the (k+1)th print to the 2nd, so
         the same leg is entered at a different price;
     (c) BUCKETING -- terciles are recut inside each variant's own selected
         set, so "thin" is a different population in every row.

This script separates them: §2 puts a clustered CI on the spread, §3 measures
the overlap between the selected sets, and §4 holds the population fixed at
the common core so only pricing varies.

In-sample only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

os.environ["POLARS_FMT_MAX_COLS"] = "20"
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


def cboot_diff(va, ga, vb, gb, seed=0):
    """Event-clustered CI on mean(a) - mean(b), resampling the union of events.

    Events are the cluster because one print settles every leg of a target
    event, and an event can contribute legs to *both* arms -- so the two arms
    must be resampled together, not independently.
    """
    ev = np.unique(np.concatenate([ga, gb]))
    idx = {e: i for i, e in enumerate(ev)}
    k = len(ev)
    ia = np.array([idx[e] for e in ga]); ib = np.array([idx[e] for e in gb])
    sa = np.bincount(ia, weights=va, minlength=k); ca = np.bincount(ia, minlength=k).astype(float)
    sb = np.bincount(ib, weights=vb, minlength=k); cb = np.bincount(ib, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    na, da = sa[pick].sum(1), ca[pick].sum(1)
    nb, db = sb[pick].sum(1), cb[pick].sum(1)
    ok = (da > 0) & (db > 0)
    b = na[ok] / da[ok] - nb[ok] / db[ok]
    d = float(va.mean() - vb.mean())
    return d, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float((b <= 0).mean())


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


def show(rows):
    with pl.Config(float_precision=2, tbl_width_chars=400, tbl_rows=60):
        print(pl.DataFrame(rows))


def main() -> None:
    d = pl.read_parquet(OUT / "leadlag_legs.parquet")
    tickers = d["target_ticker"].unique().to_list()
    raw = (scan_trades(is_only=True).filter(pl.col("ticker").is_in(tickers))
           .select("ticker", "trade_id", "count", "yes_price", "created_time").collect())
    ded = raw.unique(subset=["trade_id", "ticker", "created_time", "yes_price", "count"])
    col = (ded.group_by("ticker", "created_time")
           .agg(((pl.col("yes_price") * pl.col("count")).sum()
                 / pl.col("count").sum()).alias("vwap"),
                pl.col("count").sum().alias("sz"))
           .sort("ticker", "created_time"))
    g = (col.with_columns(pl.col("created_time").dt.epoch("ns").alias("ns"))
         .group_by("ticker", maintain_order=True)
         .agg(pl.col("ns"), pl.col("vwap"), pl.col("sz")))
    tape = {t: (np.asarray(a, np.int64), np.asarray(b, float), np.asarray(c, float))
            for t, a, b, c in zip(g["ticker"], g["ns"], g["vwap"], g["sz"])}

    z = d["z_surprise"].to_numpy()
    lim = float(np.percentile(np.abs(z), 99))
    sig = d["direction"].to_numpy().astype(float) * np.clip(z, -lim, lim)
    win = d["win"].to_numpy().astype(float)
    ev, yr = d["target_event"].to_numpy(), d["yr"].to_numpy()
    tk = d["target_ticker"].to_numpy()
    tres = d["t_res"].dt.epoch("ns").to_numpy()
    years = sorted(np.unique(yr))
    N = len(tk)

    p0 = np.full(N, np.nan); vol_pre = np.zeros(N); px_post: list = [None] * N
    for i in range(N):
        a = tape.get(tk[i])
        if a is None:
            continue
        ns, px, sz = a
        q = int(np.searchsorted(ns, tres[i], side="right"))
        if q > 0:
            p0[i] = px[q - 1]; vol_pre[i] = sz[:q].sum()
        px_post[i] = px[q:]

    def run(kconf, confirm=True):
        pc = np.full(N, np.nan); pf = np.full(N, np.nan)
        for i in range(N):
            a = px_post[i]
            if a is None:
                continue
            if len(a) < (kconf + 1 if confirm else 1):
                continue
            if confirm:
                pc[i] = np.median(a[:kconf]); pf[i] = a[kconf]
            else:
                pf[i] = a[0]
        ok = np.isfinite(p0) & np.isfinite(pf)
        if confirm:
            ok = ok & np.isfinite(pc)
        side = tercile_side(sig, np.where(ok, p0, -1.0), yr, years)
        ent = np.where(side > 0, pf, 100.0 - pf)
        m = ok & (side != 0) & (ent >= FLOOR_C)
        if confirm:
            m = m & (((pc - p0) * np.sign(side)) >= CONFIRM_C)
        gross = np.where(side > 0, 100.0 * win, 100.0 * (1 - win)) - ent
        return m.astype(bool), gross - fee_cents(ent) - spread_cents(ent) / 2.0

    V = [("k=3 (shipped)", 3, True), ("k=2", 2, True),
         ("k=1", 1, True), ("no confirm", 0, False)]
    R = {lab: run(k, c) for lab, k, c in V}

    # ---- 1. what the confirmation filter costs vs the print requirement ---
    print("=== 1. where do the positions actually go? ===")
    rows = []
    for lab, k, c in V:
        m, _ = R[lab]
        need = (k + 1) if c else 1
        havep = np.array([a is not None and len(a) >= need for a in px_post]) & np.isfinite(p0)
        rows.append(dict(variant=lab, legs_with_enough_prints=int(havep.sum()),
                         selected=int(m.sum()),
                         dropped_by_confirm_floor_tercile=int(havep.sum()) - int(m.sum())))
    show(rows)
    print("  the print-count requirement costs ~100-800 legs; the 2c confirmation")
    print("  filter costs ~6,000. k is not what sets the sample size.\n")

    # ---- 2. a CI on the spread itself ------------------------------------
    print("=== 2. thin - liquid, WITH an event-clustered CI on the difference ===")
    rows = []
    for lab, _, _ in V:
        m, net = R[lab]
        kk = m & np.isfinite(vol_pre)
        q = np.percentile(vol_pre[kk], [33.3, 66.7])
        a = kk & (vol_pre <= q[0]); b = kk & (vol_pre > q[1])
        dd, lo, hi, p = cboot_diff(net[a], ev[a], net[b], ev[b])
        rows.append(dict(variant=lab, n_thin=int(a.sum()), n_liq=int(b.sum()),
                         thin=float(net[a].mean()), liquid=float(net[b].mean()),
                         spread=dd, ci_lo=lo, ci_hi=hi, p_le0=p))
    show(rows)

    # ---- 3. how different are the selected sets? -------------------------
    print("\n=== 3. overlap of the selected sets (Jaccard on leg-rows) ===")
    labs = [v[0] for v in V]
    rows = []
    for a in labs:
        r = {"variant": a}
        for b in labs:
            ma, mb = R[a][0], R[b][0]
            r[b] = float((ma & mb).sum() / max((ma | mb).sum(), 1))
        rows.append(r)
    show(rows)
    core = R["k=3 (shipped)"][0] & R["k=2"][0] & R["k=1"][0]
    print(f"  common core of k=1,2,3: {int(core.sum())} legs "
          f"({int(core.sum())/int(R['k=3 (shipped)'][0].sum()):.0%} of the shipped 843)")

    # ---- 4. hold the population fixed; vary only the pricing -------------
    print("\n=== 4. same legs, different fill print -- pricing effect alone ===")
    rows = []
    q = np.percentile(vol_pre[core], [33.3, 66.7])
    a0 = core & (vol_pre <= q[0]); b0 = core & (vol_pre > q[1])
    for lab in ["k=3 (shipped)", "k=2", "k=1"]:
        net = R[lab][1]
        dd, lo, hi, p = cboot_diff(net[a0], ev[a0], net[b0], ev[b0])
        rows.append(dict(variant=lab, n=int(core.sum()),
                         net_all=float(net[core].mean()),
                         thin=float(net[a0].mean()), liquid=float(net[b0].mean()),
                         spread=dd, ci_lo=lo, ci_hi=hi, p_le0=p))
    show(rows)
    print("  tercile cutpoints fixed on the core, so 'thin' is the same legs")
    print("  in every row and only the entry price changes.\n")

    # ---- 5. how much do the cutpoints move? ------------------------------
    print("=== 5. how much does 'thin' itself move between variants? ===")
    rows = []
    for lab, _, _ in V:
        m, _ = R[lab]
        kk = m & np.isfinite(vol_pre)
        q = np.percentile(vol_pre[kk], [33.3, 66.7])
        rows.append(dict(variant=lab, n=int(kk.sum()),
                         cut_33=float(q[0]), cut_67=float(q[1]),
                         med_vol=float(np.median(vol_pre[kk]))))
    show(rows)


if __name__ == "__main__":
    main()
