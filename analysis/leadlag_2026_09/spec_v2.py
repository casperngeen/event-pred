#!/usr/bin/env python
"""Spec v2: VWAP-collapsed tape + median-of-3 confirmation.

    venv/bin/python analysis/leadlag_2026_09/spec_v2.py

Two changes from reports/strategy_spec.md, both responses to defects found in
the audit:

1. **VWAP tape.** Exact duplicate trade records are dropped, then every
   ``(ticker, created_time)`` group is collapsed to one synthetic print at the
   size-weighted mean. A single order sweeping the book emits many records at
   one instant; taking "the last" of them is order-dependent and therefore not
   a rule at all (verified: reordering the input changes it, while VWAP and
   max/min are invariant). The committed panel's tie-break sat at the 98th
   percentile of 60 random tie-breaks.
2. **Median-of-3 confirmation.** The confirming price is the median of the
   first three collapsed prints after ``t_res`` rather than the first one, and
   the fill moves to the fourth print so signal and execution never overlap.

Everything else is unchanged: walk-forward terciles bucketed on ``p0``,
``move >= 2c``, ``entry >= 20c``, hold to settlement, taker costs.

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
N_BOOT, N_PERM, FEE_RATE, CONTRACTS = 10000, 200, 0.07, 100
BUCKETS = [(1, 5), (5, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99)]
CONFIRM_C, FLOOR_C, KCONF = 2.0, 20.0, 3


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
    zc = np.clip(z, -lim, lim)
    dirn = d["direction"].to_numpy().astype(float)
    sig = dirn * zc
    win = d["win"].to_numpy().astype(float)
    ev, yr = d["target_event"].to_numpy(), d["yr"].to_numpy()
    tk = d["target_ticker"].to_numpy()
    tres = d["t_res"].dt.epoch("ns").to_numpy()
    gap = d["gap_days"].to_numpy()
    years = sorted(np.unique(yr))

    raw = (scan_trades(is_only=True)
           .filter(pl.col("ticker").is_in(d["target_ticker"].unique().to_list()))
           .select("ticker", "trade_id", "count", "yes_price", "created_time")
           .collect())
    ded = raw.unique(subset=["trade_id", "ticker", "created_time", "yes_price", "count"])
    col = (ded.group_by("ticker", "created_time")
           .agg(((pl.col("yes_price") * pl.col("count")).sum()
                 / pl.col("count").sum()).alias("vwap"))
           .sort("ticker", "created_time"))
    g = (col.with_columns(pl.col("created_time").dt.epoch("ns").alias("ns"))
         .group_by("ticker", maintain_order=True)
         .agg(pl.col("ns"), pl.col("vwap")))
    tape = {t: (np.asarray(a, dtype=np.int64), np.asarray(b, float))
            for t, a, b in zip(g["ticker"], g["ns"], g["vwap"])}
    print(f"raw trade rows {raw.height} -> deduped {ded.height} -> "
          f"VWAP prints {col.height}")

    N = len(tk)
    p0 = np.full(N, np.nan); pc = np.full(N, np.nan); pf = np.full(N, np.nan)
    for i in range(N):
        a = tape.get(tk[i])
        if a is None:
            continue
        ns, px = a
        q = int(np.searchsorted(ns, tres[i], side="right"))
        if q > 0:
            p0[i] = px[q - 1]
        if q + KCONF < len(px):
            pc[i] = np.median(px[q:q + KCONF])
            pf[i] = px[q + KCONF]
    ok = np.isfinite(p0) & np.isfinite(pc) & np.isfinite(pf)
    print(f"legs with a usable window (needs {KCONF + 1} prints after t_res): "
          f"{int(ok.sum())} of {N}\n")

    def evaluate(signal, confirm=True, floor=True):
        side = tercile_side(signal, np.where(ok, p0, -1.0), yr, years)
        ent = np.where(side > 0, pf, 100.0 - pf)
        m = ok & (side != 0)
        if confirm:
            m = m & (((pc - p0) * np.sign(side)) >= CONFIRM_C)
        if floor:
            m = m & (ent >= FLOOR_C)
        pay = np.where(side > 0, 100.0 * win, 100.0 * (1 - win))
        cost = fee_cents(ent) + spread_cents(ent) / 2.0
        return m, pay - ent, cost, ent, side

    m, gross, cost, ent, side = evaluate(sig)
    net = gross - cost
    o, lo, hi, p = cboot(net[m], ev[m])
    print("=== SPEC v2 headline ===")
    print(f"  positions        {int(m.sum())}   target events {len(np.unique(ev[m]))}")
    print(f"  mean entry       {ent[m].mean():.1f}c   win rate {(gross[m] > 0).mean():.3f}"
          f"   mean hold {gap[m].mean():.0f}d")
    print(f"  gross            {gross[m].mean():+.2f}c")
    print(f"  friction         {cost[m].mean():.2f}c")
    print(f"  NET              {o:+.2f}c   95% CI [{lo:+.2f}, {hi:+.2f}]   P(<=0) = {p:.3f}")
    print(f"  return on capital  equal-contract {100*net[m].sum()/ent[m].sum():+.1f}%"
          f"   equal-dollar {100*np.mean(net[m]/ent[m]):+.1f}%")

    print("\n=== by year ===")
    rows = []
    for Y in years:
        k = m & (yr == Y)
        if k.sum() < 30:
            continue
        oy, ly, hy, py = cboot(net[k], ev[k])
        rows.append(dict(year=int(Y), n=int(k.sum()), events=int(len(np.unique(ev[k]))),
                         net=oy, ci_lo=ly, ci_hi=hy, p_le0=py))
    with pl.Config(float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    print("\n=== ablation ===")
    rows = []
    for nm, cf, fl in (("full (signal+confirm+floor)", True, True),
                       ("drop floor", True, False),
                       ("drop confirm", False, True),
                       ("signal only", False, False)):
        mm, gg, cc, _, _ = evaluate(sig, cf, fl)
        nn = gg - cc
        oo, ll, hh, pp = cboot(nn[mm], ev[mm])
        rows.append(dict(variant=nm, n=int(mm.sum()), net=oo, ci_lo=ll, ci_hi=hh, p_le0=pp))
    with pl.Config(float_precision=2, tbl_width_chars=210):
        print(pl.DataFrame(rows))

    print("\n=== block-permutation null (signal shuffled within trigger series) ===")
    tev = d["trigger_event"].to_numpy(); tsr = d["trigger"].to_numpy()
    u, ri = np.unique(tev, return_inverse=True)
    tz = np.zeros(len(u)); ts = np.empty(len(u), dtype=object)
    for i2, t in enumerate(u):
        j = int(np.argmax(tev == t)); tz[i2], ts[i2] = zc[j], tsr[j]
    grp = [np.where(ts == s)[0] for s in np.unique(ts)]
    rng = np.random.default_rng(1)
    null = []
    for _ in range(N_PERM):
        zp = tz.copy()
        for gg2 in grp:
            zp[gg2] = tz[rng.permutation(gg2)]
        mm, g2, c2, _, _ = evaluate(dirn * zp[ri])
        if mm.sum() > 30:
            null.append(float((g2 - c2)[mm].mean()))
    null = np.array(null)
    r = int((null >= o).sum())
    print(f"  observed {o:+.2f}c   null {null.mean():+.2f} +/- {null.std():.2f}"
          f"   draws >= observed {r}/{len(null)}   p = {(r+1)/(len(null)+1):.4f}")
    print(f"  signal's marginal contribution over the apparatus: {o - null.mean():+.2f}c")


if __name__ == "__main__":
    main()
