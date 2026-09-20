#!/usr/bin/env python
"""Is the edge concentrated in thin contracts? The underreaction cross-section.

    venv/bin/python analysis/leadlag_2026_09/liquidity.py

`move_filter.py` C and the move-size cross-section both point at underreaction:
the market moves part of the way on the trigger and the rest arrives later. If
that reading is right, the mechanism is **slow information diffusion**, and the
standard cross-sectional prediction follows -- Hong-Lim-Stein: underreaction is
larger where fewer people are watching. Thin contracts should pay more than
liquid ones, and they should also *reprice more slowly*, which is the same
mechanism measured without reference to profit.

If instead the edge is flat in liquidity, or concentrated in the *liquid*
names, the underreaction story is in trouble and the effect needs a different
explanation.

Liquidity is measured on the target leg's own tape **strictly before t_res**,
so nothing here can see the post-trigger window the trade is taken in:

    n_pre      collapsed prints on this ticker before t_res
    vol_pre    contracts traded on this ticker before t_res
    days_pre   distinct calendar days with a print before t_res
    stale_h    hours from the last pre-t_res print to t_res
    ev_vol     contracts traded across the whole target event before t_res

Everything else -- terciles, confirmation, floor, costs -- is spec v2 unchanged.

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
CONFIRM_C, FLOOR_C, KCONF = 2.0, 20.0, 3
NS_H = 3_600_000_000_000


def fee_cents(p):
    p = np.asarray(p, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(p):
    out = np.full(np.shape(p), 2.0)
    out[np.abs(np.asarray(p, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def cboot(v, g, seed=0):
    if len(v) == 0:
        return (np.nan,) * 4
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


def show(rows, cols=None):
    with pl.Config(float_precision=2, tbl_width_chars=210, tbl_rows=40):
        print(pl.DataFrame(rows))


def main() -> None:
    d = pl.read_parquet(OUT / "leadlag_legs.parquet").drop_nulls("p0")
    z = d["z_surprise"].to_numpy()
    lim = float(np.percentile(np.abs(z), 99))
    sig = d["direction"].to_numpy().astype(float) * np.clip(z, -lim, lim)
    win = d["win"].to_numpy().astype(float)
    ev, yr = d["target_event"].to_numpy(), d["yr"].to_numpy()
    tk = d["target_ticker"].to_numpy()
    tev = d["target_event"].to_numpy()
    tgt = d["target"].to_numpy()
    tres = d["t_res"].dt.epoch("ns").to_numpy()
    years = sorted(np.unique(yr))

    raw = (scan_trades(is_only=True)
           .filter(pl.col("ticker").is_in(d["target_ticker"].unique().to_list()))
           .select("ticker", "trade_id", "count", "yes_price", "created_time")
           .collect())
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

    # ---- per-row: entry window + pre-t_res liquidity ---------------------
    N = len(tk)
    p0 = np.full(N, np.nan); pc = np.full(N, np.nan); pf = np.full(N, np.nan)
    n_pre = np.zeros(N); vol_pre = np.zeros(N); days_pre = np.zeros(N)
    stale_h = np.full(N, np.nan); first_h = np.full(N, np.nan)
    for i in range(N):
        a = tape.get(tk[i])
        if a is None:
            continue
        ns, px, sz = a
        q = int(np.searchsorted(ns, tres[i], side="right"))
        n_pre[i] = q
        if q > 0:
            p0[i] = px[q - 1]
            vol_pre[i] = sz[:q].sum()
            days_pre[i] = len(np.unique(ns[:q] // (24 * NS_H)))
            stale_h[i] = (tres[i] - ns[q - 1]) / NS_H
        if q < len(ns):
            first_h[i] = (ns[q] - tres[i]) / NS_H
        if q + KCONF < len(px):
            pc[i] = np.median(px[q:q + KCONF])
            pf[i] = px[q + KCONF]
    ok = np.isfinite(p0) & np.isfinite(pc) & np.isfinite(pf)

    # event-level pre-t_res volume: sum vol_pre over the legs of the event
    ev_vol = np.zeros(N)
    for e in np.unique(tev):
        k = tev == e
        for t in np.unique(tres[k]):
            kk = k & (tres == t)
            ev_vol[kk] = vol_pre[kk].sum()

    side = tercile_side(sig, np.where(ok, p0, -1.0), yr, years)
    ent = np.where(side > 0, pf, 100.0 - pf)
    sel = (ok & (side != 0)
           & (((pc - p0) * np.sign(side)) >= CONFIRM_C)
           & (ent >= FLOOR_C))
    broad = ok & (side != 0)          # signal only, no confirm/floor
    gross = np.where(side > 0, 100.0 * win, 100.0 * (1 - win)) - ent
    net = gross - fee_cents(ent) - spread_cents(ent) / 2.0

    print(f"rows {N}   usable window {int(ok.sum())}   "
          f"spec-v2 positions {int(sel.sum())}   signal-only {int(broad.sum())}\n")

    MEAS = [("n_pre", n_pre, "prints on this leg before t_res"),
            ("vol_pre", vol_pre, "contracts on this leg before t_res"),
            ("days_pre", days_pre, "distinct trading days before t_res"),
            ("ev_vol", ev_vol, "contracts across the whole event before t_res"),
            ("stale_h", stale_h, "hours since last print at t_res (HIGH = thin)")]

    # ---- 0. mechanism, no profit involved --------------------------------
    print("=== 0. do thin contracts reprice more slowly? ===")
    print("   hours from t_res to the first print, by pre-t_res liquidity\n")
    rows = []
    for nm, v, _ in MEAS:
        k = ok & np.isfinite(v) & np.isfinite(first_h)
        q = np.percentile(v[k], [33.3, 66.7])
        for lab, kk in (("thin", k & (v <= q[0])), ("mid", k & (v > q[0]) & (v <= q[1])),
                        ("liquid", k & (v > q[1]))):
            if nm == "stale_h":
                lab = {"thin": "liquid", "liquid": "thin", "mid": "mid"}[lab]
            rows.append(dict(measure=nm, tercile=lab, n=int(kk.sum()),
                             med_h=float(np.median(first_h[kk])),
                             mean_h=float(first_h[kk].mean()),
                             med_move=float(np.median(np.abs(pc - p0)[kk]))))
    show(rows)

    # ---- 1. net by liquidity tercile, spec v2 set ------------------------
    for label, mask in (("SPEC v2 (confirm + floor)", sel), ("SIGNAL ONLY (no filters)", broad)):
        print(f"\n=== 1. net by liquidity tercile -- {label} ===")
        rows = []
        for nm, v, _ in MEAS:
            k = mask & np.isfinite(v)
            q = np.percentile(v[k], [33.3, 66.7])
            band = [("Q1 thin", k & (v <= q[0])), ("Q2", k & (v > q[0]) & (v <= q[1])),
                    ("Q3 liquid", k & (v > q[1]))]
            if nm == "stale_h":
                band = [("Q1 thin", k & (v > q[1])), ("Q2", k & (v > q[0]) & (v <= q[1])),
                        ("Q3 liquid", k & (v <= q[0]))]
            for lab, kk in band:
                o, lo, hi, p = cboot(net[kk], ev[kk])
                rows.append(dict(measure=nm, tercile=lab, n=int(kk.sum()),
                                 events=int(len(np.unique(ev[kk]))),
                                 mean_entry=float(ent[kk].mean()),
                                 gross=float(gross[kk].mean()),
                                 net=o, ci_lo=lo, ci_hi=hi, p_le0=p))
        show(rows)

    # ---- 2. the confound: liquidity vs moneyness, series, year -----------
    print("\n=== 2. what is liquidity confounded with? (spec v2 set) ===")
    k = sel & np.isfinite(vol_pre)
    q = np.percentile(vol_pre[k], [33.3, 66.7])
    rows = []
    for lab, kk in (("Q1 thin", k & (vol_pre <= q[0])),
                    ("Q2", k & (vol_pre > q[0]) & (vol_pre <= q[1])),
                    ("Q3 liquid", k & (vol_pre > q[1]))):
        top = pl.Series("s", tgt[kk]).value_counts(sort=True).head(3)
        rows.append(dict(tercile=lab, n=int(kk.sum()),
                         mean_p0=float(p0[kk].mean()),
                         mean_entry=float(ent[kk].mean()),
                         mean_yr=float(yr[kk].mean()),
                         med_vol=float(np.median(vol_pre[kk])),
                         top_series="; ".join(f"{a}:{b}" for a, b in
                                              zip(top["s"], top["count"]))))
    show(rows)

    # ---- 3. control for moneyness: double sort ---------------------------
    print("\n=== 3. double sort -- liquidity within entry-price band (spec v2) ===")
    rows = []
    for elo, ehi in ((20, 60), (60, 85), (85, 100)):
        k = sel & (ent >= elo) & (ent < ehi) & np.isfinite(vol_pre)
        if k.sum() < 60:
            continue
        q = np.percentile(vol_pre[k], [50.0])
        for lab, kk in (("thin", k & (vol_pre <= q[0])), ("liquid", k & (vol_pre > q[0]))):
            o, lo, hi, p = cboot(net[kk], ev[kk])
            rows.append(dict(entry_band=f"{elo}-{ehi}c", half=lab, n=int(kk.sum()),
                             events=int(len(np.unique(ev[kk]))),
                             net=o, ci_lo=lo, ci_hi=hi, p_le0=p))
    show(rows)

    # ---- 4. paired within-event contrast ---------------------------------
    print("\n=== 4. within target event: thin legs minus liquid legs ===")
    print("   each event contributes one difference, so moneyness/series/date")
    print("   are all held fixed by construction\n")
    for label, mask in (("SPEC v2", sel), ("SIGNAL ONLY", broad)):
        diffs, ns_ = [], []
        for e in np.unique(ev[mask]):
            k = mask & (ev == e)
            if k.sum() < 4:
                continue
            v = vol_pre[k]
            med = np.median(v)
            a, b = net[k][v <= med], net[k][v > med]
            if len(a) == 0 or len(b) == 0:
                continue
            diffs.append(a.mean() - b.mean()); ns_.append(int(k.sum()))
        diffs = np.array(diffs)
        rng = np.random.default_rng(0)
        bs = diffs[rng.integers(0, len(diffs), (N_BOOT, len(diffs)))].mean(1)
        print(f"  {label:<12} events {len(diffs):>4}   "
              f"thin - liquid = {diffs.mean():+.2f}c   "
              f"95% CI [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]   "
              f"P(<=0) = {(bs <= 0).mean():.3f}")


    # ---- 5. capacity: how much can actually be traded in the thin cell ---
    print("\n=== 5. capacity -- volume available AFTER t_res, by liquidity ===")
    vol_post = np.zeros(N)
    for i in range(N):
        a = tape.get(tk[i])
        if a is None:
            continue
        ns, px, sz = a
        q = int(np.searchsorted(ns, tres[i], side="right"))
        vol_post[i] = sz[q:].sum()
    k = sel & np.isfinite(vol_pre)
    q = np.percentile(vol_pre[k], [33.3, 66.7])
    rows = []
    for lab, kk in (("Q1 thin", k & (vol_pre <= q[0])),
                    ("Q2", k & (vol_pre > q[0]) & (vol_pre <= q[1])),
                    ("Q3 liquid", k & (vol_pre > q[1]))):
        rows.append(dict(tercile=lab, n=int(kk.sum()),
                         net=float(net[kk].mean()),
                         med_vol_post=float(np.median(vol_post[kk])),
                         p25_vol_post=float(np.percentile(vol_post[kk], 25)),
                         med_profit_at_5pct_of_vol=float(
                             np.median(0.05 * vol_post[kk] * net[kk]) / 100.0)))
    show(rows)
    print("  med_profit_at_5pct_of_vol is USD per position, taking 5% of the")
    print("  leg's whole post-trigger volume at the modelled net edge.")


if __name__ == "__main__":
    main()
