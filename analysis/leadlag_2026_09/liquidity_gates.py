#!/usr/bin/env python
"""How much liquidity selection is already baked in, and does relaxing it
change the thin-vs-liquid gradient?

    venv/bin/python analysis/leadlag_2026_09/liquidity_gates.py

`liquidity.py` found the edge concentrated in thin legs. That test is run on a
population that has **already been filtered for liquidity four times**, and
three of the four are invisible in `build_panel.py`:

  G0  data/trades/ is a convenience sample (research_log.md §12). A leg the
      markets metadata says traded, but for which no trades were collected,
      is not in the panel at all. This gate is not relaxable -- it is the
      shape of the archive -- but it is measurable, and §1 measures it.
  G1  build_panel.py keeps a leg only if it has >= 1 print strictly AFTER
      t_res.
  G2  spec_v2 drops legs with no print BEFORE t_res (p0 is the tercile
      bucketing variable).
  G3  spec_v2 needs KCONF + 1 = 4 collapsed prints after t_res: three to form
      the median-of-3 confirmation, a fourth to fill on.

G3 is the binding one and it is mine, not the data's. It truncates precisely
the thinnest tail, so "thin" in `liquidity.py` means *thin among legs that
still printed four times after the trigger*. If the gradient is real, dropping
KCONF should extend it -- the newly admitted legs are thinner still, so they
should pay more, not less. If the gradient instead flattens or reverses when
the thinnest legs are let back in, the `liquidity.py` result is an artefact of
where the gate happened to sit.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import os
os.environ["POLARS_FMT_MAX_COLS"] = "20"
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.events.implied import THRESHOLD, classify_contract, parse_threshold
from stg.panel._io import load_markets, scan_trades

OUT = Path("analysis/leadlag_2026_09/out")
N_BOOT, FEE_RATE, CONTRACTS = 10000, 0.07, 100
BUCKETS = [(1, 5), (5, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99)]
CONFIRM_C, FLOOR_C = 2.0, 20.0
NS_H = 3_600_000_000_000


def fee_cents(p):
    p = np.asarray(p, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(p):
    out = np.full(np.shape(p), 2.0)
    out[np.abs(np.asarray(p, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def cboot(v, g, seed=0):
    if len(v) < 5:
        return (float(v.mean()) if len(v) else np.nan, np.nan, np.nan, np.nan)
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


def show(rows):
    with pl.Config(float_precision=2, tbl_width_chars=400, tbl_rows=60):
        print(pl.DataFrame(rows))


def main() -> None:
    d = pl.read_parquet(OUT / "leadlag_legs.parquet")
    tickers = d["target_ticker"].unique().to_list()

    # ---- G0: legs the metadata says traded but the archive never collected --
    print("=== 1. G0 -- how much did the archive never collect? ===")
    mk = load_markets(is_only=True)
    ev_in_panel = d["target_event"].unique().to_list()
    m = (mk.filter(pl.col("event_ticker").is_in(ev_in_panel))
         .select("event_ticker", "ticker", "yes_sub_title", "result",
                 "volume", "close_time")
         .unique(subset=["ticker"]))
    kinds = [classify_contract(t, s) for t, s in zip(m["ticker"], m["yes_sub_title"])]
    strikes = [parse_threshold(t, s) if k == THRESHOLD else None
               for t, s, k in zip(m["ticker"], m["yes_sub_title"], kinds)]
    m = (m.with_columns(pl.Series("kind", kinds),
                        pl.Series("strike", strikes, dtype=pl.Float64))
         .filter((pl.col("kind") == THRESHOLD) & pl.col("strike").is_not_null()
                 & pl.col("result").is_in(["yes", "no"])))
    have = set(tickers)
    m = m.with_columns(pl.col("ticker").is_in(list(have)).alias("in_panel"))
    tot = m.height
    rows = []
    for lab, expr in (("volume == 0", pl.col("volume") == 0),
                      ("volume 1-99", (pl.col("volume") > 0) & (pl.col("volume") < 100)),
                      ("volume 100-999", (pl.col("volume") >= 100) & (pl.col("volume") < 1000)),
                      ("volume >= 1000", pl.col("volume") >= 1000)):
        s = m.filter(expr)
        rows.append(dict(metadata_volume=lab, legs=s.height,
                         share=s.height / tot,
                         in_panel=int(s["in_panel"].sum()),
                         pct_in_panel=100.0 * s["in_panel"].mean() if s.height else np.nan))
    show(rows)
    print(f"  threshold legs listed across the {len(ev_in_panel)} target events: {tot}")
    print(f"  of those, present in the panel: {int(m['in_panel'].sum())} "
          f"({100*m['in_panel'].mean():.1f}%)")

    # ---- rebuild the tape -------------------------------------------------
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

    n_post = np.zeros(N, dtype=int)
    p0 = np.full(N, np.nan)
    vol_pre = np.zeros(N)
    px_post: list = [None] * N
    for i in range(N):
        a = tape.get(tk[i])
        if a is None:
            continue
        ns, px, sz = a
        q = int(np.searchsorted(ns, tres[i], side="right"))
        n_post[i] = len(ns) - q
        if q > 0:
            p0[i] = px[q - 1]
            vol_pre[i] = sz[:q].sum()
        px_post[i] = px[q:]

    # ---- G1/G2/G3 attrition ----------------------------------------------
    print("\n=== 2. the gate ladder (post-G0) ===")
    stages = [("panel rows (G1: >=1 post print)", np.ones(N, bool)),
              ("G2: has a pre-t_res print (p0)", np.isfinite(p0)),
              ("G3 k=1: >=2 post prints", np.isfinite(p0) & (n_post >= 2)),
              ("G3 k=2: >=3 post prints", np.isfinite(p0) & (n_post >= 3)),
              ("G3 k=3: >=4 post prints (SHIPPED)", np.isfinite(p0) & (n_post >= 4))]
    rows = []
    for lab, mm in stages:
        rows.append(dict(gate=lab, legs=int(mm.sum()),
                         events=int(len(np.unique(ev[mm]))),
                         med_vol_pre=float(np.median(vol_pre[mm])),
                         share_of_panel=float(mm.mean())))
    show(rows)

    # ---- run the spec at each KCONF, with the liquidity split -------------
    def run(kconf: int, confirm: bool = True):
        pc = np.full(N, np.nan); pf = np.full(N, np.nan)
        for i in range(N):
            a = px_post[i]
            if a is None:
                continue
            need = kconf + 1 if confirm else 1
            if len(a) < need:
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
        m2 = ok & (side != 0) & (ent >= FLOOR_C)
        if confirm:
            m2 = m2 & (((pc - p0) * np.sign(side)) >= CONFIRM_C)
        gross = np.where(side > 0, 100.0 * win, 100.0 * (1 - win)) - ent
        return m2, gross - fee_cents(ent) - spread_cents(ent) / 2.0, ent

    print("\n=== 3. headline as G3 is relaxed ===")
    rows = []
    variants = [("k=3 median-of-3 (SHIPPED)", 3, True),
                ("k=2 median-of-2", 2, True),
                ("k=1 first print confirms", 1, True),
                ("no confirmation, fill at 1st print", 0, False)]
    for lab, k, cf in variants:
        m2, net, ent = run(k, cf)
        o, lo, hi, p = cboot(net[m2], ev[m2])
        rows.append(dict(variant=lab, n=int(m2.sum()),
                         events=int(len(np.unique(ev[m2]))),
                         med_vol_pre=float(np.median(vol_pre[m2])),
                         net=o, ci_lo=lo, ci_hi=hi, p_le0=p))
    show(rows)

    print("\n=== 4. the thin-vs-liquid gradient at each gate ===")
    print("   liquidity = contracts traded on the leg before t_res\n")
    rows = []
    for lab, k, cf in variants:
        m2, net, ent = run(k, cf)
        kk = m2 & np.isfinite(vol_pre)
        q = np.percentile(vol_pre[kk], [33.3, 66.7])
        cells = {}
        for tl, sel2 in (("thin", kk & (vol_pre <= q[0])),
                         ("mid", kk & (vol_pre > q[0]) & (vol_pre <= q[1])),
                         ("liquid", kk & (vol_pre > q[1]))):
            o, lo, hi, p = cboot(net[sel2], ev[sel2])
            cells[tl] = (o, int(sel2.sum()), p)
        rows.append(dict(variant=lab,
                         n_thin=cells["thin"][1], thin=cells["thin"][0],
                         thin_p=cells["thin"][2],
                         mid=cells["mid"][0],
                         n_liq=cells["liquid"][1], liquid=cells["liquid"][0],
                         liq_p=cells["liquid"][2],
                         spread_c=cells["thin"][0] - cells["liquid"][0]))
    show(rows)

    print("\n=== 5. the legs G3 was excluding -- do they pay? ===")
    print("   admitted at k=1 but not at k=3, i.e. 2-3 post prints only\n")
    m1, net1, _ = run(1, True)
    m3, net3, _ = run(3, True)
    only1 = m1 & ~m3
    for lab, mm, nn in (("k=1 only (the thinnest admitted)", only1, net1),
                        ("k=3 core", m3, net3)):
        o, lo, hi, p = cboot(nn[mm], ev[mm])
        print(f"  {lab:<34} n={int(mm.sum()):>4}  events={len(np.unique(ev[mm])):>3}  "
              f"med_vol_pre={np.median(vol_pre[mm]):>8.0f}  "
              f"net={o:+.2f}c  CI [{lo:+.2f}, {hi:+.2f}]  P(<=0)={p:.3f}")


if __name__ == "__main__":
    main()
