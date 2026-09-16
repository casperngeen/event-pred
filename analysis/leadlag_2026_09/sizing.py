#!/usr/bin/env python
"""Does position sizing help, and does sizing up on cheap contracts help?

    venv/bin/python analysis/leadlag_2026_09/sizing.py
    (needs build_panel.py)

The intuition being tested: "at 20c my max loss per contract is only 20c, so I
can buy more of them."

Per *contract* that is right. On a **fixed budget** it is not. Kalshi is fully
collateralised -- the price you pay IS the entire stake -- so $100 spent at 20c
and $100 spent at 90c both lose $100 when wrong. What differs is the shape:
cheap contracts lose everything often and win big rarely; dear ones lose
everything rarely and win small often. So the comparison has to be made on
return per dollar deployed, not cents per contract.

Rules, all on the frozen spec's positions
-----------------------------------------
* ``equal contracts`` -- one per signal. What every number so far assumes.
* ``equal dollars``   -- ``w ~ 1/price``. This *is* "size up when cheap",
  stated precisely: each position ties up the same capital.
* ``cheap tilt``      -- ``w ~ 1/price^2``. The aggressive version.
* ``dear tilt``       -- ``w ~ price``. The control.
* ``Kelly``           -- ``w ~ f*/price`` with ``f* = (q_hat - p)/(1 - p)``, the
  Kelly fraction for a binary paying 1 at price p, with ``q_hat`` the strategy's
  realised hit rate in that entry-price bucket estimated on **prior years only**.

Kelly runs opposite to the intuition: for the same edge in percentage points it
wants more capital on expensive contracts, because they win more often.

Fractional Kelly is not a separate row -- scaling every weight by a constant
changes leverage, not relative allocation, so return on deployed capital is
identical.

Scored on return per dollar deployed, plus the spread of **per-event** returns,
since one print settles every leg of an event and the event is the risk unit.

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
KELLY_BANDS = [(20, 35), (35, 50), (50, 65), (65, 80), (80, 90), (90, 100)]
CONFIRM_C, FLOOR_C = 2.0, 20.0


def fee_cents(p):
    p = np.asarray(p, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(p):
    out = np.full(np.shape(p), 2.0)
    out[np.abs(np.asarray(p, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def boot_ratio(num, den, groups, seed=0):
    """Clustered bootstrap of sum(num)/sum(den) -- a portfolio return."""
    u, inv = np.unique(groups, return_inverse=True)
    k = len(u)
    sn = np.bincount(inv, weights=num, minlength=k)
    sd_ = np.bincount(inv, weights=den, minlength=k)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    b = sn[pick].sum(1) / np.maximum(sd_[pick].sum(1), 1e-9)
    return (float(num.sum() / den.sum()), float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)), float((b <= 0).mean()))


def main() -> None:
    d = pl.read_parquet(OUT / "leadlag_legs.parquet").drop_nulls("p0")
    z = d["z_surprise"].to_numpy()
    lim = float(np.percentile(np.abs(z), 99))
    sig = d["direction"].to_numpy().astype(float) * np.clip(z, -lim, lim)
    p0, pe = d["p0"].to_numpy(), d["p_entry"].to_numpy()
    win = d["win"].to_numpy().astype(float)
    ev, yr = d["target_event"].to_numpy(), d["yr"].to_numpy()
    tk = d["target_ticker"].to_numpy()
    tres = d["t_res"].dt.epoch("s").to_numpy()
    years = sorted(np.unique(yr))

    side = np.zeros(len(p0))
    for lo, hi in BUCKETS:
        m = (p0 >= lo) & (p0 < hi)
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

    tt = (scan_trades(is_only=True)
          .filter(pl.col("ticker").is_in(d["target_ticker"].unique().to_list()))
          .select("ticker", "yes_price", "created_time")
          .sort("ticker", "created_time").collect())
    g = (tt.with_columns(pl.col("created_time").dt.epoch("s").alias("s"))
         .group_by("ticker", maintain_order=True)
         .agg(pl.col("s"), pl.col("yes_price")))
    tape = {t: (np.asarray(a, dtype=np.int64), np.asarray(b, dtype=float))
            for t, a, b in zip(g["ticker"], g["s"], g["yes_price"])}
    ent = np.full(len(p0), np.nan)
    for i in range(len(p0)):
        arr = tape.get(tk[i])
        if arr is None:
            continue
        s_, px_ = arr
        j = int(np.searchsorted(s_, tres[i], side="right"))
        if j + 1 < len(s_):
            ent[i] = px_[j + 1] if side[i] > 0 else 100.0 - px_[j + 1]

    keep = (side != 0) & np.isfinite(ent) & (ent >= FLOOR_C)
    keep &= ((pe - p0) * np.sign(side)) >= CONFIRM_C
    i = np.where(keep)[0]
    e = ent[i]
    payoff = np.where(side > 0, 100.0 * win, 100.0 * (1 - win))[i]
    net = payoff - e - (fee_cents(e) + spread_cents(e) / 2.0)
    evs, yrs = ev[i], yr[i]
    hit = (payoff > 50).astype(float)
    print(f"positions {len(i)}   events {len(np.unique(evs))}   "
          f"mean entry {e.mean():.1f}c   hit rate {hit.mean():.3f}\n")

    kf = np.zeros(len(i))
    for lo, hi in KELLY_BANDS:
        m = (e >= lo) & (e < hi)
        for Y in years[1:]:
            tr, te = m & (yrs < Y), m & (yrs == Y)
            if tr.sum() < 40 or te.sum() == 0:
                continue
            q = hit[tr].mean()
            p_ = e[te] / 100.0
            kf[te] = np.clip((q - p_) / np.maximum(1 - p_, 1e-6), 0.0, 1.0)

    rules = {
        "equal contracts (current)": np.ones(len(i)),
        "equal dollars   (w~1/p)": 1.0 / e,
        "cheap tilt      (w~1/p^2)": 1.0 / e ** 2,
        "dear tilt       (w~p)": e.copy(),
        "Kelly           (w~f*/p)": kf / e,
    }
    rows = []
    for nm, w in rules.items():
        w = np.asarray(w, dtype=float)
        if w.sum() <= 0:
            continue
        w = w / w.mean()
        num, den = w * net, w * e
        r, lo, hi, p = boot_ratio(num, den, evs)
        u, inv = np.unique(evs, return_inverse=True)
        pn = np.bincount(inv, weights=num)
        pd_ = np.bincount(inv, weights=den)
        ok = pd_ > 1e-9
        er = pn[ok] / pd_[ok]
        rows.append(dict(rule=nm, ret_pct=100 * r, ci_lo=100 * lo, ci_hi=100 * hi,
                         p_le0=p, ev_sd=100 * float(er.std()),
                         ratio=float(er.mean() / er.std()) if er.std() > 0 else np.nan,
                         worst_ev=100 * float(er.min()),
                         cap_under_35c=float(den[e < 35].sum() / den.sum())))
    with pl.Config(tbl_rows=12, float_precision=2, tbl_width_chars=240):
        print(pl.DataFrame(rows))
    print("\nret_pct = total net / total capital deployed.")
    print("ev_sd / ratio / worst_ev are on PER-EVENT returns (the risk unit).")
    print("cap_under_35c = share of capital sitting in cheap contracts.")

    print("\n=== where the edge is, by entry price ===")
    rows = []
    for lo, hi in KELLY_BANDS:
        m = (e >= lo) & (e < hi)
        if m.sum() < 30:
            continue
        rows.append(dict(entry=f"{lo}-{hi}c", n=int(m.sum()),
                         hit=float(hit[m].mean()),
                         breakeven=float(e[m].mean() / 100),
                         edge_pp=100 * float(hit[m].mean() - e[m].mean() / 100),
                         net_c=float(net[m].mean()),
                         ret_pct=100 * float(net[m].sum() / e[m].sum())))
    with pl.Config(tbl_rows=12, float_precision=2, tbl_width_chars=210):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
