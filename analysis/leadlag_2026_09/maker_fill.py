#!/usr/bin/env python
"""Maker execution: does resting a limit order rescue the lead-lag edge?

    venv/bin/python analysis/leadlag_2026_09/maker_fill.py
    (needs build_panel.py)

``economics.py`` assumed a taker throughout: cross the spread at the first
print after the trigger resolves, pay fee + half the measured spread. That
assumption costs **1.55c of a 2.36c gross edge** -- two thirds of it -- and
leaves net +0.81c with a CI straddling zero.

``research_summary.md`` §6.4 flags this as *"the single most important execution
assumption in Phase 4, and it should be modelled explicitly rather than assumed
either way"*, and argues for resting orders: the signal predicts settlement, not
immediacy, so there is no moment you must be in by. It has never been tested.

The simulation
--------------
The trigger resolves at ``t_res``. You learn the surprise then, and the target
has not necessarily printed yet (``research_log.md`` §13: the repricing lands at
the target's first post-resolution print, median 6.2 min). So you rest a limit
at ``p0`` -- the target's last pre-resolution price, the standing market -- on
the side the signal indicates, and wait.

A resting **buy** at ``L`` fills when someone sells at or below ``L``; a resting
**sell** at ``L`` fills when someone buys at or above ``L``. Both are observable
from the trade tape, so fills are measured rather than assumed.

Adverse selection is the whole question, and it is built into the geometry: if
the signal is right and the price gaps away from you, **you do not get filled**.
If the signal is wrong and the price comes back, you do. So the maker's fill set
is selected against, and the test is whether what survives still pays.

Three things are varied
-----------------------
* **window** -- how long the order rests before it is abandoned (or crossed).
* **unfilled policy** -- give up, or cross at the prevailing price.
* **maker fee** -- Kalshi's taker formula is ``0.07*p(1-p)``; the maker side is
  reported at both the full taker fee (conservative) and zero (optimistic),
  since the exact schedule has varied. No spread is paid either way: not
  crossing is the point.

Bucketing for the trading rule uses ``p0``, not ``p_entry``, so the maker and
taker arms decide on identical information.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel._io import scan_trades
from stg.splits import assert_no_oos

OUT = Path("analysis/leadlag_2026_09/out")
N_BOOT = 10000
FEE_RATE, CONTRACTS = 0.07, 100
BUCKETS = [(1, 5), (5, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99)]


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
    d = pl.read_parquet(OUT / "leadlag_legs.parquet").drop_nulls("p0")
    assert_no_oos(d, time_col="close_time")
    z = d["z_surprise"].to_numpy()
    lim = float(np.percentile(np.abs(z), 99))
    d = d.with_columns(
        pl.Series("sig", d["direction"].to_numpy() * np.clip(z, -lim, lim)))
    print(f"rows with a pre-resolution price: {d.height}   "
          f"target events: {d['target_event'].n_unique()}")

    # ---- trade tape for every target leg
    tickers = d["target_ticker"].unique().to_list()
    tt = (scan_trades(is_only=True)
          .filter(pl.col("ticker").is_in(tickers))
          .select("ticker", "yes_price", "created_time")
          .sort("ticker", "created_time").collect())
    tape: dict = {}
    g = (tt.with_columns(pl.col("created_time").dt.epoch("s").alias("s"))
         .group_by("ticker", maintain_order=True)
         .agg(pl.col("s"), pl.col("yes_price")))
    for tk, s_, px_ in zip(g["ticker"], g["s"], g["yes_price"]):
        tape[tk] = (np.asarray(s_, dtype=np.int64), np.asarray(px_, dtype=float))
    print(f"legs with a tape: {len(tape)}\n")

    # ---- walk-forward tercile rule, bucketed on p0
    p0 = d["p0"].to_numpy()
    sig = d["sig"].to_numpy()
    yr = d["yr"].to_numpy()
    years = sorted(np.unique(yr))
    side = np.zeros(len(p0))
    for lo, hi in BUCKETS:
        m = (p0 >= lo) & (p0 < hi)
        if m.sum() < 60:
            continue
        idx = np.where(m)[0]
        for Y in years[1:]:
            tr = idx[yr[idx] < Y]
            te = idx[yr[idx] == Y]
            if len(tr) < 60 or len(te) == 0:
                continue
            t = np.percentile(sig[tr], [33.3, 66.7])
            side[te] = np.where(sig[te] > t[1], 1.0,
                                np.where(sig[te] <= t[0], -1.0, 0.0))
    d = d.with_columns(pl.Series("side", side))
    pos = d.filter(pl.col("side") != 0)
    print(f"walk-forward positions: {pos.height}   "
          f"events: {pos['target_event'].n_unique()}\n")

    # ---- simulate
    t_res = pos["t_res"].dt.epoch("s").to_numpy()
    tk = pos["target_ticker"].to_numpy()
    L = pos["p0"].to_numpy().astype(float)
    sd = pos["side"].to_numpy()
    p_ent = pos["p_entry"].to_numpy().astype(float)

    WINDOWS = [("1 hour", 3600), ("1 day", 86400), ("to close", 10 ** 9)]
    results = {}
    for label, W in WINDOWS:
        filled = np.zeros(len(L), dtype=bool)
        for i in range(len(L)):
            arr = tape.get(tk[i])
            if arr is None:
                continue
            s_, px_ = arr
            j = int(np.searchsorted(s_, t_res[i], side="right"))
            k = int(np.searchsorted(s_, t_res[i] + W, side="right"))
            if j >= k:
                continue
            win_px = px_[j:k]
            # resting buy fills on a sale at or below the limit; resting sell
            # fills on a purchase at or above it
            filled[i] = bool((win_px <= L[i]).any() if sd[i] > 0
                             else (win_px >= L[i]).any())
        results[label] = filled
        print(f"fill rate, {label:<9}: {filled.mean():.3f}")

    win = pos["win"].to_numpy().astype(float)
    ev = pos["target_event"].to_numpy()

    print("\n=== adverse selection: are the fills the losing side? ===")
    print("gross = payoff - entry, before any fee. Compare filled vs unfilled")
    print("at the SAME limit price, so the difference is selection alone.\n")
    rows = []
    for label, _ in WINDOWS:
        f = results[label]
        entry = np.where(sd > 0, L, 100.0 - L)
        payoff = np.where(sd > 0, 100.0 * win, 100.0 * (1 - win))
        gross = payoff - entry
        if f.sum() < 30 or (~f).sum() < 30:
            continue
        rows.append(dict(window=label, n_filled=int(f.sum()),
                         fill_rate=float(f.mean()),
                         gross_filled=float(gross[f].mean()),
                         gross_unfilled=float(gross[~f].mean()),
                         selection=float(gross[f].mean() - gross[~f].mean())))
    with pl.Config(float_precision=2, tbl_width_chars=210):
        print(pl.DataFrame(rows))

    print("\n=== the comparison: taker vs maker ===")
    print("Taker crosses at p_entry paying fee + half spread. Maker rests at p0,")
    print("pays no spread, and is only in the trade when filled.\n")
    entry_t = np.where(sd > 0, p_ent, 100.0 - p_ent)
    payoff = np.where(sd > 0, 100.0 * win, 100.0 * (1 - win))
    net_t = payoff - entry_t - (fee_cents(entry_t) + spread_cents(entry_t) / 2.0)
    obs, lo, hi, pneg = cboot(net_t, ev)
    rows = [dict(arm="taker @ p_entry", n=len(net_t), fill_rate=1.0,
                 net=obs, ci_lo=lo, ci_hi=hi, p_le0=pneg)]

    entry_m = np.where(sd > 0, L, 100.0 - L)
    for label, _ in WINDOWS:
        f = results[label]
        if f.sum() < 30:
            continue
        for fee_mult, fee_name in ((1.0, "taker fee"), (0.0, "no fee")):
            net_m = (payoff[f] - entry_m[f] - fee_mult * fee_cents(entry_m[f]))
            obs, lo, hi, pneg = cboot(net_m, ev[f])
            rows.append(dict(arm=f"maker {label}, {fee_name}",
                             n=int(f.sum()), fill_rate=float(f.mean()),
                             net=obs, ci_lo=lo, ci_hi=hi, p_le0=pneg))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=220):
        print(pl.DataFrame(rows))

    print("\n=== if unfilled, cross instead of giving up ===")
    print("Rest for the window; if still unfilled, take at p_entry.\n")
    rows = []
    for label, _ in WINDOWS:
        f = results[label]
        if f.sum() < 30:
            continue
        for fee_mult, fee_name in ((1.0, "taker fee"), (0.0, "no fee")):
            net = np.where(
                f,
                payoff - entry_m - fee_mult * fee_cents(entry_m),
                payoff - entry_t - (fee_cents(entry_t) + spread_cents(entry_t) / 2.0))
            obs, lo, hi, pneg = cboot(net, ev)
            rows.append(dict(arm=f"rest {label} then cross, {fee_name}",
                             n=len(net), net=obs, ci_lo=lo, ci_hi=hi, p_le0=pneg))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=220):
        print(pl.DataFrame(rows))

    # ------------------------------------------------------------------
    print("\n=== how do you capture the trades that never filled? ===")
    print("Two routes, and the counterfactual +14.83c is available on neither.")
    print("That number is measured AT p0 -- a price that stops existing the")
    print("moment the news is out. It measures how far the market moved away")
    print("from you, not money sitting on a table.\n")

    print("--- route 1: be faster. Take at p0 before the first post-news print.")
    print("    The upper bound on a latency play: you always get the stale price.")
    entry_p0 = np.where(sd > 0, L, 100.0 - L)
    gross_p0 = payoff - entry_p0
    net_p0 = gross_p0 - (fee_cents(entry_p0) + spread_cents(entry_p0) / 2.0)
    gross_pe = payoff - entry_t
    rows = []
    for nm, gr, nt, en in (("take at p0 (infinitely fast)", gross_p0, net_p0, entry_p0),
                           ("take at p_entry (realistic)", gross_pe, net_t, entry_t)):
        obs, lo, hi, pneg = cboot(nt, ev)
        rows.append(dict(arm=nm, n=len(nt), mean_entry=float(en.mean()),
                         gross=float(gr.mean()), net=obs,
                         ci_lo=lo, ci_hi=hi, p_le0=pneg))
    with pl.Config(float_precision=2, tbl_width_chars=220):
        print(pl.DataFrame(rows))
    print("    The gap between the two rows is what the 6.2-minute repricing")
    print("    window (research_log §13) is actually worth per contract.")

    print("\n--- route 2: pay up. Rest a more aggressive limit.")
    print("    offset k moves the limit k cents toward the market, so the order")
    print("    fills more often and at a worse price. k = 0 is the maker arm;")
    print("    large k converges to the taker.\n")
    W = 86400
    rows = []
    for k in (0, 1, 2, 3, 5, 8):
        lim_k = np.where(sd > 0, L + k, L - k)
        lim_k = np.clip(lim_k, 1.0, 99.0)
        filled = np.zeros(len(L), dtype=bool)
        for i in range(len(L)):
            arr = tape.get(tk[i])
            if arr is None:
                continue
            s_, px_ = arr
            j = int(np.searchsorted(s_, t_res[i], side="right"))
            e_ = int(np.searchsorted(s_, t_res[i] + W, side="right"))
            if j >= e_:
                continue
            wp = px_[j:e_]
            filled[i] = bool((wp <= lim_k[i]).any() if sd[i] > 0
                             else (wp >= lim_k[i]).any())
        if filled.sum() < 30:
            continue
        ent = np.where(sd > 0, lim_k, 100.0 - lim_k)
        net_k = payoff[filled] - ent[filled] - fee_cents(ent[filled])
        obs, lo, hi, pneg = cboot(net_k, ev[filled])
        rows.append(dict(offset_c=k, fill_rate=float(filled.mean()),
                         n=int(filled.sum()), net_no_fee=obs,
                         ci_lo=lo, ci_hi=hi, p_le0=pneg))
    with pl.Config(tbl_rows=10, float_precision=2, tbl_width_chars=220):
        print(pl.DataFrame(rows))
    print("    (zero maker fee assumed throughout, i.e. the optimistic case)")

    pos.with_columns([pl.Series(f"filled_{k.replace(' ', '_')}", v)
                      for k, v in results.items()]).write_parquet(
        OUT / "maker_fill.parquet")
    print(f"\nwrote {OUT / 'maker_fill.parquet'}")
    print("\nCaveat: a fill is recorded whenever ANY trade crossed the limit in")
    print("the window. That ignores queue position, so these fill rates are an")
    print("upper bound and the maker arms are optimistic on that axis.")


if __name__ == "__main__":
    main()
