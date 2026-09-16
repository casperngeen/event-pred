#!/usr/bin/env python
"""Does an early exit beat holding to settlement?

    venv/bin/python analysis/leadlag_2026_09/exit_rules.py
    (needs build_panel.py)

The frozen spec holds to settlement and pays for one crossing. Adding an exit
costs a second crossing -- ~1.83c at the 71c mean entry, which is 34% of the
+5.41c net -- so an exit rule has to beat holding by more than that before it is
worth anything.

The test is also diagnostic rather than merely economic. If a take-profit beats
holding, the signal is predicting a **repricing** and the position should be
closed once it has happened. If holding wins, the signal is predicting the
**settlement** and any exit is leaving money on the table. The spec assumes the
second; this checks it.

Mechanics
---------
Position value is tracked in its own units: ``v = price`` for a long YES,
``v = 100 - price`` for a short. Entry cost ``e`` is that value at the second
post-news print. Then walk the trade tape forward:

* take profit when ``v - e >= TP``
* stop out when  ``v - e <= -SL``
* otherwise hold to settlement, where ``v`` becomes the 0/100 payoff

An early exit pays ``fee(v) + spread(v)/2`` again; holding pays nothing more.
Whichever trigger comes first wins; both are evaluated on the same tape, so a
position cannot take profit and stop out in the same run.

Filters are the frozen ones: walk-forward terciles, confirmation >= 2c, entry
price >= 20c.

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

    # entry = second post-resolution print; remember where we are on the tape
    ent = np.full(len(p0), np.nan)
    jent = np.full(len(p0), -1, dtype=int)
    for i in range(len(p0)):
        arr = tape.get(tk[i])
        if arr is None:
            continue
        s_, px_ = arr
        j = int(np.searchsorted(s_, tres[i], side="right"))
        if j + 1 < len(s_):
            jent[i] = j + 1
            ent[i] = px_[j + 1] if side[i] > 0 else 100.0 - px_[j + 1]

    keep = (side != 0) & np.isfinite(ent) & (ent >= FLOOR_C)
    keep &= ((pe - p0) * np.sign(side)) >= CONFIRM_C
    idx = np.where(keep)[0]
    print(f"positions under the frozen filters: {len(idx)}   "
          f"events: {len(np.unique(ev[idx]))}\n")

    payoff = np.where(side > 0, 100.0 * win, 100.0 * (1 - win))
    entry_cost = fee_cents(ent) + spread_cents(ent) / 2.0

    def run(tp, sl):
        """Returns net per position and the share exited early."""
        net = np.empty(len(idx))
        early = np.zeros(len(idx), dtype=bool)
        for n, i in enumerate(idx):
            s_, px_ = tape[tk[i]]
            v = px_[jent[i] + 1:] if side[i] > 0 else 100.0 - px_[jent[i] + 1:]
            exit_v = None
            if len(v):
                hit = np.full(len(v), False)
                if tp is not None:
                    hit |= v >= ent[i] + tp
                if sl is not None:
                    hit |= v <= ent[i] - sl
                w = np.where(hit)[0]
                if len(w):
                    exit_v = float(v[w[0]])
            if exit_v is None:
                net[n] = payoff[i] - ent[i] - entry_cost[i]
            else:
                net[n] = (exit_v - ent[i] - entry_cost[i]
                          - (fee_cents(exit_v) + spread_cents(exit_v) / 2.0))
                early[n] = True
        return net, early

    rows = []
    for tp in (None, 5, 10, 15, 20, 30):
        for sl in (None, 20, 30):
            net, early = run(tp, sl)
            o, lo, hi, p = cboot(net, ev[idx])
            rows.append(dict(take_profit=("hold" if tp is None else tp),
                             stop_loss=("none" if sl is None else sl),
                             exited_early=float(early.mean()),
                             net=o, ci_lo=lo, ci_hi=hi, p_le0=p))
    t = pl.DataFrame(rows)
    print("net per contract; 'hold' = no take-profit. Entry and any exit both")
    print("pay fee + half spread; settlement pays nothing.\n")
    with pl.Config(tbl_rows=30, float_precision=2, tbl_width_chars=210):
        print(t.sort("net", descending=True))

    base = t.filter((pl.col("take_profit") == "hold") & (pl.col("stop_loss") == "none"))
    print(f"\nbaseline (hold to settlement, no stop): "
          f"{base['net'][0]:+.2f}c")
    best = t.sort("net", descending=True).row(0, named=True)
    print(f"best grid cell: TP={best['take_profit']} SL={best['stop_loss']}  "
          f"{best['net']:+.2f}c  (exited early {best['exited_early']:.0%})")
    print("\nA cell only matters if it beats the baseline by more than the ~1.8c")
    print("second crossing it pays -- and the grid was searched in-sample.")


if __name__ == "__main__":
    main()
