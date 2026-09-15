#!/usr/bin/env python
"""Does requiring the market to move your way first improve the trade?

    venv/bin/python analysis/leadlag_2026_09/move_filter.py
    (needs maker_fill.py)

``maker_fill.py`` found that positions which never filled at a resting limit
would have been worth +14.83c gross at ``p0``. That number is unattainable --
``p0`` stops existing once the news is out. But decomposing it shows **+7.81c of
that edge is still there at ``p_entry``**, the first post-news print, which *is*
transactable. The repricing takes 7.02c; it does not take all of it.

And the thing that distinguishes those positions is observable before you trade:
they are the ones where the market **already moved in the signal's direction**
(median +4c, 81% moved more than 1c) against a median of 0c for everything else.

So: take the same walk-forward signal, but only trade when the first post-news
print has already moved at least ``k`` cents the way the signal predicted. No
look-ahead -- ``p_entry`` is observable at the decision point.

Two controls, because "trade with the move" has an obvious alternative reading:

* **momentum only** -- trade the direction of the move and ignore the signal
  entirely. If this works as well, the signal is decoration.
* **disagreement** -- the market moved, but the signal points the other way.
  If this pays too, the filter is just selecting large moves.

In-sample only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("analysis/leadlag_2026_09/out")
N_BOOT, FEE_RATE, CONTRACTS = 10000, 0.07, 100


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
    sd = d["side"].to_numpy()
    p0, pe = d["p0"].to_numpy(), d["p_entry"].to_numpy()
    win = d["win"].to_numpy().astype(float)
    ev = d["target_event"].to_numpy()
    f = d["filled_to_close"].to_numpy()

    def net_of(side):
        ent = np.where(side > 0, pe, 100.0 - pe)
        pay = np.where(side > 0, 100.0 * win, 100.0 * (1 - win))
        return pay - ent - (fee_cents(ent) + spread_cents(ent) / 2.0)

    move = (pe - p0) * np.sign(sd)
    print("=== how big is the repricing, p0 -> first post-news print? ===")
    print("signed so positive = moved away from a resting limit\n")
    for nm, m in (("ALL", np.ones(len(move), bool)),
                  ("filled by close", f), ("NEVER filled", ~f)):
        x = move[m]
        print(f"  {nm:<16} n={int(m.sum()):5d}  mean {x.mean():+6.2f}c  "
              f"median {np.median(x):+5.1f}c  p90 {np.percentile(x, 90):+6.1f}c  "
              f"share>1c {np.mean(x > 1):.2f}  share>5c {np.mean(x > 5):.2f}")
    print("\n-> the median leg does not move at all; a thin tail gaps several")
    print("   cents. There is no smooth walk through intermediate prices to")
    print("   catch, which is why a limit 8c away barely raises the fill rate.")

    ent0 = np.where(sd > 0, p0, 100.0 - p0)
    entE = np.where(sd > 0, pe, 100.0 - pe)
    pay = np.where(sd > 0, 100.0 * win, 100.0 * (1 - win))
    print("\n=== what the +14.83c is made of ===")
    print(f"  never-filled set (n={int((~f).sum())}):")
    print(f"    gross at p0 (unattainable)    {(pay - ent0)[~f].mean():+6.2f}c")
    print(f"    gross at p_entry (attainable) {(pay - entE)[~f].mean():+6.2f}c")
    print(f"    taken by the repricing        {((pay - ent0) - (pay - entE))[~f].mean():+6.2f}c")

    n = net_of(sd)
    print("\n=== A. signal, filtered on the market having already moved my way ===\n")
    rows = []
    for k in (-99, 0, 1, 2, 3, 5):
        m = move >= k
        if m.sum() < 40:
            continue
        o, lo, hi, p = cboot(n[m], ev[m])
        rows.append(dict(min_move_c=("none" if k < -50 else k), n=int(m.sum()),
                         events=int(len(np.unique(ev[m]))), net=o,
                         ci_lo=lo, ci_hi=hi, p_le0=p))
    with pl.Config(tbl_rows=10, float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    print("\n=== B. control: momentum only, signal ignored ===\n")
    sm = np.sign(pe - p0)
    nm_ = net_of(sm)
    rows = []
    for k in (0, 2, 3, 5):
        m = (np.abs(pe - p0) >= k) & (sm != 0)
        if m.sum() < 40:
            continue
        o, lo, hi, p = cboot(nm_[m], ev[m])
        rows.append(dict(min_abs_move_c=k, n=int(m.sum()),
                         events=int(len(np.unique(ev[m]))), net=o,
                         ci_lo=lo, ci_hi=hi, p_le0=p))
    with pl.Config(tbl_rows=10, float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    print("\n=== C. control: the market moved, but the signal disagrees ===\n")
    agree = np.sign(pe - p0) == np.sign(sd)
    rows = []
    for nm2, mask in (("signal + move agree, >=3c", (move >= 3)),
                      ("move >=3c, signal disagrees",
                       (np.abs(pe - p0) >= 3) & (~agree) & (sd != 0))):
        if mask.sum() < 40:
            continue
        o, lo, hi, p = cboot(n[mask], ev[mask])
        rows.append(dict(arm=nm2, n=int(mask.sum()),
                         events=int(len(np.unique(ev[mask]))), net=o,
                         ci_lo=lo, ci_hi=hi, p_le0=p))
    with pl.Config(tbl_rows=10, float_precision=2, tbl_width_chars=210):
        print(pl.DataFrame(rows))

    print("\n=== D. is the move just a proxy for surprise magnitude? ===")
    print("If filtering on |signal| does the same job, the move adds nothing.\n")
    sig = np.abs(d["sig"].to_numpy())
    rows = []
    for q in (0.0, 0.33, 0.5, 0.66):
        thr = np.quantile(sig, q)
        m = sig >= thr
        o, lo, hi, p = cboot(n[m], ev[m])
        rows.append(dict(min_abs_signal_q=q, n=int(m.sum()), net=o,
                         ci_lo=lo, ci_hi=hi, p_le0=p))
    with pl.Config(tbl_rows=10, float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    print("\nCaveat: the move threshold is chosen in-sample and five were tried.")
    print("Best cell is P(<=0) = 0.12 with a CI still spanning zero, so this is")
    print("a direction worth a walk-forward test, not an established result.")


if __name__ == "__main__":
    main()
