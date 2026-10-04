#!/usr/bin/env python
"""The frozen OOS test (strategy_spec.md §9). Run once.

    venv/bin/python analysis/leadlag_2026_09/oos_test.py              # in-sample check only
    venv/bin/python analysis/leadlag_2026_09/oos_test.py --run-oos    # the test

The rule is the original one on the VWAP tape -- ``tie_handling.py``'s
``run("vwap")``, and its functions are imported from there, not copied:

* ``p0`` is the last print at or before ``t_res``; the confirming price is the
  first print after it; the fill is the second.
* ``signal = direction * clip(z_surprise, +-lim)``, ``lim`` the in-sample p99
  of ``|z|``.
* Tercile cutpoints per ``p0`` bucket are fitted on **all in-sample rows** and
  applied to 2026 unchanged. A bucket needs 60 in-sample rows to trade.
* ``move >= 2c`` in the side's direction, entry ``>= 20c``, taker fee plus half
  the moneyness spread, hold to settlement, bootstrap clustered on
  ``target_event``.

Without ``--run-oos`` it only reproduces ``tie_handling.py``'s in-sample VWAP
row through the same cutpoint code, and never opens a 2026 file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stg.panel._io import scan_trades  # noqa: E402
from tie_handling import (BUCKETS, CONFIRM_C, FLOOR_C, cboot, fee_cents,  # noqa: E402
                          spread_cents)

OUT = Path("analysis/leadlag_2026_09/out")
BUILT = Path("data/_OOS_DO_NOT_USE_built_2026")
MIN_BUCKET = 60


def vwap_tape(trades: pl.DataFrame) -> dict:
    """ticker -> (epoch-ns, price) with each instant collapsed to its VWAP."""
    col = (trades.group_by("ticker", "created_time")
           .agg(((pl.col("yes_price") * pl.col("count")).sum()
                 / pl.col("count").sum()).alias("px"))
           .sort("ticker", "created_time")
           .with_columns(pl.col("created_time").dt.epoch("ns").alias("ns"))
           .group_by("ticker", maintain_order=True)
           .agg(pl.col("ns"), pl.col("px")))
    return {t: (np.asarray(a, dtype=np.int64), np.asarray(b, float))
            for t, a, b in zip(col["ticker"], col["ns"], col["px"])}


def prices(d: pl.DataFrame, tape: dict):
    """p0, first print after t_res (confirmation), second print (fill)."""
    tk = d["target_ticker"].to_numpy()
    tres = d["t_res"].dt.epoch("ns").to_numpy()
    n = len(tk)
    p0, pf, p2 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        arr = tape.get(tk[i])
        if arr is None:
            continue
        ns, px = arr
        q = int(np.searchsorted(ns, tres[i], side="right"))
        if q > 0:
            p0[i] = px[q - 1]
        if q < len(px):
            pf[i] = px[q]
        if q + 1 < len(px):
            p2[i] = px[q + 1]
    return p0, pf, p2, np.isfinite(p0) & np.isfinite(pf) & np.isfinite(p2)


def cutpoints(sig, bv) -> dict:
    cuts = {}
    for lo, hi in BUCKETS:
        m = (bv >= lo) & (bv < hi)
        if m.sum() >= MIN_BUCKET:
            cuts[(lo, hi)] = np.percentile(sig[m], [33.3, 66.7])
    return cuts


def apply(sig, bv, cuts) -> np.ndarray:
    side = np.zeros(len(sig))
    for (lo, hi), t in cuts.items():
        m = (bv >= lo) & (bv < hi)
        side[m] = np.where(sig[m] > t[1], 1.0, np.where(sig[m] <= t[0], -1.0, 0.0))
    return side


def score(side, ok, p0, pf, p2, win, ev) -> dict:
    ent = np.where(side > 0, p2, 100.0 - p2)
    m = ok & (side != 0) & (ent >= FLOOR_C) & (((pf - p0) * np.sign(side)) >= CONFIRM_C)
    pay = np.where(side > 0, 100.0 * win, 100.0 * (1 - win))
    gross = pay - ent
    cost = fee_cents(ent) + spread_cents(ent) / 2.0
    o, lo, hi, p = cboot((gross - cost)[m], ev[m])
    return dict(n=int(m.sum()), target_events=int(len(np.unique(ev[m]))),
                gross=float(gross[m].mean()), friction=float(cost[m].mean()),
                net=o, ci_lo=lo, ci_hi=hi, p_le0=p)


def load(path: Path, tape: dict, lim=None):
    d = pl.read_parquet(path).drop_nulls("p0")
    z = d["z_surprise"].to_numpy()
    lim = float(np.percentile(np.abs(z), 99)) if lim is None else lim
    sig = d["direction"].to_numpy().astype(float) * np.clip(z, -lim, lim)
    p0, pf, p2, ok = prices(d, tape)
    return d, sig, lim, p0, pf, p2, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-oos", action="store_true")
    a = ap.parse_args()

    # --- in-sample: frozen limit and cutpoints ----------------------------
    is_path = OUT / "leadlag_legs.parquet"
    tk = pl.read_parquet(is_path)["target_ticker"].unique().to_list()
    raw = (scan_trades(is_only=True).filter(pl.col("ticker").is_in(tk))
           .select("ticker", "trade_id", "count", "yes_price", "created_time").collect())
    ded = raw.unique(subset=["trade_id", "ticker", "created_time", "yes_price", "count"])
    d, sig, lim, p0, pf, p2, ok = load(is_path, vwap_tape(ded))
    bv = np.where(ok, p0, -1.0)
    cuts = cutpoints(sig, bv)

    # Check: walk-forward through the same two functions must reproduce
    # tie_handling.py's VWAP row (910 positions, +4.39c, [+0.02, +8.54]).
    yr = d["yr"].to_numpy()
    years = sorted(np.unique(yr))
    wf = np.zeros(len(sig))
    for Y in years[1:]:
        fit, te = yr < Y, yr == Y
        side_y = apply(sig, np.where(te, bv, -1.0), cutpoints(sig[fit], bv[fit]))
        wf[te] = side_y[te]
    ref = score(wf, ok, p0, pf, p2, d["win"].to_numpy().astype(float),
                d["target_event"].to_numpy())
    print("in-sample check (walk-forward, must match tie_handling.py vwap row):")
    print("  " + "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in ref.items()))
    print(f"\nfrozen: winsorisation |z| <= {lim:.3f}; cutpoints fitted on "
          f"{len(sig)} in-sample rows")
    for (lo, hi), t in cuts.items():
        print(f"  p0 in [{lo:>2}, {hi:>2})  sell <= {t[0]:+.3f}   buy > {t[1]:+.3f}")

    if not a.run_oos:
        print("\n2026 not touched. Re-run with --run-oos for the test.")
        return

    # --- the test ----------------------------------------------------------
    otape = vwap_tape(pl.read_parquet(BUILT / "tape_vwap.parquet"))
    od, osig, _, op0, opf, op2, ook = load(BUILT / "leadlag_legs.parquet", otape, lim)
    side = apply(osig, np.where(ook, op0, -1.0), cuts)
    r = score(side, ook, op0, opf, op2, od["win"].to_numpy().astype(float),
              od["target_event"].to_numpy())
    print("\n=== FROZEN OOS TEST (strategy_spec.md §9) ===")
    print(f"  positions      {r['n']}   target events {r['target_events']}")
    print(f"  gross          {r['gross']:+.2f}c")
    print(f"  friction       {r['friction']:.2f}c")
    print(f"  NET            {r['net']:+.2f}c   95% CI [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
          f"   P(<=0) = {r['p_le0']:.3f}")


if __name__ == "__main__":
    main()
