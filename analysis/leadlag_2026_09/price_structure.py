#!/usr/bin/env python
"""How the market behaves as a function of the entry price, before any signal.

    venv/bin/python analysis/leadlag_2026_09/price_structure.py

The baseline any lead-lag model has to beat, resolved by initial price. Nothing
here uses the trigger at all -- it is the market's own settlement forecast,
scored against outcomes, cut by where the contract was trading when the trigger
resolved.

Why the cut matters, and why it is kept in raw YES terms
--------------------------------------------------------
A leg at 10c and a leg at 90c are not the same bet seen from two sides. Three
things differ at once:

1. **Calibration.** Whether the market is right at 10c is an empirically
   separate question from whether it is right at 90c; a venue can be unbiased
   in the middle and biased in one wing only, and folding to ``min(p, 100-p)``
   would average exactly that asymmetry away.
2. **Payoff geometry.** Buying YES at 10c risks 10 to win 90. Buying at 90c
   risks 90 to win 10. Equal *probability* edges are wildly unequal in return
   on capital, and Kalshi positions are fully collateralised, so capital is
   locked at notional either way.
3. **Fee incidence.** Kalshi's ``0.07*p(1-p)`` peaks at 50c and falls toward
   both wings, so the hurdle a signal must clear is not flat in price.

So every table below is in YES-price buckets, unfolded.

Clustering
----------
On ``target_event`` throughout. One print settles every leg of an event, and
the same target event is matched by several different triggers, so rows sharing
a target event are the same underlying draw. The leg count is never the sample
size.

In-sample only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("analysis/leadlag_2026_09/out")
N_BOOT = 10000
FEE_RATE, CONTRACTS = 0.07, 100

# Unfolded YES-price buckets. Deliberately finer in the wings, where the
# interesting asymmetries live and where most of the ladder sits.
BUCKETS = [(1, 5), (5, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99)]


def fee_cents(price_cents) -> np.ndarray:
    p = np.asarray(price_cents, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def cluster_boot(vals: np.ndarray, groups: np.ndarray, seed: int = 0):
    """Resample whole clusters. Returns (obs, lo, hi, P(mean<=0))."""
    uniq, inv = np.unique(groups, return_inverse=True)
    buckets = [vals[inv == i] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(buckets))
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(idx, len(idx), replace=True)
        boot[b] = np.concatenate([buckets[i] for i in pick]).mean()
    return float(vals.mean()), float(np.percentile(boot, 2.5)), \
        float(np.percentile(boot, 97.5)), float((boot <= 0).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    d = pl.read_parquet(OUT / "leadlag_legs.parquet")
    p = d["p_entry"].to_numpy() / 100.0
    y = d["win"].to_numpy().astype(float)
    ev = d["target_event"].to_numpy()

    print(f"rows {d.height}   trigger events {d['trigger_event'].n_unique()}   "
          f"target events {d['target_event'].n_unique()}   pairs "
          f"{d.select(['trigger', 'target']).unique().height}")
    print(f"\nmarket, pooled:  Brier {brier(p, y):.4f}   log loss {logloss(p, y):.4f}"
          f"   base rate {y.mean():.4f}")
    base = np.full_like(p, y.mean())
    print(f"base rate only:  Brier {brier(base, y):.4f}   log loss {logloss(base, y):.4f}")
    print("-> the market's own quote is the benchmark; a signal has to beat THIS.")

    # ------------------------------------------------------------ calibration
    print("\n=== 1. is the market calibrated, by entry price? ===")
    print("edge_pp = realised - paid, in percentage points. Negative means the")
    print("YES side was overpriced at that price. Clustered on target_event.\n")
    rows = []
    for lo, hi in BUCKETS:
        m = (d["p_entry"] >= lo).to_numpy() & (d["p_entry"] < hi).to_numpy()
        if m.sum() < 30:
            continue
        e = 100.0 * y[m] - d["p_entry"].to_numpy()[m]
        obs, clo, chi, pneg = cluster_boot(e, ev[m])
        rows.append(dict(band=f"{lo}-{hi}c", n_legs=int(m.sum()),
                         n_events=int(len(np.unique(ev[m]))),
                         paid=float(d["p_entry"].to_numpy()[m].mean()),
                         realised=100.0 * float(y[m].mean()),
                         edge_pp=obs, ci_lo=clo, ci_hi=chi,
                         sig="yes" if (clo > 0 or chi < 0) else ""))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    # ------------------------------------------------------- payoff asymmetry
    print("\n=== 2. the same edge is worth different amounts at different prices ===")
    print("A flat +2pp edge, bought at each price. Fee charged once (settlement")
    print("is not a trade); no exit spread. 'ret_on_cap' is net / capital, and")
    print("capital is the full notional because Kalshi collateralises.\n")
    rows = []
    for px in (5, 10, 25, 50, 75, 90, 95):
        edge = 2.0
        gross = edge                       # pp of edge == cents of expected P&L
        f = float(fee_cents([px])[0])
        net = gross - f
        rows.append(dict(price=px, fee=f, gross=gross, net=net,
                         ret_on_cap=100.0 * net / px,
                         breakeven_pp=f))
    with pl.Config(float_precision=2):
        print(pl.DataFrame(rows))
    print("\n-> the fee hurdle is ~3x heavier at 50c than at 5c, and the SAME")
    print("   net cent is worth 20x more per unit of capital at 5c than at 95c.")
    print("   Any 'where should the model be right' question has to net these.")

    # --------------------------------------------------------- reliability
    print("\n=== 3. reliability, finer grid (the calibration curve) ===")
    edges = [1, 3, 5, 8, 12, 20, 30, 40, 50, 60, 70, 80, 88, 92, 95, 97, 99]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d["p_entry"] >= lo).to_numpy() & (d["p_entry"] < hi).to_numpy()
        if m.sum() < 25:
            continue
        rows.append(dict(band=f"{lo}-{hi}", n=int(m.sum()),
                         n_ev=int(len(np.unique(ev[m]))),
                         paid=float(d["p_entry"].to_numpy()[m].mean()),
                         realised=100.0 * float(y[m].mean()),
                         diff=100.0 * float(y[m].mean()) - float(d["p_entry"].to_numpy()[m].mean())))
    with pl.Config(tbl_rows=30, float_precision=1):
        print(pl.DataFrame(rows))

    # ------------------------------------------------------------ by horizon
    print("\n=== 4. how long is the capital locked? ===")
    print("gap_days = trigger resolution -> target close. This is the hold.\n")
    with pl.Config(tbl_rows=20, float_precision=2):
        print(d.with_columns(
            pl.when(pl.col("gap_days") < 3).then(pl.lit("0-3d"))
              .when(pl.col("gap_days") < 10).then(pl.lit("3-10d"))
              .when(pl.col("gap_days") < 30).then(pl.lit("10-30d"))
              .otherwise(pl.lit("30-60d")).alias("hold")
        ).group_by("hold").agg(
            pl.len().alias("legs"),
            pl.col("target_event").n_unique().alias("events"),
            pl.col("gap_days").median().alias("med_days"),
            pl.col("p_entry").mean().alias("mean_p"),
            (100.0 * pl.col("win").mean()).alias("realised"),
        ).sort("med_days"))


if __name__ == "__main__":
    main()
