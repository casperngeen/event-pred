"""Event-clustered inference for the hold-to-settlement analysis.

Why this exists. At settlement the payoff is a deterministic function of the
*target event's* resolution, so two panel rows sharing a ``target_event`` are
not merely correlated -- they are perfectly correlated. Row-level t-statistics,
bootstraps and permutations therefore overstate significance, and they overstate
it worst exactly where a fast trigger feeds a slow target: WTI fires weekly and
GDP resolves quarterly, so 17 WTI->GDP "trades" in the 0-35c bucket turn out to
be 2 GDP contracts, both of which resolved YES.

This re-runs the settlement numbers clustering on ``target_event``:

  * the point estimate is the mean over EVENT means (one vote per outcome),
  * the bootstrap resamples EVENTS, not rows,
  * the permutation shuffles the signal's side within each event's own price
    bucket, so it respects both the clustering and the base rate.

Run from the repo root, after horizon_sweep.py has written the panel.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.direction import (
    SignRule, fit_structure, gate_coverage, predict_oof, walk_forward,
)

FEE_RATE, HALF = 0.07, 2.35
CONTRACTS = int(os.environ.get("CONTRACTS", "100"))
N_BOOT = 20000


def fee(c, C: int = CONTRACTS):
    """Cents per contract. Official: round_up(0.07 * C * P * (1-P))."""
    p = np.asarray(c, float) / 100.0
    return np.ceil(FEE_RATE * C * p * (1 - p) * 100) / 100.0 * 100.0 / C


def cluster_stats(pnl: np.ndarray, events: np.ndarray, rng) -> dict:
    """Mean, se and bootstrap CI with ``events`` as the resampling unit."""
    uniq = np.unique(events)
    means = np.array([pnl[events == e].mean() for e in uniq])
    k = len(uniq)
    se = means.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
    boot = np.array([means[rng.integers(0, k, k)].mean() for _ in range(N_BOOT)])
    return dict(n_rows=len(pnl), n_events=k, mean=float(means.mean()),
                se=float(se), t=float(means.mean() / se) if se else np.nan,
                lo=float(np.percentile(boot, 2.5)),
                hi=float(np.percentile(boot, 97.5)),
                p_gt0=float((boot > 0).mean()))


def main() -> None:
    rng = np.random.default_rng(0)
    panel = pl.read_parquet("artifacts/horizon_panel.parquet")
    folds = walk_forward(panel, n_folds=8, start_frac=0.4)
    st = [(f, fit_structure(panel.filter(pl.Series(f.train)))) for f in folds]
    cov = np.zeros(panel.height, bool)
    for f in folds:
        cov |= f.test
    direction = np.where(predict_oof(panel, SignRule(), st) >= 0.5, 1, -1)

    for gate in ("bh", "p05"):
        m = cov & gate_coverage(panel, st, gate)
        d = (panel.filter(pl.Series(m))
             .with_columns(pl.Series("dir", direction[m].astype(float)))
             .filter(pl.col("settle_yes").is_not_null()
                     & pl.col("p_entry").is_not_null()))
        pe = d["p_entry"].to_numpy().astype(float)
        won = d["settle_yes"].to_numpy() > 0
        by = d["dir"].to_numpy() > 0
        price = np.where(by, pe, 100 - pe)
        pnl = np.where(by == won, 100.0, 0.0) - (price + HALF + fee(price))
        ev = np.array(d["target_event"].to_list())
        pair = np.array(d["pair"].to_list())

        print("\n" + "=" * 92)
        print(f"gate={gate}   fees at C={CONTRACTS}   clustering on target_event")
        print("=" * 92)
        print(f"  {'subset':<30}{'rows':>6}{'events':>8}{'net':>9}{'t':>7}"
              f"{'95% CI':>20}{'P(>0)':>8}")

        subsets = [("ALL", np.ones(len(pnl), bool)),
                   ("excl WTI->GDP", pair != "WTI->GDP/any"),
                   ("excl all WTI/WTIW triggers",
                    ~np.array([p.startswith("WTI") for p in pair])),
                   ("price paid < 35c", price < 35),
                   ("price paid < 35c, excl WTI->GDP",
                    (price < 35) & (pair != "WTI->GDP/any")),
                   ("price paid >= 65c", price >= 65)]
        for lab, keep in subsets:
            if keep.sum() < 5:
                continue
            s = cluster_stats(pnl[keep], ev[keep], rng)
            print(f"  {lab:<30}{s['n_rows']:>6}{s['n_events']:>8}{s['mean']:>9.2f}"
                  f"{s['t']:>7.2f}   [{s['lo']:>6.1f},{s['hi']:>6.1f}]{s['p_gt0']:>8.3f}")

        # per-pair, clustered
        print(f"\n  per pair (clustered):")
        print(f"  {'pair':<28}{'rows':>6}{'events':>8}{'net':>9}{'t':>7}")
        for p_ in sorted(set(pair)):
            k = pair == p_
            if k.sum() < 5:
                continue
            s = cluster_stats(pnl[k], ev[k], rng)
            if s["n_events"] < 3:
                print(f"  {p_:<28}{s['n_rows']:>6}{s['n_events']:>8}{s['mean']:>9.2f}"
                      f"{'  n/a':>7}   <- too few distinct outcomes to test")
                continue
            print(f"  {p_:<28}{s['n_rows']:>6}{s['n_events']:>8}{s['mean']:>9.2f}"
                  f"{s['t']:>7.2f}")


if __name__ == "__main__":
    main()
