#!/usr/bin/env python
"""Weight the jumps instead of thresholding them: does B's response scale with A's jump?

    venv/bin/python -W ignore analysis/spillover_2026_09/dose.py > analysis/spillover_2026_09/out/dose.txt

Needs ``jumps.py``. Uses every jump found at the lowest threshold (≥ 5c, 2,198
jumps), so small jumps count but with small weight, rather than being dropped
or counted fully. For each (jump, target) pair, the excess response is the
theory-signed response after the real jump minus the same target's response at
that jump's placebo time. Regressed on the jump's size (mean |ΔA| of A's jumping
contracts, cents):

    excess = c + β·(size − 5)          β > 0: bigger jumps move B further
                                        in the theory direction

with SEs clustered by source jump, per horizon; and the mean excess by size
band. Co-jumps and same-release pairs are excluded.
"""
from __future__ import annotations

import numpy as np
import polars as pl

OUT = "analysis/spillover_2026_09/out"
HORIZONS = ["+1h", "+4h", "+24h", "+72h"]

R = (pl.read_parquet(f"{OUT}/jumps.parquet")
     .filter((pl.col("thresh") == 5.0) & ~pl.col("same_release")))
real = R.filter((pl.col("kind") == "real") & ~pl.col("cojump"))
plac = R.filter(pl.col("kind") == "placebo")
key = ["jump", "B"]


def ols_cluster(x, y, g):
    X = np.column_stack([np.ones(len(x)), x])
    inv = np.linalg.inv(X.T @ X)
    b = inv @ X.T @ y
    e = y - X @ b
    u, ix = np.unique(g, return_inverse=True)
    S = np.zeros((2, 2))
    for k in range(len(u)):
        m = ix == k
        s = X[m].T @ e[m]
        S += np.outer(s, s)
    V = inv @ S @ inv * len(u) / (len(u) - 1)
    return b, np.sqrt(np.diag(V))


print("Dose-response of the spillover to the size of the source jump (all jumps ≥ 5c).")
print("excess = theory-signed response after the jump − the same target's response at the")
print("jump's placebo time, cents. β: cents of excess per extra cent of jump size.\n")
print(f"{'horizon':8} {'pairs':>6} {'jumps':>6} {'c (at 5c)':>10} {'β per cent':>11} {'t(β)':>6} | "
      f"{'mean excess by jump size band (n)':>60}")
bands = [(5, 7), (7, 10), (10, 15), (15, 25), (25, 100)]
for h in HORIZONS:
    d = (real.select(*key, "size", pl.col(h).alias("r"))
         .join(plac.select(*key, pl.col(h).alias("p")), on=key, how="inner")
         .filter(pl.col("r").is_finite() & pl.col("p").is_finite())
         .with_columns((pl.col("r") - pl.col("p")).alias("ex")))
    if d.height < 50:
        continue
    b, se = ols_cluster(d["size"].to_numpy() - 5, d["ex"].to_numpy(), d["jump"].to_numpy())
    cells = []
    for lo, hi in bands:
        x = d.filter((pl.col("size") >= lo) & (pl.col("size") < hi))["ex"]
        cells.append(f"{lo}-{hi}c {x.mean():+.2f} ({x.len()})" if x.len() else f"{lo}-{hi}c –")
    print(f"{h:8} {d.height:>6} {d['jump'].n_unique():>6} {b[0]:>+10.3f} {b[1]:>+11.4f} "
          f"{b[1] / se[1]:>+6.2f} | " + "  ".join(cells))
