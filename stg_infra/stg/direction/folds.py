"""Expanding walk-forward folds over trigger-resolution time.

Same discipline as ``stg.models.train``: expanding train window, a
``PURGE_DAYS`` gap before the test slice so a label window cannot straddle the
boundary, and the 2026 wall never approached. The clock here is ``t0`` -- the
*trigger's* resolution time -- because that is the moment a live user of the
model would have to decide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from stg.splits import PURGE_DAYS


@dataclass(frozen=True)
class Fold:
    i: int
    cut: np.datetime64
    train: np.ndarray   # boolean mask over panel rows
    test: np.ndarray


def _cuts(t0: np.ndarray, n_folds: int, start_frac: float) -> list[np.datetime64]:
    d = np.sort(np.unique(t0))
    idx = [min(len(d) - 1, int(len(d) * (start_frac + (1 - start_frac) * i / n_folds)))
           for i in range(n_folds)]
    return [d[i] for i in idx] + [d[-1] + np.timedelta64(1, "D")]


def walk_forward(
    panel: pl.DataFrame,
    *,
    n_folds: int = 6,
    start_frac: float = 0.5,
    purge_days: int = PURGE_DAYS,
    min_train: int = 100,
) -> list[Fold]:
    """Folds with at least ``min_train`` training rows and a non-empty test slice."""
    t0 = panel["t0"].to_numpy().astype("datetime64[ns]")
    purge = np.timedelta64(purge_days, "D")
    cuts = _cuts(t0, n_folds, start_frac)
    out: list[Fold] = []
    for i in range(n_folds):
        tr = t0 < (cuts[i] - purge)
        te = (t0 >= cuts[i]) & (t0 < cuts[i + 1])
        if tr.sum() < min_train or te.sum() == 0:
            continue
        out.append(Fold(i=i, cut=cuts[i], train=tr, test=te))
    return out
