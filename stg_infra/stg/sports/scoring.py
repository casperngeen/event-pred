"""The standard evaluation table for out-of-sample predictions on the per-night sample tables.

Every function takes walk-forward output (one row per node x night, with ``pred`` and the label) and
scores it the same way for every model, so the linear baselines, no-graph models and graph rungs
are comparable cell by cell:

    r2           1 - SSE / sum(y^2): against predicting no move
    r2_quiet     the same on nodes whose own team did not play that night (pure propagation)
    ic           mean over nights of the within-night Spearman correlation of pred and y, and its
                 t over nights (does the model rank which contracts move, and which way)
    calib        slope of y on pred (with intercept): 1 = a 2c prediction is worth 2c
    paired_t     per-night SSE(base) - SSE(model), mean / standard error over nights (> 0: the
                 model beats the base); nights are the independent unit, since rows on a night
                 share the same results
"""
from __future__ import annotations

import numpy as np
import polars as pl

KEY = ["node", "night"]
MIN_IC_ROWS = 4


def r2(y: np.ndarray, p: np.ndarray) -> float:
    s0 = float((y ** 2).sum())
    return 1 - float(((y - p) ** 2).sum()) / s0 if s0 > 0 else np.nan


def night_ic(pred: pl.DataFrame, label: str) -> tuple[float, float, int]:
    """(mean, t, nights) of the within-night rank correlation."""
    ics = (pred.group_by("night")
           .agg(pl.corr("pred", label, method="spearman").alias("ic"), pl.len().alias("n"))
           .filter(pl.col("n") >= MIN_IC_ROWS, pl.col("ic").is_not_nan(), pl.col("ic").is_not_null())["ic"].to_numpy())
    if len(ics) < 3:
        return np.nan, np.nan, len(ics)
    sd = ics.std(ddof=1)
    return float(ics.mean()), float(ics.mean() / (sd / np.sqrt(len(ics)))) if sd > 0 else np.nan, len(ics)


def calib_slope(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.cov(p, y)[0, 1] / p.var(ddof=1)) if len(p) > 2 and p.var() > 0 else np.nan


def paired_t(pred: pl.DataFrame, base: pl.DataFrame, label: str) -> tuple[float, float]:
    """(R^2 gain, t) of pred over base on their common rows, by night."""
    j = pred.select(*KEY, label, "pred").join(base.select(*KEY, pl.col("pred").alias("base")), on=KEY)
    if j.height == 0:
        return np.nan, np.nan
    d = (j.group_by("night").agg((((pl.col(label) - pl.col("base")) ** 2) - (pl.col(label) - pl.col("pred")) ** 2)
                                 .sum().alias("d"))["d"].to_numpy())
    gain = float(d.sum() / (j[label] ** 2).sum()) if (j[label] ** 2).sum() > 0 else np.nan
    if len(d) < 3 or d.std(ddof=1) == 0:
        return gain, np.nan
    return gain, float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def score(pred: pl.DataFrame, label: str, base: pl.DataFrame | None = None) -> dict:
    """The table's row for one model's predictions (base: the comparison model's, e.g. 4b)."""
    if pred.height == 0:
        return {"n": 0}
    y, p = pred[label].to_numpy(), pred["pred"].to_numpy()
    q = pred.filter(~pl.col("played"))
    ic, ic_t, _ = night_ic(pred, label)
    out = {"n": len(y), "r2": r2(y, p), "n_quiet": q.height,
           "r2_quiet": r2(q[label].to_numpy(), q["pred"].to_numpy()) if q.height >= 30 else np.nan,
           "ic": ic, "ic_t": ic_t, "calib": calib_slope(y, p)}
    if base is not None:
        out["gain_vs_base"], out["t_vs_base"] = paired_t(pred, base, label)
    return out
