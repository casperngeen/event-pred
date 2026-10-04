"""Purged walk-forward evaluation for the per-night sample tables (stg.sports.samples).

Folds are over nights, per league-season: the first ``start_frac`` of nights only train, the rest
split into ``n_folds`` consecutive test blocks. A row trains a fold only if its label ended before
the fold's first decision (``label_end < first test t``), so no label overlaps the test period.
Feature scaling (ridge) is fitted on each fold's training rows only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl


def night_folds(nights: list, n_folds: int = 5, start_frac: float = 0.4) -> list[list]:
    """Consecutive test blocks of nights after the first start_frac of them."""
    nights = sorted(nights)
    first = int(len(nights) * start_frac)
    blocks = np.array_split(np.array(nights[first:], dtype=object), n_folds)
    return [list(b) for b in blocks if len(b)]


@dataclass
class FoldFit:
    fold: int
    coef: dict
    n_train: int
    n_test: int


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Ridge on standardised X (scaling from these rows only); returns (beta_std, mu, sd, intercept)."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    ym = y.mean()
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return beta, mu, sd, ym


def inner_split(train: pl.DataFrame, label_end: str, val_frac: float = 0.2) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(fit, val) inside a fold's training rows: val = the latest val_frac of its nights, fit = the
    rows whose labels ended before val starts (purged the same way as the outer folds)."""
    nights = sorted(train["night"].unique().to_list())
    val = train.filter(pl.col("night").is_in(nights[int(len(nights) * (1 - val_frac)):]))
    fit = train.filter(pl.col(label_end) < val["t"].min()) if val.height else train.head(0)
    return fit, val


def choose_lam(train: pl.DataFrame, features: list[str], label: str, label_end: str, grid, val_frac: float = 0.2) -> float:
    """The ridge penalty with the lowest error on the latest val_frac of the training nights, fitted
    on the earlier ones (purged the same way), so the penalty never sees the test fold."""
    fit, val = inner_split(train, label_end, val_frac)
    if fit.height < max(30, 5 * len(features)) or val.height == 0:
        return float(np.median(grid))
    X, y = fit.select(features).to_numpy().astype(float), fit[label].to_numpy().astype(float)
    Xv, yv = val.select(features).to_numpy().astype(float), val[label].to_numpy().astype(float)
    best, best_err = grid[0], np.inf
    for lam in grid:
        beta, mu, sd, b0 = ridge_fit(X, y, lam)
        err = float((((Xv - mu) / sd @ beta + b0 - yv) ** 2).sum())
        if err < best_err:
            best, best_err = lam, err
    return float(best)


def walk_forward(df: pl.DataFrame, features: list[str], label: str, label_end: str,
                 lam: float | list = 1.0, n_folds: int = 5, start_frac: float = 0.4) -> tuple[pl.DataFrame, list[FoldFit]]:
    """Out-of-sample predictions for every test row, and each fold's coefficients (original units).
    df: one league-season (and kind), with night, t, the features, the label and its end. ``lam``: a
    penalty, or a grid chosen per fold on the training rows (:func:`choose_lam`)."""
    d = df.filter(pl.col(label).is_not_null(), *[pl.col(f).is_not_null() for f in features])
    folds = night_folds(d["night"].unique().to_list(), n_folds, start_frac)
    preds, fits = [], []
    for i, test_nights in enumerate(folds):
        test = d.filter(pl.col("night").is_in(test_nights))
        start = test["t"].min()
        train = d.filter(pl.col(label_end) < start)
        if train.height < max(30, 5 * len(features)) or test.height == 0:
            continue
        lam_i = choose_lam(train, features, label, label_end, lam) if isinstance(lam, (list, tuple)) else lam
        X, y = train.select(features).to_numpy().astype(float), train[label].to_numpy().astype(float)
        beta, mu, sd, b0 = ridge_fit(X, y, lam_i)
        coef = beta / sd
        icpt = b0 - float(mu @ coef)
        Xt = test.select(features).to_numpy().astype(float)
        preds.append(test.with_columns(pl.Series("pred", Xt @ coef + icpt), pl.lit(i).alias("fold")))
        fits.append(FoldFit(i, {"intercept": icpt, "lam": lam_i, **dict(zip(features, coef))}, train.height, test.height))
    out = pl.concat(preds) if preds else d.head(0).with_columns(pl.lit(None, pl.Float64).alias("pred"),
                                                               pl.lit(None, pl.Int32).alias("fold"))
    return out, fits


def walk_forward_fn(df: pl.DataFrame, label: str, label_end: str, fit: Callable, min_train: int = 30,
                    n_folds: int = 5, start_frac: float = 0.4) -> pl.DataFrame:
    """The same purged folds as :func:`walk_forward`, for any model. ``fit(train)`` gets the fold's
    training rows (tune inside it with :func:`inner_split`) and returns ``predict(rows) -> array``.
    Rows with a null label are dropped; feature nulls are the model's business."""
    d = df.filter(pl.col(label).is_not_null())
    preds = []
    for i, test_nights in enumerate(night_folds(d["night"].unique().to_list(), n_folds, start_frac)):
        test = d.filter(pl.col("night").is_in(test_nights))
        train = d.filter(pl.col(label_end) < test["t"].min()) if test.height else d.head(0)
        if train.height < min_train or test.height == 0:
            continue
        preds.append(test.with_columns(pl.Series("pred", np.asarray(fit(train)(test), dtype=float)), pl.lit(i).alias("fold")))
    return pl.concat(preds) if preds else d.head(0).with_columns(pl.lit(None, pl.Float64).alias("pred"),
                                                                 pl.lit(None, pl.Int32).alias("fold"))


def oos_scores(pred: pl.DataFrame, label: str) -> dict:
    """R^2 against predicting zero, correlation, and sign agreement on rows that moved."""
    if pred.height == 0:
        return {"n": 0}
    y, p = pred[label].to_numpy(), pred["pred"].to_numpy()
    sse, sst0 = float(((y - p) ** 2).sum()), float((y ** 2).sum())
    moved = y != 0
    return {"n": len(y), "r2_vs_zero": 1 - sse / sst0 if sst0 > 0 else np.nan,
            "corr": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else np.nan,
            "sign_acc": float((np.sign(p[moved]) == np.sign(y[moved])).mean()) if moved.any() else np.nan}
