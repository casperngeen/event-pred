from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def _align(y_true, y_pred):
    t = pd.Series(y_true).rename("y_true")
    p = pd.Series(y_pred).rename("y_pred")
    df = pd.concat([t, p], axis=1).dropna()
    return df["y_true"], df["y_pred"]


def rmse(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def mape(y_true, y_pred, eps: float = 1e-12) -> float:
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs((yt - yp) / denom)))


def information_coefficient(y_true, y_pred, method: str = "spearman") -> float:
    yt, yp = _align(y_true, y_pred)
    if len(yt) < 3:
        return float("nan")

    if method.lower() == "pearson":
        c, _ = pearsonr(yp.values, yt.values)
    else:
        c, _ = spearmanr(yp.values, yt.values)
    return float(c)


def predictive_metrics_summary(y_true, y_pred, ic_method: str = "spearman") -> dict:
    return {
        "ic": information_coefficient(y_true, y_pred, method=ic_method),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "n_obs": int(pd.concat([pd.Series(y_true), pd.Series(y_pred)], axis=1).dropna().shape[0]),
    }
    