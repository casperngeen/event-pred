from __future__ import annotations

import math
import numpy as np


def half_life(spread: np.ndarray, *, min_len: int = 15) -> float:
    """AR(1) half-life estimate using Δs_t = a + b s_{t-1} + e and phi=1+b."""
    s = spread[np.isfinite(spread)]
    if len(s) < min_len:
        return np.nan

    ds = np.diff(s)
    s_lag = s[:-1]
    X = np.column_stack([np.ones(len(s_lag)), s_lag])

    try:
        coef = np.linalg.lstsq(X, ds, rcond=None)[0]
        b = float(coef[1])
    except Exception:
        return np.nan

    phi = 1.0 + b
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return np.nan
    return -math.log(2.0) / math.log(phi)


def rolling_beta(y: np.ndarray, x: np.ndarray, w: int, *, min_obs: int = 8) -> float:
    """OLS beta = cov(x,y)/var(x) on the last w observations."""
    yw = y[-w:]
    xw = x[-w:]
    m = np.isfinite(yw) & np.isfinite(xw)
    yw = yw[m]
    xw = xw[m]
    if len(yw) < max(min_obs, w // 2):
        return np.nan

    vx = np.var(xw, ddof=1)
    if not np.isfinite(vx) or vx <= 0:
        return np.nan

    cov = np.cov(xw, yw, ddof=1)[0, 1]
    return float(cov / vx)


def rolling_z(spread: np.ndarray, w: int) -> float:
    """Rolling z-score on last w points, returning z at the most recent time."""
    seg = spread[-w:]
    if not np.isfinite(seg).all():
        return np.nan
    mu = float(seg.mean())
    sd = float(seg.std(ddof=1))
    return (float(spread[-1]) - mu) / sd if sd > 0 else np.nan


def pairwise_corr(a: np.ndarray, b: np.ndarray, *, min_overlap: int = 10) -> float:
    """Correlation of 1st differences using only overlapping finite points."""
    da = np.diff(a)
    db = np.diff(b)
    m = np.isfinite(da) & np.isfinite(db)
    da = da[m]
    db = db[m]
    if len(da) < min_overlap:
        return np.nan
    sa = da.std(ddof=1)
    sb = db.std(ddof=1)
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(da, db)[0, 1])