"""Shared loading and regression helpers for the intraday scripts. Not a script.

Reads ``out/cells.parquet`` and ``out/paths.parquet`` (``build_paths.py``) and
holds the channel definitions, the (cells × τ) response matrix, and a batched
weighted least-squares fit reused for the bootstrap over release instants and the
sign-flip null.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("analysis/intraday_2026_09/out")
KEY = ["arm", "t_rel", "target"]
FAMS = {"kalshi": ["inflation", "labour", "growth", "policy"],
        "calendar": ["inflation", "activity", "labour", "sentiment"]}
TTYPES = ["inflation", "labour", "growth", "policy"]
MIN_NZ = 20                      # a channel needs this many cells with a non-zero signal
N_BOOT, N_FLIP = 1000, 1000

cells = pl.read_parquet(OUT / "cells.parquet")
paths = pl.read_parquet(OUT / "paths.parquet")
TAU_ALL = np.sort(paths["tau_min"].unique().to_numpy())
TAU_POST = TAU_ALL[TAU_ALL > 0]


def base_arm(arm: str) -> str:
    return arm.removesuffix("_placebo")


def tau_label(t: int) -> str:
    if t < 60 or t % 60:
        return f"{t}m"
    if t < 1440 or t % 1440:
        return f"{t // 60}h"
    return f"{t // 1440}d"


def response_matrix(c: pl.DataFrame, taus, col: str = "r", trunc: str = "trunc_k") -> np.ndarray:
    """(len(c), len(taus)) response in cents, NaN where truncated or not yet priced;
    rows aligned with ``c``."""
    taus = list(taus)
    sub = (paths.filter(pl.col("tau_min").is_in(taus))
           .join(c.select(KEY), on=KEY, how="semi")
           .select(*KEY, "tau_min", pl.when(~pl.col(trunc)).then(pl.col(col)).alias("v")))
    wide = sub.pivot(on="tau_min", index=KEY, values="v")
    wide = c.select(KEY).join(wide, on=KEY, how="left", maintain_order="left")
    return wide.select([str(t) for t in taus]).to_numpy().astype(float)


def instant_index(c: pl.DataFrame) -> np.ndarray:
    return np.unique(c["t_rel"].to_numpy(), return_inverse=True)[1]


def wls(X: np.ndarray, Y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """β (T, k) of each column of Y (n, T; NaN = missing) on X (n, k), weights w (n,)."""
    M = np.isfinite(Y)
    WM = M * (1.0 if w is None else w[:, None])
    G = np.einsum("nk,nt,nl->tkl", X, WM, X, optimize=True) + 1e-9 * np.eye(X.shape[1])
    b = np.einsum("nk,nt->tk", X, WM * np.where(M, Y, 0.0), optimize=True)
    beta = np.linalg.solve(G, b[..., None])[..., 0]
    beta[M.sum(0) < X.shape[1] + 2] = np.nan          # a bar with (almost) no untruncated cells
    return beta


def fit(X, Y, inst, rng, n_boot=N_BOOT, n_flip=N_FLIP):
    """Point estimate, bootstrap over instants, and sign-flip null.

    X's first column is the intercept; the signal columns are flipped per
    instant for the null. Returns (β (T,k), boot (B,T,k), null (F,T,k)).
    """
    n_inst = inst.max() + 1
    est = wls(X, Y)
    boot = np.stack([wls(X, Y, np.bincount(rng.integers(0, n_inst, n_inst), minlength=n_inst)[inst]
                         .astype(float)) for _ in range(n_boot)]) if n_boot else None
    null = None
    if n_flip:
        null = []
        for _ in range(n_flip):
            f = rng.choice([-1.0, 1.0], n_inst)[inst]
            Xf = X.copy()
            Xf[:, 1:] *= f[:, None]
            null.append(wls(Xf, Y))
        null = np.stack(null)
    return est, boot, null


def signal_columns(c: pl.DataFrame, arm: str) -> list[str]:
    """The family signal columns with at least MIN_NZ non-zero cells in ``c``."""
    return [f"x_{f}" for f in FAMS[base_arm(arm)] if (c[f"x_{f}"] != 0).sum() >= MIN_NZ]
