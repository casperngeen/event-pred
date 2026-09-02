"""Linear complexity ladder — promotes ``agcrn_complexity.py`` T5.

Each rung is a per-fold ridge on flat per-(snapshot, node) features taken from
the *last* step of the input window. The ladder answers: is there any
linear-capacity signal in the neighbour channel for a higher-capacity graph
model to build on? (In the August run: no — every graph rung had negative
incremental R² out of sample.)

Rungs
-----
zero                    predict 0
train_mean              per-node train-fold mean
own_level               AR-ish: [implied_mean]
own_momentum            + [d_implied_mean]
neighbour_all           + LOO mean Δ of all active nodes that snapshot
neighbour_same_release  + mean Δ of same-release group members
neighbour_stage1        + ρ̂-weighted Δ over Stage-1 in-edges (the linear
                          analogue of the frozen-prior AGCRN)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from stg.models.tensors import MODEL_FEATURES
from stg.panel.registry import same_release_groups

_F = {c: i for i, c in enumerate(MODEL_FEATURES)}
RUNGS = ("zero", "train_mean", "own_level", "own_momentum",
         "neighbour_all", "neighbour_same_release", "neighbour_stage1")


def _stage1_in_edges(nodes: list[str],
                     path: str | Path = "artifacts/adjacency_is.parquet",
                     survivors_only: bool = True) -> np.ndarray:
    """(N, N) signed ρ̂; entry [i, j] = influence of i on j (i -> j)."""
    A = np.zeros((len(nodes), len(nodes)))
    p = Path(path)
    if not p.exists():
        return A
    e = pl.read_parquet(p)
    if survivors_only and "survives" in e.columns:
        e = e.filter(pl.col("survives"))
    ni = {n: i for i, n in enumerate(nodes)}
    for r in e.iter_rows(named=True):
        if r["trigger"] in ni and r["target"] in ni:
            A[ni[r["trigger"]], ni[r["target"]]] = r["rho"]
    return A


def _same_release_matrix(nodes: list[str]) -> np.ndarray:
    A = np.zeros((len(nodes), len(nodes)))
    ni = {n: i for i, n in enumerate(nodes)}
    for members in same_release_groups().values():
        idx = [ni[m] for m in members if m in ni]
        for i in idx:
            for j in idx:
                if i != j:
                    A[i, j] = 1.0
    return A


def make_features(win: dict, nodes: list[str], rung: str) -> np.ndarray:
    """(n_samples, N, n_feat) design tensor for a rung.

    ``win`` is the output of ``sequence_windows``: uses ``Xs[:, -1]`` (last step)
    and ``Ms[:, -1]`` (active mask that snapshot).
    """
    Xlast = win["Xs"][:, -1]            # (n, N, F)
    active = win["Ms"][:, -1].astype(float)   # (n, N)
    im = Xlast[..., _F["implied_mean"]]
    dm = Xlast[..., _F["d_implied_mean"]] * active   # Δ only where active

    if rung in ("zero", "train_mean"):
        return np.zeros((*im.shape, 0))
    if rung == "own_level":
        return im[..., None]
    if rung == "own_momentum":
        return np.stack([im, dm], axis=-1)

    # neighbour aggregate
    if rung == "neighbour_all":
        tot = (dm * active).sum(1, keepdims=True)
        cnt = active.sum(1, keepdims=True)
        nb = (tot - dm * active) / np.clip(cnt - active, 1, None)
    elif rung == "neighbour_same_release":
        M = _same_release_matrix(nodes)
        nb = (dm @ M.T) / np.clip((active @ M.T), 1, None)
    elif rung == "neighbour_stage1":
        A = _stage1_in_edges(nodes)          # [i,j] = i->j
        nb = dm @ A                            # sum_i dm_i * A[i,j]
    else:
        raise ValueError(rung)
    return np.stack([im, dm, nb], axis=-1)


def _ridge(X: np.ndarray, y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    A[0, 0] -= lam
    return np.linalg.solve(A, X1.T @ y)


class LinearBaseline:
    """sklearn-ish fit/predict over flattened active (sample, node) rows."""

    def __init__(self, rung: str, nodes: list[str], lam: float = 10.0):
        self.rung = rung
        self.nodes = nodes
        self.lam = lam
        self.coef_: np.ndarray | None = None
        self.node_mean_: np.ndarray | None = None

    def _flat(self, win: dict):
        F = make_features(win, self.nodes, self.rung)       # (n, N, k)
        ym = win["ym"]
        rows_f = F[ym]
        y = win["y"][ym]
        node_ix = np.broadcast_to(np.arange(F.shape[1]), ym.shape)[ym]
        return rows_f, y, node_ix

    def fit(self, win: dict):
        Xr, y, node_ix = self._flat(win)
        self.node_mean_ = np.array([
            y[node_ix == i].mean() if (node_ix == i).any() else 0.0
            for i in range(len(self.nodes))])
        if self.rung not in ("zero", "train_mean") and Xr.shape[1]:
            self.coef_ = _ridge(Xr, y, self.lam)
        return self

    def predict(self, win: dict) -> np.ndarray:
        F = make_features(win, self.nodes, self.rung)
        if self.rung == "zero":
            return np.zeros(F.shape[:2])
        if self.rung == "train_mean":
            return np.broadcast_to(self.node_mean_, F.shape[:2]).copy()
        flat = F.reshape(-1, F.shape[-1])
        X1 = np.column_stack([np.ones(len(flat)), flat])
        return (X1 @ self.coef_).reshape(F.shape[:2])
