"""Walk-forward scoring for the direction ladder.

Everything is pooled out-of-fold: each row is predicted exactly once, by a
model that saw only rows resolving at least ``PURGE_DAYS`` before it. The
headline is **directional accuracy against the pooled base rate**, not R^2 --
the whole point of the pivot is that the label is a sign.

Significance uses the block-permutation null from ``stg.structure.stats``:
labels are shuffled *within trigger series*, which preserves the CPI/CPIYOY
same-print dependence that makes a naive per-row null ~10x too narrow
(``research_log.md`` §3).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import rankdata

from stg.direction.folds import Fold, walk_forward
from stg.direction.learners import Learner, ladder
from stg.direction.structure import FoldStructure, fit_structure

_RNG = np.random.default_rng(0)
EPS = 1e-6
GATES = ("all", "p05", "bh")


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC with tie-averaged ranks — a constant predictor must
    score exactly 0.5, or the base-rate rung looks spuriously informative."""
    n_pos, n_neg = int((y > 0).sum()), int((y < 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y > 0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _balanced(y: np.ndarray, yh: np.ndarray) -> float:
    """Mean of the two class recalls — immune to the label imbalance that makes
    raw accuracy unreadable on the small structure-covered subsets."""
    rec = [float((yh[y == c] == c).mean()) for c in (1, -1) if (y == c).any()]
    return float(np.mean(rec)) if rec else float("nan")


def _block_perm_p(y: np.ndarray, yhat: np.ndarray, blocks: np.ndarray,
                  n_perm: int = 5000, rng=None) -> tuple[float, float, float, float]:
    """One-sided p for accuracy, shuffling labels within each trigger block.

    Returns ``(accuracy, p, null_mean, null_sd)``. The null *mean* is the number
    to compare an accuracy against — not the majority base rate, which is a
    different (degenerate) strategy. A predictor that calls both classes scores
    ~50% against shuffled labels however imbalanced those labels are, so on a
    59%-up subset the majority rule and the sign rule are being measured against
    two different yardsticks."""
    rng = rng or _RNG
    obs = float((yhat == y).mean())
    idx_by_block = [np.where(blocks == b)[0] for b in np.unique(blocks)]
    null = np.empty(n_perm)
    yp = y.copy()
    for k in range(n_perm):
        for idx in idx_by_block:
            yp[idx] = rng.permutation(y[idx])
        null[k] = (yhat == yp).mean()
    return obs, float((null >= obs).mean()), float(null.mean()), float(null.std())


def run_ladder(
    panel: pl.DataFrame,
    *,
    learners: list[Learner] | None = None,
    n_folds: int = 6,
    start_frac: float = 0.5,
    min_edge_n: int = 10,
    q: float = 0.10,
    n_perm: int = 5000,
) -> tuple[pl.DataFrame, pl.DataFrame, list[tuple[Fold, FoldStructure]]]:
    """Returns ``(summary, per_fold, fold_structures)``.

    ``summary`` has one row per rung with pooled out-of-fold metrics;
    ``per_fold`` has one row per (rung, fold).
    """
    learners = learners if learners is not None else ladder()
    folds = walk_forward(panel, n_folds=n_folds, start_frac=start_frac)
    if not folds:
        raise ValueError("no usable walk-forward folds")

    structures = [(f, fit_structure(panel.filter(pl.Series(f.train)),
                                    min_n=min_edge_n, q=q)) for f in folds]

    y = panel["y"].to_numpy().astype(int)
    blocks = panel["trigger"].to_numpy()
    covered = np.zeros(panel.height, bool)
    for f in folds:
        covered |= f.test

    # A row is "covered" by the structure if its pair carries an edge, at a
    # given gate, fitted on the *training* rows of the fold that predicts it.
    # Pooled metrics on the ``bh`` subset are the honest out-of-fold version of
    # the 75.9% sign agreement in artifacts/adjacency_report.md, which
    # conditions on in-sample BH survival and so cannot be read as predictive.
    # The gates form a coverage ladder: strictness bought with sample size.
    cov = {g: gate_coverage(panel, structures, g) for g in GATES}

    rows, fold_rows = [], []
    for lrn in learners:
        p = predict_oof(panel, lrn, structures)
        for f, st in structures:
            yh = np.where(p[f.test] >= 0.5, 1, -1)
            fold_rows.append(dict(
                rung=lrn.name, fold=f.i, cut=str(np.datetime64(f.cut, "D")),
                n_train=int(f.train.sum()), n_test=int(f.test.sum()),
                n_edges=len(st.rho), n_bh=len(st.survives),
                acc=float((yh == y[f.test]).mean()),
                base=float(max((y[f.test] > 0).mean(), (y[f.test] < 0).mean())),
            ))

        for subset in GATES:
            mask = covered & cov[subset]
            if mask.sum() < 20:
                continue
            rows.append(_metrics(lrn.name, subset, y[mask], p[mask],
                                 blocks[mask], n_perm))
    return pl.DataFrame(rows), pl.DataFrame(fold_rows), structures


def predict_oof(panel: pl.DataFrame, learner: Learner,
                structures: list[tuple[Fold, FoldStructure]]) -> np.ndarray:
    """P(up) for every row, each predicted by the fold that holds it out.

    Rows in no fold's test slice stay NaN. Exposed because a tradability ledger
    needs the *predictions* a rung actually made, not just its scores.
    """
    p = np.full(panel.height, np.nan)
    for f, st in structures:
        m = learner.fit(panel.filter(pl.Series(f.train)), st)
        p[f.test] = m.predict_proba(panel.filter(pl.Series(f.test)), st)
    return p


def gate_coverage(panel: pl.DataFrame,
                  structures: list[tuple[Fold, FoldStructure]],
                  gate: str) -> np.ndarray:
    """Boolean mask of rows whose pair carries an edge at ``gate`` in the fold
    that predicts them."""
    pairs = panel["pair"].to_list()
    cov = np.zeros(panel.height, bool)
    for f, st in structures:
        cov[f.test] = st.covered([pairs[i] for i in np.where(f.test)[0]], gate)
    return cov


def _metrics(rung: str, subset: str, yt: np.ndarray, p: np.ndarray,
             blocks: np.ndarray, n_perm: int) -> dict:
    pt = np.clip(p, EPS, 1 - EPS)
    yh = np.where(pt >= 0.5, 1, -1)
    acc, pval, null_mean, null_sd = _block_perm_p(yt, yh, blocks, n_perm)
    up = (yt > 0).astype(float)
    base = float(max(up.mean(), 1 - up.mean()))
    return dict(
        rung=rung, subset=subset, n=int(yt.size), acc=acc, base_rate=base,
        skill=acc - base, bal_acc=_balanced(yt, yh), auc=_auc(yt, pt),
        logloss=float(-(up * np.log(pt) + (1 - up) * np.log(1 - pt)).mean()),
        brier=float(((pt - up) ** 2).mean()),
        perm_p=pval, perm_null=null_mean, perm_sd=null_sd,
    )


def edge_stability(structures: list[tuple[Fold, FoldStructure]]) -> pl.DataFrame:
    """How often each pair's train-only edge survives BH, and its sign churn."""
    rows: dict[str, dict] = {}
    for _f, st in structures:
        for pair, rho in st.rho.items():
            r = rows.setdefault(pair, dict(pair=pair, folds=0, bh_folds=0,
                                           rhos=[]))
            r["folds"] += 1
            r["bh_folds"] += int(pair in st.survives)
            r["rhos"].append(rho)
    out = []
    for r in rows.values():
        rh = np.array(r["rhos"])
        out.append(dict(pair=r["pair"], folds=r["folds"], bh_folds=r["bh_folds"],
                        rho_mean=float(rh.mean()), rho_min=float(rh.min()),
                        rho_max=float(rh.max()),
                        sign_stable=bool(np.all(np.sign(rh) == np.sign(rh[0])))))
    return pl.DataFrame(out).sort("bh_folds", "rho_mean", descending=True)


def coverage_composition(panel: pl.DataFrame, structures: list[tuple[Fold, FoldStructure]],
                         gate: str = "bh") -> pl.DataFrame:
    """Which pairs supply the structure-covered rows, and how they score.

    The covered subsets are small, so the first question about any result on
    them is which edges they actually came from — a result carried by one
    same-release pair is a different claim from one spread over five channels.
    """
    pairs = panel["pair"].to_list()
    y = panel["y"].to_numpy().astype(int)
    rows: list[dict] = []
    for f, st in structures:
        idx = np.where(f.test)[0]
        te_pairs = [pairs[i] for i in idx]
        cov = st.covered(te_pairs, gate)
        z = panel["z_surprise"].to_numpy()[idx]
        sig = st.signal(te_pairs, z)
        for j in np.where(cov)[0]:
            rows.append(dict(pair=te_pairs[j], fold=f.i,
                             hit=int(np.sign(sig[j]) == y[idx[j]]),
                             same_release=bool(panel["same_release"][int(idx[j])])))
    if not rows:
        return pl.DataFrame()
    return (pl.DataFrame(rows).group_by("pair")
            .agg(pl.len().alias("n"), pl.col("hit").mean().alias("sign_acc"),
                 pl.col("fold").n_unique().alias("folds"),
                 pl.col("same_release").first())
            .sort("n", descending=True))
