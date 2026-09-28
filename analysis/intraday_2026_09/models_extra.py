#!/usr/bin/env python
"""Q4 continued: tree, non-temporal and Bayesian-graph rungs on the same task.

    # needs models.py's out/oof_models{,_mean2}.parquet; add --price mean2 to every call for the bounce check
    for g in trees mlp bayes; do
        venv/bin/python -W ignore analysis/intraday_2026_09/models_extra.py --group $g \
            > analysis/intraday_2026_09/out/models_extra_$g.txt &
    done; wait
    venv/bin/python -W ignore analysis/intraday_2026_09/models_extra.py --group score \
        > analysis/intraday_2026_09/out/models_extra.txt

Each group runs in its own process and writes out/oof_extra_<group>.parquet:
LightGBM and torch each bring an OpenMP runtime, and on macOS one process with
both segfaults (torch imported first) or deadlocks (torch run after LightGBM). ``--group score`` scores them with the original rungs.

``models.py`` compares linear rungs, a GRU and an AGCRN. This adds the rungs a
reviewer would ask for next, on the same bars, labels, folds and scoring:

  6a LightGBM          gradient-boosted trees on the per-bar features (the 22 of
                       ``models.py`` + node id as a categorical + target type);
                       rounds early-stopped on the last 15% of training releases
  6b LightGBM, traded-bar loss
  6c random forest     sklearn, 200 trees, min 200 bars per leaf, 100k bars per tree
  6d MLP per bar       64 → 32 ReLU on the same features + a learned node id, no
                       recurrence and no graph (``models.rung_torch``, 2 seeds)
  6e MLP, traded-bar loss
  7a Bayes graph, hard sign
                       every edge (source → target node) × τ-bucket has its own
                       slope on the theory-signed surprise, β ≥ 0, partially
                       pooled within its channel (source family × target type) ×
                       τ-bucket: ``event_time_2026_09/bayes.py``'s hard prior
                       moved onto the intraday grid; rung 2 is its one-slope-per-
                       channel limit
  7b Bayes STG, soft sign + node path state
                       the soft prior (edge signs free, channel mean μ_g), plus
                       the path-state features (rung 3's) × τ-bucket with one
                       slope per node, pooled across nodes: §9's Bayesian STG,
                       whose node-adaptive own-state block is AGCRN's node-specific
                       part in linear form

The Bayes rungs are fitted by Gibbs sampling on sufficient statistics
(XᵀWX, XᵀWy), so the 400k bars cost nothing per iteration. Each (release,
target) cell has total weight 1 across its bars, so the likelihood counts cells,
not the ~50 overlapping bars of each, and the pooling isn't swamped by
pseudo-replication. 2 chains × (400 burn-in + 800 draws); predictions from the
posterior mean.

The original 8 rungs are read back from ``out/oof_models{suffix}.parquet`` and
scored jointly with these (``models.score``: Δ vs the best linear rung 1–3b,
bootstrap over releases).
"""
from __future__ import annotations

import argparse
import sys
import time

if "trees" in sys.argv:
    import lightgbm as lgb                   # before any torch import: the other order segfaults on macOS

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.linalg import solve_triangular
from scipy.special import ndtr, ndtri
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, "analysis/intraday_2026_09")
import models as M0  # noqa: E402  (builds the panel on import; honours --price)
from models import (BUCKET, FNAMES, IS_K, LABELS, LSD, NB, OUT, PATH_F, ROLE, SEEDS, SUFFIX, TAUS,  # noqa: E402
                    TRADED, TT, XF, XOWN, YM, YS, I, J, M, N, X, folds, inst, nodes, rung_torch, score)

F = len(FNAMES)
BURN, DRAWS, CHAINS = 400, 800, 2
A_T, B_T = 2.0, 0.02          # τ² ~ InvGamma(A_T, B_T), as bayes.py
A_S, B_S = 2.0, 1.0           # σ² ~ InvGamma(A_S, B_S)
MU_SD, ALPHA_SD = 0.3, 0.1
LGB_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=500,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=4, verbose=-1)


# ------------------------------------------------------------------ trees
def tab(ii, jj, nn_):
    """Per-bar feature table: the 22 features, node id, target type."""
    return np.column_stack([X[ii, jj, nn_], nn_, TT[nn_]]).astype(np.float32)


CAT = [F, F + 1]


def rung_lgb(tr, te, fit_m, es_m, traded_only=False):
    out, info = {}, []
    ti, tj, tn = np.nonzero(M & te[:, None, None])
    Xt = tab(ti, tj, tn)
    for lab in LABELS:
        keep = YM[lab] & (TRADED[lab] if traded_only else True)
        fi, fj, fn = np.nonzero(keep & fit_m[:, None, None])
        ei, ej, en = np.nonzero(keep & es_m[:, None, None])
        preds = []
        for s in SEEDS:
            dtr = lgb.Dataset(tab(fi, fj, fn), YS[lab][fi, fj, fn], categorical_feature=CAT, free_raw_data=False)
            dva = lgb.Dataset(tab(ei, ej, en), YS[lab][ei, ej, en], categorical_feature=CAT, reference=dtr)
            b = lgb.train(dict(LGB_PARAMS, seed=s), dtr, num_boost_round=2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            preds.append(b.predict(Xt, num_iteration=b.best_iteration))
            info.append(b.best_iteration)
        p = np.full((I, J, N), np.nan)
        p[ti, tj, tn] = np.mean(preds, 0)
        out[lab] = p
    return out, f"rounds {info}"


def rung_rf(tr, te):
    out = {}
    ti, tj, tn = np.nonzero(M & te[:, None, None])
    Xt = tab(ti, tj, tn)
    for lab in LABELS:
        fi, fj, fn = np.nonzero(YM[lab] & tr[:, None, None])
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=200, max_features=0.33,
                                   max_samples=min(100_000, len(fi)), n_jobs=4, random_state=0)
        rf.fit(tab(fi, fj, fn), YS[lab][fi, fj, fn])
        p = np.full((I, J, N), np.nan)
        p[ti, tj, tn] = rf.predict(Xt)
        out[lab] = p
    return out, ""


# ------------------------------------------------------------------ MLP
class BarMLP(nn.Module):
    """The same input as the GRU at each bar, no state carried between bars."""

    def __init__(self, F, d_id=4):
        super().__init__()
        self.node = nn.Parameter(torch.randn(N, d_id) * 0.1)
        self.net = nn.Sequential(nn.Linear(F + 1 + d_id, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, m):                    # x (B, J, N, F), m (B, J, N)
        mf = m.float()[..., None]
        ids = self.node.expand(*x.shape[:2], -1, -1)
        return self.net(torch.cat([x * mf, mf, ids], -1))


# ------------------------------------------------------------------ Bayes on the graph
SRC = ["own"] + list(XF)                      # source 0: the releasing series' own z (Kalshi own cells)
S_N = len(SRC)
SRCV = np.stack([np.where(ROLE == "own", XOWN, 0.0) * IS_K[:, None]] + list(XF.values()), -1)   # (I, N, S)
# columns: NB bucket intercepts | edges (s, n, b) | node path state (f, n, b)
N_EDGE = S_N * N * NB
N_PATH = len(PATH_F) * N * NB


def edge_group(s, n, b):
    """Channel × bucket: own is one channel; family sources split by target type."""
    ch = 0 if s == 0 else 1 + (s - 1) * 4 + TT[n]
    return ch * NB + b


def path_group(f, b):
    return f * NB + b


def design(ii, jj, nn_, path, psd):
    """Sparse design (rows × k) for bars (ii, jj, nn_)."""
    n = len(ii)
    b = BUCKET[jj]
    rows, cols, vals = [np.arange(n)], [b], [np.ones(n)]
    sv = SRCV[ii, nn_]                                           # (n, S)
    for s in range(S_N):
        nz = sv[:, s] != 0
        rows.append(np.nonzero(nz)[0])
        cols.append(NB + (s * N + nn_[nz]) * NB + b[nz])
        vals.append(sv[nz, s])
    if path:
        pv = X[ii, jj, nn_][:, PATH_F] / psd
        for f in range(len(PATH_F)):
            nz = pv[:, f] != 0
            rows.append(np.nonzero(nz)[0])
            cols.append(NB + N_EDGE + (f * N + nn_[nz]) * NB + b[nz])
            vals.append(pv[nz, f])
    k = NB + N_EDGE + (N_PATH if path else 0)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, k))


def gibbs_ss(XtX, Xty, yty, n_eff, prior, path, seed):
    """One chain on sufficient statistics. prior 'hard': edge β ≥ 0 (theory
    direction; the surprises are already theory-signed), coordinate-wise
    truncated normals. 'soft': β_edge ~ N(μ_g, τ_g²), one joint Gaussian draw."""
    rng = np.random.default_rng(seed)
    k = len(Xty)
    E0, E1 = NB, NB + N_EDGE
    eg = np.array([edge_group(s, n, b) for s in range(S_N) for n in range(N) for b in range(NB)])
    G = eg.max() + 1
    grp = np.concatenate([np.full(NB, -1), eg])
    if path:
        pg = np.array([G + path_group(f, b) for f in range(len(PATH_F)) for n in range(N) for b in range(NB)])
        grp = np.concatenate([grp, pg])
        G = pg.max() + 1
    tau2 = np.full(G, B_T / (A_T - 1))
    mu = np.zeros(G)
    sig2 = 1.0
    beta = np.zeros(k)
    gi = grp[grp >= 0]
    cnt = np.bincount(gi, minlength=G)
    draws = []
    diag = np.diag(XtX).copy()
    for it in range(BURN + DRAWS):
        pvar = np.where(grp >= 0, tau2[np.maximum(grp, 0)], ALPHA_SD ** 2)
        pmean = np.where(grp >= 0, mu[np.maximum(grp, 0)], 0.0)
        if prior == "soft":
            prec = XtX / sig2 + np.diag(1 / pvar)
            L = np.linalg.cholesky(prec)
            z = solve_triangular(L, Xty / sig2 + pmean / pvar, lower=True)
            beta = solve_triangular(L.T, z + rng.standard_normal(k), lower=False)
        else:
            g = XtX @ beta
            for c in range(k):
                r = Xty[c] - g[c] + diag[c] * beta[c]
                v = 1 / (diag[c] / sig2 + 1 / pvar[c])
                mean = v * (r / sig2 + pmean[c] / pvar[c])
                if E0 <= c < E1:                                # θ ≥ 0: inverse-CDF upper tail
                    a = -mean / np.sqrt(v)
                    new = mean + np.sqrt(v) * (-ndtri(rng.uniform() * ndtr(-a)))
                else:
                    new = rng.normal(mean, np.sqrt(v))
                if new != beta[c]:
                    g += XtX[c] * (new - beta[c])          # XtX symmetric: row = column
                    beta[c] = new
        dev = beta[grp >= 0] - mu[gi]
        tau2 = 1 / rng.gamma(A_T + cnt / 2, 1 / (B_T + np.bincount(gi, dev ** 2, minlength=G) / 2))
        if prior == "soft":
            v = 1 / (cnt / tau2 + 1 / MU_SD ** 2)
            mu = rng.normal(v * np.bincount(gi, beta[grp >= 0], minlength=G) / tau2, np.sqrt(v))
        ssr = max(yty - 2 * beta @ Xty + beta @ XtX @ beta, 1e-9)
        sig2 = 1 / rng.gamma(A_S + n_eff / 2, 1 / (B_S + ssr / 2))
        if it >= BURN:
            draws.append(beta.copy())
    return np.array(draws)


def rhat(chains, used):
    """Max split-R̂ over the coefficients the data touches (``used``); the rest
    are prior draws that never enter a prediction."""
    h = [c[i * DRAWS // 2:(i + 1) * DRAWS // 2, used] for c in chains for i in range(2)]
    W = np.mean([x.var(0, ddof=1) for x in h], 0)
    B = np.var([x.mean(0) for x in h], 0, ddof=1) * (DRAWS // 2)
    ok = W > 1e-12
    n2 = DRAWS // 2
    return float(np.sqrt(((n2 - 1) / n2 * W[ok] + B[ok] / n2) / W[ok]).max())


def rung_bayes(tr, te, prior, path):
    out, info = {}, []
    ti, tj, tn = np.nonzero(M & te[:, None, None])
    for lab in LABELS:
        fi, fj, fn = np.nonzero(YM[lab] & tr[:, None, None])
        psd = X[fi, fj, fn][:, PATH_F].std(0) + 1e-9
        D = design(fi, fj, fn, path, psd)
        cell = fi * N + fn
        w = 1.0 / np.bincount(cell)[cell]                       # each (release, target) cell weighs 1
        y = YS[lab][fi, fj, fn]
        Dw = D.multiply(w[:, None]).tocsr()
        XtX = (D.T @ Dw).toarray()
        Xty = Dw.T @ y
        chains = [gibbs_ss(XtX, Xty, float((w * y * y).sum()), float(w.sum()), prior, path, s)
                  for s in range(CHAINS)]
        bm = np.concatenate(chains).mean(0)
        info.append(f"{lab} R̂ {rhat(chains, np.diag(XtX) > 0):.2f}")
        p = np.full((I, J, N), np.nan)
        for s0 in range(0, len(ti), 200_000):
            sl = slice(s0, s0 + 200_000)
            p[ti[sl], tj[sl], tn[sl]] = design(ti[sl], tj[sl], tn[sl], path, psd) @ bm
        out[lab] = p
    return out, ", ".join(info)


# ------------------------------------------------------------------ run
GROUPS = {
    "trees": {
        "6a LightGBM": lambda f: rung_lgb(*f),
        "6b LightGBM, traded-bar loss": lambda f: rung_lgb(*f, traded_only=True),
        "6c random forest": lambda f: rung_rf(f[0], f[1]),
    },
    "mlp": {
        "6d MLP per bar": lambda f: _torch(f, False),
        "6e MLP, traded-bar loss": lambda f: _torch(f, True),
    },
    "bayes": {
        "7a Bayes graph, hard sign": lambda f: rung_bayes(f[0], f[1], "hard", False),
        "7b Bayes STG, soft sign + node path": lambda f: rung_bayes(f[0], f[1], "soft", True),
    },
}


def _torch(f, traded_only):
    tr, te, fit_m, es_m = f
    runs = [rung_torch(lambda: BarMLP(F), tr, te, fit_m, es_m, s, traded_only=traded_only) for s in SEEDS]
    return ({lab: np.mean([r[0][lab] for r in runs], 0) for lab in LABELS},
            f"best/stop epoch {[(r[1]['best_epoch'], r[1]['epochs']) for r in runs]}")


def load_oof(path) -> tuple[dict, list[str]]:
    oof = pl.read_parquet(path)
    names = sorted({c.split("|")[0] for c in oof.columns if "|" in c}, key=lambda s: s.split()[0])
    oof = (oof.join(inst.select("arm", "t_rel", "i"), on=["arm", "t_rel"])
           .with_columns(pl.col("target").replace_strict({n: k for k, n in enumerate(nodes)},
                                                         return_dtype=pl.Int64).alias("n"),
                         pl.col("tau_min").replace_strict({int(t): k for k, t in enumerate(TAUS)},
                                                          return_dtype=pl.Int64).alias("j")))
    ii, jj, nn_ = (oof[c].to_numpy() for c in ("i", "j", "n"))
    pred = {}
    for m in names:
        pred[m] = {}
        for lab in LABELS:
            a = np.full((I, J, N), np.nan)
            a[ii, jj, nn_] = oof[f"{m}|{lab}"].to_numpy()
            pred[m][lab] = a
    return pred, names


def run_group(group: str) -> None:
    rungs = GROUPS[group]
    pred = {m: {lab: np.full((I, J, N), np.nan) for lab in LABELS} for m in rungs}
    for f, tr, te, fit_m, es_m in folds():
        for name, fn in rungs.items():
            t0 = time.time()
            o, msg = fn((tr, te, fit_m, es_m))
            for lab in LABELS:
                pred[name][lab][te] = o[lab][te]
            print(f"fold {f} | {name}: {time.time() - t0:.0f}s {msg}", flush=True)
    done = np.all([np.isfinite(pred[m][lab]) for m in rungs for lab in LABELS], axis=0)
    ii, jj, nn_ = np.nonzero(done & (YM["1h"] | YM["24h"]))
    path = OUT / f"oof_extra_{group}{SUFFIX}.parquet"
    pl.DataFrame({"arm": inst["arm"].to_numpy()[ii], "t_rel": inst["t_rel"].to_numpy()[ii],
                  "target": np.array(nodes)[nn_], "tau_min": TAUS[jj],
                  **{f"{m}|{lab}": pred[m][lab][ii, jj, nn_] for m in rungs for lab in LABELS}}).write_parquet(path)
    print(f"wrote {path}")


def run_score() -> None:
    pred, names = load_oof(OUT / f"oof_models{SUFFIX}.parquet")
    for g in GROUPS:
        p, n = load_oof(OUT / f"oof_extra_{g}{SUFFIX}.parquet")
        pred |= p
        names += n
    rows = score(pred, names, n_linear=4)
    pl.DataFrame(rows).write_parquet(OUT / f"models_extra_scores{SUFFIX}.parquet")
    print(f"\nwrote {OUT / f'models_extra_scores{SUFFIX}.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=(*GROUPS, "score"), required=True)
    g = ap.parse_known_args()[0].group
    run_score() if g == "score" else run_group(g)
