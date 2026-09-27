#!/usr/bin/env python
"""Bayesian hierarchical regression on the economic graph: partial pooling of
edges within a channel, per-edge credible intervals.

    venv/bin/python -W ignore analysis/event_time_2026_09/bayes.py \
        > analysis/event_time_2026_09/out/bayes.txt           # needs build_panel.py first

Data: the event-time panel (``_panel.py``), each labelled cell (release
instant t, target b) in per-series z units, regressed on the surprises of the
series that released at t through the 142 candidate edges (cross-release,
theory sign s_e = HAWKISH[a]·HAWKISH[b]):

    y_tb = α + Σ_a β_ab · z_ta + ε,   ε ~ N(0, σ²)

Two priors, both pooling edges of the same channel g (type → type):

  hard sign   β_e = s_e·θ_e,  θ_e ~ HalfNormal(τ_g)      (every edge in the theory
              direction; the channel shares the scale, hence the mean)
  soft sign   β_e ~ Normal(s_e·μ_g, τ_g²),  μ_g ~ N(0, 0.3²)
              (channel mean and each edge's sign are free: the data can
              overturn theory; this is the one that says which edges are supported)

  τ_g² ~ InvGamma(2, 0.02) (prior scale ≈ 0.14 z-units per unit surprise),
  σ² ~ InvGamma(2, 1), α ~ N(0, 0.1²).

Fitted by Gibbs sampling, written out here: every conditional is standard.
β is drawn jointly (Gaussian) under the soft prior and one edge at a time
(truncated normal) under the hard prior; τ_g², σ² are inverse-gamma, μ_g and α
normal. 2 chains × (1,000 burn-in + 2,000 draws); split-R̂ reported.

Out of sample: the walk-forward of ``models.py`` (8 folds, purged on label end),
predictions from the posterior-mean coefficients, scored against the free
linear graph (ridge), the one-slope imposed-sign rung and the zero-parameter
rule, on all labelled test cells and on the cells where the BH-channel signal
fires. In sample: the full-sample fit's channel means and per-edge support.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl
from scipy.special import ndtr, ndtri
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "stg_infra")
from stg.models.baselines import _ridge
from stg.models.train import PURGE, _fold_cuts, _label_end

sys.path.insert(0, "analysis/event_time_2026_09")
from _panel import FEATS, G_all, G_bh, N, TYPE, Zf, label_sd, nodes, windows  # noqa: E402

Rf = FEATS.index("released")
EDGES = [(a, b) for a, b in zip(*np.nonzero(G_all))]
E = len(EDGES)
SIGN = np.array([G_all[a, b] for a, b in EDGES])
CHAN = [f"{TYPE[nodes[a]]}→{TYPE[nodes[b]]}" for a, b in EDGES]
GROUPS = sorted(set(CHAN))
GIDX = np.array([GROUPS.index(c) for c in CHAN])
A_T, B_T = 2.0, 0.02          # τ² ~ InvGamma(A_T, B_T)
A_S, B_S = 2.0, 1.0           # σ² ~ InvGamma(A_S, B_S)
MU_SD, ALPHA_SD = 0.3, 0.1
BURN, DRAWS, CHAINS = 1000, 2000, 2
YCLIP = 4.0                   # winsorise z-unit labels for estimation only


def cells(w, rows, lsd):
    """Design for the labelled cells of windows ``rows``: X (n × E), y, meta."""
    z = w["Xs"][:, -1, :, Zf] * (w["Xs"][:, -1, :, Rf] > 0)
    j, b = np.nonzero(w["ym"] & rows[:, None])
    X = np.zeros((len(j), E))
    for e, (a, bb) in enumerate(EDGES):
        m = b == bb
        X[m, e] = z[j[m], a]
    y = w["y"][j, b] / lsd[b]
    return X, y, j, b


def rtnorm_pos(mean, sd, rng):
    """One draw from N(mean, sd²) truncated to [0, ∞), by inverse CDF of the
    upper tail (stable far into the tail): z > a with a = −mean/sd."""
    a = -mean / sd
    return mean + sd * (-ndtri(rng.uniform() * ndtr(-a)))


def gibbs(X, y, prior, seed, sign=None, gidx=None, burn=BURN, draws=DRAWS):
    """One chain. ``sign``/``gidx`` (theory sign and channel index per edge,
    one per column of X) default to this panel's 142 edges."""
    SIGN = globals()["SIGN"] if sign is None else sign
    GIDX = globals()["GIDX"] if gidx is None else gidx
    E, G = len(SIGN), int(GIDX.max()) + 1
    rng = np.random.default_rng(seed)
    n = len(y)
    yc = np.clip(y, -YCLIP, YCLIP)
    beta = np.zeros(E)
    tau2 = np.full(G, B_T / (A_T - 1))
    mu = np.zeros(G)
    sig2, alpha = 1.0, 0.0
    XtX_diag = (X ** 2).sum(0)
    XtX = X.T @ X if prior == "soft" else None                 # fixed across iterations
    NZ = [np.nonzero(X[:, e])[0] for e in range(E)]          # cells each edge touches
    out = {"beta": [], "mu": [], "tau": [], "sig": []}
    for it in range(burn + draws):
        r = yc - alpha
        if prior == "soft":
            prec = XtX / sig2 + np.diag(1 / tau2[GIDX])
            m0 = SIGN * mu[GIDX] / tau2[GIDX]
            L = np.linalg.cholesky(prec)
            mean = np.linalg.solve(L.T, np.linalg.solve(L, X.T @ r / sig2 + m0))
            beta = mean + np.linalg.solve(L.T, rng.standard_normal(E))
            for g in range(G):                    # μ_g | β, τ_g
                ix = GIDX == g
                v = 1 / (ix.sum() / tau2[g] + 1 / MU_SD ** 2)
                mu[g] = rng.normal(v * (SIGN[ix] * beta[ix]).sum() / tau2[g], np.sqrt(v))
            dev = beta - SIGN * mu[GIDX]
        else:
            res = r - X @ beta
            for e in range(E):                    # θ_e | rest: truncated normal ≥ 0
                if XtX_diag[e] == 0:
                    theta = abs(rng.normal(0, np.sqrt(tau2[GIDX[e]])))
                else:
                    ix = NZ[e]
                    xs = X[ix, e] * SIGN[e]
                    rr = res[ix] + xs * (SIGN[e] * beta[e])
                    v = 1 / (XtX_diag[e] / sig2 + 1 / tau2[GIDX[e]])
                    mean = v * (xs @ rr) / sig2
                    theta = rtnorm_pos(mean, np.sqrt(v), rng)
                    res[ix] = rr - xs * theta
                beta[e] = SIGN[e] * theta
            dev = beta
        for g in range(G):                        # τ_g² | deviations
            ix = GIDX == g
            tau2[g] = 1 / rng.gamma(A_T + ix.sum() / 2, 1 / (B_T + (dev[ix] ** 2).sum() / 2))
        e_ = yc - alpha - X @ beta
        sig2 = 1 / rng.gamma(A_S + n / 2, 1 / (B_S + (e_ ** 2).sum() / 2))
        v = 1 / (n / sig2 + 1 / ALPHA_SD ** 2)
        alpha = rng.normal(v * (yc - X @ beta).sum() / sig2, np.sqrt(v))
        if it >= burn:
            out["beta"].append(beta.copy()); out["mu"].append(mu.copy())
            out["tau"].append(np.sqrt(tau2)); out["sig"].append(np.sqrt(sig2))
            out.setdefault("alpha", []).append(alpha)
    return {k: np.array(v) for k, v in out.items()}


def fit(X, y, prior, sign=None, gidx=None, burn=BURN, draws=DRAWS):
    chains = [gibbs(X, y, prior, s, sign, gidx, burn, draws) for s in range(CHAINS)]
    out = {k: np.concatenate([c[k] for c in chains]) for k in chains[0]}
    # split-R̂ on β
    halves = [c["beta"][i * draws // 2:(i + 1) * draws // 2] for c in chains for i in range(2)]
    W = np.mean([h.var(0, ddof=1) for h in halves], 0)
    B = np.var([h.mean(0) for h in halves], 0, ddof=1) * (draws // 2)
    rhat = np.sqrt(((draws // 2 - 1) / (draws // 2) * W + B / (draws // 2)) / np.maximum(W, 1e-12))
    out["rhat_max"] = float(np.nanmax(rhat[W > 0])) if (W > 0).any() else np.nan
    return out


def scores(y, p):
    s = (y != 0) & (p != 0)
    up, pu = y[s] > 0, p[s] > 0
    bal = np.mean([(pu & up).sum() / max(up.sum(), 1), (~pu & ~up).sum() / max((~up).sum(), 1)])
    nz = y != 0
    auc = roc_auc_score(y[nz] > 0, p[nz]) if len(np.unique(y[nz] > 0)) == 2 else np.nan
    return 1 - ((y - p) ** 2).sum() / (y ** 2).sum(), bal, auc


def ridge_free(X, y, lam=10.0):
    coef = _ridge(X, np.clip(y, -YCLIP, YCLIP), lam)
    return coef[0], coef[1:]


def one_slope(X, y):
    s = X @ SIGN
    coef = _ridge(s[:, None], np.clip(y, -YCLIP, YCLIP), 10.0)
    return coef[0], coef[1] * SIGN


def main():
    for k in ("imm", "settle"):
        w = windows(k)
        lsd = label_sd(k)
        dates, ends = w["dates"], _label_end(w)
        cuts = _fold_cuts(dates, 8)
        P = {m: [] for m in ("zero-param rule", "linear, one slope", "linear free (ridge)",
                             "Bayes hierarchical, hard sign", "Bayes hierarchical, soft sign")}
        Y, FIRE, rh = [], [], []
        for i in range(8):
            tr = ends < (cuts[i] - PURGE)
            te = (dates >= cuts[i]) & (dates < cuts[i + 1])
            if tr.sum() < 30 or te.sum() == 0:
                continue
            Xtr, ytr, _, _ = cells(w, tr, lsd)
            Xte, yte, jte, bte = cells(w, te, lsd)
            Y.append(yte)
            zte = w["Xs"][:, -1, :, Zf] * (w["Xs"][:, -1, :, Rf] > 0)
            FIRE.append((zte @ G_bh)[jte, bte] != 0)
            P["zero-param rule"].append(Xte @ SIGN)
            a0, b0 = one_slope(Xtr, ytr)
            P["linear, one slope"].append(a0 + Xte @ b0)
            a0, b0 = ridge_free(Xtr, ytr)
            P["linear free (ridge)"].append(a0 + Xte @ b0)
            for prior, name in (("hard", "Bayes hierarchical, hard sign"), ("soft", "Bayes hierarchical, soft sign")):
                d = fit(Xtr, ytr, prior)
                rh.append(d["rhat_max"])
                P[name].append(d["alpha"].mean() + Xte @ d["beta"].mean(0))
            print(f"[{k}] fold {i}: train cells {len(ytr)}, test cells {len(yte)}", flush=True)
        y = np.concatenate(Y)
        fire = np.concatenate(FIRE)
        print(f"\n{'=' * 100}\nlabel '{k}': out-of-fold, {len(y)} test cells ({int(fire.sum())} where the "
              f"BH-channel signal fires); max split-R̂ over folds {np.nanmax(rh):.3f}\n{'=' * 100}")
        print(f"{'model':34} {'R²':>8} {'bal acc':>7} {'AUC':>6} | {'firing: R²':>10} {'bal acc':>7} {'AUC':>6}")
        for name, ps in P.items():
            p = np.concatenate(ps)
            r_all, b_all, a_all = scores(y, p)
            r_f, b_f, a_f = scores(y[fire], p[fire])
            r_all_s = "–" if name == "zero-param rule" else f"{r_all:+.4f}"
            r_f_s = "–" if name == "zero-param rule" else f"{r_f:+.4f}"
            print(f"{name:34} {r_all_s:>8} {b_all:>7.3f} {a_all:>6.3f} | {r_f_s:>10} {b_f:>7.3f} {a_f:>6.3f}")

        # full-sample fit: channel means and per-edge support
        Xa, ya, _, _ = cells(w, np.ones(len(dates), bool), lsd)
        soft, hard = fit(Xa, ya, "soft"), fit(Xa, ya, "hard")
        print(f"\nfull-sample fit ({len(ya)} cells; max split-R̂ soft {soft['rhat_max']:.3f}, "
              f"hard {hard['rhat_max']:.3f})")
        print("\nchannel means μ_g, soft prior (theory-signed: + = the theory direction), with the")
        print("number of edges and how many the data supports (P(theory sign) ≥ 0.95):")
        print(f"{'channel':22} {'edges':>5} {'μ_g':>7} {'90% CI':>18} {'P(μ>0)':>7} {'supported':>9} "
              f"{'hard: mean θ':>12}")
        nx = (Xa != 0).sum(0)
        ptheory = ((soft["beta"] * SIGN) > 0).mean(0)
        for g, ch in enumerate(GROUPS):
            ix = GIDX == g
            mu = soft["mu"][:, g]
            lo, hi = np.percentile(mu, [5, 95])
            sup = int(((ptheory >= 0.95) & ix).sum())
            th = (hard["beta"][:, ix] * SIGN[ix]).mean()
            print(f"{ch:22} {ix.sum():>5} {mu.mean():>+7.3f} [{lo:+.3f}, {hi:+.3f}] {(mu > 0).mean():>7.2f} "
                  f"{sup:>4}/{ix.sum():<4} {th:>12.3f}")
        print("\nedges the data supports (soft prior, P(theory sign) ≥ 0.95), with their data count:")
        order = np.argsort(-ptheory)
        shown = 0
        for e in order:
            if ptheory[e] < 0.95:
                break
            a, b = EDGES[e]
            bm = soft["beta"][:, e]
            lo, hi = np.percentile(bm, [5, 95])
            print(f"  {nodes[a]:>13} → {nodes[b]:<13} {CHAN[e]:22} β {bm.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"P {ptheory[e]:.3f}  cells {nx[e]}")
            shown += 1
        print(f"  ({shown} of {E} edges; against 5% of {E} ≈ {0.05 * E:.0f} expected at P ≥ 0.95 if the "
              f"data carried no signal and the prior were centred)")
        print(f"edges the data contradicts (P(theory sign) ≤ 0.05): {int((ptheory <= 0.05).sum())}")
        pl.DataFrame({"edge": [f"{nodes[a]}→{nodes[b]}" for a, b in EDGES], "channel": CHAN,
                      "cells": nx, "soft_mean": soft["beta"].mean(0), "p_theory": ptheory,
                      "hard_mean": hard["beta"].mean(0)}).write_parquet(
            f"analysis/event_time_2026_09/out/bayes_edges_{k}.parquet")


if __name__ == "__main__":
    main()
