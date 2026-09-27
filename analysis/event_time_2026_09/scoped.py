#!/usr/bin/env python
"""Scope the graph to established relations: can it learn them then?

    venv/bin/python -W ignore analysis/event_time_2026_09/scoped.py \
        > analysis/event_time_2026_09/out/scoped.txt     # needs build_panel.py and models.py first
    venv/bin/python -W ignore analysis/event_time_2026_09/scoped.py --scopes t2 t2wf \
        > analysis/event_time_2026_09/out/scoped_t2.txt

Fitting on every cell of every series may be too noisy for the graph to find
the few relations that exist. Here each model sees only a scope — a set of
edges, the series they touch, and only the cells those edges can explain (a
target's label at an instant where one of its in-scope sources released) —
for training *and* testing.

Scopes:

  hub     labour and inflation releases → FED (10 edges, 2 negative: U3,
          JOBLESSCLAIMS). The Stage-1 structure. Fixed in advance.
  BH      the three BH channels, labour→labour, inflation→inflation,
          labour→policy (25 edges, 6 negative). Fixed in advance.
  wf      walk-forward selected: in each fold, the edges whose Spearman(z_a, y_b)
          on that fold's *training* windows has p < 0.10 and n ≥ 8, signed by
          the data. Not chosen with the test period in view.
  t2      every type→type channel whose theory-signed slope (y_b on
          Σ_a HAWKISH[a]·HAWKISH[b]·z_a over the channel's cells, intercept,
          SE clustered by release instant) has t ≥ 2 on the full sample, per
          label. On this panel that is labour→policy (+ growth→policy on imm).
  t2wf    the same rule applied inside each fold to its training windows only.

The fixed scopes (hub, BH, t2) were identified on 2021–2025 data that includes the 2024–25
test blocks, so they are an optimistic upper bound; the wf scope is honest.

Models inside a scope (same walk-forward, purge and training as ``models.py``):
  zero-param rule      side = sign Σ_a G[a, b]·z_a (no fitting)
  linear sign imposed  y_b = α + β·Σ_a G[a, b]·z_a (one slope)
  linear free          y_b = α_b + Σ_a W_ab·z_a on the scope's edges (no sign prior)
  AGCRN adaptive       no prior, softmax graph over the scope's series
  AGCRN signed         no prior, directed signed graph E_dst E_srcᵀ
  AGCRN frozen         the scope's signed graph frozen in (for wf: the data signs)

Scored on the scope's firing test cells: R², balanced accuracy, AUC, balanced
sign accuracy of the effective edges (∂ŷ_b/∂z_a) against theory, and for the
settle label gross P&L vs a random side. The same AGCRN and linear models
trained on *all* cells (``oof_all.parquet``) are scored on the same scoped cells
— the direct test of whether scoping the training helps.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl
import torch
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN
from stg.models.baselines import _ridge
from stg.models.tensors import apply_feature_scaler, apply_label_scaler, fit_feature_scaler
from stg.models.train import PURGE, _fold_cuts, fit_fold
from stg.structure.stats import spearman, spearman_p

sys.path.insert(0, "analysis/event_time_2026_09")
from _panel import (END, FEATS, G_all, G_bh, HAWKISH, L, M, N, TYPE, X, Y, Zf, instants,  # noqa: E402
                    label_sd, ni, nodes)

SEEDS = (0, 1, 2)
N_BOOT = 2000
torch.set_num_threads(4)
Rf = FEATS.index("released")
F = X.shape[-1]
rng = np.random.default_rng(0)

fed = ni["FED"]
G_hub = np.zeros((N, N))
for a in range(N):
    if G_all[a, fed] != 0 and TYPE[nodes[a]] in ("labour", "inflation"):
        G_hub[a, fed] = G_all[a, fed]


def rule(t): print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)


# ------------------------------------------------------------------ data
def all_windows(k):
    idx = np.array([t for t in range(L - 1, len(instants)) if np.isfinite(Y[k][t]).any()])
    return dict(t=idx, Xs=np.stack([X[t - L + 1:t + 1] for t in idx]),
                Ms=np.stack([M[t - L + 1:t + 1] for t in idx]),
                y=np.nan_to_num(Y[k][idx]), ym=np.isfinite(Y[k][idx]),
                end=END[k][idx], dates=instants[idx])


def firing(w, G):
    """(window, node): target b is labelled and an in-scope source released."""
    rel = w["Xs"][:, -1, :, Rf] > 0
    return w["ym"] & ((rel.astype(float) @ (G != 0).astype(float)) > 0)


def label_end(w, mask):
    e = np.where(mask, w["end"], np.datetime64("NaT"))
    out = np.array([row[~np.isnat(row)].max() if (~np.isnat(row)).any() else np.datetime64("NaT")
                    for row in e])
    return out.astype("datetime64[us]")


# ------------------------------------------------------------------ models
def z_last(Xs):
    return Xs[:, -1, :, Zf] * (Xs[:, -1, :, Rf] > 0)


def fit_linear_imposed(Xs, y, ym, G):
    s = z_last(Xs) @ G
    coef = _ridge(s[ym][:, None], y[ym], 10.0)
    return lambda Xt: coef[0] + coef[1] * (z_last(Xt) @ G)


def fit_linear_free(Xs, y, ym, G):
    z = z_last(Xs)
    W, c = np.zeros((N, N)), np.zeros(N)
    for b in range(N):
        par, rows = np.nonzero(G[:, b])[0], ym[:, b]
        if len(par) and rows.sum() >= 8:
            coef = _ridge(z[rows][:, par], y[rows, b], 10.0)
            c[b], W[par, b] = coef[0], coef[1:]
    return (lambda Xt: z_last(Xt) @ W + c), W


def make_agcrn(kind, G):
    cfg = dict(hidden=16, d_emb=2, masking="per_step", zero_head=True, n_horizons=1)
    if kind == "adaptive":
        return lambda: AGCRN(N, F, embedding="hybrid", **cfg)
    if kind == "signed":
        return lambda: AGCRN(N, F, embedding="learned", adjacency="signed", **cfg)
    return lambda: AGCRN(N, F, embedding="hybrid", adjacency="stage1", stage1_adj=G, **cfg)


def fit_agcrn(kind, G, w, fit, es, te, sd_y, seed):
    """Train on the scope: inactive series are masked out of the inputs (they
    send nothing and their state is frozen), and the loss is on firing cells."""
    torch.manual_seed(seed)
    scope = ((G != 0).any(0) | (G != 0).any(1))
    Ms = w["Ms"] & scope[None, None, :]
    mu, sd = fit_feature_scaler(w["Xs"][fit], Ms[fit])

    def prep(m):
        return (torch.tensor(apply_feature_scaler(w["Xs"][m], mu, sd)), torch.tensor(Ms[m]),
                torch.tensor(apply_label_scaler(w["y"][m], sd_y), dtype=torch.float32),
                torch.tensor(w["ym_s"][m]))
    model, _ = fit_fold(make_agcrn(kind, G), prep(fit), prep(es))
    Xt, Mt, _, _ = prep(te)
    Xg = Xt.clone().requires_grad_(True)
    out = model(Xg, Mt).squeeze(-1)
    rel, lab = w["Xs"][te][:, -1, :, Rf] > 0, w["ym_s"][te]
    J = {}
    for b in np.nonzero((G != 0).any(0))[0]:
        g, = torch.autograd.grad(out[:, b].sum(), Xg, retain_graph=True)
        g = g[:, -1, :, Zf].numpy() / sd[:, Zf]
        for a in np.nonzero(G[:, b])[0]:
            sel = rel[:, a] & lab[:, b]
            if sel.any():
                J[(a, b)] = float(g[sel, a].mean())
    return out.detach().numpy(), J


# ------------------------------------------------------------------ walk-forward
def run_scope(k, scope):
    """Returns out-of-fold predictions (z units) per model on firing test cells,
    and effective edges per model."""
    w = all_windows(k)
    sd_y = label_sd(k)
    cuts = _fold_cuts(w["dates"], 8)
    preds = {m: np.full(w["y"].shape, np.nan) for m in MODELS}
    edges = {m: {} for m in MODELS}
    test_mask = np.zeros(w["y"].shape, bool)
    edges_used = []
    for i in range(8):
        ends_all = label_end(w, w["ym"])
        if scope == "wf":
            tr0 = ends_all < (cuts[i] - PURGE)
            G = select_edges(w, tr0, sd_y)
        elif scope == "t2wf":
            tr0 = ends_all < (cuts[i] - PURGE)
            G = select_channels(w, tr0, sd_y)
        elif scope == "t2":
            G = select_channels(w, np.ones(len(w["dates"]), bool), sd_y)
        else:
            G = {"hub": G_hub, "BH": G_bh}[scope]
        CHOSEN.append(sorted({(TYPE[nodes[a]], TYPE[nodes[b]]) for a, b in zip(*np.nonzero(G))}))
        if not (G != 0).any():
            continue
        edges_used.append(int((G != 0).sum()))
        fm = firing(w, G)
        ws = {**w, "ym_s": fm}
        ends = label_end(w, fm)
        tr = ~np.isnat(ends) & (ends < (cuts[i] - PURGE))
        te = (w["dates"] >= cuts[i]) & (w["dates"] < cuts[i + 1]) & fm.any(1)
        if tr.sum() < 20 or te.sum() == 0:
            continue
        test_mask[te] |= fm[te]
        y_tr = apply_label_scaler(w["y"][tr], sd_y)
        f = fit_linear_imposed(w["Xs"][tr], y_tr, fm[tr], G)
        preds["linear sign imposed"][te] = f(w["Xs"][te])
        f, W = fit_linear_free(w["Xs"][tr], y_tr, fm[tr], G)
        preds["linear free"][te] = f(w["Xs"][te])
        for a, b in zip(*np.nonzero(G)):
            edges["linear free"].setdefault((a, b), []).append(W[a, b])
        preds["zero-param rule"][te] = z_last(w["Xs"][te]) @ G
        # early-stop split inside the fold, as run_torch
        tdates = np.sort(w["dates"][tr])
        es_cut = tdates[int(len(tdates) * 0.85)]
        fit = tr & ~np.isnat(ends) & (ends < es_cut)
        es = tr & (w["dates"] >= es_cut)
        if fit.sum() < 15 or es.sum() < 3:
            fit, es = tr, tr
        for kind in ("adaptive", "signed", "frozen"):
            name = f"AGCRN {kind}"
            runs = []
            for seed in SEEDS:
                p, J = fit_agcrn(kind, G, ws, fit, es, te, sd_y, seed)
                runs.append(p)
                for e, v in J.items():
                    edges[name].setdefault(e, []).append(v)
            preds[name][te] = np.mean(runs, 0)
    return w, test_mask, preds, edges, edges_used


def select_edges(w, tr, sd_y):
    """Edges with p < 0.10 and n ≥ 8 on the training windows, signed by the data."""
    z, rel = z_last(w["Xs"]), w["Xs"][:, -1, :, Rf] > 0
    y = w["y"] / sd_y
    G = np.zeros((N, N))
    for a, b in zip(*np.nonzero(G_all)):
        sel = tr & rel[:, a] & w["ym"][:, b]
        if sel.sum() >= 8:
            r = spearman(z[sel, a], y[sel, b])
            if np.isfinite(r) and spearman_p(r, int(sel.sum())) < 0.10:
                G[a, b] = np.sign(r)
    return G


CHANNELS = {}
for _a, _b in zip(*np.nonzero(G_all)):
    CHANNELS.setdefault((TYPE[nodes[_a]], TYPE[nodes[_b]]), []).append((_a, _b))


def channel_t(w, tr, sd_y, c):
    """Theory-signed slope t for channel c on windows tr (intercept, SE
    clustered by window = release instant). NaN if fewer than 10 cells."""
    Gc = np.zeros((N, N))
    for a, b in CHANNELS[c]:
        Gc[a, b] = G_all[a, b]
    S = z_last(w["Xs"]) @ Gc
    cell = tr[:, None] & w["ym"] & (S != 0) & (Gc != 0).any(0)[None, :]
    j, b = np.nonzero(cell)
    if len(j) < 10 or len(np.unique(j)) < 3:
        return np.nan
    y, s = w["y"][j, b] / sd_y[b], S[j, b]
    Xd = np.column_stack([np.ones(len(y)), s])
    inv = np.linalg.pinv(Xd.T @ Xd)
    beta = inv @ Xd.T @ y
    e = y - Xd @ beta
    meat = sum(np.outer(Xd[j == u].T @ e[j == u], Xd[j == u].T @ e[j == u]) for u in np.unique(j))
    g = len(np.unique(j))
    V = inv @ meat @ inv * g / (g - 1)
    return beta[1] / np.sqrt(V[1, 1]) if V[1, 1] > 0 else np.nan


def select_channels(w, tr, sd_y, thresh=2.0):
    G = np.zeros((N, N))
    for c, edges in CHANNELS.items():
        t = channel_t(w, tr, sd_y, c)
        if np.isfinite(t) and t >= thresh:
            for a, b in edges:
                G[a, b] = G_all[a, b]
    return G


MODELS = ["zero-param rule", "linear sign imposed", "linear free", "AGCRN adaptive",
          "AGCRN signed", "AGCRN frozen"]


# ------------------------------------------------------------------ scoring
def scores(y, p):
    s = (y != 0) & (p != 0)
    up, pu = y[s] > 0, p[s] > 0
    bal = np.mean([(pu & up).sum() / max(up.sum(), 1), (~pu & ~up).sum() / max((~up).sum(), 1)])
    nz = y != 0
    auc = roc_auc_score(y[nz] > 0, p[nz]) if len(np.unique(y[nz] > 0)) == 2 and np.ptp(p[nz]) > 0 else np.nan
    return 1 - ((y - p) ** 2).sum() / (y ** 2).sum(), bal, auc


def pnl(y_c, p, inst):
    side = np.sign(p)
    t = side != 0
    side, y_c, inst = side[t], y_c[t], inst[t]
    q = (side < 0).mean()
    ex = side * y_c - (1 - 2 * q) * y_c
    u, inv = np.unique(inst, return_inverse=True)
    E, C = np.bincount(inv, ex, len(u)), np.bincount(inv, None, len(u))
    bs = [E[b].sum() / C[b].sum() for b in (rng.integers(0, len(u), len(u)) for _ in range(N_BOOT))]
    return (side * y_c).mean(), ex.mean(), np.percentile(bs, [2.5, 97.5]), q


def sign_bal(edges):
    theory = np.sign(G_all)
    ok = [(np.sign(np.mean(v)), theory[e]) for e, v in edges.items() if np.mean(v) != 0]
    if not ok:
        return np.nan, np.nan, 0
    m, t = np.array(ok).T
    hp = (m[t > 0] > 0).mean() if (t > 0).any() else np.nan
    hn = (m[t < 0] < 0).mean() if (t < 0).any() else np.nan
    tab = [[int(((t > 0) & (m > 0)).sum()), int(((t > 0) & (m < 0)).sum())],
           [int(((t < 0) & (m > 0)).sum()), int(((t < 0) & (m < 0)).sum())]]
    p = fisher_exact(tab, alternative="greater")[1] if (t < 0).any() else np.nan
    return np.nanmean([hp, hn]), p, len(ok)


CHOSEN: list = []


def main():
    import argparse  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--scopes", nargs="*", default=["hub", "BH", "wf"],
                    choices=["hub", "BH", "wf", "t2", "t2wf"])
    SCOPES = ap.parse_args().scopes
    oof_all = pl.read_parquet("analysis/event_time_2026_09/out/oof_all.parquet")
    FULL = {"linear own + econ (BH ch.)": "linear own+econ, trained on ALL cells",
            "linear econ signal only (BH ch.)": "linear econ (BH), trained on ALL cells",
            "AGCRN adaptive (hybrid), no prior": "AGCRN adaptive, trained on ALL cells",
            "AGCRN frozen econ graph (BH ch.)": "AGCRN frozen BH, trained on ALL cells"}

    for k in ("imm", "settle"):
        for scope in SCOPES:
            CHOSEN.clear()
            w, tm, preds, edges, used = run_scope(k, scope)
            sd_y = label_sd(k)
            y = w["y"] / sd_y
            rule(f"label '{k}', scope '{scope}' — {int(tm.sum())} firing test cells on "
                 f"{int(tm.any(1).sum())} instants; edges per fold {used}")
            chans = sorted({f"{a}→{b}" for fold in CHOSEN for a, b in fold})
            print(f"channels in scope (any fold): {', '.join(chans)}; folds with each: "
                  + ", ".join(f"{c} {sum(c in {f'{a}→{b}' for a, b in fold} for fold in CHOSEN)}/{len(CHOSEN)}"
                              for c in chans))
            print(f"{'model':42} {'R²':>8} {'bal acc':>7} {'AUC':>6} | {'sign bal':>8} {'p':>6} {'edges':>5}"
                  + (f" | {'gross ¢':>8} {'vs random side ¢ [95% CI]':>28} {'short':>5}" if k == "settle" else ""))
            rows = [(m, preds[m][tm], edges.get(m, {})) for m in MODELS]
            cell_t, cell_n = np.nonzero(tm)
            key = pl.DataFrame({"instant": w["dates"][cell_t].astype("datetime64[us]"),
                                "series": [nodes[i] for i in cell_n],
                                "i": np.arange(len(cell_t))})
            for mname, label in FULL.items():
                d = (oof_all.filter((pl.col("label") == k) & (pl.col("model") == mname))
                     .select(pl.col("instant").cast(pl.Datetime("us")), "series", "pred_c"))
                j = key.join(d, on=["instant", "series"], how="left").sort("i")
                p = j["pred_c"].to_numpy().astype(float) / sd_y[cell_n]
                rows.append((label, p, {}))
            for name, p, e in rows:
                ok = np.isfinite(p)
                r2, bal, auc = scores(y[tm][ok], p[ok])
                r2s = "–" if name == "zero-param rule" else f"{r2:+.4f}"   # raw Σz is not a forecast
                sb, sp, ne = sign_bal(e) if e else (np.nan, np.nan, 0)
                line = (f"{name:42} {r2s:>8} {bal:>7.3f} {auc:>6.3f} | "
                        f"{sb:>8.2f} {sp:>6.3f} {ne:>5}" if e else
                        f"{name:42} {r2s:>8} {bal:>7.3f} {auc:>6.3f} | {'':>8} {'':>6} {'':>5}")
                if k == "settle":
                    g, ex, ci, q = pnl(w["y"][tm][ok], p[ok], w["dates"][cell_t][ok])
                    line += f" | {g:>+8.2f} {ex:>+8.2f} [{ci[0]:+6.2f},{ci[1]:+6.2f}] {q:>5.2f}"
                print(line, flush=True)
        print("\nsign bal: balanced sign accuracy of the effective edges (mean over folds and seeds)\n"
              "against theory, 0.5 = no sign learned; p: one-sided Fisher exact. Rows 'trained on ALL\n"
              "cells' are the models.py predictions scored on the same scoped cells.")



if __name__ == "__main__":
    main()
