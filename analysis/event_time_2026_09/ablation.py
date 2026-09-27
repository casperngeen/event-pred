#!/usr/bin/env python
"""Two questions about the graph models on the release clock.

    venv/bin/python analysis/event_time_2026_09/ablation.py \
        > analysis/event_time_2026_09/out/ablation.txt      # needs build_panel.py first

A  STG vs temporal-only vs spatial-only. The same AGCRN (per-step masking,
   zero-init head, hidden 16, d_emb 2, hybrid embedding) with one part removed:

     STG            graph + 6-instant history
     temporal only  history, no graph (frozen all-zero adjacency: no messages)
     spatial only   graph, no history (window of 1 instant)
     neither        no graph, no history (a per-node network on the current state)

   each with the adaptive graph and, where it applies, the economic BH graph
   frozen in. Linear analogues on the same windows: own state (last instant),
   own-state lags (all 6), the economic signal, and own lags + signal.

B  Do the graph models learn the *signs*? Each fitted model's effective edge
   ∂ŷ_b/∂z_a, averaged over held-out windows where a released and b is
   labelled (and over folds and seeds), compared with the theory sign
   HAWKISH[a]·HAWKISH[b]:

     no prior         AGCRN adaptive (softmax: the graph cannot carry a sign),
                      AGCRN signed (E_dst E_srcᵀ: it can), a free linear graph
                      (ridge of y_b on every candidate parent's z)
     structure prior  AGCRN adaptive + the BH edges as an unsigned soft prior
     signed prior     AGCRN with the signed BH / all-edge graph frozen in

   Also: agreement with the *empirical* sign (full-sample Spearman of z_a with
   y_b, n ≥ 5), which is as far as any learner could get from this data.

Walk-forward and training exactly as ``models.py`` (8 folds, purged on label
end, 3 seeds). Out-of-fold predictions go to ``out/oof_ablation.parquet`` for
``metrics.py`` and ``returns.py``.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

import numpy as np
import polars as pl
import torch
from scipy.stats import fisher_exact

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN, count_params
from stg.models.baselines import _ridge
from stg.models.train import run_linear, run_torch
from stg.structure.stats import spearman

sys.path.insert(0, "analysis/event_time_2026_09")
from _panel import (FEATS, G_all, G_bh, N, F, OWN, Ridge, Zf, label_sd,  # noqa: E402
                    nodes, windows)

SEEDS = (0, 1, 2)
torch.set_num_threads(4)
Rf = FEATS.index("released")
CAND, BH = G_all != 0, G_bh != 0
ZEROS = np.zeros((N, N))


def rule(t): print("\n" + "=" * 96 + f"\n{t}\n" + "=" * 96, flush=True)


def last1(w):
    """The same windows cut to the final instant (no history)."""
    return {**w, "Xs": w["Xs"][:, -1:], "Ms": w["Ms"][:, -1:]}


class RidgeLags(Ridge):
    """Pooled ridge on own-state features at every lag, ± the econ signal.
    18+ raw features on a few hundred cells: standardised on the training
    cells and clipped at ±3 sd — ``d_q50_7d`` reaches 87 sd out of sample, and
    unclipped the rung scores R² −0.55 from a handful of cells. (The
    last-instant ``Ridge`` has 3 features and is left as in ``models.py``.)"""

    def _f(self, win):
        x = win["Xs"]
        cols = [x[:, l, :, FEATS.index(c)] for l in range(x.shape[1]) for c in OWN]
        if self.G is not None:
            z = x[:, -1, :, Zf] * (x[:, -1, :, Rf] > 0)
            cols.append(z @ self.G)
        Fm = np.stack(cols, -1)
        if not hasattr(self, "mu"):
            self.mu, self.sd = Fm[win["ym"]].mean(0), Fm[win["ym"]].std(0) + 1e-9
        return np.clip((Fm - self.mu) / self.sd, -3, 3)


class FreeGraph:
    """Per-target ridge of y_b on every candidate parent's surprise: a linear
    graph with no prior (one free coefficient per candidate edge)."""
    fits: list = []

    def fit(self, win):
        z = win["Xs"][:, -1, :, Zf] * (win["Xs"][:, -1, :, Rf] > 0)
        self.W, self.c = np.zeros((N, N)), np.zeros(N)
        for b in range(N):
            par, rows = np.nonzero(CAND[:, b])[0], win["ym"][:, b]
            if len(par) and rows.sum() >= 10:
                coef = _ridge(z[rows][:, par], win["y"][rows, b], 10.0)
                self.c[b], self.W[par, b] = coef[0], coef[1:]
        FreeGraph.fits.append(self.W.copy())
        return self

    def predict(self, win):
        z = win["Xs"][:, -1, :, Zf] * (win["Xs"][:, -1, :, Rf] > 0)
        return z @ self.W + self.c


def agcrn(G=None, **kw):
    cfg = dict(hidden=16, d_emb=2, masking="per_step", zero_head=True, embedding="hybrid") | kw
    if G is not None:
        cfg |= dict(adjacency="stage1", stage1_adj=G)
    return lambda: AGCRN(N, F, n_horizons=1, **cfg)


def jacobian_reader(store):
    """on_fold hook: accumulate ∂ŷ_b/∂z_a (per unit raw z) over test windows.
    ``hook.released`` / ``hook.labelled`` are the (window, node) masks at the
    window's last instant, set by the caller; ``te`` selects the fold's rows."""
    def hook(model, Xt, Mt, te, seed, sd_f):
        Xg = Xt.clone().requires_grad_(True)
        out = model(Xg, Mt).squeeze(-1)
        for b in range(N):
            g, = torch.autograd.grad(out[:, b].sum(), Xg, retain_graph=True)
            g = g[:, -1, :, Zf].numpy() / sd_f[:, Zf]
            for a in np.nonzero(CAND[:, b])[0]:
                sel = hook.released[te][:, a] & hook.labelled[te][:, b]
                if sel.any():
                    store[(a, b)].append(g[sel, a].mean())
    return hook


def sign_table(name, W, rows):
    """W: (N, N) effective/learned edges, NaN where no read-out.

    Most theory edges are positive (94 of 142; 19 of 25 BH), so a model whose
    edges are all positive — "move with your neighbours' surprise" — already
    agrees 66% / 76% of the time. The test is therefore *balanced* sign
    accuracy, the mean of the hit rates on theory-positive and theory-negative
    edges (0.5 = no sign learned), with a one-sided Fisher exact test of the
    2×2 (theory sign × model sign) table.
    """
    theory = np.sign(G_all)
    out = {"model": name}
    for sub, mask in (("all", CAND), ("BH", BH)):
        ok = mask & np.isfinite(W) & (W != 0)
        t, m = theory[ok] > 0, np.sign(W[ok]) > 0
        tab = [[int((t & m).sum()), int((t & ~m).sum())], [int((~t & m).sum()), int((~t & ~m).sum())]]
        hp = tab[0][0] / max(t.sum(), 1)
        hn = tab[1][1] / max((~t).sum(), 1) if (~t).any() else np.nan
        emp = ok & (EMP != 0)
        out[sub] = dict(n=int(ok.sum()), pos=float(t.mean()) if ok.any() else np.nan,
                        raw=float((t == m).mean()) if ok.any() else np.nan,
                        hp=hp, hn=hn, bal=np.nanmean([hp, hn]),
                        p=fisher_exact(tab, alternative="greater")[1] if (~t).any() else np.nan,
                        emp=float((np.sign(W[emp]) == EMP[emp]).mean()) if emp.any() else np.nan)
    rows.append(out)
    EDGES.append(pl.DataFrame([dict(label=k, model=name, source=nodes[a], target=nodes[b],
                                    edge=float(W[a, b]), theory=float(theory[a, b]))
                               for a, b in zip(*np.nonzero(CAND & np.isfinite(W)))]))


OOF = []


def keep(k, name, r, w, lsd):
    for j, d in enumerate(w["dates"]):
        for i in np.nonzero(r["_mask"][j])[0]:
            OOF.append(dict(label=k, instant=d.astype("datetime64[us]").item(), series=nodes[i],
                            model=name, pred_c=float(r["_preds"][j, i] * lsd[i]),
                            y_c=float(r["_y"][j, i] * lsd[i])))


def summary(name, r, params="–", extra=""):
    y, p, m = r["_y"], r["_preds"], r["_mask"]
    s = m & (y != 0) & (p != 0)
    r2 = 1 - ((y - p)[m] ** 2).sum() / (y[m] ** 2).sum()
    print(f"{name:46} {params:>6} {r2:>+8.4f} {(np.sign(y[s]) == np.sign(p[s])).mean():>6.3f}"
          f"{extra}", flush=True)


SIGNS = []
EDGES = []
for k in ("imm", "settle"):
    w = windows(k)
    lsd = label_sd(k)
    # which (window, node) cells released / are labelled at the window's last instant
    rel_w = w["Xs"][:, -1, :, Rf] > 0
    lab_w = w["ym"]
    # empirical sign per edge: full-sample Spearman of z_a with y_b (n ≥ 5)
    z_w = w["Xs"][:, -1, :, Zf] * rel_w
    EMP = np.zeros((N, N))
    for a, b in zip(*np.nonzero(CAND)):
        sel = rel_w[:, a] & lab_w[:, b]
        if sel.sum() >= 5:
            rho = spearman(z_w[sel, a], w["y"][sel, b])
            EMP[a, b] = np.sign(rho) if np.isfinite(rho) else 0

    rule(f"A  STG vs TEMPORAL-ONLY vs SPATIAL-ONLY — label '{k}' (per-node z units)")
    print(f"{'model':46} {'params':>6} {'R² vs 0':>8} {'dir≠0':>6}")
    for name, f, win in [
            ("linear own state (last instant)", lambda: Ridge(True), w),
            ("linear temporal: own-state lags", lambda: RidgeLags(True), w),
            ("linear spatial: econ signal (BH ch.)", lambda: Ridge(False, G_bh), w),
            ("linear spatial: econ signal (all)", lambda: Ridge(False, G_all), w),
            ("linear both: own lags + econ (BH ch.)", lambda: RidgeLags(True, G_bh), w)]:
        r = run_linear(f, win, lsd)
        summary(name, r)
        keep(k, name, r, win, lsd)
    for name, f, win in [
            ("AGCRN STG: adaptive graph + history", agcrn(), w),
            ("AGCRN temporal only: no graph", agcrn(ZEROS), w),
            ("AGCRN spatial only: adaptive graph, no history", agcrn(), last1(w)),
            ("AGCRN neither: no graph, no history", agcrn(ZEROS), last1(w)),
            ("AGCRN STG: econ BH graph + history", agcrn(G_bh), w),
            ("AGCRN spatial only: econ BH graph, no history", agcrn(G_bh), last1(w))]:
        t0 = time.time()
        r = run_torch(f, win, lsd, seeds=SEEDS)
        summary(name, r, str(count_params(f())), f"  ±{r['r2_vs_zero_sd']:.3f} [{time.time() - t0:.0f}s]")
        keep(k, name, r, win, lsd)

    rule(f"B  DO THE GRAPH MODELS LEARN THE SIGNS? — label '{k}'")
    print("effective edge ∂ŷ_b/∂z_a on held-out windows vs the theory sign HAWKISH[a]·HAWKISH[b];\n"
          "the test is balanced sign accuracy (table footnote). For reference, the full-sample empirical\n"
          f"sign (Spearman of z_a with y_b, n ≥ 5) agrees with theory on "
          f"{(EMP[CAND & (EMP != 0)] == np.sign(G_all)[CAND & (EMP != 0)]).mean():.3f} of "
          f"{int((CAND & (EMP != 0)).sum())} edges, {(EMP[BH & (EMP != 0)] == np.sign(G_bh)[BH & (EMP != 0)]).mean():.3f} "
          f"of {int((BH & (EMP != 0)).sum())} BH edges.\n")
    rows = []
    for name, f in [("AGCRN adaptive, no prior", agcrn()),
                    ("AGCRN signed directed, no prior", agcrn(embedding="learned", adjacency="signed")),
                    ("AGCRN adaptive + unsigned BH prior (λ=2)", agcrn(prior_adj=G_bh, prior_lambda=2.0)),
                    ("AGCRN frozen signed BH graph", agcrn(G_bh)),
                    ("AGCRN frozen signed all-edge graph", agcrn(G_all))]:
        store = defaultdict(list)
        hook = jacobian_reader(store)
        hook.released, hook.labelled = rel_w, lab_w
        r = run_torch(f, w, lsd, seeds=SEEDS, on_fold=hook)
        if "signed directed" in name or "prior" in name:
            keep(k, name, r, w, lsd)
        J = np.full((N, N), np.nan)
        for (a, b), v in store.items():
            J[a, b] = np.mean(v)
        sign_table(name, J, rows)
    FreeGraph.fits = []
    r = run_linear(lambda: FreeGraph(), w, lsd)
    keep(k, "linear free graph, no prior", r, w, lsd)
    Wf = np.mean(FreeGraph.fits, 0)
    sign_table("linear free graph, no prior (mean coef.)", np.where(CAND, Wf, np.nan), rows)
    print(f"{'':44} {'— all 142 candidate edges —':^40} | {'— 25 BH-channel edges —':^40}")
    print(f"{'model':44} {'n':>4} {'%+':>5} {'raw':>5} {'hit+':>5} {'hit−':>5} {'bal':>5} {'p':>6} | "
          f"{'n':>4} {'%+':>5} {'raw':>5} {'hit+':>5} {'hit−':>5} {'bal':>5} {'p':>6}")
    f = lambda v: f"{v:.2f}" if np.isfinite(v) else "  –"
    for r_ in rows:
        cells_ = []
        for sub in ("all", "BH"):
            d = r_[sub]
            cells_.append(f"{d['n']:>4} {f(d['pos']):>5} {f(d['raw']):>5} {f(d['hp']):>5} "
                          f"{f(d['hn']):>5} {f(d['bal']):>5} {d['p']:>6.3f}")
        print(f"{r_['model']:44} {cells_[0]} | {cells_[1]}")
    print("%+ = share of theory-positive edges among those read out (= raw agreement of an\n"
          "all-positive model); hit+/hit− = sign hit rate on theory-positive/-negative edges;\n"
          "bal = their mean (0.5 = no sign learned); p = one-sided Fisher exact.")
    print("\nThe frozen-graph rungs are given the theory sign in the adjacency; they can still\n"
          "reverse it through the node weights, so their row measures whether the model keeps it.")

pl.DataFrame(OOF).write_parquet("analysis/event_time_2026_09/out/oof_ablation.parquet")
pl.concat(EDGES).write_parquet("analysis/event_time_2026_09/out/ablation_edges.parquet")
