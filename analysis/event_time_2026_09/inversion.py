#!/usr/bin/env python
"""Why does the scoped AGCRN rank below chance when its edge signs are right?

    venv/bin/python -W ignore analysis/event_time_2026_09/inversion.py \
        > analysis/event_time_2026_09/out/inversion.txt      # needs scoped.py's inputs

In the labour→FED scopes (``scoped.py`` t2 / t2wf) every AGCRN scores AUC
0.23–0.41 on the immediate label, including the frozen one whose effective
edges have the theory sign on every edge. Three explanations, each with a test,
all run inside the same pipeline as ``scoped.py`` (same windows, scope masks,
fit / early-stop split, feature and label scalers, ``fit_fold``, firing-cell
loss and test cells):

  alignment / sign convention
      pass-through, untrained   ŷ_b = Σ_a G[a,b]·z_a, read back from the *scaled*
                                inputs: the zero-parameter rule computed inside
                                the model pipeline. Must equal the rule.
      pass-through, trained     scale·(that) + bias, fitted by ``fit_fold`` on the
                                scaled labels, from scale = 0 (lr 0.05, ≤1000
                                epochs). A negative fitted scale would mean the
                                training labels are inverted against the
                                evaluation labels.
  the history (recurrent layer learning "fade the recent move")
      AGCRN frozen, no history  window of 1 instant
  the non-surprise inputs (own state)
      AGCRN frozen, surprise only    inputs = [released, z]
      AGCRN frozen, neither          both removed

and the two proposed fixes:

      sign-fixed magnitudes     y_b = c_b + Σ_a w_ab·G[a,b]·z_a with w_ab ≥ 0
                                (ridge-regularised NNLS): the data learns sizes,
                                theory fixes signs
      rule + AGCRN residual     the imposed-sign linear rung as the base, and a
                                frozen, zero-initialised AGCRN trained only on the
                                residual, so it starts exactly at the base

Scopes: t2 and t2wf (labour→FED; the inversion), and hub (no inversion, as a
contrast). Both labels. Early stopping, seeds and folds as ``scoped.py``.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import nnls

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN
from stg.models.tensors import apply_feature_scaler, apply_label_scaler, fit_feature_scaler
from stg.models.train import PURGE, _fold_cuts, fit_fold

sys.path.insert(0, "analysis/event_time_2026_09")
import scoped as S  # noqa: E402
from _panel import label_sd  # noqa: E402

N, F, Zf, Rf, SEEDS = S.N, S.F, S.Zf, S.Rf, S.SEEDS
torch.set_num_threads(4)


class PassThrough(nn.Module):
    """ŷ_b = scale·Σ_a G[a,b]·z_a·1[a released, active] + bias, reading raw z and
    the release flag back from the standardised inputs."""

    def __init__(self, G, mu, sd, cols, train=True):
        super().__init__()
        self.register_buffer("A", torch.tensor(G, dtype=torch.float32))      # [source, target]
        zi, ri = cols.index(Zf), cols.index(Rf)
        self.zi, self.ri = zi, ri
        self.register_buffer("mz", torch.tensor(mu[:, Zf], dtype=torch.float32))
        self.register_buffer("sz", torch.tensor(sd[:, Zf], dtype=torch.float32))
        self.register_buffer("mr", torch.tensor(mu[:, Rf], dtype=torch.float32))
        self.register_buffer("sr", torch.tensor(sd[:, Rf], dtype=torch.float32))
        # trained: start at 0 so the fitted sign comes from the labels, not the init
        self.scale = nn.Parameter(torch.zeros(1) if train else torch.ones(1), requires_grad=train)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=train)

    def forward(self, seq, mask):
        x = seq[:, -1]
        z = x[..., self.zi] * self.sz + self.mz
        rel = (x[..., self.ri] * self.sr + self.mr) > 0.5
        z = z * rel * mask[:, -1]
        return (self.scale * (z @ self.A) + self.bias)[..., None]


def agcrn_frozen(G, f_in):
    return lambda: AGCRN(N, f_in, hidden=16, d_emb=2, masking="per_step", zero_head=True,
                         n_horizons=1, embedding="hybrid", adjacency="stage1", stage1_adj=G)


def fit_torch(build, w, G, fit, es, te, sd_y, seed, cols, hist, y_override=None):
    """Same preparation as scoped.fit_agcrn, with feature subset / history options.
    ``build(mu, sd)`` returns the model; an untrained PassThrough is not fitted."""
    torch.manual_seed(seed)
    scope = (G != 0).any(0) | (G != 0).any(1)
    Xs = w["Xs"] if hist else w["Xs"][:, -1:]
    Ms = (w["Ms"] if hist else w["Ms"][:, -1:]) & scope[None, None, :]
    mu, sd = fit_feature_scaler(Xs[fit], Ms[fit])
    y = apply_label_scaler(w["y"], sd_y) if y_override is None else y_override

    def prep(m):
        return (torch.tensor(apply_feature_scaler(Xs[m], mu, sd)[..., cols]), torch.tensor(Ms[m]),
                torch.tensor(y[m], dtype=torch.float32), torch.tensor(w["ym_s"][m]))
    model = build(mu, sd)
    if isinstance(model, PassThrough):
        if model.scale.requires_grad:      # 2 parameters: train it to convergence
            model, _ = fit_fold(lambda: model, prep(fit), prep(es), lr=0.05, max_epochs=1000,
                                patience=100)
    else:
        model, _ = fit_fold(lambda: model, prep(fit), prep(es))
    Xt, Mt, _, _ = prep(te)
    with torch.no_grad():
        out = model(Xt, Mt).squeeze(-1).numpy()
    scale = float(model.scale) if isinstance(model, PassThrough) else np.nan
    return out, scale


def sign_fixed(Xs, y, ym, G, lam=10.0):
    """Per target: y_b = c_b + Σ_a w_ab·(G_ab·z_a), w ≥ 0, ridge via row augmentation."""
    z = S.z_last(Xs)
    W, c = np.zeros((N, N)), np.zeros(N)
    for b in range(N):
        par, rows = np.nonzero(G[:, b])[0], ym[:, b]
        if len(par) == 0 or rows.sum() < 8:
            continue
        Fm = z[rows][:, par] * G[par, b][None, :]
        fm, yb = Fm.mean(0), y[rows, b].mean()
        A = np.vstack([Fm - fm, np.sqrt(lam) * np.eye(len(par))])
        rhs = np.concatenate([y[rows, b] - yb, np.zeros(len(par))])
        wv, _ = nnls(A, rhs)
        W[par, b] = wv * G[par, b]
        c[b] = yb - fm @ wv
    return lambda Xt: S.z_last(Xt) @ W + c


VARIANTS = ["zero-param rule", "pass-through, untrained", "pass-through, trained",
            "AGCRN frozen (as scoped.py)", "AGCRN frozen, no history",
            "AGCRN frozen, surprise only", "AGCRN frozen, neither",
            "linear sign imposed", "sign-fixed magnitudes (w ≥ 0)", "rule + AGCRN residual"]


def run(k, scope):
    w = S.all_windows(k)
    sd_y = label_sd(k)
    cuts = _fold_cuts(w["dates"], 8)
    P = {v: np.full(w["y"].shape, np.nan) for v in VARIANTS}
    tm = np.zeros(w["y"].shape, bool)
    scales = []
    allc, zc = list(range(F)), [Rf, Zf]
    for i in range(8):
        ends_all = S.label_end(w, w["ym"])
        if scope == "t2wf":
            G = S.select_channels(w, ends_all < (cuts[i] - PURGE), sd_y)
        elif scope == "t2":
            G = S.select_channels(w, np.ones(len(w["dates"]), bool), sd_y)
        else:
            G = S.G_hub
        if not (G != 0).any():
            continue
        fm = S.firing(w, G)
        ws = {**w, "ym_s": fm}
        ends = S.label_end(w, fm)
        tr = ~np.isnat(ends) & (ends < (cuts[i] - PURGE))
        te = (w["dates"] >= cuts[i]) & (w["dates"] < cuts[i + 1]) & fm.any(1)
        if tr.sum() < 20 or te.sum() == 0:
            continue
        tm[te] |= fm[te]
        tdates = np.sort(w["dates"][tr])
        es_cut = tdates[int(len(tdates) * 0.85)]
        fit = tr & ~np.isnat(ends) & (ends < es_cut)
        es = tr & (w["dates"] >= es_cut)
        if fit.sum() < 15 or es.sum() < 3:
            fit, es = tr, tr
        y_tr = apply_label_scaler(w["y"][tr], sd_y)
        P["zero-param rule"][te] = S.z_last(w["Xs"][te]) @ G
        base = S.fit_linear_imposed(w["Xs"][tr], y_tr, fm[tr], G)
        P["linear sign imposed"][te] = base(w["Xs"][te])
        P["sign-fixed magnitudes (w ≥ 0)"][te] = sign_fixed(w["Xs"][tr], y_tr, fm[tr], G)(w["Xs"][te])
        runs = {v: [] for v in VARIANTS[1:7] + ["rule + AGCRN residual"]}
        for seed in SEEDS:
            p, _ = fit_torch(lambda mu, sd: PassThrough(G, mu, sd, allc, train=False),
                             ws, G, fit, es, te, sd_y, seed, allc, True)
            runs["pass-through, untrained"].append(p)
            p, sc = fit_torch(lambda mu, sd: PassThrough(G, mu, sd, allc, train=True),
                              ws, G, fit, es, te, sd_y, seed, allc, True)
            runs["pass-through, trained"].append(p)
            scales.append(sc)
            for name, cols, hist in (("AGCRN frozen (as scoped.py)", allc, True),
                                     ("AGCRN frozen, no history", allc, False),
                                     ("AGCRN frozen, surprise only", zc, True),
                                     ("AGCRN frozen, neither", zc, False)):
                p, _ = fit_torch(lambda mu, sd, c=cols: agcrn_frozen(G, len(c))(), ws, G, fit, es,
                                 te, sd_y, seed, cols, hist)
                runs[name].append(p)
            # residual: AGCRN (zero-init, so it starts at the base) fits y − base
            resid = apply_label_scaler(w["y"], sd_y) - base(w["Xs"])
            p, _ = fit_torch(lambda mu, sd: agcrn_frozen(G, F)(), ws, G, fit, es, te, sd_y, seed,
                             allc, True, y_override=resid)
            runs["rule + AGCRN residual"].append(p + base(w["Xs"][te]))
        for v, r in runs.items():
            P[v][te] = np.mean(r, 0)
    return w, tm, P, scales


for k in ("imm", "settle"):
    for scope in ("t2", "t2wf", "hub"):
        w, tm, P, scales = run(k, scope)
        sd_y = label_sd(k)
        y = w["y"] / sd_y
        cell_t, _ = np.nonzero(tm)
        S.rule(f"label '{k}', scope '{scope}' — {int(tm.sum())} firing test cells; "
               f"trained pass-through scale per fold×seed: min {np.min(scales):+.3f}, "
               f"median {np.median(scales):+.3f}, share < 0 {np.mean(np.array(scales) < 0):.2f}")
        head = f"{'variant':34} {'R²':>8} {'bal acc':>7} {'AUC':>6}"
        if k == "settle":
            head += f" | {'gross ¢':>8} {'vs random side ¢ [95% CI]':>28}"
        print(head)
        for v in VARIANTS:
            p = P[v][tm]
            r2, bal, auc = S.scores(y[tm], p)
            r2s = "–" if v in ("zero-param rule", "pass-through, untrained") else f"{r2:+.4f}"
            line = f"{v:34} {r2s:>8} {bal:>7.3f} {auc:>6.3f}"
            if k == "settle":
                g, ex, ci, _ = S.pnl(w["y"][tm], p, w["dates"][cell_t])
                line += f" | {g:>+8.2f} {ex:>+8.2f} [{ci[0]:+6.2f},{ci[1]:+6.2f}]"
            print(line, flush=True)
        same = np.allclose(P["zero-param rule"][tm], P["pass-through, untrained"][tm], atol=1e-4)
        print(f"pass-through (untrained) reproduces the rule on every test cell: {same}")
