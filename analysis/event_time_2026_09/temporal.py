#!/usr/bin/env python
"""Temporal encoding in calendar time: a Bayesian STG, and the same fixes on AGCRN.

    venv/bin/python -W ignore analysis/event_time_2026_09/temporal.py \
        > analysis/event_time_2026_09/out/temporal.txt       # needs build_panel.py first

The event-time panel's steps are release instants, irregularly spaced (median
gap one week, a 6-instant window spans ~54 days). The AGCRN's GRU sees only
their order, and the Bayesian model (``bayes.py``) has no temporal part at
all. Two calendar-time encodings, both computed from releases strictly before
the instant (no look-ahead):

  1  lag windows   per series, the sum of its surprises released in the last
                   24 h, and in the 1–7 days before that
  2  decay         per series, Σ exp(−Δt / h)·z over its releases in the last
                   30 days, h = 1 day and 7 days

A  Bayesian STG (soft-sign hierarchical, ``bayes.fit``): the spatial model
   (142 edges pooled within 15 channels) plus theory-signed channel signals of
   the lagged / decayed surprises (15 per window or decay constant, pooled within
   it), the target's own recent surprise (one coefficient per series, pooled), and
   a node-specific own-state block (p_lead, d_q50_7d, flow7: one coefficient
   per series × feature, pooled across series: AGCRN's node-adaptive weights with
   shrinkage). Ablation: spatial only, + each temporal block, + own state, full.

B  AGCRN (per-step masking, zero head, hidden 16, d_emb 2), adaptive graph and
   the frozen BH economic graph, with:
     base              the ``models.py`` configuration: 6-instant GRU, order only
     + calendar lags   encoding 1 and 2 (h = 7 d) as extra node inputs
     + lags, no GRU    the same inputs, window of 1 (no recurrence)
     + learned decay   an elapsed-time input and a GRU-D-style gate that shrinks
                       the hidden state by σ(b_n − softplus(w_n)·Δt) before each
                       step (per node, learned), plus the calendar lags

Walk-forward as ``models.py`` (8 folds, purged on label end), both labels.
Metrics: R² vs 0, balanced accuracy, AUC on all labelled test cells and on
those where the BH-channel signal fires.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN
from stg.models.train import PURGE, _fold_cuts, _label_end, run_torch

sys.path.insert(0, "analysis/event_time_2026_09")
import bayes  # noqa: E402
from _panel import (END, FEATS, G_all, G_bh, L, M, N, X, Y, Z, instants,  # noqa: E402
                    label_sd)

torch.set_num_threads(4)
SEEDS = (0, 1, 2)
HR, DAY = np.timedelta64(1, "h"), np.timedelta64(1, "D")
T = len(instants)
OWNF = ["p_lead", "d_q50_7d", "flow7"]


# ------------------------------------------------------------------ calendar-time features
def past_sums():
    """(T, N) per-series sums of surprises released strictly before each instant."""
    R24, R7, D1, D7 = (np.zeros((T, N)) for _ in range(4))
    for i in range(T):
        dt = (instants[i] - instants[:i]) / HR                     # hours since earlier instants
        if not len(dt):
            continue
        z = Z[:i]
        R24[i] = z[(dt > 0) & (dt <= 24)].sum(0)
        R7[i] = z[(dt > 24) & (dt <= 7 * 24)].sum(0)
        w30 = (dt > 0) & (dt <= 30 * 24)
        D1[i] = (np.exp(-dt[w30] / 24)[:, None] * z[w30]).sum(0)
        D7[i] = (np.exp(-dt[w30] / (7 * 24))[:, None] * z[w30]).sum(0)
    return {"lag 24h": R24, "lag 1-7d": R7, "decay 1d": D1, "decay 7d": D7}


PAST = past_sums()
DT = np.r_[0.0, np.diff(instants) / HR]                              # hours since the previous instant
print(f"instants with a release in the previous 24 h: {(np.abs(PAST['lag 24h']).sum(1) > 0).mean():.0%}; "
      f"in the 1–7 days before: {(np.abs(PAST['lag 1-7d']).sum(1) > 0).mean():.0%}")


# ------------------------------------------------------------------ A: Bayesian STG
def inst_index(dates):
    return np.searchsorted(instants, dates)


def design(k, w, rows, lsd, blocks, own_stats=None):
    """Spatial edge block plus the requested temporal blocks, for labelled cells
    of windows ``rows``. Returns X, y, sign, gidx, meta."""
    XE, y, j, b = bayes.cells(w, rows, lsd)
    ti = inst_index(w["dates"][j])
    cols, sign, gid = [XE], [bayes.SIGN], [bayes.GIDX]
    ng = len(bayes.GROUPS)
    for name in blocks:
        if name in PAST:                                            # 15 theory-signed channel signals
            S = PAST[name]
            B = np.zeros((len(y), len(bayes.GROUPS)))
            for e, (a, bb) in enumerate(bayes.EDGES):
                m = b == bb
                B[m, bayes.GIDX[e]] += bayes.SIGN[e] * S[ti[m], a]
            own = S[ti, b][:, None] * (np.arange(N)[None, :] == b[:, None])    # own recent surprise
            cols += [B, own]
            sign += [np.ones(B.shape[1]), np.ones(N)]
            gid += [np.full(B.shape[1], ng), np.full(N, ng + 1)]
            ng += 2
        elif name == "own state":                                    # node-adaptive, pooled per feature
            F = np.stack([w["Xs"][j, -1, b, FEATS.index(f)] for f in OWNF], 1)
            mu, sd = own_stats if own_stats is not None else (F.mean(0), F.std(0) + 1e-9)
            F = np.clip((F - mu) / sd, -4, 4)
            for f in range(len(OWNF)):
                cols.append(F[:, [f]] * (np.arange(N)[None, :] == b[:, None]))
                sign.append(np.ones(N)); gid.append(np.full(N, ng)); ng += 1
    return np.hstack(cols), y, np.concatenate(sign), np.concatenate(gid), (j, b)


VARIANTS = {
    "spatial only (bayes.py)": [],
    "+ lag windows (24h, 1-7d)": ["lag 24h", "lag 1-7d"],
    "+ decay h = 1d": ["decay 1d"],
    "+ decay h = 7d": ["decay 7d"],
    "+ own state (node-adaptive)": ["own state"],
    "full: lags + own state": ["lag 24h", "lag 1-7d", "own state"],
    "full: decay 7d + own state": ["decay 7d", "own state"],
}


def run_bayes(k):
    from _panel import windows
    w = windows(k)
    lsd = label_sd(k)
    dates, ends = w["dates"], _label_end(w)
    cuts = _fold_cuts(dates, 8)
    P = {v: [] for v in VARIANTS}
    Ys, FIRE, rh = [], [], []
    for i in range(8):
        tr = ends < (cuts[i] - PURGE)
        te = (dates >= cuts[i]) & (dates < cuts[i + 1])
        if tr.sum() < 30 or te.sum() == 0:
            continue
        for v, blocks in VARIANTS.items():
            stats = None
            if "own state" in blocks:                               # scale on the training cells only
                _, _, _, _, (jj, bb) = design(k, w, tr, lsd, [])
                Fraw = np.stack([w["Xs"][jj, -1, bb, FEATS.index(f)] for f in OWNF], 1)
                stats = (Fraw.mean(0), Fraw.std(0) + 1e-9)
            Xtr, ytr, sgn, gidx, _ = design(k, w, tr, lsd, blocks, stats)
            Xte, yte, _, _, (jte, bte) = design(k, w, te, lsd, blocks, stats)
            d = bayes.fit(Xtr, ytr, "soft", sign=sgn, gidx=gidx, burn=500, draws=1000)
            rh.append(d["rhat_max"])
            P[v].append(d["alpha"].mean() + Xte @ d["beta"].mean(0))
            if v == "spatial only (bayes.py)":
                Ys.append(yte)
                z = w["Xs"][:, -1, :, FEATS.index("z")] * (w["Xs"][:, -1, :, FEATS.index("released")] > 0)
                FIRE.append((z @ G_bh)[jte, bte] != 0)
        print(f"  [{k}] fold {i} done", flush=True)
    y, fire = np.concatenate(Ys), np.concatenate(FIRE)
    print(f"\nA  BAYESIAN STG — label '{k}': {len(y)} test cells ({int(fire.sum())} BH-firing); "
          f"max split-R̂ {np.nanmax(rh):.3f}")
    print(f"{'variant':32} {'R²':>8} {'bal':>6} {'AUC':>6} | {'firing R²':>9} {'bal':>6} {'AUC':>6}")
    for v in VARIANTS:
        p = np.concatenate(P[v])
        a, f = bayes.scores(y, p), bayes.scores(y[fire], p[fire])
        print(f"{v:32} {a[0]:>+8.4f} {a[1]:>6.3f} {a[2]:>6.3f} | {f[0]:>+9.4f} {f[1]:>6.3f} {f[2]:>6.3f}")
    # full-sample group means of the full lag model
    allrows = np.ones(len(dates), bool)
    _, _, _, _, (jj, bb) = design(k, w, allrows, lsd, [])
    Fraw = np.stack([w["Xs"][jj, -1, bb, FEATS.index(f)] for f in OWNF], 1)
    blocks = ["lag 24h", "lag 1-7d", "decay 7d", "own state"]
    Xa, ya, sgn, gidx, _ = design(k, w, allrows, lsd, blocks, (Fraw.mean(0), Fraw.std(0) + 1e-9))
    d = bayes.fit(Xa, ya, "soft", sign=sgn, gidx=gidx, burn=500, draws=1000)
    names = (["channel " + g for g in bayes.GROUPS]
             + [f"{b_} {part}" for b_ in ("lag 24h", "lag 1-7d", "decay 7d")
                for part in ("(channels)", "(own series)")]
             + [f"own state: {f}" for f in OWNF])
    print(f"\nfull-sample group means (lags, decay 7d and own state together; max split-R̂ "
          f"{d['rhat_max']:.3f}); + = theory direction for channel blocks")
    for g in range(len(bayes.GROUPS), int(gidx.max()) + 1):
        mu = d["mu"][:, g]
        lo, hi = np.percentile(mu, [5, 95])
        print(f"  {names[g]:34} μ {mu.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  P(μ>0) {(mu > 0).mean():.2f}")


# ------------------------------------------------------------------ B: AGCRN
class DecayAGCRN(AGCRN):
    """Per-step AGCRN whose hidden state is shrunk by a learned, per-node gate of
    the elapsed time before each step: h ← σ(b_n − softplus(w_n)·Δt)·h. The
    (scaled) elapsed time is read from input feature ``dt_index``."""

    def __init__(self, *a, dt_index: int, **kw):
        super().__init__(*a, **kw)
        self.dt_index = dt_index
        self.dw = nn.Parameter(torch.zeros(self.n_nodes))
        self.db = nn.Parameter(torch.full((self.n_nodes,), 3.0))

    def forward(self, seq, seq_mask):
        B, Lw, Nn, _ = seq.shape
        h = torch.zeros(B, Nn, self.hidden, device=seq.device)
        for t in range(Lw):
            x_t = seq[:, t]
            gate = torch.sigmoid(self.db - TF.softplus(self.dw) * x_t[..., self.dt_index])
            h = gate[..., None] * h
            active = seq_mask[:, t]
            m = active.float()[..., None]
            x_t = torch.cat([x_t * m, m], dim=-1)
            E = self._embed(x_t)
            A = self._adjacency(E, active)
            h = m * self.cell(x_t, h, E, A) + (1 - m) * h
        return self.head(self.drop(h))


def windows_aug(k, extra, Lw):
    """``_panel.windows`` with extra node features appended (T, N, f) and window Lw."""
    Xa = np.concatenate([X] + [e[..., None].astype(np.float32) for e in extra], -1)
    idx = [t for t in range(L - 1, T) if np.isfinite(Y[k][t]).any()]
    ym = np.stack([np.isfinite(Y[k][t]) for t in idx])
    ends = np.array([END[k][t][ym[j]].max() for j, t in enumerate(idx)])
    return {"Xs": np.stack([Xa[t - Lw + 1:t + 1] for t in idx]),
            "Ms": np.stack([M[t - Lw + 1:t + 1] for t in idx]),
            "y": np.nan_to_num(np.stack([Y[k][t] for t in idx])), "ym": ym,
            "dates": instants[idx], "label_end": ends.astype("datetime64[us]")}


def run_agcrn(k):
    lsd = label_sd(k)
    lagf = [PAST["lag 24h"], PAST["lag 1-7d"], PAST["decay 7d"]]
    dtf = np.repeat(np.log1p(DT)[:, None], N, 1)
    setups = {
        "base (6-instant GRU, order only)": ([], L, None),
        "+ calendar lags": (lagf, L, None),
        "+ calendar lags, no GRU (window 1)": (lagf, 1, None),
        "+ calendar lags + learned decay": (lagf + [dtf], L, "decay"),
    }
    print(f"\nB  AGCRN — label '{k}'")
    print(f"{'graph':10} {'temporal encoding':36} {'R²':>8} {'bal':>6} {'AUC':>6} | "
          f"{'firing R²':>9} {'bal':>6} {'AUC':>6}")
    for gname, G in (("adaptive", None), ("econ BH", G_bh)):
        for sname, (extra, Lw, kind) in setups.items():
            w = windows_aug(k, extra, Lw)
            F = w["Xs"].shape[-1]
            cfg = dict(hidden=16, d_emb=2, masking="per_step", zero_head=True, n_horizons=1,
                       embedding="hybrid")
            if G is not None:
                cfg |= dict(adjacency="stage1", stage1_adj=G)
            if kind == "decay":
                make = lambda F=F, cfg=cfg: DecayAGCRN(N, F, dt_index=F - 1, **cfg)   # noqa: E731
            else:
                make = lambda F=F, cfg=cfg: AGCRN(N, F, **cfg)                         # noqa: E731
            t0 = time.time()
            r = run_torch(make, w, lsd, seeds=SEEDS)
            yv, pv, mv = r["_y"], r["_preds"], r["_mask"]
            z = w["Xs"][:, -1, :, FEATS.index("z")] * (w["Xs"][:, -1, :, FEATS.index("released")] > 0)
            fm = mv & ((z @ G_bh) != 0)
            a, f = bayes.scores(yv[mv], pv[mv]), bayes.scores(yv[fm], pv[fm])
            print(f"{gname:10} {sname:36} {a[0]:>+8.4f} {a[1]:>6.3f} {a[2]:>6.3f} | "
                  f"{f[0]:>+9.4f} {f[1]:>6.3f} {f[2]:>6.3f}  [{time.time() - t0:.0f}s]", flush=True)


for k in ("imm", "settle"):
    run_bayes(k)
    run_agcrn(k)
