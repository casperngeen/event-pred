#!/usr/bin/env python
"""Can AGCRN represent the negative rival edges once the neighbourhood is supplied?

    venv/bin/python -W ignore analysis/sports_2026_09/signfix.py > analysis/sports_2026_09/out/signfix.txt

In ``graphs.py`` the standard AGCRN recovers the self edges but none of the 120
negative rival edges. Its adjacency softmax(ReLU(EEᵀ)) is non-negative, but the
conv weight on that support could be negative, so "rivals push you down" is
representable in principle. What blocks it is that the learned adjacency also
aggregates each node's *own* surprise, so a negative weight on it would flip
the self effect too. The signed variant (E_dst E_srcᵀ) removes the constraint
but must find which pairs are neighbours as well as their sign.

Two fixes, both supplying the neighbourhood from metadata (the schedule read
from game titles, fitted on the training days only; no self-loops) and
learning only the sign and size:

  AGCRN + schedule support     adjacency = the schedule graph, row-normalised
                               and frozen, as the second support [I, A_sched]:
                               the conv weight on it carries the rival sign,
                               the identity support carries the self sign
  AGCRN signed, schedule mask  signed adjacency E_dst E_srcᵀ, restricted to the
                               schedule graph's edges

compared with the standard adaptive AGCRN, the unrestricted signed AGCRN and
(A) frozen in. Same data, features, training and read-out as ``graphs.py``.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl
import torch

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN                                    # read-only use
from stg.models.tensors import apply_feature_scaler, fit_feature_scaler
from stg.models.train import fit_fold

sys.path.insert(0, "analysis/sports_2026_09")
import graphs as G  # noqa: E402

N, WIN, CUTS = G.N, G.WIN, ["2025-07-15", G.SEASON_END]


class MaskedSigned(AGCRN):
    """Signed directed adjacency E_dst E_srcᵀ, zero outside a supplied mask."""

    def __init__(self, *a, mask=None, **kw):
        super().__init__(*a, **kw)
        self.register_buffer("mask_in", torch.tensor(mask.T, dtype=torch.float32))  # [target, source]

    def _adjacency(self, E, active):
        A_in = (self.E_dst @ E.t()) * self.mask_in
        return A_in[None] * active[:, None, :].float()


def fit_read(make, m, seed):
    """Train like graphs.torch_graph and read the effective edges ∂ŷ_b/∂z_a."""
    torch.manual_seed(seed)
    WIN, YSD = G.fold_data(m)
    wi = np.isin(WIN["day"], G.days[m])
    d = WIN["day"][wi]
    es_cut = np.sort(d)[int(len(d) * 0.85)]
    fit, es = wi & (WIN["day"] < es_cut), wi & (WIN["day"] >= es_cut)
    mu, sd = fit_feature_scaler(WIN["Xs"][fit], WIN["Ms"][fit])

    def prep(s):
        return (torch.tensor(apply_feature_scaler(WIN["Xs"][s], mu, sd)), torch.tensor(WIN["Ms"][s]),
                torch.tensor(WIN["y"][s] / YSD, dtype=torch.float32), torch.tensor(WIN["ym"][s]))
    model, _ = fit_fold(make, prep(fit), prep(es))
    X, Mk = prep(wi)[:2]
    X = X.clone().requires_grad_(True)
    out = model(X, Mk).squeeze(-1)
    played, lab = WIN["Xs"][wi][:, -1, :, 1] > 0, WIN["ym"][wi]
    J = np.zeros((N, N))
    for b in range(N):
        g, = torch.autograd.grad(out[:, b].sum(), X, retain_graph=True)
        g = g[:, -1, :, 0].numpy()
        for a in range(N):
            sel = played[:, a] & lab[:, b]
            if sel.any():
                J[a, b] = g[sel, a].mean()
    return J


F = WIN["Xs"].shape[-1]
cfg = dict(hidden=16, d_emb=6, masking="per_step", zero_head=True, n_horizons=1)
rows = []
for cut in CUTS:
    m = G.days <= np.datetime64(cut)
    S = G.cooccur(cut)
    np.fill_diagonal(S, 0.0)
    mask = (S > 0).astype(float)
    makers = {
        "AGCRN adaptive (standard)": lambda: AGCRN(N, F, embedding="hybrid", **cfg),
        "AGCRN signed, unrestricted": lambda: AGCRN(N, F, embedding="learned", adjacency="signed", **cfg),
        "AGCRN + schedule support (frozen, unsigned)": lambda: AGCRN(N, F, embedding="hybrid",
                                                                    adjacency="stage1", stage1_adj=S, **cfg),
        "AGCRN signed, schedule mask": lambda: MaskedSigned(N, F, embedding="learned", adjacency="signed",
                                                            mask=mask, **cfg),
        "(A) frozen in AGCRN (check)": lambda: AGCRN(N, F, embedding="hybrid", adjacency="stage1",
                                                     stage1_adj=G.W_A, **cfg),
    }
    for name, make in makers.items():
        J = np.mean([fit_read(make, m, s) for s in G.SEEDS], 0)
        rows.append(dict(model=name, cut=cut, days=int(m.sum()), **G.score(J)))
    print(f"cut {cut} done ({int(m.sum())} days)", flush=True)

R = pl.DataFrame(rows)
pl.Config.set_tbl_rows(20); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(46)
print("\nEffective edges vs the known graph (self +, division rival −). hit_rival = share of\n"
      "the 120 rival edges with the right (negative) sign; sign_bal = mean of self and rival hits.\n")
print(R.select("model", "cut", "days", *[pl.col(c).round(3) for c in
                                         ("auroc", "auroc_rival", "hit_self", "hit_rival", "sign_bal")]))
R.write_parquet(f"{G.OUT}/signfix.parquet")
