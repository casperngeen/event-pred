"""Checklist step 3 — apply the fixes cumulatively, re-run the honest comparison.

Same walk-forward harness as ``scripts/train_agcrn.py`` (8 expanding IS folds,
PURGE_DAYS gap, early stop on the last 15% of train), same target (k=3
per-series z-scored Δ implied_mean), 3 seeds. In-sample only. Run from
event-pred/:

    venv/bin/python analysis/agcrn_checklist_2026_09/step3_fixes.py \
        > analysis/agcrn_checklist_2026_09/out/step3_fixes.txt

Ladder (each AGCRN rung adds one fix to the rung above; minimal config):

  A0  as on main                         legacy masking, learned E / shared-MLP E
  A1  + zero-init head                   starts at predict-zero (step 1d/1e)
  A2  + per-step masking                 padding cannot leak (step 1a)
  A3  + series-id (event-type) embedding hybrid E = MLP([x_t, e_series])
  A4  + event-study prior                logits + λ|ρ̂|, all searched Stage-1 pairs
  A5  + top-3 sparsification             stops uniform averaging (step 2a)
  A6  + cut capacity                     shared weights, dropout 0.2, wd 1e-3
  A7  + surprise on the triggering node  s_pit channel (step 2e)
  and: frozen Stage-1 graph, orientation-fixed; default size at A2.

Linear rungs on the same folds, plus a surprise-aware ridge for A7's inputs.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN, LinearBaseline, build_labels, build_tensor, count_params, sequence_windows
from stg.models.baselines import _ridge, _stage1_in_edges, make_features
from stg.models.tensors import add_surprise_channel, global_label_sd
from stg.models.train import _fold_cuts, run_linear, run_torch, scale_diagnostics

NP = "artifacts/panels/node_panel_event.parquet"
SP = "artifacts/panels/surprise_panel.parquet"
K, L, SEEDS = 3, 12, (0, 1, 2)
torch.set_num_threads(4)

pt = build_tensor(pl.read_parquet(NP))
pts = add_surprise_channel(pt, pl.read_parquet(SP))
Y, lm = build_labels(pt, k=K, kind="belief_z")
win = sequence_windows(pt, Y, lm, L=L)
win_s = sequence_windows(pts, Y, lm, L=L)
lsd = global_label_sd(Y, lm, "belief_z")
S1_all = _stage1_in_edges(pt.nodes, survivors_only=False)
S1_bh = _stage1_in_edges(pt.nodes, survivors_only=True)


class SurpriseRidge(LinearBaseline):
    """own_momentum + own surprise + ρ̂-weighted in-neighbour surprise."""

    def __init__(self, nodes):
        super().__init__("own_momentum", nodes)

    def _feats(self, win):
        base = make_features(win, self.nodes, "own_momentum")
        s = win["Xs"][:, -1, :, -1] * win["Ms"][:, -1]
        return np.concatenate([base, s[..., None], (s @ S1_all)[..., None]], axis=-1)

    def fit(self, win):
        F, ym = self._feats(win), win["ym"]
        self.coef_ = _ridge(F[ym], win["y"][ym], self.lam)
        return self

    def predict(self, win):
        F = self._feats(win)
        flat = F.reshape(-1, F.shape[-1])
        return (np.column_stack([np.ones(len(flat)), flat]) @ self.coef_).reshape(F.shape[:2])


def agcrn(F=pt.F, **kw):
    cfg = dict(hidden=16, d_emb=2) | kw
    return lambda: AGCRN(pt.N, F, n_horizons=1, **cfg)


STEP = dict(masking="per_step", zero_head=True)
A3 = dict(STEP, embedding="hybrid")
A4 = dict(A3, prior_adj=S1_all, prior_lambda=2.0)
A5 = dict(A4, topk=3)
A6 = dict(A5, weights="shared", dropout=0.2)
LADDER = [
    # name, factory, window dict, weight decay
    ("A0 main / learned E", agcrn(), win, 1e-4),
    ("A0 main / shared-MLP E", agcrn(embedding="shared_mlp"), win, 1e-4),
    ("A1 +zero head / learned", agcrn(zero_head=True), win, 1e-4),
    ("A1 +zero head / shared-MLP", agcrn(embedding="shared_mlp", zero_head=True), win, 1e-4),
    ("A2 +per-step mask / learned", agcrn(**STEP), win, 1e-4),
    ("A2 +per-step mask / shared-MLP", agcrn(embedding="shared_mlp", **STEP), win, 1e-4),
    ("A3 +series-id emb (hybrid)", agcrn(**A3), win, 1e-4),
    ("A4 +prior λ=2", agcrn(**A4), win, 1e-4),
    ("A5 +top-3", agcrn(**A5), win, 1e-4),
    ("A6 +shared W, dropout, wd", agcrn(**A6), win, 1e-3),
    ("A7 +surprise channel", agcrn(F=pts.F, **A6), win_s, 1e-3),
    ("frozen Stage-1 (fixed orient.)", agcrn(adjacency="stage1", stage1_adj=S1_bh, **STEP), win, 1e-4),
    ("A2 default size / learned", agcrn(hidden=64, d_emb=10, **STEP), win, 1e-4),
]


def dir_acc_nonzero(r):
    y, p, m = r["_y"], r["_preds"], r["_mask"]
    sel = m & (y != 0) & (p != 0)
    return float((np.sign(y[sel]) == np.sign(p[sel])).mean()) if sel.any() else np.nan


def fold_r2(r, w):
    """Per-fold R² vs zero of the (seed-averaged) predictions."""
    cuts = _fold_cuts(w["dates"], 8)
    out = []
    for i in range(8):
        te = (w["dates"] >= cuts[i]) & (w["dates"] < cuts[i + 1])
        m = r["_mask"] & te[:, None]
        if m.sum() == 0:
            continue
        yt, yp = r["_y"][m], r["_preds"][m]
        out.append(1 - ((yt - yp) ** 2).sum() / (yt ** 2).sum())
    return np.array(out)


print(f"k={K}, L={L}, 8 folds, seeds {SEEDS}; labels in per-series z units")
print("R² vs 0 and dir acc: seed-mean of per-seed metrics (as the main report).")
print("dir≠0: directional accuracy excluding zero labels. pred_sd / corr / R²@α*:")
print("on seed-averaged predictions (R²@α* uses eval labels: an upper bound).")
print("folds>0 / folds>lin: folds where the model beats zero / linear own_momentum.\n")
hdr = (f"{'model':34} {'params':>7} {'R² vs 0':>15} {'dir':>6} {'dir≠0':>6} {'pred_sd':>8} "
       f"{'corr':>7} {'R²@α*':>8} {'folds>0':>8} {'folds>lin':>9}")
print(hdr); print("-" * len(hdr))

rows = []
ref = None
for name, f, w in [("linear zero", lambda: LinearBaseline("zero", pt.nodes), win),
                   ("linear own_momentum", lambda: LinearBaseline("own_momentum", pt.nodes), win),
                   ("linear neighbour_stage1", lambda: LinearBaseline("neighbour_stage1", pt.nodes), win),
                   ("linear +surprise", lambda: SurpriseRidge(pt.nodes), win_s)]:
    r = run_linear(f, w, lsd)
    s = scale_diagnostics(r["_y"], r["_preds"], r["_mask"])
    fr = fold_r2(r, w)
    if name == "linear own_momentum":
        ref = fr
    rows.append(dict(model=name, r2=r["r2_vs_zero"], **s))
    print(f"{name:34} {'–':>7} {r['r2_vs_zero']:>+15.4f} {r['dir_acc']:>6.3f} "
          f"{dir_acc_nonzero(r):>6.3f} {s['pred_sd']:>8.3f} {s['corr']:>+7.3f} "
          f"{s['r2_at_alpha']:>+8.4f} {(fr > 0).sum():>5}/{len(fr)} "
          f"{'' if ref is None or name == 'linear own_momentum' else f'{(fr > ref).sum()}/{len(fr)}':>9}",
          flush=True)

for name, f, w, wd in LADDER:
    t0 = time.time()
    r = run_torch(f, w, lsd, seeds=SEEDS, wd=wd)
    s = scale_diagnostics(r["_y"], r["_preds"], r["_mask"])
    fr = fold_r2(r, w)
    best = [h["best_epoch"] for h in r["_hist"]]
    rows.append(dict(model=name, r2=r["r2_vs_zero"], r2_sd=r["r2_vs_zero_sd"], **s))
    print(f"{name:34} {count_params(f()):>7} {r['r2_vs_zero']:>+8.4f} ±{r['r2_vs_zero_sd']:.3f} "
          f"{r['dir_acc']:>6.3f} {dir_acc_nonzero(r):>6.3f} {s['pred_sd']:>8.3f} "
          f"{s['corr']:>+7.3f} {s['r2_at_alpha']:>+8.4f} {(fr > 0).sum():>5}/{len(fr)} "
          f"{(fr > ref).sum():>6}/{len(fr)}"
          f"   [{time.time() - t0:.0f}s; best epoch median {int(np.median(best))}, "
          f"=0 in {np.mean(np.array(best) == 0):.0%} of fold-fits]", flush=True)

pl.DataFrame(rows).write_parquet("analysis/agcrn_checklist_2026_09/out/step3_fixes.parquet")
