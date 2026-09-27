"""Checklist step 6 — is Ã's agreement with Stage-1 more than chance?

Step 2c reported rank corr(Ã, |ρ̂|) of +0.03…+0.32 over the searched pairs
with no null. This puts two nulls under it. Last walk-forward fold, minimal
config, 3 seeds (Ã averaged over seeds), same training as step 2. In-sample
only. Run from event-pred/:

    venv/bin/python analysis/agcrn_checklist_2026_09/step6_recovery_null.py \
        > analysis/agcrn_checklist_2026_09/out/step6_recovery_null.txt

Ã is read per held-out window at the last step (row = receiver), and an edge
source→target is averaged only over windows where **both** nodes are active,
so a pair's weight does not depend on how often it is co-active.

  node-label null   Relabel ρ̂'s endpoint nodes (A_π[π(i), π(j)] = A[i, j]),
                    then score Ã against the permuted searched set. Asks: does
                    Ã line up with *these* markets' structure rather than a
                    same-shaped structure on other markets?
  value null        Keep the searched set, shuffle |ρ̂| across it. Asks: within
                    the pairs Stage-1 searched, does Ã rank strong edges above
                    weak ones? This one is immune to Ã and ρ̂ both favouring
                    frequently-traded nodes.

Also the BH survivors' mean percentile among all co-active off-diagonal Ã
entries, under the node-label null. The prior variant (λ|ρ̂| in the logits)
is a positive control: it should pass both nulls.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl
import torch
from scipy.stats import rankdata

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN, build_labels, build_tensor, sequence_windows
from stg.models.baselines import _stage1_in_edges
from stg.models.tensors import apply_feature_scaler, apply_label_scaler, fit_feature_scaler, global_label_sd
from stg.models.train import PURGE, _fold_cuts, fit_fold

NP = "artifacts/panels/node_panel_event.parquet"
K, L, SEEDS, N_PERM = 3, 12, (0, 1, 2), 5000
torch.set_num_threads(4)

pt = build_tensor(pl.read_parquet(NP))
Y, lm = build_labels(pt, k=K, kind="belief_z")
win = sequence_windows(pt, Y, lm, L=L)
lsd = global_label_sd(Y, lm, "belief_z")
S1_all = _stage1_in_edges(pt.nodes, survivors_only=False)     # [trigger, target]
S1_bh = _stage1_in_edges(pt.nodes, survivors_only=True)
np.fill_diagonal(S1_all, 0.0); np.fill_diagonal(S1_bh, 0.0)

MIN = dict(hidden=16, d_emb=2)
STEP = dict(masking="per_step", zero_head=True)
VARIANTS = {
    "per_step/learned": dict(embedding="learned", **STEP),
    "per_step/shared_mlp": dict(embedding="shared_mlp", **STEP),
    "per_step/hybrid": dict(embedding="hybrid", **STEP),
    "per_step/hybrid+prior (control)": dict(embedding="hybrid", prior_adj=S1_all, prior_lambda=2.0, **STEP),
}

dates = win["dates"]
cuts = _fold_cuts(dates, 8)
tr = dates < (cuts[-2] - PURGE)
te = dates >= cuts[-2]
tdates = np.sort(dates[tr]); es_cut = tdates[int(len(tdates) * 0.85)]
fm, em = tr & (dates < es_cut), tr & (dates >= es_cut)
mu, sd = fit_feature_scaler(win["Xs"][fm], win["Ms"][fm])


def tensors(sel):
    return (torch.tensor(apply_feature_scaler(win["Xs"][sel], mu, sd)),
            torch.tensor(win["Ms"][sel]),
            torch.tensor(apply_label_scaler(win["y"][sel], lsd), dtype=torch.float32),
            torch.tensor(win["ym"][sel]))


Tf, Te, Tt = tensors(fm), tensors(em), tensors(te)
act = win["Ms"][te][:, -1]                                       # (B, N)


@torch.no_grad()
def per_sample_adj(mdl: AGCRN, X, M) -> np.ndarray:
    mdl.eval()
    x_t, a = X[:, -1], M[:, -1]
    m = a.float()[..., None]
    A = mdl._adjacency(mdl._embed(torch.cat([x_t * m, m], dim=-1)), a)
    return (A.expand(X.shape[0], -1, -1) if A.dim() == 2 else A).numpy()


def edge_matrix(A: np.ndarray) -> np.ndarray:
    """[source, target] mean Ã over windows where both are active; nan if never."""
    co = act[:, :, None] & act[:, None, :]                       # [b, target, source]
    s = np.where(co, A, 0.0).sum(0)
    n = co.sum(0)
    with np.errstate(invalid="ignore"):
        return (s / np.where(n > 0, n, np.nan)).T


ENDS = np.nonzero((S1_all != 0).any(0) | (S1_all != 0).any(1))[0]


def relabel(A: np.ndarray, perm: np.ndarray) -> np.ndarray:
    B = np.zeros_like(A)
    B[np.ix_(perm, perm)] = A
    return B


def draw_perm(rng) -> np.ndarray:
    """Permute the searched graph's endpoint nodes (survivors are a subset)."""
    perm = np.arange(pt.N)
    perm[ENDS] = rng.permutation(ENDS)
    return perm


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def score(E: np.ndarray, R: np.ndarray, Rbh: np.ndarray):
    """rank corr over R's searched (and co-active) pairs; survivors' mean percentile."""
    off = ~np.eye(pt.N, dtype=bool) & np.isfinite(E)
    sel = (R != 0) & off
    rc = spearman(E[sel], np.abs(R[sel])) if sel.sum() > 3 else np.nan
    pool = np.sort(E[off])
    sv = (Rbh != 0) & off
    pct = float(np.mean(np.searchsorted(pool, E[sv]) / len(pool))) if sv.any() else np.nan
    return rc, pct, int(sel.sum())


print(f"last fold: fit {fm.sum()} / es {em.sum()} / test {te.sum()} windows; seeds {SEEDS}")
print(f"searched ρ̂ pairs {int((S1_all != 0).sum())}, BH survivors "
      f"{[(pt.nodes[i], pt.nodes[j]) for i, j in zip(*np.nonzero(S1_bh))]}")
print(f"nulls: {N_PERM} draws each; p one-sided (null ≥ real), +1 smoothed\n")
print(f"{'variant':32} {'pairs':>5} {'rank corr':>9} {'p node':>7} {'p value':>8} "
      f"{'surv pct':>9} {'p node':>7} {'seed rc range':>16}")

rng = np.random.default_rng(0)
for name, kw in VARIANTS.items():
    f = lambda kw=kw: AGCRN(pt.N, pt.F, n_horizons=1, **MIN, **kw)
    Es, rcs = [], []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        mdl, _ = fit_fold(f, Tf, Te)
        E = edge_matrix(per_sample_adj(mdl, Tt[0], Tt[1]))
        Es.append(E)
        rcs.append(score(E, S1_all, S1_bh)[0])
    E = np.nanmean(np.stack(Es), 0)
    rc, pct, n = score(E, S1_all, S1_bh)

    node_rc, node_pct = [], []
    for _ in range(N_PERM):
        perm = draw_perm(rng)
        r_, p_, _ = score(E, relabel(S1_all, perm), relabel(S1_bh, perm))
        node_rc.append(r_); node_pct.append(p_)
    off = ~np.eye(pt.N, dtype=bool) & np.isfinite(E)
    sel = (S1_all != 0) & off
    e_sel, r_sel = E[sel], np.abs(S1_all[sel])
    val_rc = [spearman(e_sel, rng.permutation(r_sel)) for _ in range(N_PERM)]

    node_rc, node_pct, val_rc = map(lambda a: np.array(a)[np.isfinite(a)], (node_rc, node_pct, val_rc))
    p_node = (1 + (node_rc >= rc).sum()) / (1 + len(node_rc))
    p_val = (1 + (val_rc >= rc).sum()) / (1 + len(val_rc))
    p_pct = (1 + (node_pct >= pct).sum()) / (1 + len(node_pct))
    print(f"{name:32} {n:>5} {rc:>+9.3f} {p_node:>7.3f} {p_val:>8.3f} {pct:>9.2f} {p_pct:>7.3f} "
          f"{min(rcs):>+7.3f}…{max(rcs):+.3f}", flush=True)
    print(f"{'':32} null medians: node rc {np.median(node_rc):+.3f} "
          f"[95% {np.quantile(node_rc, .95):+.3f}], value rc {np.median(val_rc):+.3f} "
          f"[95% {np.quantile(val_rc, .95):+.3f}], surv pct {np.median(node_pct):.2f}", flush=True)
