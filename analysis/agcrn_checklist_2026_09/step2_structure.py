"""Checklist step 2 — structural reasons AGCRN may fail.

In-sample only. Run from event-pred/ (after nothing — standalone):

    venv/bin/python analysis/agcrn_checklist_2026_09/step2_structure.py \
        > analysis/agcrn_checklist_2026_09/out/step2_structure.txt

  2a  Row entropy of Ã       — is the graph near-uniform (every node averages all)?
  2b  Similarity vs influence — does Ã track feature similarity, and is it stable?
  2c  Recovery of Stage-1     — does Ã put weight on the event-study edges? (oriented)
  2d  Capacity                — parameters vs effective labels, per variant
  2e  Surprise input          — is there a directional channel to propagate at all?

Every variant is trained on the last walk-forward fold with the harness's
``fit_fold`` (200 epochs, patience 15) and Ã is read per held-out window, at
the window's last step. Minimal config (d_emb=2, hidden=16), seed 0.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN, build_labels, build_tensor, count_params, sequence_windows
from stg.models.baselines import _stage1_in_edges
from stg.models.capacity import within_snapshot_icc
from stg.models.tensors import (
    add_surprise_channel, apply_feature_scaler, apply_label_scaler,
    fit_feature_scaler, global_label_sd,
)
from stg.models.train import PURGE, _fold_cuts, fit_fold

NP = "artifacts/panels/node_panel_event.parquet"
SP = "artifacts/panels/surprise_panel.parquet"
K, L = 3, 12
torch.set_num_threads(4)


def rule(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


pt = build_tensor(pl.read_parquet(NP))
Y, lm = build_labels(pt, k=K, kind="belief_z")
win = sequence_windows(pt, Y, lm, L=L)
lsd = global_label_sd(Y, lm, "belief_z")
ni = {n: i for i, n in enumerate(pt.nodes)}
S1_all = _stage1_in_edges(pt.nodes, survivors_only=False)     # [trigger, target] signed rho
S1_bh = _stage1_in_edges(pt.nodes, survivors_only=True)

MIN = dict(hidden=16, d_emb=2)
VARIANTS = {
    "legacy/learned": dict(embedding="learned"),
    "legacy/shared_mlp": dict(embedding="shared_mlp"),
    "per_step/learned": dict(embedding="learned", masking="per_step", zero_head=True),
    "per_step/shared_mlp": dict(embedding="shared_mlp", masking="per_step", zero_head=True),
    "per_step/hybrid": dict(embedding="hybrid", masking="per_step", zero_head=True),
    "per_step/hybrid+prior": dict(embedding="hybrid", masking="per_step", zero_head=True,
                                  prior_adj=S1_all, prior_lambda=2.0),
    "per_step/hybrid+prior+top3": dict(embedding="hybrid", masking="per_step",
                                       zero_head=True, prior_adj=S1_all,
                                       prior_lambda=2.0, topk=3),
}

# ---- last fold
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
print(f"last fold: fit {fm.sum()} / es {em.sum()} / test {te.sum()} windows; "
      f"test from {dates[te].min()}")


@torch.no_grad()
def per_sample_adj(mdl: AGCRN, X, M) -> np.ndarray:
    """(B, N, N) Ã at the last step, row = receiver.

    Legacy builds one (N, N) Ã for the whole batch (mask = active anywhere in
    batch x window; shared-MLP E pooled over the batch). The harness predicts
    the held-out slice as one batch, so that shared Ã is what each window
    actually used — broadcast it.
    """
    mdl.eval()
    if mdl.masking == "legacy":
        return np.broadcast_to(mdl.learned_adjacency(X, M), (X.shape[0], mdl.n_nodes,
                                                              mdl.n_nodes)).copy()
    x_t, act = X[:, -1], M[:, -1]
    m = act.float()[..., None]
    x_t = torch.cat([x_t * m, m], dim=-1)
    A = mdl._adjacency(mdl._embed(x_t), act)
    return (A.expand(X.shape[0], -1, -1) if A.dim() == 2 else A).numpy()


def entropy_stats(A: np.ndarray, act: np.ndarray) -> dict:
    """Per (window, active receiver): normalised row entropy over the row's
    support, top-1 weight, and weight on padded senders."""
    H, top, pad, eff = [], [], [], []
    for b in range(A.shape[0]):
        a = act[b]
        n_act = a.sum()
        for i in np.nonzero(a)[0]:
            row = A[b, i]
            p = row[row > 0]
            if p.size == 0 or n_act < 2:
                continue
            p = p / p.sum()
            h = -(p * np.log(p)).sum()
            H.append(h / np.log(n_act))
            eff.append(np.exp(h))
            top.append(row.max() / max(row.sum(), 1e-12))
            pad.append(row[~a].sum() / max(row.sum(), 1e-12))
    return dict(H=np.mean(H), top1=np.mean(top), pad=np.mean(pad), eff_nb=np.mean(eff))


# =====================================================================
rule("2a  ROW ENTROPY OF Ã  (held-out windows, last step, active receivers)")
print("H/log(n_active) = 1 means uniform over the active nodes (every node just")
print("averages the graph). eff. nbrs = exp(H). pad mass = weight on PADDED senders")
print("(junk). 'init' is the same model before training.\n")
print(f"{'variant':28} {'':6} {'H/logn':>7} {'eff nbrs':>9} {'top-1 w':>8} "
      f"{'pad mass':>9} {'ISMPMI col':>11} {'Δ from init':>12} {'best ep':>8}")
act_t = win["Ms"][te][:, -1]
trained: dict[str, tuple] = {}
for name, kw in VARIANTS.items():
    f = lambda kw=kw: AGCRN(pt.N, pt.F, n_horizons=1, **MIN, **kw)
    torch.manual_seed(0)
    A0 = per_sample_adj(f(), Tt[0], Tt[1])
    torch.manual_seed(0)
    mdl, h = fit_fold(f, Tf, Te)
    A1 = per_sample_adj(mdl, Tt[0], Tt[1])
    trained[name] = (mdl, A1)
    rel = np.abs(A1 - A0).sum() / max(np.abs(A0).sum(), 1e-12)
    for tag, A in (("init", A0), ("train", A1)):
        s = entropy_stats(A, act_t)
        col = A[:, :, ni["ISMPMI"]].sum() / A.sum()
        extra = f"{rel:>12.1%} {h['best_epoch']:>8}" if tag == "train" else ""
        print(f"{name if tag == 'init' else '':28} {tag:6} {s['H']:>7.3f} {s['eff_nb']:>9.2f} "
              f"{s['top1']:>8.3f} {s['pad']:>9.1%} {col:>11.1%} {extra}", flush=True)
print(f"\nmean active nodes per held-out window (last step): {act_t.sum(1).mean():.1f}; "
      f"ISMPMI active in {act_t[:, ni['ISMPMI']].mean():.0%} of them")

# =====================================================================
rule("2b  SIMILARITY vs INFLUENCE, AND STABILITY OVER TIME")
print("corr(Ã_ij, cos-sim of i,j's scaled features at that step), over active")
print("off-diagonal pairs: high = Ã is a feature-similarity kernel. CV = sd/mean of")
print("Ã_ij across held-out windows, averaged over pairs active together in ≥10")
print("windows: high = the 'relation' is re-drawn every snapshot, not learned.\n")
Xt = Tt[0][:, -1].numpy()
print(f"{'variant':28} {'corr w/ sim':>12} {'CV over time':>13}")
for name, (_, A) in trained.items():
    a_v, s_v = [], []
    pair_vals: dict = {}
    for b in range(A.shape[0]):
        idx = np.nonzero(act_t[b])[0]
        if idx.size < 2:
            continue
        Z = Xt[b, idx]
        Zn = Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-8, None)
        C = Zn @ Zn.T
        for u, i in enumerate(idx):
            for v, j in enumerate(idx):
                if i != j:
                    a_v.append(A[b, i, j]); s_v.append(C[u, v])
                    pair_vals.setdefault((i, j), []).append(A[b, i, j])
    cv = [np.std(v) / np.mean(v) for v in pair_vals.values() if len(v) >= 10 and np.mean(v) > 0]
    print(f"{name:28} {np.corrcoef(a_v, s_v)[0, 1]:>+12.3f} {np.mean(cv):>13.3f}")

# =====================================================================
rule("2c  DOES Ã RECOVER THE STAGE-1 EDGES?  (orientation-correct)")
print("Ã is read with row = receiver, so edge source→target = Ã[target, source].")
print("rank corr with |ρ̂| over the searched pairs; BH survivors in the node set:")
print(f"{[(pt.nodes[i], pt.nodes[j]) for i, j in zip(*np.nonzero(S1_bh))]}\n")
searched = [(i, j) for i, j in zip(*np.nonzero(S1_all)) if i != j]
print(f"{'variant':28} {'rank corr':>10} {'survivor pct':>13} {'mass searched':>14}")

for name, (_, A) in trained.items():
    Am = A.mean(0)
    edge = Am.T                                                 # [source, target]
    rc = spearmanr([edge[i, j] for i, j in searched],
                   [abs(S1_all[i, j]) for i, j in searched]).correlation
    off = edge[~np.eye(pt.N, dtype=bool)]
    pct = [float((off < edge[i, j]).mean()) for i, j in zip(*np.nonzero(S1_bh))]
    mass = sum(edge[i, j] for i, j in searched) / off.sum()
    print(f"{name:28} {rc:>+10.3f} {str([round(p, 2) for p in pct]):>13} {mass:>14.1%}")
print(f"\n(searched pairs cover {len(searched)}/{pt.N * (pt.N - 1)} ordered pairs = "
      f"{len(searched) / (pt.N * (pt.N - 1)):.0%} — the uniform-Ã share)")
print("survivor pct = percentile of the survivor edge among all off-diagonal Ã entries")

# =====================================================================
rule("2d  CAPACITY — parameters vs the supervision available")
lp = pl.DataFrame([{"t_idx": t, "y": float(Y[t, i] / lsd[i])}
                   for t in range(pt.T) for i in range(pt.N) if lm[t, i]])
eff = within_snapshot_icc(lp)
print(f"labelled node-snapshots {lp.height}; ICC {eff['icc']:.3f}; effective n "
      f"{eff['n_effective']:.0f}. Overlapping k={K} labels cut that again by ~{K}x.")
print(f"linear ridge: 2-4 coefficients.\n")
print(f"{'variant':44} {'params':>8} {'per eff. label':>15}")
for name, kw in [("minimal/learned (pool)", dict()),
                 ("minimal/shared_mlp (pool)", dict(embedding="shared_mlp")),
                 ("minimal/per_step/hybrid (pool)", dict(embedding="hybrid", masking="per_step")),
                 ("minimal/per_step/hybrid (shared weights)",
                  dict(embedding="hybrid", masking="per_step", weights="shared")),
                 ("default/learned (pool)", dict(hidden=64, d_emb=10)),
                 ("default/shared_mlp (pool)", dict(hidden=64, d_emb=10, embedding="shared_mlp"))]:
    cfg = {**MIN, **kw}
    p = count_params(AGCRN(pt.N, pt.F, n_horizons=1, **cfg))
    print(f"{name:44} {p:>8,} {p / eff['n_effective']:>15.2f}")

# =====================================================================
rule("2e  SURPRISE — is there a directional input to propagate?")
print("MODEL_FEATURES carry no surprise. The only 'news' channel is d_implied_mean,")
print("the node's own recent belief change. With the surprise on the triggering")
print("node added (add_surprise_channel, s_pit ∈ [-1, 1]):\n")
pts = add_surprise_channel(pt, pl.read_parquet(SP))
S = pts.X[..., -1]
print(f"cells carrying a surprise: {(S != 0).sum()} of {pt.mask.sum()} active cells "
      f"({(S != 0).sum() / pt.mask.sum():.1%}); snapshots with ≥1 surprise: "
      f"{(S != 0).any(1).mean():.0%}")
z = Y / lsd[None]
# neighbour surprise pushed through the Stage-1 graph: nb[t, j] = Σ_i s[t, i] ρ_ij
for label, A in (("all searched ρ̂", S1_all), ("BH survivors", S1_bh)):
    nb = S @ A
    sel = lm & (nb != 0)
    if sel.sum() > 10:
        c = np.corrcoef(nb[sel], z[sel])[0, 1]
        agree = np.mean(np.sign(nb[sel]) == np.sign(z[sel]))
        print(f"  ρ̂-weighted in-neighbour surprise ({label:15}) vs own k={K} Δz: "
              f"n={sel.sum():4}  corr {c:+.3f}  sign agree {agree:.3f}")
own = lm & (S != 0)
print(f"  own surprise vs own k={K} Δz (post-release drift):                "
      f"n={own.sum():4}  corr {np.corrcoef(S[own], z[own])[0, 1]:+.3f}  "
      f"sign agree {np.mean(np.sign(S[own]) == np.sign(z[own])):.3f}")
print("\nsign agree includes zero labels as misses; see step 3 for dir. acc on y≠0.")
