"""Checklist step 1 — rule out training and implementation issues.

Each check targets a failure that can cripple AGCRN while leaving the linear
ladder untouched. In-sample only. Run from event-pred/:

    venv/bin/python analysis/agcrn_checklist_2026_09/step1_implementation.py \
        > analysis/agcrn_checklist_2026_09/out/step1_implementation.txt

  1a  Padding and masking   — does padding reach the real nodes?
  1b  Target scaling        — is the loss dominated by one series' units?
  1c  Can it overfit?       — drive train loss to ~0 on 16 windows
  1d  Train vs val loss     — under- or over-fitting, and where early stopping lands
  1e  Zero baseline         — is "linear wins" really "predicting ~0 wins"?

Minimal config (d_emb=2, hidden=16) throughout except where stated; the
Bai-default (d_emb=10, hidden=64) is checked in 1c.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN, LinearBaseline, build_labels, build_tensor, sequence_windows
from stg.models.baselines import _stage1_in_edges
from stg.models.tensors import (
    MODEL_FEATURES, apply_feature_scaler, apply_label_scaler, fit_feature_scaler,
    global_label_sd,
)
from stg.models.train import (
    PURGE, _fold_cuts, _masked_huber, fit_fold, run_linear, run_torch, scale_diagnostics,
)

NP = "artifacts/panels/node_panel_event.parquet"
K, L = 3, 12
torch.set_num_threads(4)


def rule(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


pt = build_tensor(pl.read_parquet(NP))
Y, lm = build_labels(pt, k=K, kind="belief_z")
win = sequence_windows(pt, Y, lm, L=L)
lsd = global_label_sd(Y, lm, "belief_z")
ni = {n: i for i, n in enumerate(pt.nodes)}
MIN = dict(hidden=16, d_emb=2)
DEF = dict(hidden=64, d_emb=10)


def model(emb="learned", masking="legacy", cfg=MIN, **kw):
    return lambda: AGCRN(pt.N, pt.F, n_horizons=1, embedding=emb, masking=masking,
                         **cfg, **kw)


VARIANTS = [(f"{m}/{e}", model(e, m)) for m in ("legacy", "per_step")
            for e in ("learned", "shared_mlp")]

# =====================================================================
rule("1a  PADDING AND MASKING")
act = win["Ms"]
print(f"nodes N={pt.N}, snapshots T={pt.T}; active cells {pt.mask.mean():.1%} "
      f"(mean {pt.mask.sum(1).mean():.1f} active nodes / snapshot)")
print(f"nodes active *somewhere* in the full training batch+window: "
      f"{int(act.any(axis=(0, 1)).sum())}/{pt.N}  <- legacy mask")
print(f"nodes active at the window's last step (mean): {act[:, -1].sum(1).mean():.1f}")
print("\nper-node active share (fraction of snapshots with a real belief):")
share = pt.mask.mean(0)
print("  " + "  ".join(f"{n}={s:.0%}" for n, s in sorted(zip(pt.nodes, share), key=lambda x: x[1])))

mu, sd = fit_feature_scaler(win["Xs"], act)
Xs = apply_feature_scaler(win["Xs"], mu, sd)
print("\nPadded cells are zero-filled in raw units, then standardised with the")
print("active cells' mean/sd, so a padded cell becomes -mu/sd:")
print(f"{'feature':18} {'|z| active':>11} {'|z| padded':>11} {'max |z| pad':>12}")
for f, n in enumerate(MODEL_FEATURES):
    a, p = np.abs(Xs[act][:, f]), np.abs(Xs[~act][:, f])
    print(f"{n:18} {a.mean():>11.2f} {p.mean():>11.2f} {p.max():>12.1f}")

# sensitivity: scramble only padded cells, measure change on labelled nodes
x = torch.tensor(Xs[:128]); m = torch.tensor(act[:128]); ym = torch.tensor(win["ym"][:128])
x2 = x.clone(); x2[~m] = torch.randn_like(x2[~m]) * 5
print("\nScramble ONLY padded cells (N(0,5²)), measure the change in the prediction")
print("on real labelled nodes (untrained model, seed 0). 0 = padding cannot leak.")
for name, f in VARIANTS:
    torch.manual_seed(0)
    mdl = f().eval()
    with torch.no_grad():
        d = (mdl(x, m) - mdl(x2, m)).squeeze(-1)[ym].abs()
    print(f"  {name:22} mean |Δpred| {d.mean():.4f}   max {d.max():.4f}")

# batch dependence of the legacy shared-MLP embedding
torch.manual_seed(0)
mdl = model("shared_mlp", "legacy")().eval()
with torch.no_grad():
    full = mdl(x, m)[0]
    alone = mdl(x[:1], m[:1])[0]
print(f"\nlegacy shared_mlp: E = MLP(x_t).mean(over batch), so sample 0's prediction")
print(f"depends on the other samples: |pred(batch) - pred(alone)| max = "
      f"{(full - alone).abs().max():.4f} (per_step: per-sample E, 0 by construction)")

# Stage-1 orientation
s1 = _stage1_in_edges(pt.nodes)
edges = [(pt.nodes[i], pt.nodes[j]) for i, j in zip(*np.nonzero(s1))]
print(f"\nStage-1 BH survivors inside the node set: {edges}")
print("The model aggregates A @ x (row = receiver). The main-branch code used the")
print("[trigger, target] matrix directly, so the receivers were the TRIGGERS:")
print(f"  FED in-weights, as on main:  "
      f"{ {pt.nodes[j]: round(float(v), 2) for j, v in enumerate(s1[ni['FED']]) if v} or 'none'}")
print(f"  FED in-weights, fixed (A.T): "
      f"{ {pt.nodes[j]: round(float(v), 2) for j, v in enumerate(s1[:, ni['FED']]) if v} }")

# =====================================================================
rule("1b  TARGET SCALING")
print("Raw Δimplied_mean (k=3) sd per node vs labelled-cell count. The harness")
print("divides by this sd (global_label_sd) before the loss, so every node's")
print("target is unit-variance. Loss share = node's share of Σ y² over labelled cells.")
raw = [(n, Y[lm[:, i], i]) for i, n in enumerate(pt.nodes)]
tot_raw = sum((v ** 2).sum() for _, v in raw)
tot_z = sum(((v / lsd[i]) ** 2).sum() for i, (_, v) in enumerate(raw))
print(f"{'node':12} {'n':>5} {'raw sd':>11} {'loss share raw':>15} {'loss share z':>13}")
for i, (n, v) in enumerate(raw):
    print(f"{n:12} {v.size:>5} {lsd[i]:>11.4g} {(v ** 2).sum() / tot_raw:>15.1%} "
          f"{((v / lsd[i]) ** 2).sum() / tot_z:>13.1%}")
zall = np.concatenate([v / lsd[i] for i, (_, v) in enumerate(raw)])
print(f"\nz-scored labels: sd {zall.std():.2f}, |z|>3 share {np.mean(np.abs(zall) > 3):.1%}, "
      f"max |z| {np.abs(zall).max():.1f}; Huber δ=1 caps the tail's gradient.")
print("Share of zero labels (no belief change over k snapshots): "
      f"{np.mean(zall == 0):.1%}")


# =====================================================================
def tensors(sel, mu, sd):
    return (torch.tensor(apply_feature_scaler(win["Xs"][sel], mu, sd)),
            torch.tensor(win["Ms"][sel]),
            torch.tensor(apply_label_scaler(win["y"][sel], lsd), dtype=torch.float32),
            torch.tensor(win["ym"][sel]))


def zero_loss(t):
    return _masked_huber(torch.zeros_like(t[2]), t[2], t[3]).item()


rule("1c  CAN IT OVERFIT?  16 windows, 3000 full-batch Adam steps, lr 3e-3, no wd")
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(np.nonzero(win["ym"].any(1))[0], 16, replace=False))
sel = np.zeros(len(win["dates"]), bool); sel[idx] = True
mu16, sd16 = fit_feature_scaler(win["Xs"][sel], win["Ms"][sel])
T16 = tensors(sel, mu16, sd16)
print(f"{int(T16[3].sum())} labelled cells; predict-zero Huber loss {zero_loss(T16):.4f}")
print(f"{'variant':34} {'params':>7} {'loss@0':>8} {'loss@300':>9} {'loss@3000':>10} {'/zero':>7}")
for cfg_name, cfg in (("minimal", MIN), ("default", DEF)):
    for name, f in [(f"{cfg_name}/{m}/{e}", model(e, m, cfg)) for m in ("legacy", "per_step")
                    for e in ("learned", "shared_mlp")]:
        torch.manual_seed(0)
        mdl = f()
        opt = torch.optim.Adam(mdl.parameters(), lr=3e-3)
        ls = []
        for _ in range(3000):
            opt.zero_grad()
            loss = _masked_huber(mdl(T16[0], T16[1]).squeeze(-1), T16[2], T16[3])
            loss.backward(); opt.step(); ls.append(loss.item())
        n_p = sum(p.numel() for p in mdl.parameters())
        print(f"{name:34} {n_p:>7} {ls[0]:>8.4f} {ls[299]:>9.4f} {ls[-1]:>10.4f} "
              f"{ls[-1] / zero_loss(T16):>7.3f}", flush=True)

# =====================================================================
rule("1d  TRAIN vs VALIDATION LOSS  (walk-forward harness, 8 folds, seed 0)")
print("Loss is masked Huber on z-scored labels; each value is shown as a ratio to")
print("predict-zero's loss on the same cells (<1 beats zero). 'fit' = training")
print("part of the fold, 'es' = its early-stop tail, 'test' = the held-out slice.")
print("sd init / sd kept: prediction sd on the test slice at initialisation and")
print("for the state early stopping kept (labels: sd ~0.86 in z units).\n")
dates = win["dates"]
cuts = _fold_cuts(dates, 8)
fold_ix = []
for i in range(8):
    tr = dates < (cuts[i] - PURGE)
    te = (dates >= cuts[i]) & (dates < cuts[i + 1])
    if tr.sum() < 40 or te.sum() == 0:
        continue
    tdates = np.sort(dates[tr]); es_cut = tdates[int(len(tdates) * 0.85)]
    fold_ix.append((i, tr & (dates < es_cut), tr & (dates >= es_cut), te))

settings = [("harness (200 ep, patience 15, lr 1e-3)", dict(max_epochs=200, patience=15)),
            ("long (600 ep, no early stop, lr 1e-3)", dict(max_epochs=600, patience=10**9))]
for sname, kw in settings:
    print(f"--- {sname}")
    print(f"{'variant':22} {'fold':>4} {'ep run':>6} {'best ep':>7} {'fit@0':>6} "
          f"{'fit@best':>8} {'fit@end':>8} {'es@best':>8} {'es@end':>7} {'test':>6} "
          f"{'sd init':>7} {'sd kept':>7}")
    for name, f in VARIANTS:
        for (i, fm, em, te) in (fold_ix[len(fold_ix) // 2], fold_ix[-1]):
            mu_f, sd_f = fit_feature_scaler(win["Xs"][fm], win["Ms"][fm])
            Tf, Te, Tt = tensors(fm, mu_f, sd_f), tensors(em, mu_f, sd_f), tensors(te, mu_f, sd_f)
            torch.manual_seed(0)
            with torch.no_grad():
                sd0 = f().eval()(Tt[0], Tt[1]).squeeze(-1)[Tt[3]].std().item()
            torch.manual_seed(0)
            mdl, h = fit_fold(f, Tf, Te, **kw)
            with torch.no_grad():
                pt_ = mdl(Tt[0], Tt[1]).squeeze(-1)
                tl = _masked_huber(pt_, Tt[2], Tt[3]).item()
                sd1 = pt_[Tt[3]].std().item()
            zf, ze, zt = zero_loss(Tf), zero_loss(Te), zero_loss(Tt)
            b = h["best_epoch"]
            print(f"{name:22} {i:>4} {h['epochs']:>6} {b:>7} {h['train'][0] / zf:>6.2f} "
                  f"{h['train'][b] / zf:>8.2f} {h['train'][-1] / zf:>8.2f} "
                  f"{h['es'][b] / ze:>8.2f} {h['es'][-1] / ze:>7.2f} {tl / zt:>6.2f} "
                  f"{sd0:>7.3f} {sd1:>7.3f}", flush=True)

# =====================================================================
rule("1e  ZERO BASELINE — pooled walk-forward, 8 folds, 3 seeds")
print("pred_sd: spread of the model's predictions (labels have sd ~0.86).")
print("corr / r2_at_alpha: how much directional signal survives if the")
print("predictions were optimally rescaled (uses eval labels: an upper bound).\n")
print(f"{'model':30} {'R² vs 0':>9} {'dir acc':>8} {'pred_sd':>8} {'corr':>7} "
      f"{'alpha*':>7} {'R²@alpha*':>10}")
for rung in ("zero", "train_mean", "own_level", "own_momentum", "neighbour_all",
             "neighbour_stage1"):
    r = run_linear(lambda rung=rung: LinearBaseline(rung, pt.nodes), win, lsd)
    s = scale_diagnostics(r["_y"], r["_preds"], r["_mask"])
    print(f"{'linear/' + rung:30} {r['r2_vs_zero']:>+9.4f} {r['dir_acc']:>8.3f} "
          f"{s['pred_sd']:>8.3f} {s['corr']:>+7.3f} {s['alpha_star']:>+7.3f} "
          f"{s['r2_at_alpha']:>+10.4f}")
for name, f in VARIANTS:
    t0 = time.time()
    r = run_torch(f, win, lsd, seeds=(0, 1, 2))
    s = scale_diagnostics(r["_y"], r["_preds"], r["_mask"])
    print(f"{'AGCRN/' + name:30} {r['r2_vs_zero']:>+9.4f} {r['dir_acc']:>8.3f} "
          f"{s['pred_sd']:>8.3f} {s['corr']:>+7.3f} {s['alpha_star']:>+7.3f} "
          f"{s['r2_at_alpha']:>+10.4f}   ({time.time() - t0:.0f}s; seed-avg preds)",
          flush=True)
print("\nNote: AGCRN R² / dir acc are the seed-mean of per-seed metrics (as in the")
print("main report); scale diagnostics are on the seed-averaged predictions.")
