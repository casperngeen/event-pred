#!/usr/bin/env python
"""Event-time graph models: does the economic prior predict anything on the
release clock?

    venv/bin/python analysis/event_time_2026_09/models.py \
        > analysis/event_time_2026_09/out/models.txt       # needs build_panel.py first

Same question as ``agcrn_checklist_2026_09/step4_economic_prior.py``, on the
event-time panel instead of the daily one, for both labels:

  imm     immediate repricing of the lead contract (≤3 prints, ≤24h), cents
  settle  settlement residual from the first post-release print, cents

  A  Zero-parameter signal: Σ_a HAWKISH[a]·HAWKISH[b]·z_a against the label.
     Sign agreement with a CI from a cluster bootstrap over release instants.
  B  Walk-forward (8 expanding folds, purged on label end, 3 seeds):
     linear rungs and AGCRN (per-step masking, zero-init head, minimal size)
     with an adaptive graph, and with the economic graph frozen in.

A step is a release instant; a window is the last ``L`` instants. Input-active
node = has a lead-contract price at t⁻ or is releasing at t (so the surprise is
never masked out). Missing ladder statistics are imputed with the node's median
over the panel — a units choice about features only, never labels — and flagged
by ``has_ladder``. In-sample only.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, "stg_infra")
from stg.models import count_params
from stg.models.train import _fold_cuts, run_linear, run_torch, scale_diagnostics

sys.path.insert(0, "analysis/event_time_2026_09")
from _panel import *  # noqa: E402,F401,F403  tensors, graphs, windows, rungs (shared with ablation.py)

SEEDS, N_BOOT = (0, 1, 2), 2000
torch.set_num_threads(4)


def rule(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


print(f"instants T={T}, nodes N={N}, features F={F}, window L={L}")
print(f"input-active cells {M.mean():.0%}; econ graph: {int((G_all != 0).sum())} edges "
      f"({int((G_all < 0).sum())} negative), BH channels: {int((G_bh != 0).sum())}")
for k in Y:
    y = Y[k][np.isfinite(Y[k])]
    print(f"label {k:6}: n={y.size}, sd {y.std():.1f}c, share zero {np.mean(y == 0):.1%}, "
          f"share >0 {np.mean(y > 0):.1%}")

# =====================================================================
rule("A  ZERO-PARAMETER ECONOMIC SIGNAL vs THE LABEL")
print("signal[t, b] = Σ_a G[a, b]·z[t, a]; sign agreement on cells with signal ≠ 0")
print("and label ≠ 0. 95% CI: bootstrap resampling whole release instants.\n")
rng = np.random.default_rng(0)


def agree_ci(sig, y):
    sel = (sig != 0) & np.isfinite(y) & (y != 0)
    t_idx = np.nonzero(sel.any(1))[0]
    hit = (np.sign(sig) == np.sign(np.nan_to_num(y))) & sel
    h_t, n_t = hit[t_idx].sum(1), sel[t_idx].sum(1)
    est = h_t.sum() / max(n_t.sum(), 1)
    bs = []
    for _ in range(N_BOOT):
        b = rng.integers(0, len(t_idx), len(t_idx))
        bs.append(h_t[b].sum() / max(n_t[b].sum(), 1))
    return int(n_t.sum()), len(t_idx), est, np.percentile(bs, [2.5, 97.5])


print(f"{'label':7} {'graph / channel':26} {'cells':>6} {'instants':>9} {'agree':>7} {'95% CI':>16}")
for k in Y:
    for gname, G in (("all cross-release", G_all), ("BH channels", G_bh)):
        n, nt, est, ci = agree_ci(Z @ G, Y[k])
        print(f"{k:7} {gname:26} {n:>6} {nt:>9} {est:>7.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    for ch in sorted({(TYPE[nodes[a]], TYPE[nodes[b]]) for a, b in zip(*np.nonzero(G_all))}):
        Gc = G_all * np.array([[(TYPE[p], TYPE[q]) == ch for q in nodes] for p in nodes])
        n, nt, est, ci = agree_ci(Z @ Gc, Y[k])
        if n >= 15:
            print(f"{'':7} {'  ' + ch[0] + '→' + ch[1]:26} {n:>6} {nt:>9} {est:>7.3f} "
                  f"[{ci[0]:.3f}, {ci[1]:.3f}]")
    lab = Y[k][np.isfinite(Y[k]) & (Y[k] != 0)]
    print(f"{'':7} {'(base rate: share > 0)':26} {lab.size:>6} {'':>9} {np.mean(lab > 0):>7.3f}\n")


# =====================================================================
def fold_r2(r, w):
    cuts = _fold_cuts(w["dates"], 8)
    out = []
    for i in range(8):
        m = r["_mask"] & ((w["dates"] >= cuts[i]) & (w["dates"] < cuts[i + 1]))[:, None]
        if m.sum():
            yt, yp = r["_y"][m], r["_preds"][m]
            out.append(1 - ((yt - yp) ** 2).sum() / (yt ** 2).sum())
    return np.array(out)


def fires(w, G):
    x = w["Xs"][:, -1]
    return (x[..., Zf] * (x[..., FEATS.index("released")] > 0)) @ G != 0


def report(name, r, w, params="–", extra=""):
    s = scale_diagnostics(r["_y"], r["_preds"], r["_mask"])
    fr = fold_r2(r, w)
    y, p, m = r["_y"], r["_preds"], r["_mask"]
    sel = m & (y != 0) & (p != 0)
    da = (np.sign(y[sel]) == np.sign(p[sel])).mean() if sel.any() else np.nan
    # restricted to cells where the BH-channel signal fires
    fm = m & fires(w, G_bh)
    r2f = 1 - ((y[fm] - p[fm]) ** 2).sum() / (y[fm] ** 2).sum()
    sf = fm & (y != 0) & (p != 0)
    daf = (np.sign(y[sf]) == np.sign(p[sf])).mean() if sf.any() else np.nan
    sd = f" ±{r['r2_vs_zero_sd']:.3f}" if "r2_vs_zero_sd" in r else ""
    print(f"{name:40} {params:>6} {r['r2_vs_zero']:>+8.4f}{sd:7} {da:>6.3f} "
          f"{s['pred_sd']:>8.3f} {s['corr']:>+7.3f} {s['r2_at_alpha']:>+8.4f} "
          f"{(fr > 0).sum():>3}/{len(fr)} {int(m.sum()):>6} | {r2f:>+8.4f} {daf:>6.3f} "
          f"{int(fm.sum()):>4}{extra}", flush=True)


OOF = []      # out-of-fold predictions; settle rows are read by backtest.py / exits.py


def keep_oof(k, name, r, w, lsd):
    for j, d in enumerate(w["dates"]):
        for i in np.nonzero(r["_mask"][j])[0]:
            OOF.append(dict(label=k, instant=d.astype("datetime64[us]").item(), series=nodes[i],
                            model=name, pred_c=float(r["_preds"][j, i] * lsd[i]),
                            y_c=float(r["_y"][j, i] * lsd[i])))


for k in Y:
    rule(f"B  WALK-FORWARD — label '{k}' (per-node z units; 8 folds, purged on label end)")
    w = windows(k)
    lsd = label_sd(k)
    print(f"{len(w['dates'])} windows; labelled cells {int(w['ym'].sum())}\n")
    print(f"{'model':40} {'params':>6} {'R² vs 0':>15} {'dir≠0':>6} {'pred_sd':>8} "
          f"{'corr':>7} {'R²@α*':>8} {'f>0':>5} {'n eval':>6} | {'R² fire':>8} {'dir fire':>6} {'n':>4}")
    for name, f in [("linear zero", lambda: Ridge(False)),
                    ("linear own state", lambda: Ridge(True)),
                    ("linear econ signal only (all)", lambda: Ridge(False, G_all)),
                    ("linear econ signal only (BH ch.)", lambda: Ridge(False, G_bh)),
                    ("linear own + econ (all)", lambda: Ridge(True, G_all)),
                    ("linear own + econ (BH ch.)", lambda: Ridge(True, G_bh))]:
        r = run_linear(f, w, lsd)
        report(name, r, w)
        keep_oof(k, name, r, w, lsd)
    for name, f in [("AGCRN adaptive (hybrid), no prior", agcrn(embedding="hybrid")),
                    ("AGCRN frozen econ graph (all)", agcrn(G_all)),
                    ("AGCRN frozen econ graph (BH ch.)", agcrn(G_bh)),
                    ("AGCRN frozen econ (all), shared W", agcrn(G_all, weights="shared"))]:
        t0 = time.time()
        r = run_torch(f, w, lsd, seeds=SEEDS)
        report(name, r, w, str(count_params(f())), f"  [{time.time() - t0:.0f}s]")
        keep_oof(k, name, r, w, lsd)
    print("\nR² vs 0 for AGCRN is the seed mean; dir≠0 / pred_sd / corr / R²@α* are")
    print("on seed-averaged predictions. f>0 = folds with R² > 0. Right of '|': the same")
    print("predictions scored only on cells where the BH-channel economic signal fires.")

oof = pl.DataFrame(OOF)
oof.write_parquet("analysis/event_time_2026_09/out/oof_all.parquet")
(oof.filter(pl.col("label") == "settle").select("instant", "series", "model", "pred_c")
 .write_parquet("analysis/event_time_2026_09/out/oof_settle.parquet"))
