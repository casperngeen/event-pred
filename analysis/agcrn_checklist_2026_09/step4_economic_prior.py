"""Checklist step 4 — AGCRN with the economic sign prior of strategy_spec.md §3.3.

The step-3 priors were *estimated* (Stage-1 ρ̂). This is the a-priori one the
lead-lag signal uses, with zero fitted parameters:

    edge A→B = HAWKISH[A] × HAWKISH[B]   (cross-release, A ≠ B)
    input    = trigger's z_surprise = surprise / implied_std, winsorised at |z| q99

placed on the trigger node on the first snapshot on/after its close date. A
softmax graph cannot carry the −1 edges (U3, JOBLESSCLAIMS), so the prior goes
in as a *frozen signed* adjacency. Also run restricted to the three BH-surviving
channels of leadlag_findings.md (labour→labour, inflation→inflation,
labour→policy).

In-sample only. Run from event-pred/:

    venv/bin/python analysis/agcrn_checklist_2026_09/step4_economic_prior.py \
        > analysis/agcrn_checklist_2026_09/out/step4_economic_prior.txt

  4a  Is the zero-parameter signal visible at snapshot resolution at all?
  4b  Walk-forward: linear and AGCRN rungs carrying the prior
"""
from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
import torch
from scipy.stats import binomtest

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN, LinearBaseline, build_labels, build_tensor, count_params, sequence_windows
from stg.models.baselines import _ridge, make_features
from stg.models.tensors import add_surprise_channel, global_label_sd
from stg.models.train import _fold_cuts, run_linear, run_torch, scale_diagnostics
from stg.panel.registry import is_same_release

NP = "artifacts/panels/node_panel_event.parquet"
SP = "artifacts/panels/surprise_panel.parquet"
K, L, SEEDS = 3, 12, (0, 1, 2)
MIN_TRIGGER_EVENTS = 10
torch.set_num_threads(4)

# strategy_spec.md §3.3 / analysis/leadlag_2026_09/build_panel.py, verbatim
HAWKISH = {
    "CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
    "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1, "CPIFOOD": +1,
    "CPIAPPAREL": +1,
    "PAYROLLS": +1, "ADP": +1,
    "U3": -1, "JOBLESSCLAIMS": -1,
    "GDP": +1, "ISMPMI": +1,
    "FED": +1,
}
TYPE = {
    **{s: "inflation" for s in ("CPI", "CPICORE", "CPIYOY", "CPICOREYOY", "CPIGAS",
                                "CPIUSEDCAR", "CPISHELTER", "CPIFOOD", "CPIAPPAREL",
                                "PCECORE")},
    **{s: "labour" for s in ("PAYROLLS", "U3", "JOBLESSCLAIMS", "ADP")},
    "GDP": "growth", "ISMPMI": "growth", "FED": "policy",
}
BH_CHANNELS = {("labour", "labour"), ("inflation", "inflation"), ("labour", "policy")}


def rule(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


pt = build_tensor(pl.read_parquet(NP))
nodes = pt.nodes
ni = {n: i for i, n in enumerate(nodes)}

# ---- trigger input: winsorised z_surprise, usable triggers only
sp = pl.read_parquet(SP).with_columns(
    (pl.col("surprise") / pl.col("implied_std")).alias("z"))
sp = sp.filter(pl.col("z").is_finite())
usable = {s for s, n in sp.group_by("series").len().iter_rows() if n >= MIN_TRIGGER_EVENTS}
triggers = sorted(usable & set(HAWKISH) & set(nodes))
cap = float(sp["z"].abs().quantile(0.99))
sp = sp.filter(pl.col("series").is_in(triggers)).with_columns(
    pl.col("z").clip(-cap, cap).alias("z_w"))
pts = add_surprise_channel(pt, sp, col="z_w")
Z = pts.X[..., -1]                                              # (T, N)

# ---- economic signed graph, [trigger, target]
def econ_graph(channels: set | None = None) -> np.ndarray:
    A = np.zeros((pt.N, pt.N))
    for a in triggers:
        for b in nodes:
            if b == a or b not in HAWKISH or is_same_release(a, b):
                continue
            if channels is not None and (TYPE[a], TYPE[b]) not in channels:
                continue
            A[ni[a], ni[b]] = HAWKISH[a] * HAWKISH[b]
    return A


G_all, G_bh = econ_graph(), econ_graph(BH_CHANNELS)

print(f"triggers (≥{MIN_TRIGGER_EVENTS} usable events, in HAWKISH and the node set): {triggers}")
print(f"z winsorised at ±{cap:.2f}; cells carrying a z: {(Z != 0).sum()}")
for name, G in (("all cross-release", G_all), ("BH channels", G_bh)):
    print(f"graph '{name}': {int((G != 0).sum())} edges, "
          f"{int((G < 0).sum())} negative; targets reached: "
          f"{sorted({nodes[j] for j in np.nonzero(G.any(0))[0]})}")

# =====================================================================
rule("4a  IS THE ZERO-PARAMETER SIGNAL VISIBLE AT SNAPSHOT RESOLUTION?")
print("signal[t, j] = Σ_i G[i, j]·z[t, i]: fires on the snapshot a trigger resolves.")
print("Label A (the AGCRN target): Δ implied_mean_j from t to t+k — starts at the")
print("  end-of-day snapshot AFTER the release, so the same-day reaction is excluded.")
print("Label B (diagnostic, not causal): t−1 to t+k−1 — includes the release day.")
print("Sign agreement on cells with signal ≠ 0 and label ≠ 0; binomial p vs 0.5")
print("(treats cells as independent — optimistic; one trigger fans out to many targets).\n")
T = pt.T
Y, lm = build_labels(pt, k=K, kind="belief_z")
lsd = global_label_sd(Y, lm, "belief_z")
im = pt.implied_mean
ok = np.isfinite(im)
YB = np.full_like(Y, np.nan); lmB = np.zeros_like(lm)
for t in range(1, T - K + 1):
    both = ok[t - 1] & ok[t + K - 1]
    YB[t, both] = im[t + K - 1, both] - im[t - 1, both]
    lmB[t, both] = True
print(f"{'graph':18} {'label':9} {'n':>5} {'sign agree':>11} {'p':>8} {'corr':>7}")
for gname, G in (("all cross-release", G_all), ("BH channels", G_bh)):
    sig = Z @ G
    for lname, Yl, ml in (("A t→t+k", Y, lm), ("B t−1→", YB, lmB)):
        z = Yl / lsd[None]
        sel = ml & (sig != 0) & (Yl != 0) & np.isfinite(Yl)
        n = int(sel.sum())
        hits = int((np.sign(sig[sel]) == np.sign(z[sel])).sum())
        p = binomtest(hits, n, 0.5).pvalue if n else np.nan
        c = np.corrcoef(sig[sel], z[sel])[0, 1] if n > 2 else np.nan
        print(f"{gname:18} {lname:9} {n:>5} {hits / max(n, 1):>11.3f} {p:>8.3f} {c:>+7.3f}")
    # per channel, label B
    for ch in sorted({(TYPE[nodes[i]], TYPE[nodes[j]]) for i, j in zip(*np.nonzero(G))}):
        Gc = np.where([[(TYPE.get(a), TYPE.get(b)) == ch for b in nodes] for a in nodes], G, 0)
        s = Z @ Gc
        for lname, Yl, ml in (("A", Y, lm), ("B", YB, lmB)):
            sel = ml & (s != 0) & (Yl != 0) & np.isfinite(Yl)
            if sel.sum() >= 10:
                h = (np.sign(s[sel]) == np.sign(Yl[sel])).mean()
                print(f"    {ch[0] + '→' + ch[1]:22} label {lname}: n={int(sel.sum()):4} "
                      f"agree {h:.3f}")

# =====================================================================
rule("4b  WALK-FORWARD — rungs carrying the economic prior (8 folds, 3 seeds)")
win_s = sequence_windows(pts, Y, lm, L=L)


class EconRidge(LinearBaseline):
    """own_momentum + the zero-parameter economic signal (fitted slope only)."""

    def __init__(self, nodes, G):
        super().__init__("own_momentum", nodes)
        self.G = G

    def _feats(self, win):
        base = make_features(win, self.nodes, "own_momentum")
        z = win["Xs"][:, -1, :, -1] * win["Ms"][:, -1]
        return np.concatenate([base, (z @ self.G)[..., None]], axis=-1)

    def fit(self, win):
        F, ym = self._feats(win), win["ym"]
        self.coef_ = _ridge(F[ym], win["y"][ym], self.lam)
        return self

    def predict(self, win):
        F = self._feats(win)
        flat = F.reshape(-1, F.shape[-1])
        return (np.column_stack([np.ones(len(flat)), flat]) @ self.coef_).reshape(F.shape[:2])


def fold_r2(r, w):
    cuts = _fold_cuts(w["dates"], 8)
    out = []
    for i in range(8):
        m = r["_mask"] & ((w["dates"] >= cuts[i]) & (w["dates"] < cuts[i + 1]))[:, None]
        if m.sum():
            yt, yp = r["_y"][m], r["_preds"][m]
            out.append(1 - ((yt - yp) ** 2).sum() / (yt ** 2).sum())
    return np.array(out)


def agcrn(G=None, **kw):
    cfg = dict(hidden=16, d_emb=2, masking="per_step", zero_head=True) | kw
    if G is not None:
        cfg |= dict(adjacency="stage1", stage1_adj=G)
    return lambda: AGCRN(pt.N, pts.F, n_horizons=1, **cfg)


hdr = (f"{'model':42} {'params':>7} {'R² vs 0':>15} {'dir≠0':>6} {'pred_sd':>8} "
       f"{'corr':>7} {'R²@α*':>8} {'folds>0':>8} {'folds>lin':>9}")
print(hdr); print("-" * len(hdr))
ref = None
for name, f in [("linear own_momentum", lambda: LinearBaseline("own_momentum", nodes)),
                ("linear + econ signal (all)", lambda: EconRidge(nodes, G_all)),
                ("linear + econ signal (BH channels)", lambda: EconRidge(nodes, G_bh))]:
    r = run_linear(f, win_s, lsd)
    s = scale_diagnostics(r["_y"], r["_preds"], r["_mask"])
    fr = fold_r2(r, win_s)
    ref = fr if ref is None else ref
    y, p, m = r["_y"], r["_preds"], r["_mask"]
    sel = m & (y != 0) & (p != 0)
    da = (np.sign(y[sel]) == np.sign(p[sel])).mean()
    print(f"{name:42} {'–':>7} {r['r2_vs_zero']:>+15.4f} {da:>6.3f} {s['pred_sd']:>8.3f} "
          f"{s['corr']:>+7.3f} {s['r2_at_alpha']:>+8.4f} {(fr > 0).sum():>5}/{len(fr)} "
          f"{(fr > ref).sum():>6}/{len(fr)}", flush=True)

for name, f in [("AGCRN frozen econ graph (all) + z", agcrn(G_all)),
                ("AGCRN frozen econ graph (BH ch.) + z", agcrn(G_bh)),
                ("AGCRN frozen econ (all) + z, shared W", agcrn(G_all, weights="shared")),
                ("AGCRN adaptive (hybrid) + z, no prior", agcrn(embedding="hybrid"))]:
    t0 = time.time()
    r = run_torch(f, win_s, lsd, seeds=SEEDS)
    s = scale_diagnostics(r["_y"], r["_preds"], r["_mask"])
    fr = fold_r2(r, win_s)
    y, p, m = r["_y"], r["_preds"], r["_mask"]
    sel = m & (y != 0) & (p != 0)
    da = (np.sign(y[sel]) == np.sign(p[sel])).mean()
    print(f"{name:42} {count_params(f()):>7} {r['r2_vs_zero']:>+8.4f} ±{r['r2_vs_zero_sd']:.3f} "
          f"{da:>6.3f} {s['pred_sd']:>8.3f} {s['corr']:>+7.3f} {s['r2_at_alpha']:>+8.4f} "
          f"{(fr > 0).sum():>5}/{len(fr)} {(fr > ref).sum():>6}/{len(fr)}   "
          f"[{time.time() - t0:.0f}s]", flush=True)
print("\nAll AGCRN rungs: per-step masking, zero-init head, minimal size. dir≠0 is on")
print("seed-averaged predictions; R² vs 0 is the seed mean of per-seed R².")
