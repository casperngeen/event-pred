"""Checklist step 5 — the ablation rungs step 3 was missing: graph-free and shuffled-graph.

Same harness, target and folds as step 3 (k=3 per-series z-scored Δ implied_mean,
8 expanding IS folds, 3 seeds). Surprise (s_pit) sits on the triggering node,
as in A7. In-sample only. Run from event-pred/:

    venv/bin/python analysis/agcrn_checklist_2026_09/step5_ablation_nulls.py \
        > analysis/agcrn_checklist_2026_09/out/step5_ablation_nulls.txt

  5a  Graph-free surprise rungs (linear): own surprise only; the cross-section's
      mean surprise broadcast to every node (a complete, unweighted graph).
  5b  Structure null (linear): surprise pushed through the Stage-1 ρ̂ graph,
      against the same graph with node labels permuted (200 draws).
  5c  Structure null (AGCRN): frozen signed Stage-1 graph + surprise channel,
      real vs 5 node-label permutations, for all searched ρ̂ and for the BH
      survivors.

A permutation relabels nodes, A_π[π(i), π(j)] = A[i, j], so every edge keeps
its weight and sign and every node keeps its in/out degree profile — only
*which* markets are connected changes. π permutes the graph's own endpoints
(nodes touching at least one edge), so a shuffled graph lives on the same
node set and cannot win or lose merely by pointing at rarely-active nodes.
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
N_PERM_LIN, N_PERM_NN = 200, 5
torch.set_num_threads(4)


def rule(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


pt = build_tensor(pl.read_parquet(NP))
pts = add_surprise_channel(pt, pl.read_parquet(SP))
Y, lm = build_labels(pt, k=K, kind="belief_z")
win = sequence_windows(pt, Y, lm, L=L)
win_s = sequence_windows(pts, Y, lm, L=L)
lsd = global_label_sd(Y, lm, "belief_z")
S1_all = _stage1_in_edges(pt.nodes, survivors_only=False)     # [trigger, target] signed ρ̂
S1_bh = _stage1_in_edges(pt.nodes, survivors_only=True)
np.fill_diagonal(S1_all, 0.0); np.fill_diagonal(S1_bh, 0.0)


def permuted(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Relabel A's endpoint nodes; redraw until the edge set actually changes."""
    ends = np.nonzero((A != 0).any(0) | (A != 0).any(1))[0]
    for _ in range(1000):
        perm = np.arange(A.shape[0])
        perm[ends] = rng.permutation(ends)
        B = np.zeros_like(A)
        B[np.ix_(perm, perm)] = A
        if not np.array_equal(B != 0, A != 0):
            return B
    raise RuntimeError("no permutation changes the edge set")


class SurpriseRidge(LinearBaseline):
    """own_momentum + a surprise term. ``mode``:

    own     — the node's own surprise (graph-free)
    pooled  — mean surprise over the snapshot's active triggers, same for every
              node (complete unweighted graph: surprise without structure)
    graph   — own surprise + Σ_i s_i A[i, j] (as step 3's ``linear +surprise``)
    """

    def __init__(self, nodes, mode: str, A: np.ndarray | None = None):
        super().__init__("own_momentum", nodes)
        self.mode, self.A = mode, A

    def _feats(self, win):
        base = make_features(win, self.nodes, "own_momentum")
        s = win["Xs"][:, -1, :, -1] * win["Ms"][:, -1]
        if self.mode == "own":
            extra = [s]
        elif self.mode == "pooled":
            n_trig = np.maximum((s != 0).sum(1, keepdims=True), 1)
            extra = [np.broadcast_to(s.sum(1, keepdims=True) / n_trig, s.shape)]
        elif self.mode == "graph":
            extra = [s, s @ self.A]
        else:
            raise ValueError(self.mode)
        return np.concatenate([base] + [e[..., None] for e in extra], axis=-1)

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
        te = (w["dates"] >= cuts[i]) & (w["dates"] < cuts[i + 1])
        m = r["_mask"] & te[:, None]
        if m.sum():
            yt, yp = r["_y"][m], r["_preds"][m]
            out.append(1 - ((yt - yp) ** 2).sum() / (yt ** 2).sum())
    return np.array(out)


hdr = (f"{'model':44} {'params':>7} {'R² vs 0':>15} {'pred_sd':>8} {'corr':>7} "
       f"{'R²@α*':>8} {'folds>0':>8} {'folds>lin':>9}")


def show(name, r, w, ref, params="–"):
    s = scale_diagnostics(r["_y"], r["_preds"], r["_mask"])
    fr = fold_r2(r, w)
    sd = f" ±{r['r2_vs_zero_sd']:.3f}" if "r2_vs_zero_sd" in r else ""
    vs = "" if ref is None else f"{(fr > ref).sum()}/{len(fr)}"
    print(f"{name:44} {params:>7} {r['r2_vs_zero']:>+8.4f}{sd:<7} {s['pred_sd']:>8.3f} "
          f"{s['corr']:>+7.3f} {s['r2_at_alpha']:>+8.4f} {(fr > 0).sum():>5}/{len(fr)} {vs:>9}",
          flush=True)
    return dict(model=name, r2=r["r2_vs_zero"], r2_sd=r.get("r2_vs_zero_sd", 0.0), **s), fr


rows = []
print(f"k={K}, L={L}, 8 folds, seeds {SEEDS}; labels in per-series z units")
print(f"Stage-1 graphs: all searched ρ̂ {int((S1_all != 0).sum())} edges on "
      f"{int(((S1_all != 0).any(0) | (S1_all != 0).any(1)).sum())} nodes; BH survivors "
      f"{int((S1_bh != 0).sum())} edges {[(pt.nodes[i], pt.nodes[j]) for i, j in zip(*np.nonzero(S1_bh))]}")

# =====================================================================
rule("5a  GRAPH-FREE SURPRISE RUNGS (linear)")
print(hdr); print("-" * len(hdr))
r = run_linear(lambda: LinearBaseline("own_momentum", pt.nodes), win, lsd)
row, ref = show("linear own_momentum", r, win, None); rows.append(row)
for name, mode, A in [("linear + own surprise (no graph)", "own", None),
                      ("linear + pooled surprise (complete graph)", "pooled", None),
                      ("linear + surprise via Stage-1 ρ̂ (all)", "graph", S1_all),
                      ("linear + surprise via Stage-1 ρ̂ (BH)", "graph", S1_bh)]:
    r = run_linear(lambda mode=mode, A=A: SurpriseRidge(pt.nodes, mode, A), win_s, lsd)
    row, _ = show(name, r, win_s, ref); rows.append(row)

# =====================================================================
rule(f"5b  STRUCTURE NULL, LINEAR — Stage-1 ρ̂ vs {N_PERM_LIN} node-label permutations")
print("p = share of permuted graphs scoring ≥ the real one (one-sided, +1 smoothed).\n")
rng = np.random.default_rng(0)
for gname, A in (("all searched ρ̂", S1_all), ("BH survivors", S1_bh)):
    real = run_linear(lambda A=A: SurpriseRidge(pt.nodes, "graph", A), win_s, lsd)
    real_c = scale_diagnostics(real["_y"], real["_preds"], real["_mask"])["corr"]
    null_r2, null_c = [], []
    for _ in range(N_PERM_LIN):
        B = permuted(A, rng)
        r = run_linear(lambda B=B: SurpriseRidge(pt.nodes, "graph", B), win_s, lsd)
        null_r2.append(r["r2_vs_zero"])
        null_c.append(scale_diagnostics(r["_y"], r["_preds"], r["_mask"])["corr"])
    null_r2, null_c = np.array(null_r2), np.array(null_c)
    p_r2 = (1 + (null_r2 >= real["r2_vs_zero"]).sum()) / (1 + len(null_r2))
    p_c = (1 + (null_c >= real_c).sum()) / (1 + len(null_c))
    print(f"{gname:16} R² real {real['r2_vs_zero']:+.4f} | null median {np.median(null_r2):+.4f} "
          f"[5%,95%] [{np.quantile(null_r2, .05):+.4f}, {np.quantile(null_r2, .95):+.4f}] | p {p_r2:.3f}")
    print(f"{'':16} corr real {real_c:+.4f} | null median {np.median(null_c):+.4f} "
          f"[5%,95%] [{np.quantile(null_c, .05):+.4f}, {np.quantile(null_c, .95):+.4f}] | p {p_c:.3f}",
          flush=True)
    rows.append(dict(model=f"5b null {gname}", r2=float(np.median(null_r2)),
                     r2_sd=float(null_r2.std()), corr=float(np.median(null_c)),
                     p_r2=float(p_r2), p_corr=float(p_c)))

# =====================================================================
rule(f"5c  STRUCTURE NULL, AGCRN — frozen signed Stage-1 graph + surprise, real vs {N_PERM_NN} permutations")
print(hdr); print("-" * len(hdr))
STEP = dict(masking="per_step", zero_head=True, hidden=16, d_emb=2)
rng = np.random.default_rng(1)
for gname, A in (("all ρ̂", S1_all), ("BH", S1_bh)):
    graphs = [("real", A)] + [(f"perm {i + 1}", permuted(A, rng)) for i in range(N_PERM_NN)]
    null = []
    for tag, G in graphs:
        f = lambda G=G: AGCRN(pt.N, pts.F, n_horizons=1, adjacency="stage1", stage1_adj=G, **STEP)
        t0 = time.time()
        r = run_torch(f, win_s, lsd, seeds=SEEDS, wd=1e-4)
        row, _ = show(f"frozen {gname} + surprise, {tag}", r, win_s, ref, str(count_params(f())))
        print(f"{'':44} [{time.time() - t0:.0f}s]", flush=True)
        rows.append(row)
        if tag != "real":
            null.append(r["r2_vs_zero"])
        else:
            real = r["r2_vs_zero"]
    print(f"-> {gname}: real {real:+.4f}, permuted mean {np.mean(null):+.4f} "
          f"(range {min(null):+.4f} … {max(null):+.4f}); real ranks "
          f"{1 + sum(x > real for x in null)} of {1 + len(null)}\n", flush=True)

pl.DataFrame(rows, strict=False, infer_schema_length=None).write_parquet("analysis/agcrn_checklist_2026_09/out/step5_ablation_nulls.parquet")
