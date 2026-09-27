#!/usr/bin/env python
"""Recovery test: can a spatial-temporal graph learn a *known* economic graph
from data shaped like ours, and how much data does it need?

    venv/bin/python analysis/recovery_2026_09/recovery.py --tmult 1 3 --reps 10 --tag a
    venv/bin/python analysis/recovery_2026_09/recovery.py --tmult 10 --reps 4 --tag b
    venv/bin/python analysis/recovery_2026_09/recovery.py --tmult 30 --reps 2 --tag c
    venv/bin/python analysis/recovery_2026_09/recovery.py --tmult 3 10 --rhos 0.6 \
        --reps 3 --truth pos --tag pos                       # sign diagnostic
    venv/bin/python analysis/recovery_2026_09/recovery.py --summarise \
        > analysis/recovery_2026_09/out/recovery.txt

Semi-synthetic. Everything except the cross-series signal comes from the real
event-time panel (``analysis/event_time_2026_09/out/event_nodes.parquet``,
build it first):

  calendar   each synthetic step copies a real release instant (bootstrap):
             which series release, which are active, which are labelled, and
             every non-surprise input feature, over the last L steps
  surprise   fresh draws z ~ N(0, C) on the releasing series, clipped at the
             real winsorisation (±2.89); C is the real cross-series surprise
             correlation on co-released instants, so the CPI family stays one
             factor and PAYROLLS/U3 stay a near-independent double trigger
  noise      the real immediate-repricing label of the target, resampled per
             node (heavy tails and the 15% zeros kept; any real signal broken)
  signal     y[t, b] = σ_b · (e + κ · Σ_a W*[a, b] · z[t, a]),  κ = ρ/√(1−ρ²)

W* (``--truth bh``) is the structure the event-time study imposes: HAWKISH
signs on the three BH channels (labour→labour, inflation→inflation,
labour→policy), cross-release only — 25 directed, signed edges (6 negative,
rank 4) out of the 142 candidate trigger→target pairs. ``--truth pos`` is |W*|,
a diagnostic that removes the negative edges. ρ is the per-edge correlation
with one parent firing; ρ = 0 is the null. ``--tmult`` lengthens the calendar:
1 = the real 154 instants.

Learners, all fitted on the first 85% of steps (torch models early-stop on its
last 15%, Adam lr 3e-3, ≤600 epochs, patience 30), all scored on how well they
rank the 142 candidates:

  pairwise        the Stage-1 estimator: Spearman(z_a, y_b) on instants where
                  a released, BH-FDR q = 0.10 (asymptotic p — synthetic labels
                  are mostly untied on the cells that carry signal)
  lasso           y_b = Σ_a W_ab z_a per target over the candidates, CV'd —
                  a one-layer linear graph, full rank
  low-rank graph  the same with W = (E_src E_dstᵀ) ⊙ candidates, rank 4, by
                  gradient descent — the smallest STG that can express W*
  AGCRN           the event-time configuration (per-step masking, zero-init
                  head, hidden 16, d_emb 2, learned embedding). Its adjacency
                  softmax(ReLU(EEᵀ)) is symmetric and non-negative, and the
                  message Ã·x reaches the receiver without its sender's identity
  AGCRN signed    identical, except the adjacency is E_dst E_srcᵀ (d = 4):
                  directed and signed, no softmax
  … + prior      sign ablation (``--learners agcrn agcrn_prior agcrn_frozen signed``):
                  the true graph given as an unsigned soft prior, or frozen in
                  with its signs
  … z-only input  ablation (``--learners agcrn_z signed_z --no-baselines``): the
                  same two models fed only [released, z], so the surprise is not
                  one of 15 features × 6 lags

Edge read-outs: the learner's own matrix (ρ̂, coefficients, Ã, E_dst E_srcᵀ), and
for the torch models the effective edge ∂ŷ_b/∂z_a averaged over held-out
windows where a released and b is labelled. Metrics: AUROC of |score| for true
vs false candidates; sign accuracy on the true edges; precision at K = 25;
discovery power / false discovery proportion where there is a discovery set;
held-out R² vs zero against the oracle R² (the planted signal itself).
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
import zlib

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN
from stg.models.tensors import apply_feature_scaler, fit_feature_scaler
from stg.models.train import fit_fold
from stg.panel.registry import is_same_release
from stg.structure.stats import benjamini_hochberg, spearman, spearman_p

PANEL = "analysis/event_time_2026_09/out/event_nodes.parquet"
OUT = "analysis/recovery_2026_09/out"
L, ZCLIP, Q, RANK = 6, 2.89, 0.10, 4
TRAIN = dict(max_epochs=600, patience=30, lr=3e-3)
torch.set_num_threads(3)

HAWKISH = {
    "CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
    "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1, "CPIFOOD": +1,
    "CPIAPPAREL": +1, "PAYROLLS": +1, "ADP": +1, "U3": -1, "JOBLESSCLAIMS": -1,
    "GDP": +1, "ISMPMI": +1, "FED": +1,
}
TYPE = {**{s: "inflation" for s in ("CPI", "CPICORE", "CPIYOY", "CPICOREYOY", "CPIGAS",
                                    "CPIUSEDCAR", "CPISHELTER", "CPIFOOD", "CPIAPPAREL",
                                    "PCECORE")},
        **{s: "labour" for s in ("PAYROLLS", "U3", "JOBLESSCLAIMS", "ADP")},
        "GDP": "growth", "ISMPMI": "growth", "FED": "policy"}
BH_CHANNELS = {("labour", "labour"), ("inflation", "inflation"), ("labour", "policy")}
FEATS = ["q50", "sigma_iqr", "skew_q", "d_q50_7d", "n_legs", "med_age_h", "fresh24",
         "vol7", "flow7", "days_to_close", "p_lead", "lead_age_h",
         "has_ladder", "released", "z"]
LOG = {"med_age_h", "vol7", "lead_age_h"}
Zf, Rf = FEATS.index("z"), FEATS.index("released")


# ------------------------------------------------------------ real panel
# Same construction as analysis/event_time_2026_09/models.py.
def load_panel():
    panel = pl.read_parquet(PANEL).with_columns(
        pl.col("q50").is_not_null().cast(pl.Float64).alias("has_ladder"),
        pl.col("released").cast(pl.Float64))
    nodes = sorted(panel["series"].unique().to_list())
    instants = (panel["instant"].dt.replace_time_zone(None).unique().sort()
                .to_numpy().astype("datetime64[us]"))
    ni = {n: i for i, n in enumerate(nodes)}
    ti = {t: i for i, t in enumerate(instants.tolist())}
    T, N, F = len(instants), len(nodes), len(FEATS)
    X = np.full((T, N, F), np.nan)
    M = np.zeros((T, N), bool)
    Y = np.full((T, N), np.nan)
    for r in panel.iter_rows(named=True):
        t, i = ti[np.datetime64(r["instant"].replace(tzinfo=None), "us").tolist()], ni[r["series"]]
        for f, c in enumerate(FEATS):
            v = r.get(c)
            if v is not None:
                X[t, i, f] = np.log1p(max(v, 0)) if c in LOG else float(v)
        M[t, i] = r.get("p_lead") is not None or bool(r["released"])
        if r.get("y_imm") is not None and M[t, i]:
            Y[t, i] = r["y_imm"]
    for i in range(N):
        for f in range(F):
            col = X[M[:, i], i, f]
            med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
            X[:, i, f] = np.where(np.isfinite(X[:, i, f]), X[:, i, f], med)
    R = X[..., Rf] > 0
    Z = np.where(R, X[..., Zf], 0.0)
    return nodes, X.astype(np.float32), M, R, Z, Y


nodes, X0, M0, R0, Z0, Y0 = load_panel()
T0, N, F = X0.shape
ni = {n: i for i, n in enumerate(nodes)}
YM0 = np.isfinite(Y0)
SIG = np.array([np.nanstd(Y0[:, i]) if YM0[:, i].sum() > 1 else 1.0 for i in range(N)])
SIG[~np.isfinite(SIG) | (SIG < 1e-9)] = 1.0
_all = (Y0[YM0] / np.broadcast_to(SIG, Y0.shape)[YM0])
POOL = [(Y0[YM0[:, i], i] / SIG[i]) if YM0[:, i].sum() >= 10 else _all for i in range(N)]

triggers = [n for n in nodes if R0[:, ni[n]].any()]
CAND = np.zeros((N, N), bool)
W_BH = np.zeros((N, N))
for a in triggers:
    for b in nodes:
        if a != b and not is_same_release(a, b):
            CAND[ni[a], ni[b]] = True
            if (TYPE[a], TYPE[b]) in BH_CHANNELS:
                W_BH[ni[a], ni[b]] = HAWKISH[a] * HAWKISH[b]
TRUTHS = {"bh": W_BH, "pos": np.abs(W_BH)}
ca, cb = np.nonzero(CAND)           # candidate pairs, fixed order


def surprise_corr():
    """Pairwise correlation of real surprises on co-released instants, PSD."""
    C = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            both = R0[:, i] & R0[:, j]
            if both.sum() >= 5 and Z0[both, i].std() > 0 and Z0[both, j].std() > 0:
                C[i, j] = C[j, i] = np.corrcoef(Z0[both, i], Z0[both, j])[0, 1]
    w, V = np.linalg.eigh(C)
    C = V @ np.diag(np.clip(w, 1e-3, None)) @ V.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


CHOL = np.linalg.cholesky(surprise_corr())


# ------------------------------------------------------------ generator
def generate(tmult: float, rho: float, W: np.ndarray, rng: np.random.Generator) -> dict:
    Ts = int(round(T0 * tmult))
    src = rng.integers(0, T0, Ts)
    R, M, YM = R0[src], M0[src], YM0[src]
    z = np.clip(rng.standard_normal((Ts, N)) @ CHOL.T, -ZCLIP, ZCLIP) * R
    X = X0[src].copy()
    X[..., Zf] = z
    kappa = rho / np.sqrt(1 - rho ** 2)
    S = z @ W
    e = np.stack([rng.choice(POOL[i], Ts) for i in range(N)], 1)
    Y = np.where(YM, SIG * (e + kappa * S), np.nan)
    oracle = np.where(YM, SIG * kappa * S, np.nan)
    return dict(X=X, M=M, R=R, z=z, Y=Y, YM=YM, oracle=oracle, T=Ts)


def windows(d: dict) -> dict:
    idx = np.arange(L - 1, d["T"])
    idx = idx[d["YM"][idx].any(1)]
    Xs = np.stack([d["X"][t - L + 1:t + 1] for t in idx])
    Ms = np.stack([d["M"][t - L + 1:t + 1] for t in idx])
    return dict(idx=idx, Xs=Xs, Ms=Ms, y=np.nan_to_num(d["Y"][idx] / SIG), ym=d["YM"][idx],
                R=d["R"][idx], z=d["z"][idx], oracle=np.nan_to_num(d["oracle"][idx] / SIG))


# ------------------------------------------------------------ metrics
def edge_metrics(score: np.ndarray, W: np.ndarray, found: np.ndarray | None = None,
                 signed: bool = True) -> dict:
    s = np.nan_to_num(score[ca, cb])
    t = W[ca, cb] != 0
    k = int(t.sum())
    live = np.abs(s).max() > 0
    out = dict(auroc=roc_auc_score(t, np.abs(s)) if live else 0.5,
               prec_k=float(t[np.argsort(-np.abs(s), kind="stable")[:k]].mean()) if live else np.nan)
    st, wt = s[t], np.sign(W[ca, cb][t])
    out["sign_acc"] = (float((np.sign(st[st != 0]) == wt[st != 0]).mean())
                       if signed and (st != 0).any() else np.nan)
    # balanced: mean hit rate on positive and negative true edges. 19 of 25 true
    # edges are positive, so an all-positive read-out scores sign_acc 0.76.
    hits = [float((np.sign(st[(wt == v) & (st != 0)]) == v).mean())
            for v in (1, -1) if ((wt == v) & (st != 0)).any()]
    out["sign_bal"] = float(np.mean(hits)) if signed and len(hits) == 2 else np.nan
    if found is not None:
        f = found[ca, cb]
        out["power"] = float(f[t].mean())
        out["fdp"] = float((f & ~t).sum() / max(f.sum(), 1))
    return out


def r2(y, p, m):
    return float(1 - ((y - p)[m] ** 2).sum() / max((y[m] ** 2).sum(), 1e-12))


# ------------------------------------------------------------ learners
def pairwise(w, tr, W):
    rho = np.full((N, N), np.nan)
    p = np.full((N, N), np.nan)
    for a, b in zip(ca, cb):
        sel = tr & w["R"][:, a] & w["ym"][:, b]
        if sel.sum() >= 10:
            rho[a, b] = spearman(w["z"][sel, a], w["y"][sel, b])
            p[a, b] = spearman_p(rho[a, b], int(sel.sum()))
    found = np.zeros((N, N), bool)
    found[ca, cb] = benjamini_hochberg(p[ca, cb], Q)
    return edge_metrics(rho, W, found)


def lasso(w, tr, te, W):
    B = np.zeros((N, N))
    pred = np.zeros_like(w["y"])
    for b in range(N):
        par = np.nonzero(CAND[:, b])[0]
        rows = tr & w["ym"][:, b]
        if len(par) == 0 or rows.sum() < 20:
            continue
        m = LassoCV(cv=5, alphas=30, max_iter=5000).fit(w["z"][rows][:, par], w["y"][rows, b])
        B[par, b] = m.coef_
        pred[:, b] = m.predict(w["z"][:, par])
    return {**edge_metrics(B, W, B != 0), "r2": r2(w["y"], pred, w["ym"] & te[:, None])}


class LowRankGraph(nn.Module):
    """ŷ_b = c_b + Σ_a [(E_src E_dstᵀ) ⊙ candidates]_ab · z_a, on raw z."""

    def __init__(self):
        super().__init__()
        self.Es = nn.Parameter(torch.randn(N, RANK) * 0.1)
        self.Ed = nn.Parameter(torch.randn(N, RANK) * 0.1)
        self.c = nn.Parameter(torch.zeros(N))
        self.register_buffer("mask", torch.tensor(CAND, dtype=torch.float32))

    def adj(self):                                     # [source, target]
        return (self.Es @ self.Ed.t()) * self.mask

    def forward(self, seq, seq_mask):
        z = seq[:, -1, :, Zf] * (seq[:, -1, :, Rf] > 0)
        return (z @ self.adj() + self.c)[..., None]


class SignedAGCRN(AGCRN):
    """AGCRN with a directed, signed adjacency E_dst E_srcᵀ in place of the softmax."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.E_dst = nn.Parameter(torch.randn(self.n_nodes, self.d_emb) * 0.05)

    def adj(self):                                     # [source, target]
        return self.E @ self.E_dst.t()

    def _adjacency(self, E, active):
        A_in = self.E_dst @ self.E.t()                 # [target, source]
        return A_in[None] * active[:, None, :].float()


def torch_learner(w, fit, es, te, W, kind, seed):
    torch.manual_seed(seed)
    raw = kind == "lowrank"
    zonly = kind.endswith("_z")          # ablation: inputs = [released, z] only
    cols = [Rf, Zf] if zonly else list(range(F))
    zi = cols.index(Zf)
    mu, sd = fit_feature_scaler(w["Xs"][fit], w["Ms"][fit])

    def prep(m):
        x = w["Xs"][m] if raw else apply_feature_scaler(w["Xs"][m], mu, sd)[..., cols]
        return (torch.tensor(x, dtype=torch.float32), torch.tensor(w["Ms"][m]),
                torch.tensor(w["y"][m], dtype=torch.float32), torch.tensor(w["ym"][m]))
    cfg = dict(n_horizons=1, hidden=16, masking="per_step", zero_head=True, embedding="learned")
    make = {"lowrank": LowRankGraph,
            "agcrn": lambda: AGCRN(N, F, d_emb=2, **cfg),
            "signed": lambda: SignedAGCRN(N, F, d_emb=RANK, **cfg),
            "agcrn_z": lambda: AGCRN(N, 2, d_emb=2, **cfg),
            "agcrn_prior": lambda: AGCRN(N, F, d_emb=2, prior_adj=W, prior_lambda=2.0, **cfg),
            "agcrn_frozen": lambda: AGCRN(N, F, d_emb=2, adjacency="stage1", stage1_adj=W, **cfg),
            "signed_z": lambda: SignedAGCRN(N, 2, d_emb=RANK, **cfg)}[kind]
    t0 = time.time()
    model, hist = fit_fold(make, prep(fit), prep(es), **TRAIN)
    Xt, Mt, _, _ = prep(te)
    Xt.requires_grad_(True)
    out = model(Xt, Mt).squeeze(-1)                                  # (B, N)
    Rt, ymt = w["R"][te], w["ym"][te]
    zscale = np.ones(N) if raw else sd[:, Zf]
    J = np.full((N, N), np.nan)                                      # [source, target]
    for b in range(N):
        if not ymt[:, b].any():
            continue
        g, = torch.autograd.grad(out[:, b].sum(), Xt, retain_graph=True)
        g = g[:, -1, :, zi].numpy() / zscale                          # per unit of raw z
        for a in np.nonzero(CAND[:, b])[0]:
            sel = Rt[:, a] & ymt[:, b]
            if sel.any():
                J[a, b] = g[sel, a].mean()
    if kind.startswith("agcrn"):
        own = model.learned_adjacency(Xt.detach(), Mt).T              # Ã ≥ 0
    else:
        own = model.adj().detach().numpy()
    info = dict(r2=r2(w["y"][te], out.detach().numpy(), ymt), best_epoch=hist["best_epoch"],
                epochs=hist["epochs"], secs=time.time() - t0)
    return ({**edge_metrics(J, W), **info},
            {**edge_metrics(np.where(CAND, own, 0.0), W, signed=not kind.startswith("agcrn")), **info})


LEARNERS = {"lowrank": "low-rank graph", "agcrn": "AGCRN", "signed": "AGCRN signed",
            "agcrn_z": "AGCRN, z-only input", "signed_z": "AGCRN signed, z-only input",
            "agcrn_prior": "AGCRN + true graph as unsigned prior",
            "agcrn_frozen": "AGCRN + true signed graph frozen"}


# ------------------------------------------------------------ run
def run(tmults, rhos, reps, truth, tag, learners, baselines):
    W = TRUTHS[truth]
    rows = []
    for tmult in tmults:
        for rho in rhos:
            for rep in range(reps):
                rng = np.random.default_rng(zlib.crc32(f"{truth}-{tmult}-{rho}-{rep}".encode()))
                d = generate(tmult, rho, W, rng)
                w = windows(d)
                n = len(w["idx"])
                ix = np.arange(n)
                fit, es = ix < int(n * 0.70), (ix >= int(n * 0.70)) & (ix < int(n * 0.85))
                tr, te = fit | es, ix >= int(n * 0.85)
                base = dict(truth=truth, tmult=tmult, T=d["T"], rho=rho, rep=rep,
                            oracle_r2=r2(w["y"][te], w["oracle"][te], w["ym"][te]),
                            n_edge_med=float(np.median([(tr & w["R"][:, a] & w["ym"][:, b]).sum()
                                                        for a, b in zip(*np.nonzero(W))])))
                if baselines:
                    rows.append({**base, "learner": "pairwise (Stage-1)", **pairwise(w, tr, W)})
                    rows.append({**base, "learner": "lasso", **lasso(w, tr, te, W)})
                log = []
                for kind in learners:
                    name = LEARNERS[kind]
                    jac, own = torch_learner(w, fit, es, te, W, kind, seed=rep)
                    rows.append({**base, "learner": f"{name}: ∂ŷ/∂z", **jac})
                    if kind != "lowrank":           # low-rank: own matrix == Jacobian
                        rows.append({**base, "learner": f"{name}: own adjacency", **own})
                    log.append(f"{kind} {jac['secs']:.0f}s ep {jac['best_epoch']}/{jac['epochs']}")
                print(f"{truth} tmult {tmult:>4} rho {rho:.1f} rep {rep} T={d['T']}  "
                      + " | ".join(log), flush=True)
                pl.DataFrame(rows, infer_schema_length=None).write_parquet(
                    f"{OUT}/runs_{tag}.parquet")


def summarise():
    df = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{OUT}/runs_*.parquet"))],
                   how="diagonal_relaxed")
    # a rerun of the same (learner, setting, rep) sees identical data: keep one
    # row, preferring the rerun, which carries sign_bal
    if "sign_bal" in df.columns:
        df = (df.sort(pl.col("sign_bal").is_null())
              .unique(["truth", "learner", "tmult", "rho", "rep"], keep="first", maintain_order=True))
    TRUE = W_BH != 0
    print(f"nodes N={N}, real instants T={T0}; candidates {int(CAND.sum())} trigger→target "
          f"pairs; true edges {int(TRUE.sum())} ({int((W_BH < 0).sum())} negative, "
          f"rank {np.linalg.matrix_rank(W_BH)})")
    # anchors: the real effect on the imm label, two ways
    real = []
    for a, b in zip(ca, cb):
        sel = R0[:, a] & YM0[:, b]
        if sel.sum() >= 10:
            real.append((TRUE[a, b], np.sign(W_BH[a, b]) or 1.0,
                         spearman(Z0[sel, a], Y0[sel, b]), int(sel.sum())))
    tr_ = [(s * r, n) for t, s, r, n in real if t and np.isfinite(r)]
    fa_ = [r for t, s, r, n in real if not t and np.isfinite(r)]
    sig, y = Z0 @ W_BH, np.nan_to_num(Y0)
    sel = (sig != 0) & YM0 & (y != 0)
    agree = float((np.sign(sig[sel]) == np.sign(y[sel])).mean())
    print(f"real panel, imm label:\n"
          f"  per-edge theory-signed Spearman, {len(tr_)} true edges with n≥10: mean "
          f"{np.mean([r for r, _ in tr_]):+.3f} (median n {int(np.median([n for _, n in tr_]))}); "
          f"mean |ρ| on the {len(fa_)} other candidates {np.mean(np.abs(fa_)):.3f}\n"
          f"  pooled sign agreement of Σ W*·z with y: {agree:.3f} on {int(sel.sum())} cells "
          f"→ Gaussian-equivalent ρ = sin(π(a−½)) = {np.sin(np.pi * (agree - 0.5)):.2f}")
    print("ρ below is the planted per-edge correlation with one parent firing. n/edge =\n"
          "median training instants per true edge where the source released and the\n"
          "target was labelled. ep = best epoch / epochs run (cap 600).\n")
    agg = (df.group_by("truth", "learner", "tmult", "T", "rho", maintain_order=True)
           .agg(pl.len().alias("reps"),
                *[pl.col(c).mean() for c in ("auroc", "prec_k", "sign_acc", "sign_bal", "power", "fdp",
                                             "r2", "oracle_r2", "n_edge_med", "best_epoch",
                                             "epochs")],
                pl.col("auroc").std().alias("auroc_sd"))
           .sort("truth", "learner", "rho", "tmult"))
    f = lambda v, s="{:.3f}": s.format(v) if v is not None and np.isfinite(v) else "–"
    for (truth, ln), g in agg.group_by("truth", "learner", maintain_order=True):
        print(f"\n[{truth}] {ln}")
        print(f"  {'ρ':>4} {'T':>6} {'reps':>4} {'n/edge':>6} {'AUROC':>12} {'prec@25':>8} "
              f"{'sign':>6} {'s-bal':>6} {'power':>6} {'FDP':>6} {'R² test':>8} {'oracle':>7} {'ep':>9}")
        for r in g.iter_rows(named=True):
            ep = f"{r['best_epoch']:.0f}/{r['epochs']:.0f}" if r["epochs"] is not None else ""
            print(f"  {r['rho']:>4.1f} {r['T']:>6} {r['reps']:>4} {r['n_edge_med']:>6.0f} "
                  f"{r['auroc']:>6.3f} ±{f(r['auroc_sd'], '{:.2f}'):>4} {f(r['prec_k']):>8} "
                  f"{f(r['sign_acc']):>6} {f(r.get('sign_bal')):>6} {f(r['power']):>6} {f(r['fdp']):>6} "
                  f"{f(r['r2'], '{:+.4f}'):>8} {r['oracle_r2']:>+7.4f} {ep:>9}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmult", type=float, nargs="*", default=[1, 3, 10, 30])
    ap.add_argument("--rhos", type=float, nargs="*", default=[0.0, 0.2, 0.4, 0.6])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--truth", default="bh", choices=sorted(TRUTHS))
    ap.add_argument("--tag", default="all")
    ap.add_argument("--learners", nargs="*", default=["lowrank", "agcrn", "signed"],
                    choices=sorted(LEARNERS))
    ap.add_argument("--baselines", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.summarise:
        summarise()
    else:
        run(a.tmult, a.rhos, a.reps, a.truth, a.tag, a.learners, a.baselines)
