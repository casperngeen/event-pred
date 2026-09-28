#!/usr/bin/env python
"""Can a temporal model learn the within-release path? (Q4)

    venv/bin/python -W ignore analysis/intraday_2026_09/models.py \
        > analysis/intraday_2026_09/out/models.txt        # needs build_paths.py; writes out/oof_models.parquet
    venv/bin/python -W ignore analysis/intraday_2026_09/models.py --price mean2 \
        > analysis/intraday_2026_09/out/models_mean2.txt  # bid-ask bounce check: moves on the 2-print mean

At every bar τ ≤ 24 h after a release (Kalshi and calendar arms together), predict
each target's move over the next window from what is known at τ:

  1h   p(τ′) − p(τ), τ′ = the first bar ≥ τ + 60 min
  24h  p(24 h) − p(τ)

Rungs, in order:
  1  pooled curve      s·(β_g(τ′) − β_g(τ)): the theory-signed signal s times
                       the remaining slope of one absorption curve per group g
                       (Kalshi own / Kalshi cross / calendar cross), fitted in-fold
  2  channel curve     ridge on channel (source family × target type, own) ×
                       τ-bucket signal columns and bucket intercepts
  3  linear path state rung 2 + the path so far × τ-bucket: own move, traded
                       flag, prints, minutes since last print, signed flow, the
                       source contract's move (oriented)
                       (3b: the same, fitted on bars where the target trades
                       inside the window only)
  4  GRU               one GRU over the bars per target, weights shared across
                       targets plus a learned node id; a head at every bar
                       (4b: the same, with the loss on bars where the target
                       trades inside the window only, ``torch_check.py``)
  5  AGCRN             step = bar, nodes = the 17 targets, messages = the other
                       markets' state so far; adaptive graph, and a frozen
                       signed economic graph (HAWKISH·HAWKISH)

Inputs at bar τ are as-of τ only (the path panel's as-of prices, flows and
counts, the release's surprises and the pre-release state). Labels in per-series
sd units (units choice over all labelled cells), clipped at ±5 sd for fitting.

Walk-forward: the 8 expanding folds by release instant (``_fold_cuts``); every
bar of a release is in one fold; training releases are purged when their +24 h
end is within ``PURGE`` of the fold cut. Torch rungs early-stop on the last 15%
of training releases (fit releases end before that block starts); 2 seeds.

Metrics: R² vs 0, balanced accuracy, AUC (y ≠ 0), on all labelled bars and on
bars where the target trades inside the window, by arm and for Kalshi own
cells. Δ vs the best linear rung (1–3b, by R² on that subset) with a 95% CI
from a bootstrap over release instants.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as TF

sys.path.insert(0, "stg_infra")
from stg.models.agcrn import AGCRNCell
from stg.models.train import PURGE, _fold_cuts
from stg.panel.registry import is_same_release

sys.path.insert(0, "analysis/intraday_2026_09")
from _common import KEY, OUT, TAU_POST, TTYPES, cells, paths, wls  # noqa: E402

torch.set_num_threads(4)
rng = np.random.default_rng(0)
# --price mean2: every move (features and labels) on the mean of the last two
# prints instead of the last print, the bid-ask bounce check
_ap = argparse.ArgumentParser()
_ap.add_argument("--price", choices=("last", "mean2"), default="last")
PRICE = _ap.parse_known_args()[0].price
SUFFIX = "" if PRICE == "last" else "_mean2"
SEEDS = (0, 1)
N_BOOT = 500
HAWKISH = {                       # _tape.HAWKISH; importing _tape would reload the whole tape
    "CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
    "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1, "CPIFOOD": +1, "CPIAPPAREL": +1,
    "PAYROLLS": +1, "ADP": +1, "U3": -1, "JOBLESSCLAIMS": -1,
    "GDP": +1, "ISMPMI": +1, "FED": +1,
}
FAMS = [("kalshi", f) for f in ("inflation", "labour", "growth", "policy")] + \
       [("calendar", f) for f in ("inflation", "activity", "labour", "sentiment")]
LABELS = ("1h", "24h")

# ------------------------------------------------------------------ tensors (I, J, N)
TAUS = TAU_POST[TAU_POST <= 1440]
J, J24 = len(TAUS), len(TAUS) - 1
END1 = np.array([np.searchsorted(TAUS, t + 60) for t in TAUS])          # J = none
BUCKET = np.searchsorted([15, 60, 120, 360], TAUS, side="left")         # 0..4
NB = 5
C = cells.filter(pl.col("arm").is_in(["kalshi", "calendar"]))
nodes = sorted(C["target"].unique().to_list())
N = len(nodes)
inst = C.select("arm", "t_rel").unique().sort("t_rel", "arm").with_row_index("i")
I = inst.height
C = C.join(inst, on=["arm", "t_rel"]).with_columns(
    pl.col("target").replace_strict({n: k for k, n in enumerate(nodes)}, return_dtype=pl.Int64).alias("n"))
ci, cn = C["i"].to_numpy(), C["n"].to_numpy()


def cell_arr(col, fill=0.0):
    a = np.full((I, N), fill, dtype=float)
    a[ci, cn] = C[col].fill_null(fill).to_numpy().astype(float)
    return a


EXISTS = np.zeros((I, N), bool)
EXISTS[ci, cn] = True
ROLE = np.full((I, N), "", dtype=object)
ROLE[ci, cn] = C["role"].to_numpy()
IS_K = (inst["arm"] == "kalshi").to_numpy()
XOWN, SIG, P0, NPRE = cell_arr("x_own"), cell_arr("sig"), cell_arr("p0"), cell_arr("n_pre7")
XF = {f"{a}:{f}": cell_arr(f"x_{f}") * (IS_K if a == "kalshi" else ~IS_K)[:, None] for a, f in FAMS}
TT = np.array([TTYPES.index(C.filter(pl.col("target") == n)["ttype"][0]) for n in nodes])
H = np.array([HAWKISH[n] for n in nodes], float)

P = (paths.filter(pl.col("arm").is_in(["kalshi", "calendar"]) & pl.col("tau_min").is_in(TAUS.tolist()))
     .join(C.select(*KEY, "i", "n"), on=KEY)
     .with_columns(pl.col("tau_min").replace_strict({int(t): k for k, t in enumerate(TAUS)},
                                                    return_dtype=pl.Int64).alias("j")))
pi, pj, pn = P["i"].to_numpy(), P["j"].to_numpy(), P["n"].to_numpy()


def path_arr(col, fill=np.nan):
    a = np.full((I, J, N), fill, dtype=float)
    a[pi, pj, pn] = P[col].cast(pl.Float64).fill_null(np.nan).to_numpy()
    return a


R = path_arr("r" if PRICE == "last" else "r2")
NPOST, SINCE, FLOW = path_arr("n_post", 0), path_arr("since_last_min"), path_arr("flow", 0)
TRUNC = path_arr("trunc_k", 1.0) > 0
M = EXISTS[:, None, :] & ~TRUNC & np.isfinite(R)

# source contract's move so far, oriented by HAWKISH[A]·HAWKISH[B], on Kalshi cross cells
RA = np.zeros((I, J, N))
srcA = dict(zip(C["i"].to_numpy(), C["src_A"].to_list()))
for i in np.nonzero(IS_K)[0]:
    a = srcA.get(i)
    if a in nodes:
        a = nodes.index(a)
        if ROLE[i, a] == "own":
            ra = np.where(M[i, :, a], R[i, :, a], 0.0)
            for b in range(N):
                if ROLE[i, b] == "cross":
                    RA[i, :, b] = ra * H[a] * H[b]

# labels
Y, YM, TRADED = {}, {}, {}
for lab in LABELS:
    e = np.full(J, J24) if lab == "24h" else END1
    y, ym, tr = np.full((I, J, N), np.nan), np.zeros((I, J, N), bool), np.zeros((I, J, N), bool)
    for j in range(J):
        if e[j] >= J or e[j] == j:
            continue
        ok = M[:, j] & M[:, e[j]]
        y[:, j][ok] = (R[:, e[j]] - R[:, j])[ok]
        ym[:, j] = ok
        tr[:, j] = NPOST[:, e[j]] > NPOST[:, j]
    Y[lab], YM[lab], TRADED[lab] = y, ym, tr
LSD = {lab: np.array([np.nanstd(Y[lab][:, :, n][YM[lab][:, :, n]]) for n in range(N)]) for lab in LABELS}
for lab in LABELS:
    LSD[lab][~np.isfinite(LSD[lab]) | (LSD[lab] < 1e-6)] = 1.0
YS = {lab: np.clip(np.nan_to_num(Y[lab] / LSD[lab]), -5, 5) for lab in LABELS}

# features (I, J, N, F); zero where absent
S = np.where(ROLE == "own", XOWN, np.where(ROLE == "cross", SIG, 0.0))
feats = {
    "r": R, "traded": (NPOST > 0).astype(float), "log_prints": np.log1p(NPOST),
    "log_since": np.log1p(np.clip(np.nan_to_num(SINCE), 0, None)),
    "flow": np.sign(FLOW) * np.log1p(np.abs(FLOW)), "rA": RA,
    "x_own": np.broadcast_to(XOWN[:, None], (I, J, N)), "sig": np.broadcast_to(SIG[:, None], (I, J, N)),
    "own": np.broadcast_to((ROLE == "own")[:, None], (I, J, N)).astype(float),
    "sibling": np.broadcast_to((ROLE == "sibling")[:, None], (I, J, N)).astype(float),
    "kalshi": np.broadcast_to(IS_K[:, None, None], (I, J, N)).astype(float),
    "log_tau": np.broadcast_to(np.log(TAUS)[None, :, None], (I, J, N)),
    "p0": np.broadcast_to(P0[:, None] / 100, (I, J, N)),
    "log_pre": np.broadcast_to(np.log1p(NPRE)[:, None], (I, J, N)),
    **{f"x[{k}]": np.broadcast_to(v[:, None], (I, J, N)) for k, v in XF.items()},
}
FNAMES = list(feats)
X = np.stack([np.nan_to_num(np.asarray(v, float)) for v in feats.values()], -1) * M[..., None]
X = X.astype(np.float32)
PATH_F = [FNAMES.index(k) for k in ("r", "traded", "log_prints", "log_since", "flow", "rA")]
print(f"price: {PRICE}; instants {I} (Kalshi {IS_K.sum()}, calendar {(~IS_K).sum()}), bars {J}, nodes {N}, "
      f"features {len(FNAMES)}; present (instant, bar, node) {M.sum()}; labelled: "
      + ", ".join(f"{lab} {YM[lab].sum()} (traded in window {(YM[lab] & TRADED[lab]).sum()})" for lab in LABELS))

# ------------------------------------------------------------------ folds
DATES = inst["t_rel"].to_numpy().astype("datetime64[us]")
ENDS = DATES + np.timedelta64(24, "h")
CUTS = _fold_cuts(DATES, 8)


def folds():
    for f in range(8):
        tr = ENDS < (CUTS[f] - PURGE)
        te = (DATES >= CUTS[f]) & (DATES < CUTS[f + 1])
        if tr.sum() < 40 or te.sum() == 0:
            continue
        td = np.sort(DATES[tr])
        es_cut = td[int(len(td) * 0.85)]
        fit_m = tr & (ENDS < es_cut)
        es_m = tr & (DATES >= es_cut)
        yield f, tr, te, fit_m, es_m


# ------------------------------------------------------------------ rung 1: pooled curve
GROUP = np.where((ROLE == "own") & IS_K[:, None], 0,
                 np.where((ROLE == "cross") & IS_K[:, None], 1, np.where(ROLE == "cross", 2, -1)))


def rung_curve(tr, te):
    out = {lab: np.full((I, J, N), np.nan) for lab in LABELS}
    for g in range(3):
        ii, nn_ = np.nonzero((GROUP == g) & tr[:, None])
        if len(ii) < 30:
            continue
        Yg = np.where(M[ii, :, nn_], R[ii, :, nn_], np.nan)                  # (cells, J)
        beta = np.nan_to_num(wls(np.column_stack([np.ones(len(ii)), S[ii, nn_]]), Yg)[:, 1])
        ti, tn = np.nonzero((GROUP == g) & te[:, None])
        for lab in LABELS:
            e = np.full(J, J24) if lab == "24h" else np.minimum(END1, J - 1)
            out[lab][ti, :, tn] = S[ti, tn][:, None] * (beta[e] - beta)[None, :] / LSD[lab][tn][:, None]
    return out


# ------------------------------------------------------------------ rungs 2-3: ridge
def design(ii, jj, nn_, path: bool):
    b = BUCKET[jj]
    chans = [np.where(ROLE[ii, nn_] == "own", XOWN[ii, nn_], 0.0) * IS_K[ii]]
    for v in XF.values():
        for t in range(len(TTYPES)):
            chans.append(v[ii, nn_] * (TT[nn_] == t))
    cols = [np.eye(NB)[b][:, 1:]]
    cols += [np.stack(chans, 1)[:, :, None] * np.eye(NB)[b][:, None, :]]
    if path:
        cols += [X[ii, jj, nn_][:, PATH_F][:, :, None] * np.eye(NB)[b][:, None, :]]
    return np.concatenate([c.reshape(len(ii), -1) for c in cols], 1)


def ridge_fit(rows, y, path, lam=10.0, chunk=100_000):
    k = design(*(r[:1] for r in rows), path).shape[1] + 1
    A, bvec = np.zeros((k, k)), np.zeros(k)
    for s in range(0, len(rows[0]), chunk):
        Xd = design(*(r[s:s + chunk] for r in rows), path)
        Xd = np.column_stack([np.ones(len(Xd)), Xd])
        A += Xd.T @ Xd
        bvec += Xd.T @ y[s:s + chunk]
    reg = lam * np.eye(k)
    reg[0, 0] = 0
    return np.linalg.solve(A + reg, bvec)


def rung_ridge(tr, te, path, traded_only=False):
    out = {}
    for lab in LABELS:
        ii, jj, nn_ = np.nonzero(YM[lab] & tr[:, None, None] & (TRADED[lab] if traded_only else True))
        coef = ridge_fit((ii, jj, nn_), YS[lab][ii, jj, nn_], path)
        ti, tj, tn = np.nonzero(M & te[:, None, None])
        p = np.full((I, J, N), np.nan)
        for s in range(0, len(ti), 200_000):
            sl = slice(s, s + 200_000)
            Xd = design(ti[sl], tj[sl], tn[sl], path)
            p[ti[sl], tj[sl], tn[sl]] = coef[0] + Xd @ coef[1:]
        out[lab] = p
    return out


# ------------------------------------------------------------------ rungs 4-5: torch
class PathGRU(nn.Module):
    def __init__(self, F, hidden=32, d_id=4):
        super().__init__()
        self.hidden = hidden
        self.node = nn.Parameter(torch.randn(N, d_id) * 0.1)
        self.cell = nn.GRUCell(F + 1 + d_id, hidden)
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, m):                    # x (B, J, N, F), m (B, J, N)
        B = x.shape[0]
        h = x.new_zeros(B * N, self.hidden)
        ids = self.node.repeat(B, 1)
        out = []
        for j in range(x.shape[1]):
            mj = m[:, j].reshape(B * N, 1).float()
            inp = torch.cat([x[:, j].reshape(B * N, -1) * mj, mj, ids], -1)
            h = mj * self.cell(inp, h) + (1 - mj) * h
            out.append(self.head(h).reshape(B, N, 2))
        return torch.stack(out, 1)             # (B, J, N, 2)


class PathAGCRN(nn.Module):
    """AGCRN cell stepped over the bars with a head at every bar. ``G`` (source →
    target, signed) freezes the graph; otherwise softmax(ReLU(EEᵀ)) over the
    markets present at that bar."""

    def __init__(self, F, hidden=16, d_emb=4, G=None):
        super().__init__()
        self.hidden = hidden
        self.E = nn.Parameter(torch.randn(N, d_emb) * 0.05)
        self.cell = AGCRNCell(F + 1, hidden, d_emb)
        if G is not None:
            A = torch.tensor(G, dtype=torch.float32).t()
            self.register_buffer("A_prior", A / A.abs().sum(-1, keepdim=True).clamp_min(1e-8))
        else:
            self.A_prior = None
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, m):
        B = x.shape[0]
        h = x.new_zeros(B, N, self.hidden)
        out = []
        for j in range(x.shape[1]):
            mj = m[:, j].float()[..., None]
            inp = torch.cat([x[:, j] * mj, mj], -1)
            send = m[:, j][:, None, :]
            if self.A_prior is None:
                logits = TF.relu(self.E @ self.E.t()).expand(B, -1, -1).masked_fill(~send, float("-inf"))
                A = torch.nan_to_num(TF.softmax(logits, -1), nan=0.0)
            else:
                A = self.A_prior[None] * send.float()
            h = mj * self.cell(inp, h, self.E, A) + (1 - mj) * h
            out.append(self.head(h))
        return torch.stack(out, 1)


def econ_graph():
    G = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            if a != b and not is_same_release(nodes[a], nodes[b]):
                G[a, b] = H[a] * H[b]
    return G


def _loss(pred, y, ym):
    d = (pred - y)[ym]
    return TF.huber_loss(d, torch.zeros_like(d), delta=1.0) if d.numel() else pred.sum() * 0


def rung_torch(make, tr, te, fit_m, es_m, seed, max_epochs=100, patience=10, lr=1e-3, traded_only=False,
               batch=32):
    """Fit one torch rung with minibatches of ``batch`` releases; returns
    (predictions, history). ``traded_only`` trains on bars where the target
    trades inside the window only (``torch_check.py``). Patience is in epochs."""
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    mm = M[fit_m]
    Xf = X[fit_m][mm]
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    for k in ("own", "sibling", "kalshi") + tuple(f"x[{k}]" for k in XF) + ("x_own", "sig", "rA"):
        f = FNAMES.index(k)
        mu[f], sd[f] = 0.0, max(sd[f], 1e-6)              # keep 0 = no signal / absent
    Xs = ((X - mu) / sd * M[..., None]).astype(np.float32)
    Yt = torch.tensor(np.stack([YS[lab] for lab in LABELS], -1), dtype=torch.float32)
    YMt = torch.tensor(np.stack([YM[lab] & (TRADED[lab] if traded_only else True) for lab in LABELS], -1))

    def T(msk):
        return torch.tensor(Xs[msk]), torch.tensor(M[msk]), Yt[msk], YMt[msk]
    Xa, Ma, ya, yma = T(fit_m)
    Xe, Me, ye, yme = T(es_m)
    model = make()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad, best_ep = np.inf, None, 0, -1
    with torch.no_grad():
        zero = _loss(torch.zeros_like(ye), ye, yme).item()
    hist = dict(train=[], es=[], es_zero=zero)
    for ep in range(max_epochs):
        model.train()
        tot = 0.0
        for idx in np.array_split(gen.permutation(len(Xa)), max(1, len(Xa) // batch)):
            idx = torch.tensor(idx)
            opt.zero_grad()
            loss = _loss(model(Xa[idx], Ma[idx]), ya[idx], yma[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx)
        loss = torch.tensor(tot / len(Xa))
        model.eval()
        with torch.no_grad():
            ev = _loss(model(Xe, Me), ye, yme).item()
        hist["train"].append(loss.item())
        hist["es"].append(ev)
        if ev < best - 1e-5:
            best, bad, best_ep = ev, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pt = model(torch.tensor(Xs[te]), torch.tensor(M[te])).numpy()
    out = {}
    for k, lab in enumerate(LABELS):
        p = np.full((I, J, N), np.nan)
        p[te] = np.where(M[te], pt[..., k], np.nan)
        out[lab] = p
    return out, dict(hist, epochs=ep + 1, best_epoch=best_ep)


# ------------------------------------------------------------------ run
F = len(FNAMES)
G_ECON = econ_graph()
MODELS = {
    "1 pooled curve": ("curve", None),
    "2 channel curve (ridge)": ("ridge", False),
    "3 linear path state": ("ridge", True),
    "3b linear path state, traded-bar loss": ("ridge_traded", True),
    "4 GRU per target": ("torch", lambda: PathGRU(F)),
    "4b GRU, traded-bar loss": ("torch_traded", lambda: PathGRU(F)),
    "5a AGCRN adaptive": ("torch", lambda: PathAGCRN(F)),
    "5b AGCRN econ graph": ("torch", lambda: PathAGCRN(F, G=G_ECON)),
}
SUBSETS = {
    "all": lambda: np.ones((I, J, N), bool),
    "kalshi": lambda: np.broadcast_to(IS_K[:, None, None], (I, J, N)),
    "kalshi own": lambda: np.broadcast_to(((ROLE == "own") & IS_K[:, None])[:, None], (I, J, N)),
    "kalshi cross": lambda: np.broadcast_to(((ROLE == "cross") & IS_K[:, None])[:, None], (I, J, N)),
    "calendar": lambda: np.broadcast_to(~IS_K[:, None, None], (I, J, N)),
}


def walk_forward() -> dict:
    pred = {m: {lab: np.full((I, J, N), np.nan) for lab in LABELS} for m in MODELS}
    for f, tr, te, fit_m, es_m in folds():
        t0 = time.time()
        msg = []
        for name, (kind, arg) in MODELS.items():
            if kind == "curve":
                o = rung_curve(tr, te)
            elif kind.startswith("ridge"):
                o = rung_ridge(tr, te, arg, traded_only=kind == "ridge_traded")
            else:
                runs = [rung_torch(arg, tr, te, fit_m, es_m, s, traded_only=kind == "torch_traded")
                        for s in SEEDS]
                o = {lab: np.mean([r[0][lab] for r in runs], 0) for lab in LABELS}
                msg.append(f"{name.split()[0]}: best/stop epoch "
                           f"{[(r[1]['best_epoch'], r[1]['epochs']) for r in runs]}")
            for lab in LABELS:
                pred[name][lab][te] = o[lab][te]
        print(f"fold {f}: train {tr.sum()} / test {te.sum()} instants, {time.time() - t0:.0f}s; "
              + "; ".join(msg), flush=True)
    return pred


# ------------------------------------------------------------------ scoring
def auc_w(score, pos, w, uniq_inv):
    pw = np.bincount(uniq_inv, w * pos)
    nw = np.bincount(uniq_inv, w * ~pos)
    below = np.cumsum(nw) - nw
    den = pw.sum() * nw.sum()
    return (pw * (below + 0.5 * nw)).sum() / den if den > 0 else np.nan


class Scorer:
    """All metrics as instant-weighted sums, so the bootstrap is a reweighting."""

    def __init__(self, y, p, inst_idx):
        self.y, self.p = y, p
        self.inst = np.unique(inst_idx, return_inverse=True)[1]
        self.nz = y != 0
        self.inv = np.unique(p[self.nz], return_inverse=True)[1]
        self.pos = y[self.nz] > 0
        s = self.nz & (p != 0)
        self.s_up, self.s_pu = (y > 0) & s, (p > 0) & s
        self.s = s

    def __call__(self, w):
        y, p = self.y, self.p
        r2 = 1 - (w * (y - p) ** 2).sum() / (w * y ** 2).sum()
        up, dn = self.s_up, self.s & ~self.s_up
        tpr = (w * (up & self.s_pu)).sum() / max((w * up).sum(), 1e-12)
        tnr = (w * (dn & ~self.s_pu)).sum() / max((w * dn).sum(), 1e-12)
        return np.array([r2, (tpr + tnr) / 2, auc_w(p[self.nz], self.pos, w[self.nz], self.inv)])


def score(pred: dict, models: list[str], n_linear: int = 4) -> list[dict]:
    """Print every metric per label × subset; Δ vs the best of the first ``n_linear`` rungs."""
    covered = np.all([np.isfinite(pred[m][lab]) for m in models for lab in LABELS], axis=0)
    rows = []
    for lab in LABELS:
        for sub, fn in SUBSETS.items():
            for tr_only in (False, True):
                mask = YM[lab] & covered & fn() & (TRADED[lab] if tr_only else True)
                if mask.sum() < 100:
                    continue
                ii = np.nonzero(mask)[0]
                y = (Y[lab] / LSD[lab])[mask]
                n_inst = len(np.unique(ii))
                sc = {m: Scorer(y, pred[m][lab][mask], ii) for m in models}
                est = {m: s(np.ones(len(y))) for m, s in sc.items()}
                ref = max(models[:n_linear], key=lambda m: est[m][0])
                inst_i = sc[ref].inst
                boots = {m: [] for m in models}
                for _ in range(N_BOOT):
                    w = np.bincount(rng.integers(0, n_inst, n_inst), minlength=n_inst)[inst_i].astype(float)
                    vals = {m: s(w) for m, s in sc.items()}
                    for m in models:
                        boots[m].append(vals[m] - vals[ref])
                where = "bars where the target trades in the window" if tr_only else "all labelled bars"
                print(f"\n{'=' * 128}\nlabel {lab} | {sub} | {where} | n = {len(y)}, instants = {n_inst}, "
                      f"share up (y≠0) = {np.mean(y[y != 0] > 0):.3f} | ref: {ref}\n{'=' * 128}")
                print(f"{'model':38} {'R²':>8} {'bal acc':>7} {'AUC':>6} | {'ΔR² vs ref':>24} "
                      f"{'Δbal acc':>22} {'ΔAUC':>22}")
                for m in models:
                    e = est[m]
                    extra = ""
                    if m != ref:
                        lo, hi = np.nanpercentile(np.array(boots[m]), [2.5, 97.5], axis=0)
                        d = e - est[ref]
                        extra = " ".join(f"{d[k]:>+8.4f} [{lo[k]:>+7.4f},{hi[k]:>+7.4f}]" for k in range(3))
                    print(f"{m:38} {e[0]:>+8.4f} {e[1]:>7.3f} {e[2]:>6.3f} | {extra}", flush=True)
                    rows.append(dict(label=lab, subset=sub, traded_only=tr_only, model=m, n=len(y),
                                     instants=n_inst, r2=e[0], bal_acc=e[1], auc=e[2], ref=ref))
    return rows


def path_coefficients() -> None:
    """Full-sample rung 3: the path-state coefficients by τ bucket, in label sd per feature sd."""
    names = ["r", "traded", "log_prints", "log_since", "flow", "rA"]
    edges = ["≤15m", "15m–1h", "1–2h", "2–6h", "6–24h"]
    print(f"\n{'=' * 100}\nFULL-SAMPLE RUNG 3, path-state coefficients (label sd per 1 sd of the feature), "
          f"by τ bucket\n{'=' * 100}")
    for lab in LABELS:
        ii, jj, nn_ = np.nonzero(YM[lab])
        coef = ridge_fit((ii, jj, nn_), YS[lab][ii, jj, nn_], True)
        path = coef[1 + (NB - 1) + 33 * NB:].reshape(len(names), NB)
        sd = X[ii, jj, nn_][:, PATH_F].std(0)
        print(f"\nlabel {lab}\n{'feature':12} " + " ".join(f"{e:>9}" for e in edges))
        for k, n in enumerate(names):
            print(f"{n:12} " + " ".join(f"{path[k, b] * sd[k]:>+9.4f}" for b in range(NB)))


def main() -> None:
    pred = walk_forward()
    rows = score(pred, list(MODELS))
    pl.DataFrame(rows).write_parquet(OUT / f"models_scores{SUFFIX}.parquet")
    path_coefficients()
    covered = np.all([np.isfinite(pred[m][lab]) for m in MODELS for lab in LABELS], axis=0)
    ii, jj, nn_ = np.nonzero(covered & (YM["1h"] | YM["24h"]))
    oof = pl.DataFrame({"arm": inst["arm"].to_numpy()[ii], "t_rel": inst["t_rel"].to_numpy()[ii],
                        "target": np.array(nodes)[nn_], "tau_min": TAUS[jj],
                        **{f"y_{lab}": (Y[lab] / LSD[lab])[ii, jj, nn_] for lab in LABELS},
                        **{f"traded_{lab}": TRADED[lab][ii, jj, nn_] for lab in LABELS},
                        **{f"{m}|{lab}": pred[m][lab][ii, jj, nn_] for m in MODELS for lab in LABELS}})
    oof.write_parquet(OUT / f"oof_models{SUFFIX}.parquet")
    print(f"\nwrote {OUT / f'oof_models{SUFFIX}.parquet'} and {OUT / f'models_scores{SUFFIX}.parquet'}")


if __name__ == "__main__":
    main()
