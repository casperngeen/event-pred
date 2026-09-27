#!/usr/bin/env python
"""Which way of building the graph recovers a structure we know is there? (plan step 2–3)

    venv/bin/python -W ignore analysis/sports_2026_09/graphs.py > analysis/sports_2026_09/out/graphs.txt

Needs ``build_mlb.py``. Nodes: the 30 MLB teams. Trigger: the team's game
surprise on a slate day (z = Σ 1[won] − p_pre over its games that day; 0 if it
did not play). Response: the team's *division* contract (the most responsive
futures, ``diffusion.py``) from the slate's start to its end, where it traded.
Regular season only (division contracts settle at its end, 2025-09-28).

Ground truth (A), mechanical: a→a is + (a win raises your own division odds),
a→b is − for division rivals, 0 otherwise. 900 ordered candidate pairs, 150
true edges (30 self, 120 rival).

Graphs scored against (A):
  (B)  entity-from-titles   markets linked when their team strings match
                            (game ``yes_sub_title`` vs division ``yes_sub_title``)
  (C1) title embedding      cosine of sentence-embedded team futures titles
                            (all-MiniLM-L6-v2): pure semantics of the name
  (C2) title co-occurrence  how often two teams appear in the same game title
                            (the schedule, read from text): the news-co-occurrence
                            graph of STG_stocks.pdf, unsigned
  (D)  price correlation    correlation of the teams' daily division-price changes
  (E)  learned from surprise → response:
         pairwise Spearman (+ BH), free linear graph (ridge), lasso,
         diagonal + rank-6 graph (torch), AGCRN adaptive and signed
         (effective edge ∂ŷ_b/∂z_a), and AGCRN with (A) frozen as a check
       each fitted on the season up to 2025-05-15, -06-15, -07-15, -08-15, -09-28

Metrics: AUROC of |edge| for true vs false over all 900 pairs and over the 870
off-diagonal pairs (rivals only: the cross-market part); balanced sign accuracy
on the true edges (self + vs rival −); precision at 150.
"""
from __future__ import annotations

import sys
import zlib

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN                                    # read-only use
from stg.models.tensors import apply_feature_scaler, fit_feature_scaler
from stg.models.train import fit_fold
from stg.structure.stats import benjamini_hochberg, spearman, spearman_p

OUT = "analysis/sports_2026_09/out"
SEASON_END = "2025-09-28"
CUTS = ["2025-05-15", "2025-06-15", "2025-07-15", "2025-08-15", SEASON_END]
L, SEEDS = 6, (0, 1, 2)
torch.set_num_threads(4)

daily = (pl.read_parquet(f"{OUT}/mlb_daily.parquet")
         .filter(pl.col("slate") <= pl.lit(SEASON_END).str.to_date()).sort("slate", "team"))
teams_df = pl.read_parquet(f"{OUT}/mlb_teams.parquet")
teams = sorted(teams_df["team"])
N = len(teams)
div = dict(teams_df.select("team", "division").iter_rows())
days = daily["slate"].unique().sort().to_numpy()
T = len(days)


def grid(col, fill=None):
    """slate day × team array of ``col``; NaN where missing unless ``fill`` is given."""
    g = daily.pivot(on="team", index="slate", values=col).sort("slate").select(teams)
    a = g.to_numpy().astype(float)
    return a if fill is None else np.nan_to_num(a, nan=fill)


Z = grid("z", 0.0)
PLAYED = grid("games", 0.0) > 0
DP0, DP1, DN = grid("division_p0"), grid("division_p1"), grid("division_n", 0.0)
Y = np.where((DN > 0) & np.isfinite(DP0) & np.isfinite(DP1), DP1 - DP0, np.nan)
print(f"{T} slate days × {N} teams; labelled team-days {int(np.isfinite(Y).sum())}; "
      f"team-days played {int(PLAYED.sum())}")

W_A = np.array([[1.0 if a == b else (-1.0 if div[ta] == div[tb] else 0.0)
                 for b, tb in enumerate(teams)] for a, ta in enumerate(teams)])
TRUE = W_A != 0
OFF = ~np.eye(N, dtype=bool)


def score(W, signed=True):
    s = np.nan_to_num(W).ravel()
    t = TRUE.ravel()
    out = {}
    live = np.abs(s).max() > 0
    out["auroc"] = roc_auc_score(t, np.abs(s)) if live else 0.5
    o = OFF.ravel()
    out["auroc_rival"] = roc_auc_score(t[o], np.abs(s[o])) if live and np.abs(s[o]).max() > 0 else 0.5
    out["prec"] = float(t[np.argsort(-np.abs(s), kind="stable")[:TRUE.sum()]].mean()) if live else np.nan
    if signed and live:
        hs = [(np.sign(W[TRUE & (W_A == v)]) == v).mean() for v in (1, -1)]
        out["hit_self"], out["hit_rival"], out["sign_bal"] = hs[0], hs[1], float(np.mean(hs))
    else:
        out["hit_self"] = out["hit_rival"] = out["sign_bal"] = np.nan
    return out


RESULTS = []


def record(name, W, cut="full", signed=True, n_days=T):
    RESULTS.append(dict(graph=name, cut=cut, days=n_days, **score(W, signed)))


games = pl.read_parquet(f"{OUT}/mlb_games.parquet")


def cooccur(cut):
    g = games.filter(pl.col("slate") <= pl.lit(cut).str.to_date())
    W = np.zeros((N, N))
    ix = {t: i for i, t in enumerate(teams)}
    for a, b in g.select("team", "opp").iter_rows():
        W[ix[a], ix[b]] += 1
    return W


# ------------------------------------------------------------------ (D), (E)
def corr_graph(m):
    Yc = Y[m]
    W = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            ok = np.isfinite(Yc[:, a]) & np.isfinite(Yc[:, b])
            if a != b and ok.sum() >= 10 and Yc[ok, a].std() > 0 and Yc[ok, b].std() > 0:
                W[a, b] = np.corrcoef(Yc[ok, a], Yc[ok, b])[0, 1]
    return W


def pairwise(m):
    W, P = np.zeros((N, N)), np.full((N, N), np.nan)
    for a in range(N):
        for b in range(N):
            sel = m & PLAYED[:, a] & np.isfinite(Y[:, b])
            if sel.sum() >= 10:
                r = spearman(Z[sel, a], Y[sel, b])
                if np.isfinite(r):
                    W[a, b], P[a, b] = r, spearman_p(r, int(sel.sum()))
    found = np.zeros(N * N, bool)
    ok = np.isfinite(P.ravel())
    found[ok] = benjamini_hochberg(P.ravel()[ok], 0.10)
    return W, found.reshape(N, N)


def ridge_graph(m, lam=10.0):
    W = np.zeros((N, N))
    for b in range(N):
        rows = m & np.isfinite(Y[:, b])
        if rows.sum() < 15:
            continue
        X1 = np.column_stack([np.ones(rows.sum()), Z[rows]])
        A = X1.T @ X1 + lam * np.eye(N + 1)
        A[0, 0] -= lam
        W[:, b] = np.linalg.solve(A, X1.T @ Y[rows, b])[1:]
    return W


def lasso_graph(m):
    W = np.zeros((N, N))
    for b in range(N):
        rows = m & np.isfinite(Y[:, b])
        if rows.sum() < 25:
            continue
        W[:, b] = LassoCV(cv=5, alphas=30, max_iter=5000).fit(Z[rows], Y[rows, b]).coef_
    return W


class DiagLowRank(nn.Module):
    """y_b = c_b + d_b·z_b + Σ_a (E_s E_dᵀ)_ab·z_a (off-diagonal), rank 6."""

    def __init__(self, r=6):
        super().__init__()
        self.d = nn.Parameter(torch.zeros(N))
        self.Es = nn.Parameter(torch.randn(N, r) * 0.1)
        self.Ed = nn.Parameter(torch.randn(N, r) * 0.1)
        self.c = nn.Parameter(torch.zeros(N))
        self.register_buffer("off", torch.tensor(OFF, dtype=torch.float32))

    def adj(self):
        return torch.diag(self.d) + (self.Es @ self.Ed.t()) * self.off

    def forward(self, seq, mask):
        return (seq[:, -1, :, 0] @ self.adj() + self.c)[..., None]


# node features for the torch models (per day): z, played, division price, its
# previous-slate change, pennant and championship price (as-of at slate start)
PP0, CP0 = grid("pennant_p0"), grid("champ_p0")
prev = np.vstack([np.full((1, N), np.nan), Y[:-1]])
FEATS = np.stack([Z, PLAYED.astype(float), DP0, np.nan_to_num(prev), PP0, CP0], -1)
for f in range(FEATS.shape[-1]):
    col = FEATS[..., f]
    FEATS[..., f] = np.where(np.isfinite(col), col, np.nanmedian(col, axis=0, keepdims=True))
FEATS = np.nan_to_num(FEATS).astype(np.float32)
ACTIVE = np.isfinite(DP0)


def windows():
    idx = np.arange(L - 1, T)
    return dict(Xs=np.stack([FEATS[t - L + 1:t + 1] for t in idx]),
                Ms=np.stack([ACTIVE[t - L + 1:t + 1] for t in idx]),
                y=np.nan_to_num(Y[idx]), ym=np.isfinite(Y[idx]), day=days[idx])


WIN = windows()
YSD = np.nanstd(Y)


def torch_graph(kind, m, seed, cols=None):
    torch.manual_seed(seed)
    wi = np.isin(WIN["day"], days[m])
    d = WIN["day"][wi]
    es_cut = np.sort(d)[int(len(d) * 0.85)]
    fit, es = wi & (WIN["day"] < es_cut), wi & (WIN["day"] >= es_cut)
    cols = cols or list(range(FEATS.shape[-1]))
    if kind == "lowrank":
        prep_x = lambda s: torch.tensor(WIN["Xs"][s][..., [0]])   # noqa: E731  raw z
        make = DiagLowRank
    else:
        mu, sd = fit_feature_scaler(WIN["Xs"][fit], WIN["Ms"][fit])
        prep_x = lambda s: torch.tensor(apply_feature_scaler(WIN["Xs"][s], mu, sd)[..., cols])  # noqa: E731
        cfg = dict(hidden=16, d_emb=6, masking="per_step", zero_head=True, n_horizons=1)
        make = {"adaptive": lambda: AGCRN(N, len(cols), embedding="hybrid", **cfg),
                "signed": lambda: AGCRN(N, len(cols), embedding="learned", adjacency="signed", **cfg),
                "frozenA": lambda: AGCRN(N, len(cols), embedding="hybrid", adjacency="stage1",
                                         stage1_adj=W_A, **cfg)}[kind]

    def prep(s):
        return (prep_x(s), torch.tensor(WIN["Ms"][s]),
                torch.tensor(WIN["y"][s] / YSD, dtype=torch.float32), torch.tensor(WIN["ym"][s]))
    model, _ = fit_fold(make, prep(fit), prep(es))
    if kind == "lowrank":
        return model.adj().detach().numpy()
    # effective edge ∂ŷ_b/∂z_a on the training windows where a played and b is labelled
    X, Mk = prep(wi)[:2]
    X = X.clone().requires_grad_(True)
    out = model(X, Mk).squeeze(-1)
    pl_w, lab = WIN["Xs"][wi][:, -1, :, 1] > 0, WIN["ym"][wi]
    zc = cols.index(0)
    J = np.zeros((N, N))
    for b in range(N):
        g, = torch.autograd.grad(out[:, b].sum(), X, retain_graph=True)
        g = g[:, -1, :, zc].numpy()
        for a in range(N):
            sel = pl_w[:, a] & lab[:, b]
            if sel.any():
                J[a, b] = g[sel, a].mean()
    return J


def main():
    # ------------------------------------------------------------------ (A), (B), (C)
    record("(A) mechanical (truth)", W_A)
    mk = (pl.scan_parquet(sorted(__import__("glob").glob("data/markets/*.parquet")))
          .with_columns(pl.col("event_ticker").str.split("-").list.first().alias("series"),
                        pl.col("ticker").str.split("-").list.last().alias("code"))
          .filter(pl.col("series").str.starts_with("KXMLB"))
          .select("series", "code", "title", "yes_sub_title").collect())
    ALIAS = {"A": "ATH", "ARI": "AZ", "KAN": "KC", "FLA": "MIA", "WAS": "WSH"}
    mk = mk.with_columns(pl.col("code").replace(ALIAS).alias("team"))
    norm = lambda s: " ".join(str(s).lower().replace(".", "").split())     # noqa: E731
    game_name = {t: norm(n) for t, n in mk.filter(pl.col("series") == "KXMLBGAME")
                 .group_by("team").agg(pl.col("yes_sub_title").mode().first()).iter_rows() if t in div}
    divf = mk.filter(pl.col("series").str.contains("EAST|CENT|WEST"))
    fut_name = {t: norm(n) for t, n in divf.group_by("team").agg(pl.col("yes_sub_title").first()).iter_rows()}
    W_B = np.array([[1.0 if game_name.get(ta) and game_name.get(ta) == fut_name.get(tb) else 0.0
                     for tb in teams] for ta in teams])
    print(f"(B) entity match: {int(np.trace(W_B))}/30 teams matched to their own division contract; "
          f"{int(W_B.sum() - np.trace(W_B))} cross-team matches")
    record("(B) entity from titles", W_B, signed=True)

    from sentence_transformers import SentenceTransformer  # noqa: E402

    champ_title = dict(mk.filter(pl.col("series") == "KXMLB").select("team", "title").iter_rows())
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").encode(
        [champ_title[t] for t in teams], normalize_embeddings=True)
    W_C1 = emb @ emb.T
    record("(C1) title embedding (team futures titles)", W_C1, signed=False)
    for cut in CUTS:
        m = days <= np.datetime64(cut)
        nd = int(m.sum())
        record("(C2) title co-occurrence (schedule)", cooccur(cut), cut, signed=False, n_days=nd)
        record("(D) price-change correlation", corr_graph(m), cut, signed=True, n_days=nd)
        W, found = pairwise(m)
        record("(E) pairwise Spearman", W, cut, n_days=nd)
        record("(E) pairwise Spearman, BH-selected", np.where(found, W, 0), cut, n_days=nd)
        record("(E) free linear graph (ridge)", ridge_graph(m), cut, n_days=nd)
        record("(E) lasso", lasso_graph(m), cut, n_days=nd)
        for kind, label in (("lowrank", "(E) diagonal + rank-6 graph"), ("adaptive", "(E) AGCRN adaptive"),
                            ("signed", "(E) AGCRN signed"), ("frozenA", "(A) frozen in AGCRN (check)")):
            Ws = [torch_graph(kind, m, s) for s in SEEDS]
            record(label, np.mean(Ws, 0), cut, n_days=nd)
        for kind in ("adaptive", "signed"):
            Ws = [torch_graph(kind, m, s, cols=[0, 1]) for s in SEEDS]
            record(f"(E) AGCRN {kind}, surprise-only inputs", np.mean(Ws, 0), cut, n_days=nd)
        print(f"cut {cut}: {nd} days fitted", flush=True)

    R = pl.DataFrame(RESULTS)
    R.write_parquet(f"{OUT}/graphs.parquet")
    pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(48)
    print("\nAUROC: |edge| ranks true (self + division-rival) above false edges, 0.5 = chance;\n"
          "rival: the 870 off-diagonal pairs only. sign: balanced accuracy on the true edges\n"
          "(hit rate on self + and on rival −, averaged); – for unsigned graphs.\n")
    print(R.select("graph", "cut", "days", *[pl.col(c).round(3) for c in
                                             ("auroc", "auroc_rival", "prec", "hit_self", "hit_rival", "sign_bal")]))


if __name__ == "__main__":
    main()
