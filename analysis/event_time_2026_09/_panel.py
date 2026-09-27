"""Shared construction of the event-time tensors, imported by the study's scripts.

Builds the release-instant × series tensors from ``out/event_nodes.parquet``
exactly as ``models.py`` always has (moved here verbatim so ``ablation.py`` uses
the same data): features X, activity mask M, labels Y / label ends END for
``imm`` and ``settle``, the economic graphs G_all / G_bh, the window builder, the
pooled ridge rungs and the AGCRN factory. Not a script — nothing runs on import
beyond building the tensors.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.models import AGCRN
from stg.models.baselines import _ridge
from stg.panel.registry import is_same_release

L = 6
PANEL = "analysis/event_time_2026_09/out/event_nodes.parquet"

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
OWN = ["p_lead", "d_q50_7d", "flow7"]      # linear own-state rung


# ------------------------------------------------------------------ tensors
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
Y = {k: np.full((T, N), np.nan) for k in ("imm", "settle")}
END = {k: np.full((T, N), np.datetime64("NaT"), dtype="datetime64[ns]") for k in Y}
for r in panel.iter_rows(named=True):
    t, i = ti[np.datetime64(r["instant"].replace(tzinfo=None), "us").tolist()], ni[r["series"]]
    for f, c in enumerate(FEATS):
        v = r.get(c)
        if v is not None:
            X[t, i, f] = np.log1p(max(v, 0)) if c in LOG else float(v)
    M[t, i] = r.get("p_lead") is not None or bool(r["released"])
    for k, col, end in (("imm", "y_imm", "imm_end"), ("settle", "y_settle", "settle_end")):
        if r.get(col) is not None and M[t, i]:
            Y[k][t, i] = r[col]
            END[k][t, i] = np.datetime64(int(r[end]), "ns")
# impute missing features with the node's median over its active cells
for i in range(N):
    for f in range(F):
        col = X[M[:, i], i, f]
        med = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        X[:, i, f] = np.where(np.isfinite(X[:, i, f]), X[:, i, f], med)
X = X.astype(np.float32)
Zf = FEATS.index("z")
Z = X[..., Zf] * (X[..., FEATS.index("released")] > 0)


def econ_graph(channels=None):
    trig = [n for n in nodes if (X[:, ni[n], FEATS.index("released")] > 0).any()]
    G = np.zeros((N, N))
    for a in trig:
        for b in nodes:
            if a != b and not is_same_release(a, b) and (
                    channels is None or (TYPE[a], TYPE[b]) in channels):
                G[ni[a], ni[b]] = HAWKISH[a] * HAWKISH[b]
    return G


G_all, G_bh = econ_graph(), econ_graph(BH_CHANNELS)


def windows(k):
    idx = [t for t in range(L - 1, T) if np.isfinite(Y[k][t]).any()]
    ym = np.stack([np.isfinite(Y[k][t]) for t in idx])
    ends = np.array([END[k][t][ym[j]].max() for j, t in enumerate(idx)])
    return {"Xs": np.stack([X[t - L + 1:t + 1] for t in idx]),
            "Ms": np.stack([M[t - L + 1:t + 1] for t in idx]),
            "y": np.nan_to_num(np.stack([Y[k][t] for t in idx])), "ym": ym,
            "dates": instants[idx], "label_end": ends.astype("datetime64[us]")}


class Ridge:
    """Pooled ridge on the last step: own-state features and/or the econ signal."""

    def __init__(self, own: bool, G=None):
        self.own, self.G = own, G

    def _f(self, win):
        x = win["Xs"][:, -1]
        cols = [x[..., FEATS.index(c)] for c in OWN] if self.own else []
        if self.G is not None:
            z = x[..., Zf] * (x[..., FEATS.index("released")] > 0)
            cols.append(z @ self.G)
        return np.stack(cols, -1) if cols else np.zeros((*x.shape[:2], 0))

    def fit(self, win):
        Fm = self._f(win)
        if Fm.shape[-1]:
            self.coef_ = _ridge(Fm[win["ym"]], win["y"][win["ym"]], 10.0)
        return self

    def predict(self, win):
        Fm = self._f(win)
        if not Fm.shape[-1]:
            return np.zeros(Fm.shape[:2])
        flat = Fm.reshape(-1, Fm.shape[-1])
        return (np.column_stack([np.ones(len(flat)), flat]) @ self.coef_).reshape(Fm.shape[:2])


def agcrn(G=None, **kw):
    cfg = dict(hidden=16, d_emb=2, masking="per_step", zero_head=True) | kw
    if G is not None:
        cfg |= dict(adjacency="stage1", stage1_adj=G)
    return lambda: AGCRN(N, F, n_horizons=1, **cfg)


def label_sd(k: str) -> np.ndarray:
    """Per-series label sd over every labelled cell (the units choice)."""
    lsd = np.array([np.nanstd(Y[k][:, i]) if np.isfinite(Y[k][:, i]).sum() > 1 else 1.0
                    for i in range(N)])
    lsd[~np.isfinite(lsd) | (lsd < 1e-9)] = 1.0
    return lsd
