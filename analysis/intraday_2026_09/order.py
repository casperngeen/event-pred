#!/usr/bin/env python
"""Order and lead-lag inside a release (Q3): propagation through the network, or a common reaction?

    venv/bin/python -W ignore analysis/intraday_2026_09/order.py \
        > analysis/intraday_2026_09/out/order.txt          # needs build_paths.py

At τ ∈ {15 min, 1 h, 4 h} after a release, for target B's remaining move to +24 h:

  r_B(τ → 24h) ~ 1 + sig_B + r_A(0 → τ) + r_B(0 → τ) + flow_B(0 → τ)

  sig_B   theory-signed surprise signal to B (cents per unit are comparable
          with the curves)
  r_A     the move of the source's next contract (the released series with the
          largest |z|), oriented by HAWKISH[A]·HAWKISH[B] so + = the direction
          that theory says carries to B
  r_B     B's own move so far: + = continuation (underreaction),
          − = reversal (overshoot, or bid-ask bounce)
  flow_B  signed taker flow since the release, sign·log(1 + |contracts|)

A significant r_A term over sig_B is propagation *through* the network: the
source market's move tells B something the headline number doesn't.

Cells: Kalshi cross cells with a live source contract (same cells in every
spec), Kalshi own cells (no r_A), calendar cross cells (no source market). Only
cells untruncated to +24 h. Full-sample OLS with SEs clustered by release
instant; then walk-forward (the 8 expanding folds by release instant of
``stg.models.train._fold_cuts``, training releases purged when their +24 h
ends within ``PURGE`` of the fold cut): out-of-sample R² vs zero of the nested
specs, and ΔR² against sig only with a 95% CI from a bootstrap over instants.
Bounce check: the same fits with the 2-print-mean price for every move.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.models.baselines import _ridge
from stg.models.train import PURGE, _fold_cuts

sys.path.insert(0, "analysis/intraday_2026_09")
from _common import KEY, cells, paths  # noqa: E402

HAWKISH = {                       # _tape.HAWKISH; importing _tape would reload the whole tape
    "CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
    "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1, "CPIFOOD": +1, "CPIAPPAREL": +1,
    "PAYROLLS": +1, "ADP": +1, "U3": -1, "JOBLESSCLAIMS": -1,
    "GDP": +1, "ISMPMI": +1, "FED": +1,
}
TAUS = (15, 60, 240)
N_BOOT = 2000
rng = np.random.default_rng(0)


def at(tau: int, col: str) -> pl.DataFrame:
    return (paths.filter((pl.col("tau_min") == tau) & ~pl.col("trunc_k"))
            .select(*KEY, pl.col(col).alias(f"{col}@{tau}"), pl.col("flow").alias(f"flow@{tau}")))


def frame(arm: str, role: str, tau: int, col: str) -> pl.DataFrame:
    c = cells.filter((pl.col("arm") == arm) & (pl.col("role") == role) & (pl.col("cut_k_h") >= 24))
    d = (c.join(at(tau, col), on=KEY).join(at(1440, col).select(*KEY, f"{col}@1440"), on=KEY)
         .with_columns((pl.col(f"{col}@1440") - pl.col(f"{col}@{tau}")).alias("y"),
                       pl.col(f"{col}@{tau}").alias("rB"),
                       (pl.col(f"flow@{tau}").sign() * pl.col(f"flow@{tau}").abs().log1p()).alias("fB"),
                       pl.when(pl.col("role") == "own").then(pl.col("x_own")).otherwise(pl.col("sig"))
                       .alias("s"))
         .drop_nulls(["y", "rB"]))
    if role == "cross" and arm == "kalshi":
        # the source's next contract = the own cell of src_A at the same instant
        a = (paths.filter((pl.col("arm") == arm) & (pl.col("tau_min") == tau))
             .join(cells.filter(pl.col("role") == "own").select(*KEY), on=KEY, how="semi")
             .select("t_rel", pl.col("target").alias("src_A"), pl.col(col).alias("rA_raw")))
        d = (d.join(a, on=["t_rel", "src_A"], how="inner")
             .with_columns((pl.col("rA_raw") * pl.col("src_A").replace_strict(HAWKISH, return_dtype=pl.Float64)
                            * pl.col("target").replace_strict(HAWKISH, return_dtype=pl.Float64)).alias("rA"))
             .drop_nulls("rA"))
    return d.sort("t_rel")


def ols_cluster(X, y, g):
    """OLS β and CR1 cluster-robust SEs (clusters g)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    u, inv = np.unique(g, return_inverse=True)
    S = np.zeros((X.shape[1], X.shape[1]))
    for k in range(len(u)):
        s = X[inv == k].T @ e[inv == k]
        S += np.outer(s, s)
    n, p, G = len(y), X.shape[1], len(u)
    V = XtX_inv @ S @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - p))
    return b, np.sqrt(np.diag(V))


def walk_forward(d: pl.DataFrame, specs: dict[str, list[str]]):
    """Out-of-fold predictions per spec (ridge λ = 1 on standardised columns)."""
    t = d["t_rel"].to_numpy().astype("datetime64[us]")
    ends = t + np.timedelta64(24, "h")
    cuts = _fold_cuts(t, 8)
    y = d["y"].to_numpy()
    preds = {k: np.full(len(y), np.nan) for k in specs}
    for i in range(8):
        tr = ends < (cuts[i] - PURGE)
        te = (t >= cuts[i]) & (t < cuts[i + 1])
        if tr.sum() < 30 or te.sum() == 0:
            continue
        for k, cols in specs.items():
            X = d.select(cols).to_numpy().astype(float)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            coef = _ridge((X[tr] - mu) / sd, y[tr], 1.0)
            preds[k][te] = np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd]) @ coef
    return y, preds


def oos_table(d: pl.DataFrame, specs: dict[str, list[str]], ref: str):
    y, P = walk_forward(d, specs)
    m = np.all([np.isfinite(p) for p in P.values()], axis=0)
    inst = np.unique(d["t_rel"].to_numpy()[m], return_inverse=True)[1]
    n_inst = inst.max() + 1
    ym = y[m]
    sse = {k: (ym - p[m]) ** 2 for k, p in P.items()}
    s0 = ym ** 2
    out = []
    for k in specs:
        r2 = 1 - sse[k].sum() / s0.sum()
        if k == ref:
            out.append(f"{k}: R² {r2:+.4f}")
            continue
        bs = []
        for _ in range(N_BOOT):
            w = np.bincount(rng.integers(0, n_inst, n_inst), minlength=n_inst)[inst]
            bs.append((w * (sse[ref] - sse[k])).sum() / (w * s0).sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append(f"{k}: R² {r2:+.4f} (Δ {r2 - (1 - sse[ref].sum() / s0.sum()):+.4f} [{lo:+.4f},{hi:+.4f}])")
    return f"n={int(m.sum())}, instants={n_inst}; " + " | ".join(out)


def block(title: str, arm: str, role: str, terms: list[str], col: str):
    print(f"\n{'=' * 120}\n{title}  [price: {'last print' if col == 'r' else '2-print mean'}]\n{'=' * 120}")
    print(f"{'τ':>5} {'n':>5} {'inst':>5} " + " ".join(f"{t:>24}" for t in ["const"] + terms))
    frames = {}
    for tau in TAUS:
        d = frame(arm, role, tau, col)
        frames[tau] = d
        if d.height < 30:
            print(f"{tau:>4}m  too few cells ({d.height})")
            continue
        X = np.column_stack([np.ones(d.height)] + [d[t].to_numpy().astype(float) for t in terms])
        b, se = ols_cluster(X, d["y"].to_numpy(), d["t_rel"].to_numpy())
        print(f"{tau:>4}m {d.height:>5} {d['t_rel'].n_unique():>5} " + " ".join(
            f"{bi:>+9.4f} ({bi / si:>+5.2f}t)     " for bi, si in zip(b, se)), flush=True)
    return frames


TERMS_CROSS = ["s", "rA", "rB", "fB"]
TERMS_OWN = ["s", "rB", "fB"]
for col in ("r", "r2"):
    fk = block("KALSHI CROSS: remaining move r_B(τ→24h) ~ sig + r_A + r_B + flow_B (t: clustered by instant)",
               "kalshi", "cross", TERMS_CROSS, col)
    fo = block("KALSHI OWN (releasing series' next contract): r(τ→24h) ~ z + r(0→τ) + flow",
               "kalshi", "own", TERMS_OWN, col)
    fc = block("CALENDAR CROSS: r_B(τ→24h) ~ sig + r_B + flow_B", "calendar", "cross", TERMS_OWN, col)
    if col != "r":
        continue
    fk_r = fk
    print(f"\n{'=' * 120}\nWALK-FORWARD out-of-sample R² vs zero; Δ vs 'sig' only, 95% CI (bootstrap over instants)"
          f"\n{'=' * 120}")
    for tau in TAUS:
        print(f"\nτ = {tau} min")
        if fk[tau].height >= 30:
            print("  kalshi cross   " + oos_table(fk[tau], {"sig": ["s"], "+own path": ["s", "rB", "fB"],
                                                            "+source move": ["s", "rA"],
                                                            "full": ["s", "rA", "rB", "fB"]}, "sig"))
        if fo[tau].height >= 30:
            print("  kalshi own     " + oos_table(fo[tau], {"sig": ["s"], "+own path": ["s", "rB", "fB"]}, "sig"))
        if fc[tau].height >= 30:
            print("  calendar cross " + oos_table(fc[tau], {"sig": ["s"], "+own path": ["s", "rB", "fB"]}, "sig"))

print("\nsource-contract coverage (Kalshi cross cells untruncated to 24 h with a live source contract):")
for tau in TAUS:
    n_all = cells.filter((pl.col("arm") == "kalshi") & (pl.col("role") == "cross") & (pl.col("cut_k_h") >= 24)).height
    print(f"  τ = {tau:>3} min: {fk_r[tau].height}/{n_all}; source traded by τ in "
          f"{(fk_r[tau]['rA'] != 0).mean():.2f} of them")
