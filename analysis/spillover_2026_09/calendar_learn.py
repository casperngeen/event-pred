#!/usr/bin/env python
"""Do the extra calendar releases make the graph *learnable* — without theory?

    venv/bin/python -W ignore analysis/spillover_2026_09/calendar_learn.py \
        > analysis/spillover_2026_09/out/calendar_learn.txt       # needs calendar.py first

``releases.py`` shows the Kalshi markets respond to 34 non-Kalshi releases in
the theory direction. That uses theory's signs. The graph-learning question is
whether the data alone can now find the direction of each release → market
edge, which failed for the Kalshi releases, the jumps and order flow alike.

Walk-forward, per horizon, predicting the direction of each target's move
(ΔB ≠ 0) after each release:

  learned, per edge        sign(z) × sign of Σ sign(z)·ΔB over that
                           (release, target) pair's earlier releases (≥ 5)
  learned, per family      the same, pooled over the release family
                           (inflation / activity / labour / sentiment) for
                           the target: more data per estimate
  theory                   sign(z)·HAWKISH[release]·HAWKISH[target]

Balanced accuracy (mean hit rate on up and down moves), CI bootstrap over
release instants. Placebo: the same learners after flipping every release's
surprise sign at random (one fixed draw), which must score 0.5.

Then the full-sample learned graph: per (release, target) edge with ≥ 15
responses, the sign of Spearman(z, ΔB), compared with theory — balanced sign
accuracy (theory-positive and theory-negative edges averaged) — and the edges
that survive BH-FDR on their own.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.structure.stats import benjamini_hochberg, spearman, spearman_p

OUT = "analysis/spillover_2026_09/out"
N_BOOT, MIN_HIST = 2000, 5
rng = np.random.default_rng(0)

R = pl.read_parquet(f"{OUT}/releases.parquet")
flip = dict(zip(R["rel"].unique().sort(), rng.choice([-1.0, 1.0], R["rel"].n_unique())))
R = R.with_columns(pl.col("rel").replace_strict(flip, return_dtype=pl.Float64).alias("flip"))


def walk(d, h, key, zcol):
    d = d.filter(pl.col(h).is_finite()).sort("t")
    out = []
    for _, g in d.group_by(key, maintain_order=True):
        sz = np.sign(g[zcol].to_numpy())
        dB, t = g[h].to_numpy(), g["t"].to_numpy()
        prod = sz * dB
        for i in range(len(g)):
            past = prod[:i][(t[:i] < t[i]) & (dB[:i] != 0)]
            if dB[i] == 0 or len(past) < MIN_HIST or past.mean() == 0:
                continue
            out.append(dict(rel=g["rel"][i], pred=sz[i] * np.sign(past.mean()),
                            theory=sz[i] * g["hs"][i], up=dB[i] > 0))
    return pl.DataFrame(out) if out else None


def bal(p, col):
    up = p["up"].to_numpy().astype(bool)
    hit = (p[col].to_numpy() > 0) == up
    rel = p["rel"].to_numpy()
    u, inv = np.unique(rel, return_inverse=True)
    groups = [np.nonzero(inv == k)[0] for k in range(len(u))]

    def b(ix):
        return np.nanmean([hit[ix][up[ix]].mean() if up[ix].any() else np.nan,
                           hit[ix][~up[ix]].mean() if (~up[ix]).any() else np.nan])
    bs = [b(np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))]).astype(int))
          for _ in range(N_BOOT)]
    return len(hit), b(np.arange(len(hit))), *np.nanpercentile(bs, [2.5, 97.5])


print("Out-of-sample direction of each Kalshi target's move after a non-Kalshi release;\n"
      "balanced accuracy, 0.5 = chance. The direction is learned from earlier releases only.\n")
print(f"{'horizon':7} {'surprise':9} {'predictor':20} {'n':>6} {'bal acc [95% CI]':>24}")
for h in ("+1h", "+4h", "+24h"):
    for zcol, lab in (("z", "real"), ("zflip", "flipped")):
        d = R.with_columns((pl.col("z") * pl.col("flip")).alias("zflip"))
        pe = walk(d, h, ["ev", "target"], zcol)
        pf = walk(d, h, ["family", "target"], zcol)
        for name, p, col in (("learned, per edge", pe, "pred"), ("learned, per family", pf, "pred"),
                             ("theory", pf, "theory")):
            if p is not None and name != "theory" or (name == "theory" and zcol == "z" and p is not None):
                n, b, lo, hi = bal(p, col)
                print(f"{h:7} {lab:9} {name:20} {n:>6} {b:>7.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    print()

print("Full-sample learned graph: sign of Spearman(z, ΔB) per (release, target) edge, n ≥ 15, +4h")
edges = []
for (ev, tgt), g in sorted(R.filter(pl.col("+4h").is_finite()).group_by("ev", "target"), key=lambda kv: kv[0]):
    if g.height < 15:
        continue
    r = spearman(g["z"].to_numpy(), g["+4h"].to_numpy())
    if np.isfinite(r):
        edges.append(dict(ev=ev, target=tgt, n=g.height, rho=r, p=spearman_p(r, g.height),
                          theory=float(np.sign(g["hs"][0]))))
E = pl.DataFrame(edges)
E = E.with_columns(pl.Series("bh", benjamini_hochberg(E["p"].to_numpy(), 0.10)))
pos, neg = E.filter(pl.col("theory") > 0), E.filter(pl.col("theory") < 0)
hp = float((np.sign(pos["rho"]) > 0).mean()) if pos.height else np.nan
hn = float((np.sign(neg["rho"]) < 0).mean()) if neg.height else np.nan
print(f"  {E.height} edges (theory + {pos.height}, − {neg.height}); learned sign matches theory on "
      f"{hp:.2f} of + edges and {hn:.2f} of − edges: balanced {np.nanmean([hp, hn]):.3f}")
print(f"  edges surviving BH on their own (q = 0.10): {int(E['bh'].sum())}")
for r in E.filter(pl.col("bh")).sort("p").iter_rows(named=True):
    print(f"    {r['ev']:38} → {r['target']:14} ρ {r['rho']:+.3f}  n {r['n']:>3}  p {r['p']:.4f}  "
          f"theory {'+' if r['theory'] > 0 else '−'}")
