#!/usr/bin/env python
"""Every out-of-fold model on the event-time panel, scored on every metric.

    venv/bin/python analysis/event_time_2026_09/metrics.py \
        > analysis/event_time_2026_09/out/metrics.txt   # needs models.py (and ablation.py) first

Reads ``out/oof_all.parquet`` (``models.py``) and, if present,
``out/oof_ablation.parquet`` (``ablation.py``). Scores each model on both
labels, in per-series z units (label and prediction divided by the series'
label sd, as the models were trained):

  R²        R² vs predict-zero
  acc       directional accuracy on cells with y ≠ 0 and ŷ ≠ 0
  bal acc   mean of the up and down hit rates (immune to the ~54% base rate)
  F1 up     F1 of the "up" class;  F1 macro: mean of the up and down F1
  AUC       ROC AUC of ŷ for y > 0 (threshold-free)

Right of '|': AUC and balanced accuracy minus the reference linear rung (the
linear rung with the best R² on that label), with a 95% CI from a bootstrap
over release instants. A graph model "beats linear" on a metric only if that
CI excludes zero.
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score

OUT = "analysis/event_time_2026_09/out"
N_BOOT = 2000
rng = np.random.default_rng(0)

panel = pl.read_parquet(f"{OUT}/event_nodes.parquet").with_columns(
    pl.col("instant").dt.replace_time_zone(None).cast(pl.Datetime("us")))
active = pl.col("p_lead").is_not_null() | pl.col("released")
# per-series label sd, exactly as models.py computes it (ddof 0, all labelled cells)
LSD = {k: dict(panel.filter(pl.col(c).is_not_null() & active).group_by("series")
               .agg(pl.col(c).std(ddof=0)).iter_rows())
       for k, c in (("imm", "y_imm"), ("settle", "y_settle"))}

frames = [pl.read_parquet(f"{OUT}/oof_all.parquet")]
if os.path.exists(f"{OUT}/oof_ablation.parquet"):
    frames.append(pl.read_parquet(f"{OUT}/oof_ablation.parquet"))
oof = pl.concat(frames, how="diagonal_relaxed")


def scores(y, p):
    s = (y != 0) & (p != 0)
    up, pu = y[s] > 0, p[s] > 0
    tpr = (pu & up).sum() / max(up.sum(), 1)
    tnr = (~pu & ~up).sum() / max((~up).sum(), 1)
    nz = y != 0
    return dict(r2=1 - ((y - p) ** 2).sum() / (y ** 2).sum(),
                acc=(pu == up).mean(), bal=(tpr + tnr) / 2,
                f1=f1_score(up, pu), f1m=f1_score(up, pu, average="macro"),
                auc=roc_auc_score(y[nz] > 0, p[nz]))


def boot_delta(d: pl.DataFrame, ref: str, model: str):
    """Δ AUC and Δ balanced accuracy (model − ref), bootstrap over instants."""
    w = (d.filter(pl.col("model").is_in([ref, model]))
         .pivot(on="model", index=["instant", "series"], values="p").drop_nulls()
         .join(d.select("instant", "series", "y").unique(), on=["instant", "series"]))
    w = w.filter(pl.col("y") != 0)
    inst = w["instant"].to_numpy()
    u, inv = np.unique(inst, return_inverse=True)
    groups = [np.nonzero(inv == g)[0] for g in range(len(u))]
    y, a, b = w["y"].to_numpy(), w[model].to_numpy(), w[ref].to_numpy()

    def stat(ix):
        sa, sb = scores(y[ix], a[ix]), scores(y[ix], b[ix])
        return sa["auc"] - sb["auc"], sa["bal"] - sb["bal"]
    est = stat(np.arange(len(y)))
    bs = []
    for _ in range(N_BOOT):
        ix = np.concatenate([groups[g] for g in rng.integers(0, len(u), len(u))])
        if len(np.unique(y[ix] > 0)) == 2:
            bs.append(stat(ix))
    bs = np.array(bs)
    return est, np.percentile(bs, [2.5, 97.5], axis=0)


for label in ("imm", "settle"):
    lsd = LSD[label]
    d = (oof.filter(pl.col("label") == label)
         .with_columns(pl.col("series").replace_strict(lsd, return_dtype=pl.Float64).alias("sd"))
         .with_columns((pl.col("y_c") / pl.col("sd")).alias("y"),
                       (pl.col("pred_c") / pl.col("sd")).alias("p")))
    models = [m for m in d["model"].unique(maintain_order=True).to_list() if m != "linear zero"]
    res = {m: scores(*(d.filter(pl.col("model") == m).select("y", "p").to_numpy().T))
           for m in models}
    lin = [m for m in models if m.startswith("linear") and m != "linear zero"]
    ref = max(lin, key=lambda m: res[m]["r2"])
    base = d.filter((pl.col("model") == ref) & (pl.col("y") != 0))["y"].to_numpy()
    print("\n" + "=" * 118)
    print(f"label '{label}'  —  reference linear rung: {ref}  |  share up among y≠0: "
          f"{np.mean(base > 0):.3f}")
    print("=" * 118)
    print(f"{'model':42} {'R²':>8} {'acc':>6} {'bal acc':>7} {'F1 up':>6} {'F1 mac':>6} {'AUC':>6} | "
          f"{'ΔAUC vs ref':>22} {'Δbal acc vs ref':>22}")
    for m in models:
        r = res[m]
        extra = ""
        if m != ref:
            (da, db), ci = boot_delta(d, ref, m)
            extra = (f"{da:+.3f} [{ci[0, 0]:+.3f},{ci[1, 0]:+.3f}] "
                     f"{db:+.3f} [{ci[0, 1]:+.3f},{ci[1, 1]:+.3f}]")
        print(f"{m:42} {r['r2']:>+8.4f} {r['acc']:>6.3f} {r['bal']:>7.3f} {r['f1']:>6.3f} "
              f"{r['f1m']:>6.3f} {r['auc']:>6.3f} | {extra}", flush=True)
