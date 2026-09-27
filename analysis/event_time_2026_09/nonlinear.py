#!/usr/bin/env python
"""Is there a non-linear signal on the release clock?

    venv/bin/python -W ignore analysis/event_time_2026_09/nonlinear.py \
        > analysis/event_time_2026_09/out/nonlinear.txt     # needs build_panel.py first

The linear rungs in ``models.py`` score R² ≈ 0 and the recovery test says a
flexible learner cannot find structure at this n on its own. Both leave open a
non-linear signal of a *specific* shape, which costs one parameter and so is
testable here. Earlier findings name three candidates:

  sign, not size    only rank/sign of the surprise has survived (research_log §1)
  price level       a probability is bounded: the same news moves a 50c contract
                    more than a 95c one (leadlag §2: content confined to 10-75c)
  state             larger updates where the prior is wider (Bayesian updating)
                    or the leg thinner (liquidity_findings)

The signal is s[t, b] = Σ_a G[a, b]·z[t, a] (HAWKISH signs, cross-release), on
all 142 edges or the 3 BH channels. Labels are per-series z units (cents / node
sd). Two parts:

  A  Shape. Theory-signed response by quantile of s — descriptive, in-sample.
  B  Out-of-sample. Walk-forward (6 expanding folds over the last 60% of
     instants, training rows must have resolved before the fold starts), on the
     cells where s ≠ 0. One-parameter shapes against the linear rung:
        linear y = a + b·s        parent-mean  y = a + b·s/k (k firing parents)
        sign   y = a + b·sign(s)
        tanh   y = a + b·tanh(s)  asym   y = a + b₊·s⁺ + b₋·s⁻
        p-scaled   y = a + b·s·4p(1−p)   (the bounded-probability form)
        σ-scaled   y = a + b·s + c·s·σ̃   (σ̃ = within-series rank of the IQR width)
     each also fitted through the origin (no news, no expected move — the
     intercept otherwise carries each fold's mean shift), and a small
     gradient-boosted model on [s, p, σ̃, liquidity rank,
     days to close], with the same model minus s, so its gain from the surprise
     is measured on top of whatever non-linear own-state predictability exists.
     ΔR² vs linear with a bootstrap CI over release instants; for the boosted
     model, a permutation p that shuffles each series' surprises across its own
     releases (which series release when is kept; the surprise–label link is not).
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "stg_infra")
from stg.models.train import _fold_cuts
from stg.panel.registry import is_same_release

PANEL = "analysis/event_time_2026_09/out/event_nodes.parquet"
N_FOLDS, N_BOOT, N_PERM = 6, 2000, 100
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
rng = np.random.default_rng(0)


def rule(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


# ------------------------------------------------------------------ data
panel = pl.read_parquet(PANEL).with_columns(
    pl.col("instant").dt.replace_time_zone(None).cast(pl.Datetime("us")))
# state features, made comparable across series
panel = panel.with_columns(
    (pl.col("sigma_iqr").rank() / pl.col("sigma_iqr").count()).over("series")
    .fill_null(0.5).alias("sig_r"),
    (pl.col("vol7").rank() / pl.col("vol7").count()).over("series").alias("liq_r"))
Zt = (panel.filter(pl.col("released") & pl.col("z").is_not_null())
      .select("instant", "series", "z"))
instants = np.sort(panel["instant"].unique().to_numpy())
nodes = sorted(panel["series"].unique().to_list())


def graph(channels=None):
    return {(a, b): HAWKISH[a] * HAWKISH[b]
            for a in nodes for b in nodes
            if a != b and not is_same_release(a, b)
            and (channels is None or (TYPE[a], TYPE[b]) in channels)}


def signal(G: dict, zt: pl.DataFrame) -> pl.DataFrame:
    """s[instant, target] = Σ_a G[a, b]·z[instant, a]."""
    e = pl.DataFrame([{"series": a, "target": b, "g": g} for (a, b), g in G.items()])
    return (zt.join(e, on="series").group_by("instant", "target")
            .agg((pl.col("z") * pl.col("g")).sum().alias("s"), pl.len().alias("k")))


def cells(label: str, G: dict, zt: pl.DataFrame = Zt) -> pl.DataFrame:
    y, end, pcol = {"imm": ("y_imm", "imm_end", "p_lead"),
                    "settle": ("y_settle", "settle_end", "p_entry")}[label]
    d = (panel.filter(pl.col(y).is_not_null())
         .join(signal(G, zt).rename({"target": "series"}), on=["instant", "series"], how="left")
         .with_columns(pl.col("s").fill_null(0.0), pl.col("k").fill_null(0),
                       (pl.col(pcol) / 100).clip(0.01, 0.99).alias("p"),
                       (pl.col(y) / pl.col(y).std().over("series")).alias("y"),
                       pl.from_epoch(pl.col(end), time_unit="ns").cast(pl.Datetime("us"))
                       .alias("end")))
    return d.filter(pl.col("y").is_finite())


# ------------------------------------------------------------------ models
def design(d: pl.DataFrame, form: str, icpt: bool = True) -> np.ndarray:
    s, p, sg = d["s"].to_numpy(), d["p"].to_numpy(), d["sig_r"].to_numpy()
    k = np.maximum(d["k"].to_numpy(), 1)
    cols = {"linear": [s], "parent-mean": [s / k], "sign": [np.sign(s)], "tanh": [np.tanh(s)],
            "asym": [np.clip(s, 0, None), np.clip(s, None, 0)],
            "p-scaled": [s * 4 * p * (1 - p)],
            "σ-scaled": [s, s * (sg - 0.5)]}[form]
    return np.column_stack(([np.ones(len(s))] if icpt else []) + cols)


def ols(Xtr, ytr, Xte):
    return Xte @ np.linalg.lstsq(Xtr, ytr, rcond=None)[0]


STATE = ["p", "sig_r", "liq_r", "days_to_close"]


def boost(dtr, dte, with_s=True):
    cols = (["s"] if with_s else []) + STATE
    m = HistGradientBoostingRegressor(max_depth=2, max_iter=100, learning_rate=0.05,
                                      min_samples_leaf=20, l2_regularization=1.0,
                                      random_state=0)
    m.fit(dtr.select(cols).to_numpy(), dtr["y"].to_numpy())
    return m.predict(dte.select(cols).to_numpy())


def walk_forward(d: pl.DataFrame, predict) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions; NaN where a row is never in a test block."""
    cuts = _fold_cuts(instants, N_FOLDS)
    t, end = d["instant"].to_numpy(), d["end"].to_numpy()
    pred = np.full(len(d), np.nan)
    for i in range(N_FOLDS):
        tr = end < cuts[i]
        te = (t >= cuts[i]) & (t < cuts[i + 1])
        if tr.sum() >= 30 and te.any():
            pred[te] = predict(d.filter(pl.Series(tr)), d.filter(pl.Series(te)))
    return pred, ~np.isnan(pred)


def r2_boot(y, preds: dict, inst, ref="linear"):
    """R² vs zero per model; ΔR² vs ``ref`` with a CI resampling instants."""
    u, inv = np.unique(inst, return_inverse=True)
    sse = {k: np.bincount(inv, (y - p) ** 2, len(u)) for k, p in preds.items()}
    ss0 = np.bincount(inv, y ** 2, len(u))
    out = {}
    for k in preds:
        est = 1 - sse[k].sum() / ss0.sum()
        bs = []
        for _ in range(N_BOOT):
            b = rng.integers(0, len(u), len(u))
            bs.append((sse[ref][b].sum() - sse[k][b].sum()) / ss0[b].sum())
        out[k] = (est, est - (1 - sse[ref].sum() / ss0.sum()), np.percentile(bs, [2.5, 97.5]))
    return out


# ------------------------------------------------------------------ A: shape
rule("A  SHAPE — theory-signed response by quantile of the signal (in-sample)")
print("mean y (per-series z units) in quintiles of s among cells with s ≠ 0;\n"
      "a linear signal rises steadily, a sign signal steps at 0, a saturating one\n"
      "flattens at the ends. p-band: the same, split by the target's price.\n")
for label in ("imm", "settle"):
    for gname, G in (("all edges", graph()), ("BH channels", graph(BH_CHANNELS))):
        d = cells(label, G).filter(pl.col("s") != 0)
        q = np.quantile(d["s"], [0.2, 0.4, 0.6, 0.8])
        d = d.with_columns(pl.col("s").cut(list(q), labels=[f"q{i}" for i in range(1, 6)])
                           .alias("bin"))
        t = d.group_by("bin").agg(pl.col("s").mean().alias("s"), pl.col("y").mean(),
                                  pl.len()).sort("bin")
        row = "  ".join(f"{r['s']:+.2f}→{r['y']:+.2f}" for r in t.iter_rows(named=True))
        print(f"{label:6} {gname:12} n={len(d):>4}  s→y: {row}")
        band = (d.with_columns(pl.when(pl.col("p") < 0.25).then(pl.lit("p<25"))
                               .when(pl.col("p") > 0.75).then(pl.lit("p>75"))
                               .otherwise(pl.lit("25-75")).alias("pb"))
                .group_by("pb").agg(
                    (pl.col("y") * pl.col("s").sign()).mean().alias("signed"),
                    pl.len()).sort("pb"))
        print(f"{'':20} theory-signed mean y by price band: " + "  ".join(
            f"{r['pb']} {r['signed']:+.3f} (n={r['len']})" for r in band.iter_rows(named=True)))

# ------------------------------------------------------------------ B: OOS
FORMS = ["linear", "parent-mean", "sign", "tanh", "asym", "p-scaled", "σ-scaled"]
for label in ("imm", "settle"):
    for gname, G in (("all edges", graph()), ("BH channels", graph(BH_CHANNELS))):
        rule(f"B  WALK-FORWARD — label '{label}', {gname}, cells with s ≠ 0")
        dall = cells(label, G)
        d = dall.filter(pl.col("s") != 0)
        preds, masks = {}, []
        for icpt in (True, False):
            for f in FORMS:
                key = f if icpt else f"{f} (no intercept)"
                preds[key], m = walk_forward(d, lambda a, b, f=f, i=icpt: ols(
                    design(a, f, i), a["y"].to_numpy(), design(b, f, i)))
                masks.append(m)
        preds["boost: state only"], m1 = walk_forward(d, lambda a, b: boost(a, b, False))
        preds["boost: s + state"], m2 = walk_forward(d, boost)
        mask = np.logical_and.reduce(masks + [m1, m2])
        y = d["y"].to_numpy()[mask]
        inst = d["instant"].to_numpy()[mask]
        P = {k: v[mask] for k, v in preds.items()}
        res = {**r2_boot(y, {k: v for k, v in P.items() if "no intercept" not in k}, inst),
               **r2_boot(y, {k: v for k, v in P.items() if "no intercept" in k}, inst,
                         ref="linear (no intercept)")}
        print(f"{len(y)} test cells on {len(np.unique(inst))} instants. ΔR² is against the "
              f"linear rung\nof the same intercept choice; boosted models against linear.\n")
        print(f"{'model':32} {'R² vs 0':>9} {'ΔR² vs linear':>14} {'95% CI':>20}")
        for k, (est, dlt, ci) in res.items():
            print(f"{k:32} {est:>+9.4f} {dlt:>+14.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        # what the surprise adds to the boosted model, against each series'
        # surprises shuffled across its own releases (release pattern kept)
        gain = res["boost: s + state"][0] - res["boost: state only"][0]
        null = []
        for k in range(N_PERM):
            zp = Zt.with_columns(pl.col("z").shuffle(seed=k).over("series"))
            dp = cells(label, G, zp)
            dp = dp.filter(pl.col("s") != 0)
            pp, mp = walk_forward(dp, boost)
            sp, ms = walk_forward(dp, lambda a, b: boost(a, b, False))
            mm = mp & ms
            yp = dp["y"].to_numpy()[mm]
            null.append(((yp - sp[mm]) ** 2).sum() / (yp ** 2).sum()
                        - ((yp - pp[mm]) ** 2).sum() / (yp ** 2).sum())
        null = np.array(null)
        print(f"\nboosted model's gain from s: {gain:+.4f}; permutation null "
              f"mean {null.mean():+.4f}, 95th pct {np.percentile(null, 95):+.4f}, "
              f"p = {(1 + (null >= gain).sum()) / (1 + N_PERM):.3f}")
