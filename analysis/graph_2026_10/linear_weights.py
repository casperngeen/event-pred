"""Chunk 4a: constant relation weights, out of sample (purged walk-forward over nights).

On the live set (samples.live), one ridge model per league x kind x horizon:

    relation:  total_h = a + b_own * z_own + b_rival * x_rival
    trading:   rem_h   = a + b_own * z_own + b_rival * x_rival + b_moved * moved

total_h is the whole response to the night (from before its first game); rem_h is what was still to
come after the decision. In the trading model, b_moved < 0 means reversal of what already happened
(over-reaction), and z weights > 0 with moved controlled mean the market still under-reacted.

Champ (every team a rival): x_rival = -z_own exactly, so it is dropped there. Units: cents (wins
nodes: wins); a second pass in log-odds of the price (single-leg kinds) is the robustness check.

    venv/bin/python -W ignore analysis/graph_2026_10/linear_weights.py > analysis/graph_2026_10/out/linear_weights.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.samples import HORIZONS                         # noqa: E402
from stg.sports.walkforward import oos_scores, walk_forward     # noqa: E402

SAMPLES = Path("data/sports/derived/samples")
OUT = Path(__file__).resolve().parent / "out"
LAM = 1.0


def logit(v: pl.Expr) -> pl.Expr:
    p = v.clip(0.5, 99.5) / 100
    return (p / (1 - p)).log()


def load() -> pl.DataFrame:
    s = pl.concat([pl.read_parquet(f) for f in sorted(SAMPLES.glob("*.parquet"))], how="diagonal_relaxed")
    s = s.filter(pl.col("live"))
    lo = []
    for h in HORIZONS:            # the same labels in log-odds (single-leg kinds)
        v_h = pl.col("value_pre") + pl.col(f"total_{h}")
        lo += [(logit(v_h) - logit(pl.col("value_pre"))).alias(f"lo_total_{h}"),
               (logit(v_h) - logit(pl.col("value_t"))).alias(f"lo_rem_{h}")]
    return s.with_columns(*lo, (logit(pl.col("value_t")) - logit(pl.col("value_pre"))).alias("lo_moved"))


def run(s: pl.DataFrame, units: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    scores, coefs = [], []
    pre = "" if units == "cents" else "lo_"
    for (lg, kind), g in s.group_by("league", "kind"):
        if units == "logodds" and kind == "wins":
            continue
        rival = ["x_rival"] if abs(np.corrcoef(g["z_own"], g["x_rival"])[0, 1]) < 0.999 else []
        for model, label, feats in (("relation", "total", ["z_own", *rival]),
                                    ("trading", "rem", ["z_own", *rival, f"{pre}moved" if pre else "moved"])):
            for h in HORIZONS:
                pred, fits = walk_forward(g, feats, f"{pre}{label}_{h}", f"label_end_{h}", lam=LAM)
                sc = oos_scores(pred, f"{pre}{label}_{h}")
                scores.append({"league": lg, "kind": kind, "model": model, "horizon": h, **sc})
                for f in fits:
                    for k, v in f.coef.items():
                        coefs.append({"league": lg, "kind": kind, "model": model, "horizon": h,
                                      "fold": f.fold, "term": k.replace("lo_", ""), "coef": v})
    return pl.DataFrame(scores), pl.DataFrame(coefs)


def show(scores: pl.DataFrame, coefs: pl.DataFrame, model: str, horizons=("1h", "12h", "24h")) -> None:
    c = (coefs.filter(pl.col("model") == model, ~pl.col("term").is_in(["intercept", "lam"]), pl.col("horizon").is_in(horizons))
         .group_by("league", "kind", "horizon", "term")
         .agg(pl.col("coef").mean().alias("m"), pl.col("coef").std().alias("sd"), pl.len().alias("folds"))
         .with_columns(pl.format("{} ({})", pl.col("m").round(2), pl.col("sd").round(2)).alias("cell"))
         .pivot(on="term", index=["league", "kind", "horizon"], values="cell"))
    sc = scores.filter(pl.col("model") == model, pl.col("horizon").is_in(horizons)).select(
        "league", "kind", "horizon", "n", pl.col("r2_vs_zero").round(3), pl.col("corr").round(2),
        pl.col("sign_acc").round(2))
    order = {h: i for i, h in enumerate(HORIZONS)}
    print(sc.join(c, on=["league", "kind", "horizon"], how="left")
          .with_columns(pl.col("horizon").replace_strict(order).alias("_h")).sort("league", "kind", "_h").drop("_h"))


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(230)
    s = load()
    OUT.mkdir(exist_ok=True)
    print(__doc__.split("\n\n")[1], "\n")
    print(f"live rows: {s.height:,}; ridge lambda {LAM} on standardised features, purged expanding folds over "
          "nights (first 40% train only, 5 test folds). Coefficients: mean (sd across folds), original units.\n")
    for units in ("cents", "logodds"):
        scores, coefs = run(s, units)
        scores.write_parquet(OUT / f"linear_4a_scores_{units}.parquet")
        coefs.write_parquet(OUT / f"linear_4a_coefs_{units}.parquet")
        for model in ("relation", "trading"):
            print(f"######## {model} model, {units}: OOS R^2 vs zero, corr, sign accuracy; coefficients")
            show(scores, coefs, model)
            print()


if __name__ == "__main__":
    main()
