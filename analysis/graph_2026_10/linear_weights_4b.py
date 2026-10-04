"""Chunk 4b: relation weights that depend on the state (the linear form of the weight function g).

Live set, one ridge model per league x kind x horizon, against 4a's constant weights:

    4a  y = a + b_own z_own + b_riv x_rival                                       (+ b_mv moved, trading)
    4b  y = 4a + sum_f c_f f + sum_f [ d_f z_own f + e_f x_rival f ]
        f in: pq = p(1 - p) of the price before the night (where leverage lives; 0 at the extremes),
              frac_remaining (of the season), games_behind (of the division / conference leader)

The f main effects ask whether anything known before the night predicts the move (efficiency says
no). Wins nodes have no price level: f = frac_remaining, games_behind. Penalty chosen per fold on the
training rows' latest nights, for both models. Champ: x_rival = -z_own, so no rival terms.

    venv/bin/python -W ignore analysis/graph_2026_10/linear_weights_4b.py > analysis/graph_2026_10/out/linear_weights_4b.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stg.sports.samples import HORIZONS                         # noqa: E402
from stg.sports.walkforward import oos_scores, walk_forward     # noqa: E402
from linear_weights import load                                 # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]


def with_state(s: pl.DataFrame) -> pl.DataFrame:
    p = pl.col("value_pre") / 100
    s = s.with_columns(pl.when(pl.col("kind") == "wins").then(None).otherwise(p * (1 - p)).alias("pq"),
                       pl.col("games_behind").fill_null(0.0))
    for f in ("pq", "frac_remaining", "games_behind"):
        s = s.with_columns((pl.col("z_own") * pl.col(f)).alias(f"z_own*{f}"),
                           (pl.col("x_rival") * pl.col(f)).alias(f"x_rival*{f}"))
    return s


def specs(g: pl.DataFrame, kind: str, trading: bool) -> dict[str, list[str]]:
    rival = abs(np.corrcoef(g["z_own"], g["x_rival"])[0, 1]) < 0.999
    shocks = ["z_own"] + (["x_rival"] if rival else [])
    state = (["pq"] if kind != "wins" else []) + ["frac_remaining", "games_behind"]
    base = shocks + (["moved"] if trading else [])
    inter = [f"{s}*{f}" for s in shocks for f in state]
    return {"4a": base, "4b": base + state + inter}


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(230)
    s = with_state(load())
    OUT.mkdir(exist_ok=True)
    scores, coefs = [], []
    for (lg, kind), g in s.group_by("league", "kind"):
        for model, label in (("relation", "total"), ("trading", "rem")):
            for name, feats in specs(g, kind, model == "trading").items():
                for h in HORIZONS:
                    pred, fits = walk_forward(g, feats, f"{label}_{h}", f"label_end_{h}", lam=GRID)
                    sc = oos_scores(pred, f"{label}_{h}")
                    scores.append({"league": lg, "kind": kind, "model": model, "spec": name, "horizon": h, **sc,
                                   "lam_med": float(np.median([f.coef["lam"] for f in fits])) if fits else None})
                    for f in fits:
                        coefs += [{"league": lg, "kind": kind, "model": model, "spec": name, "horizon": h,
                                   "fold": f.fold, "term": k, "coef": v} for k, v in f.coef.items()
                                  if k not in ("intercept", "lam")]
    sc, cf = pl.DataFrame(scores), pl.DataFrame(coefs)
    sc.write_parquet(OUT / "linear_4b_scores.parquet")
    cf.write_parquet(OUT / "linear_4b_coefs.parquet")
    order = {h: i for i, h in enumerate(HORIZONS)}

    print(__doc__.split("\n\n")[1], "\n")
    for model in ("relation", "trading"):
        print(f"######## {model}: OOS R^2 vs zero, 4a (constant) vs 4b (state-dependent); median chosen penalty")
        w = (sc.filter(pl.col("model") == model)
             .pivot(on="spec", index=["league", "kind", "horizon"], values=["r2_vs_zero", "lam_med"])
             .with_columns((pl.col("r2_vs_zero_4b") - pl.col("r2_vs_zero_4a")).round(3).alias("gain"),
                           pl.col("r2_vs_zero_4a").round(3), pl.col("r2_vs_zero_4b").round(3),
                           pl.col("horizon").replace_strict(order).alias("_h"))
             .sort("league", "kind", "_h").drop("_h"))
        print(w)
        print()

    print("######## relation, 4b interaction coefficients (mean over folds, t = mean / sd across folds), 12h and 24h")
    keep = cf.filter(pl.col("model") == "relation", pl.col("spec") == "4b", pl.col("horizon").is_in(["12h", "24h"]),
                     pl.col("term").str.contains(r"\*"))
    print(keep.group_by("league", "kind", "horizon", "term").agg(
        pl.col("coef").mean().round(3).alias("mean"),
        (pl.col("coef").mean() / pl.col("coef").std()).round(1).alias("t_folds"))
        .sort("league", "kind", "horizon", "term"))


if __name__ == "__main__":
    main()
