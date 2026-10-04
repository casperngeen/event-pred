"""Trade only when the expected edge beats the full cost (4b trading model, out of sample).

    cost  = spread_t / 2 + spread_exit / 2 + fee(entry) + fee(exit),  fee(p) = 7 p (1 - p) cents
            (the exit spread is unknown at entry: spread_t stands in for it in the rule)
    edge  = c * |pred|, c = the calibration slope of realised rem_h on pred, fitted on each fold's
            training predictions only (inner walk-forward), so it never sees the test fold
    trade when edge > m * cost, for m in MARGINS

1. calibration: realised mid move vs predicted, by bins of the out-of-sample prediction
2. the sweep: trades, gross and net cents per contract (net: actual spreads and fees on both legs)

    venv/bin/python -W ignore analysis/graph_2026_10/trade_edge.py > analysis/graph_2026_10/out/trade_edge.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stg.sports.asof import SportsData                          # noqa: E402
from stg.sports.walkforward import walk_forward                 # noqa: E402
from checks_4a import exits                                     # noqa: E402
from linear_weights import load                                 # noqa: E402
from linear_weights_4b import GRID, specs, with_state           # noqa: E402

FEE = 7.0
MARGINS = [0.0, 1.0, 1.5, 2.0, 3.0]


def fee(x: pl.Expr) -> pl.Expr:
    return FEE * (x / 100) * (1 - x / 100)


def calibrated(g: pl.DataFrame, feats: list[str], h: str) -> pl.DataFrame:
    """OOS predictions, each scaled by a slope fitted on that fold's own earlier OOS predictions:
    a second walk-forward over the first one's output (the slope only uses folds before the test)."""
    p, _ = walk_forward(g, feats, f"rem_{h}", f"label_end_{h}", lam=GRID)
    if p.height == 0:
        return p
    out = []
    for f in sorted(p["fold"].unique().to_list()):
        prev = p.filter(pl.col("fold") < f)
        cur = p.filter(pl.col("fold") == f)
        if prev.height >= 30 and prev["pred"].std() > 0:
            x, y = prev["pred"].to_numpy(), prev[f"rem_{h}"].to_numpy()
            c = float(np.clip((x @ y) / (x @ x), 0, 2))
        else:
            c = None                                  # no earlier folds: cannot calibrate, no trades
        out.append(cur.with_columns(pl.lit(c, pl.Float64).alias("calib")))
    return pl.concat(out)


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(230)
    d = SportsData()
    s = with_state(load()).filter(pl.col("kind") != "wins")
    allp = []
    for h in ("12h", "24h"):
        preds = [calibrated(g, specs(g, kind, True)["4b"], h).with_columns(pl.lit(h).alias("h"))
                 for (lg, kind), g in s.group_by("league", "kind")]
        p = exits(d, pl.concat([x for x in preds if x.height], how="diagonal_relaxed"), h)
        mid_h = pl.col("value_t") + pl.col(f"rem_{h}")
        up = pl.col("pred") > 0
        p = p.with_columns(
            pl.col(f"rem_{h}").alias("rem"),
            (pl.col("spread_t") + 2 * fee(pl.col("value_t"))).alias("cost_rule"),
            (pl.col("calib") * pl.col("pred").abs()).alias("edge"),
            pl.when(up).then(mid_h - pl.col("spread_exit") / 2 - pl.col("value_t") - pl.col("spread_t") / 2)
            .otherwise(pl.col("value_t") - pl.col("spread_t") / 2 - mid_h - pl.col("spread_exit") / 2).alias("gross"),
            (fee(pl.col("value_t")) + fee(mid_h)).alias("fees"),
            (pl.col("pred").sign() * pl.col(f"rem_{h}")).alias("mid"))
        allp.append(p.filter(pl.col("spread_exit").is_not_null(), ~pl.col("one_sided_exit").fill_null(True)))
    p = pl.concat(allp, how="diagonal_relaxed")

    print(__doc__.split("\n\n")[1], "\n")
    print("1. calibration (all single-leg live kinds): realised rem vs prediction, by |prediction| bin (cents)")
    print(p.with_columns(pl.col("pred").abs().cut([0.25, 0.5, 1, 2, 4]).alias("|pred| bin"))
          .group_by("h", "|pred| bin").agg(pl.len().alias("n"), pl.col("pred").abs().mean().round(2).alias("mean_|pred|"),
                                          pl.col("mid").mean().round(2).alias("realised_in_pred_direction"),
                                          pl.col("calib").median().round(2).alias("calib_slope_med"))
          .sort("h", "|pred| bin"))

    print("\n2. trade when calibrated edge > m x (spread + both fees); net uses actual exit spread and fees")
    rows = []
    for m in MARGINS:
        t = p.filter(pl.col("calib").is_not_null(), pl.col("edge") > m * pl.col("cost_rule"))
        for (h, lg, kind), g in t.group_by("h", "league", "kind"):
            rows.append({"m": m, "h": h, "league": lg, "kind": kind, "trades": g.height,
                         "gross": round(g["gross"].mean(), 2), "net": round((g["gross"] - g["fees"]).mean(), 2),
                         "net_total": round((g["gross"] - g["fees"]).sum(), 1)})
        if t.height:
            rows.append({"m": m, "h": "all", "league": "all", "kind": "all", "trades": t.height,
                         "gross": round(t["gross"].mean(), 2), "net": round((t["gross"] - t["fees"]).mean(), 2),
                         "net_total": round((t["gross"] - t["fees"]).sum(), 1)})
    r = pl.DataFrame(rows)
    print(r.filter(pl.col("h") == "all").select("m", "trades", "gross", "net", "net_total"))
    print("\n   by league / kind (cells with >= 10 trades):")
    print(r.filter(pl.col("h") != "all", pl.col("trades") >= 10).sort("h", "league", "kind", "m"))


if __name__ == "__main__":
    main()
