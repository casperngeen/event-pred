"""Is the graph's gain on totals just "this line is out of step with its neighbours"? A hand-made test,
no training: the ladder gap of each contract, added to the saved forecasts of the no-graph and full models.

Ladders (per game, per 5-s step; market prices are never delayed, so all of this is known at the step):
    total ladder   total contracts, line = strike, P = price of "over line"
    margin ladder  winner + spread contracts on the home team's winning margin: line = side x strike
                   (winner: 0), P = price for home contracts, 1 - price for away contracts (P is then
                   always "home margin > line")
Ladder gap of a contract: its P minus the P interpolated (in probit units) between the nearest distinct
line below and the nearest distinct line above, its own line left out; in cents, in the contract's own
direction (sign flipped back for away contracts). Missing when it has no line on one side.

Scores (folds 2-5; every slope is fitted walk-forward on the earlier test folds only): R^2 of the change in
the traded price at +60 s / +120 s against no change, for
    gap only          beta x gap
    no_graph          the no-message-passing model (seed 0)
    no_graph + gap    its forecast + beta x gap
    full              the full model (seed 0)
    full + gap        its forecast + beta x gap
on the rows where the gap exists (coverage shown), and the slope beta itself.

    venv/bin/python -W ignore analysis/graph_2026_10/b12_ladder_gap.py
    -> out/b12_ladder_gap.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import ndtr, ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
from stg.models.ingame import Panel                     # noqa: E402
from stg.sports.loaders import derived_dir              # noqa: E402

RES = Path("results/ingame")
OUT = Path(__file__).resolve().parent / "out"
TAG = "_d10_h30-60-120-1800_tpf"
ARMS = {"full": f"gru{TAG}_seed0", "no_graph": f"gru{TAG}_r0_seed0"}


def load(L: str, pn: Panel, arm: str) -> pl.DataFrame:
    parts = []
    for k in range(1, 6):
        z = np.load(RES / f"{L}_fold{k}_{ARMS[arm]}.npz")
        step = z["step"]
        first = np.r_[True, step[1:] != step[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(step)), 0))
        node = pn.step_node_ptr[step] + (np.arange(len(step)) - start)
        parts.append(pl.DataFrame({"fold": np.full(len(step), k), "ri": np.arange(len(step)), "step": step, "node": node, "kind": z["kind"],
                                   "value": z["value"], "d60": z["delta"][:, 1], "d120": z["delta"][:, 2],
                                   "t60": z["value_h"][:, 1] - z["value"], "t120": z["value_h"][:, 2] - z["value"],
                                   "mid": np.where(z["has_bid"] & z["has_ask"], (z["entry_bid"] + z["entry_ask"]) / 2, np.nan)}))
    return pl.concat(parts)


def ladders(f: pl.DataFrame, pn: Panel) -> pl.DataFrame:
    """Winner / spread / total rows with their ladder, line on it, and direction (-1 for away contracts)."""
    node = f["node"].to_numpy()
    strike, side = np.asarray(pn.node_strike[node], np.float64), np.asarray(pn.node_side[node], np.int64)
    f = f.with_columns(pl.Series("strike", strike), pl.Series("side", side)).filter(pl.col("kind") <= 2)
    return f.with_columns(
        pl.when(pl.col("kind") == 2).then(pl.lit("total")).otherwise(pl.lit("margin")).alias("ladder"),
        pl.when(pl.col("kind") == 2).then(pl.col("strike")).when(pl.col("kind") == 0).then(0.0)
          .otherwise(pl.col("side") * pl.col("strike")).alias("line"),
        pl.when((pl.col("kind") < 2) & (pl.col("side") < 0)).then(-1.0).otherwise(1.0).alias("sgn"))


def ladder_gap(f: pl.DataFrame, price: str = "value", name: str = "gap") -> pl.DataFrame:
    """Adds `name`: the contract's price minus the price interpolated from its nearest lines below and above
    (probit units, own line left out), in cents in the contract's own direction. Rows whose price is
    missing take no part (no gap, and they are not anyone's neighbour)."""
    p = (pl.col(price) / 100).clip(0.02, 0.98)
    g = f.select("step", "ladder", "line", "sgn", pl.when(pl.col("sgn") > 0).then(p).otherwise(1 - p).alias("P")).filter(pl.col("P").is_not_nan())
    g = g.with_columns(pl.Series("z", ndtri(g["P"].to_numpy())))
    lines = (g.group_by("step", "ladder", "line").agg(pl.col("z").mean()).sort("step", "ladder", "line")
             .with_columns(pl.col("line").shift(1).over("step", "ladder").alias("l_lo"), pl.col("z").shift(1).over("step", "ladder").alias("z_lo"),
                           pl.col("line").shift(-1).over("step", "ladder").alias("l_hi"), pl.col("z").shift(-1).over("step", "ladder").alias("z_hi"))
             .drop("z"))
    h = f.with_columns(pl.when(pl.col("sgn") > 0).then(p).otherwise(1 - p).alias("_P")).join(lines, on=["step", "ladder", "line"], how="left")
    zf = pl.col("z_lo") + (pl.col("z_hi") - pl.col("z_lo")) * (pl.col("line") - pl.col("l_lo")) / (pl.col("l_hi") - pl.col("l_lo"))
    fair = ndtr(h.select(zf).to_series().fill_null(np.nan).to_numpy())
    return (h.with_columns(pl.Series("_fair", fair)).with_columns((pl.col("sgn") * 100 * (pl.col("_P") - pl.col("_fair"))).fill_nan(None).alias(name))
            .drop("_P", "_fair", "l_lo", "z_lo", "l_hi", "z_hi"))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - ((y - p) ** 2).sum() / (y ** 2).sum())


def walk_forward(g: pl.DataFrame, t: str, base: str | None) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Predictions base + beta x gap on folds 2-5, beta fitted on the earlier folds (least squares on the
    residual of the base forecast)."""
    ys, ps, betas = [], [], []
    for k in range(2, 6):
        tr, te = g.filter(pl.col("fold") < k), g.filter(pl.col("fold") == k)
        r = tr[t].to_numpy() - (tr[base].to_numpy() if base else 0)
        x = tr["gap"].to_numpy()
        b = float((x * r).sum() / (x * x).sum())
        betas.append(b)
        ys.append(te[t].to_numpy())
        ps.append((te[base].to_numpy() if base else 0) + b * te["gap"].to_numpy())
    return np.concatenate(ys), np.concatenate(ps), betas


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(60)
    rows = []
    for L in ("NBA", "MLB", "NFL"):
        pn = Panel(derived_dir() / "ingame" / L / "panel_v3f")
        full = ladder_gap(ladders(load(L, pn, "full"), pn))
        ng = load(L, pn, "no_graph").select("step", "node", pl.col("d60").alias("n60"), pl.col("d120").alias("n120"))
        f = full.join(ng, on=["step", "node"], how="inner")
        for kind in ("spread", "total"):
            k = f.filter(pl.col("kind") == {"spread": 1, "total": 2}[kind])
            for h in ("60", "120"):
                t = f"t{h}"
                a = k.filter(pl.col(t).is_finite())
                cov = a["gap"].is_not_null().mean()
                g = a.filter(pl.col("gap").is_finite())
                g25 = g.filter(pl.col("fold") >= 2)
                y = g25[t].to_numpy()
                r = dict(league=L, kind=kind, horizon=f"{h}s", rows=g25.height, coverage=round(cov, 2),
                         gap_sd=round(g["gap"].std(), 1))
                _, p, b = walk_forward(g, t, None)
                r["gap only"] = round(100 * r2(y, p), 2)
                r["beta (gap only)"] = round(float(np.mean(b)), 2)
                r["no_graph"] = round(100 * r2(y, g25[f"n{h}"].to_numpy()), 2)
                _, p, b = walk_forward(g, t, f"n{h}")
                r["no_graph + gap"] = round(100 * r2(y, p), 2)
                r["beta (on no_graph)"] = round(float(np.mean(b)), 2)
                r["full"] = round(100 * r2(y, g25[f"d{h}"].to_numpy()), 2)
                _, p, b = walk_forward(g, t, f"d{h}")
                r["full + gap"] = round(100 * r2(y, p), 2)
                r["beta (on full)"] = round(float(np.mean(b)), 2)
                rows.append(r)
        print(L, "done", flush=True)
    t = pl.DataFrame(rows)
    txt = ("Ladder gap test (R^2 in %, change in traded price vs no change, folds 2-5, rows with a gap; slopes walk-forward)\n"
           + str(t))
    (OUT / "b12_ladder_gap.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
