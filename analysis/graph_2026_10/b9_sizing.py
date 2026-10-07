"""Size by signal and width instead of hard cut-offs. Current model, NBA spreads, 120 s hold, fixed exit,
offset 1c, strictly-through fills, all test folds, feed delays 0 / 5 / 10 s (trade log has |signal| >= 5).
size = s(signal) x w(width), each between 0 and 1:
    s linear     (|signal| - 5) / 10, capped at 1 (0 at 5c, 1 at >= 15c)
    s quadratic  the linear value squared
    w linear     width / 10, capped at 1, 0 at width <= 0; missing width (a side > 120 s old) -> w_missing
Hard rules are size 1 inside the cut-offs. Compared on: trades placed, total size (sum of sizes), P&L (sum of size x net,
cents), net per contract, and t of the mean P&L per game over every game in the log (scale-free).

    venv/bin/python -W ignore analysis/graph_2026_10/b9_sizing.py
    -> out/b9_sizing.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b7_maker_liquidity import liquidity               # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402

a = pl.col("signal").abs()
lin_s = ((a - 5) / 10).clip(0, 1)
lin_w = (pl.col("width") / 10).clip(0, 1)
step = lambda c, x: pl.when(c).then(1.0).otherwise(0.0)   # noqa: E731
RULES = {
    "hard: signal >= 8, width >= 9": step((a >= 8) & (pl.col("width") >= 9), 1),
    "hard: signal >= 8, width >= 5": step((a >= 8) & (pl.col("width") >= 5), 1),
    "hard: signal >= 8, any width": step(a >= 8, 1),
    "linear signal only": lin_s,
    "quadratic signal only": lin_s ** 2,
    "linear signal x linear width (missing 0)": lin_s * lin_w.fill_null(0),
    "linear signal x linear width (missing 1)": lin_s * lin_w.fill_null(1),
    "quadratic signal x linear width (missing 0)": lin_s ** 2 * lin_w.fill_null(0),
    "quadratic signal x linear width (missing 1)": lin_s ** 2 * lin_w.fill_null(1),
    "quadratic signal x linear width (missing 0.5)": lin_s ** 2 * lin_w.fill_null(0.5),
}


def game_t(g: pl.DataFrame, games: pl.Series) -> float:
    per = pl.DataFrame({"game_id": games}).join(g.group_by("game_id").agg(pl.col("pnl").sum()), on="game_id", how="left").fill_null(0)["pnl"].to_numpy()
    return float(per.mean() / per.std(ddof=1) * np.sqrt(len(per)))


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_fmt_str_lengths(50)
    parts = []
    for d in (0, 5, 10):
        f = pl.read_parquet(OUT / f"b7_maker_trades_NBA_d{d}_h30-60-120-1800_tpf.parquet").filter(
            pl.col("k") == 1, pl.col("through") == 0, pl.col("kind") == "spread", pl.col("hold") == "120s")
        parts.append(with_net(f).with_columns(pl.lit(d).alias("delay")))
    f = liquidity(pl.concat(parts, how="diagonal_relaxed"))
    rows, folds = [], []
    for name, size in RULES.items():
        r = dict(rule=name)
        for d in (0, 5, 10):
            x = f.filter(pl.col("delay") == d)
            games = x["game_id"].unique()
            g = x.with_columns(size.alias("size")).filter(pl.col("size") > 0).with_columns((pl.col("size") * pl.col("net")).alias("pnl"))
            c = g["size"].sum()
            r |= {f"d{d}_trades": g.height, f"d{d}_total_size": round(c), f"d{d}_pnl": round(g["pnl"].sum()), f"d{d}_per_contract": round(g["pnl"].sum() / c, 2),
                  f"d{d}_t": round(game_t(g, games), 1)}
            if d == 10:
                fr = dict(rule=name)
                for k in sorted(x["fold"].unique().to_list()):
                    y = g.filter(pl.col("fold") == k)
                    fr[f"fold{k}"] = round(y["pnl"].sum() / y["size"].sum(), 1) if y.height else None
                folds.append(fr)
        rows.append(r)
    t = pl.DataFrame(rows)
    text = ("Current model, NBA spreads, 120 s hold, all test folds: hard cut-offs vs sizing by signal and width\n"
            "trades = trades placed (size > 0); total_size = sum of sizes (contracts at full size 1); pnl = cents; per_contract = pnl / total_size; t = mean P&L per game / its standard error (all games)\n"
            + "\n".join(f"\n######## delay {d} s\n" + str(t.select("rule", pl.col(f"^d{d}_.*$"))) for d in (0, 5, 10))
            + "\n\n######## net per contract by test fold (10 s delay)\n" + str(pl.DataFrame(folds)))
    (OUT / "b9_sizing.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
