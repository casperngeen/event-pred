"""Whole-contract sizing for the live runners: the frozen size (0-1, ((|signal| - 5) / 10)^2 x width / 10, each capped
at 1) times a scale K, rounded to whole contracts, on the test folds of the deployed models (NBA / NFL at 60 s, MLB
pitch model at 10 s with its skip window; b33_ablation_table.trades: frozen rule, fixed exit at 120 s, fees).

    frozen (fractional)  the current backtest: P&L x the 0-1 size, so "contracts" can be 0.24
    round, K             round(K x size) contracts (half up); orders rounding to 0 are not placed
    at least 1, K        max(1, round(K x size)): every order the frozen rule places, at least 1 contract
    flat 1               1 contract on every order

A fill is the same at any of these sizes: the backtest fills an order only when a print trades strictly through its
limit, which empties our price level first. Per variant and contract type: orders placed, contracts, profit ($),
net c/contract, game-level t, folds up, and profit per unit of K (comparable across K and with the fractional rule).

    venv/bin/python -W ignore analysis/graph_2026_10/b34_integer_sizing.py -> out/b34_integer_sizing.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b33_ablation_table import FULL, OUT, V, per_game, trades, tstat   # noqa: E402

DEPLOYED = {"MLB": FULL, "NBA": V("deployed", "delay 60 s", delay=60), "NFL": V("deployed", "delay 60 s", delay=60)}
KS = (1, 2, 3, 5, 10)


def variants() -> list[tuple[str, int | None, pl.Expr]]:
    v = [("frozen (fractional)", 1, pl.col("size"))]
    for k in KS:
        n = (pl.col("size") * k + 0.5).floor()
        v += [(f"round, K={k}", k, n), (f"at least 1, K={k}", k, pl.max_horizontal(n, pl.lit(1.0)))]
    return v + [("flat 1", 1, pl.lit(1.0))]


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_hide_column_data_types(True)
    rows = []
    for L, v in DEPLOYED.items():
        t = trades(v, L)
        for kind in ("spread", "total"):
            g = t.filter(pl.col("kind") == kind)
            games = g["game_id"].unique()
            for name, k, n in variants():
                x = g.with_columns(n.alias("n")).filter(pl.col("n") > 0).with_columns((pl.col("n") * pl.col("net")).alias("pnl"))
                folds = x.group_by("fold").agg(pl.col("pnl").sum())
                c = x["n"].sum()
                rows.append(dict(contract=f"{L} {kind}", sizing=name, orders=x.height, contracts=round(c, 1),
                                 max_order=x["n"].max(), usd=round(x["pnl"].sum() / 100, 1),
                                 net_c=round(x["pnl"].sum() / c, 2) if c else None, t_game=tstat(per_game(x, games)),
                                 folds_up=f"{(folds['pnl'] > 0).sum()}/{folds.height}",
                                 usd_per_K=round(x["pnl"].sum() / 100 / k, 1)))
        print(L, "done", flush=True)
    df = pl.DataFrame(rows)
    lines = [__doc__.split("\n\n")[0], ""]
    for c in df["contract"].unique(maintain_order=True):
        lines += [f"================ {c}", str(df.filter(pl.col("contract") == c).drop("contract")), ""]
    (OUT / "b34_integer_sizing.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
