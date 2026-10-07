"""Fixed exit vs holding to settlement, on the same entries: the frozen candidates (120 s hold, frozen sizing)
with the frozen model (no futures, average of 5 seeds' forecasts), all test folds.

    fixed exit   as b7_maker_trades.py: at fill + 120 s rest 1c past the last print, else take, else settle
    hold         keep the contract to its result: side x (100 y - entry) - maker fee at entry (no exit fee)
Metrics as b14_nofutures_pnl.py, plus win rates; diff_$ / diff_t = hold minus fixed exit per game (paired t over games).

    venv/bin/python -W ignore analysis/graph_2026_10/b16_exit_vs_hold.py
    -> out/b16_exit_vs_hold.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b14_nofutures_pnl import SIZE, metrics, per_game   # noqa: E402
from b6_maker_exit import MAKER_RATE                    # noqa: E402
from b7_maker_liquidity import liquidity               # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402

TAG = "_h30-60-120-1800_tp_ens5"
STRATS = [("NBA", "spread"), ("MLB", "total"), ("NBA", "total"), ("NFL", "total")]


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_hide_column_data_types(True)
    q = (pl.col("entry") / 100).clip(0, 1)
    rows, how, wins = [], [], []
    for L in sorted({s[0] for s in STRATS}):
        f = (liquidity(with_net(pl.read_parquet(OUT / f"b7_maker_trades_{L}_d10{TAG}.parquet")
                                .filter(pl.col("k") == 1, pl.col("through") == 0, pl.col("hold") == "120s")))
             .with_columns(SIZE.alias("size")).filter(pl.col("size") > 0)
             .with_columns(pl.col("first_play").dt.convert_time_zone("America/New_York").dt.date().alias("day"),
                           (pl.col("side") * (100 * pl.col("y") - pl.col("entry"))).alias("hold_gross")))
        f = f.with_columns((pl.col("hold_gross") - MAKER_RATE * q * (1 - q)).alias("hold_net"))
        for _, kind in [s for s in STRATS if s[0] == L]:
            x = f.filter(pl.col("kind") == kind)
            games, days = x["game_id"].unique().sort(), x.select("day").unique()
            fixed = x.with_columns((pl.col("size") * pl.col("net")).alias("pnl"))
            hold = x.with_columns(pl.col("hold_gross").alias("gross"), (pl.col("size") * pl.col("hold_net")).alias("pnl"))
            base = per_game(fixed, games)
            for v, g in (("fixed exit", fixed), ("hold to settlement", hold)):
                rows.append(dict(strategy=f"{L} {kind}", exit=v) | metrics(g, games, days, base, v == "fixed exit"))
                n = g["pnl"] / g["size"]                       # net cents per contract of each trade
                w = g["size"]
                pg = per_game(g, games)
                wins.append(dict(strategy=f"{L} {kind}", exit=v, trades=g.height,
                                 win_pct=round(100 * float((n > 0).mean()), 1), loss_pct=round(100 * float((n < 0).mean()), 1),
                                 win_pct_sized=round(100 * float(w.filter(n > 0).sum() / w.sum()), 1),
                                 avg_win_c=round(float(n.filter(n > 0).mean()), 1), avg_loss_c=round(float(n.filter(n < 0).mean()), 1),
                                 games_up_pct=round(100 * float((pg > 0).mean()), 1)))
            how.append(x.group_by("how").agg(pl.col("size").sum()).with_columns(
                (100 * pl.col("size") / pl.col("size").sum()).round(1).alias("pct"), pl.lit(f"{L} {kind}").alias("strategy")))
    t = pl.DataFrame(rows).drop("trades")
    h = pl.concat(how).pivot(on="how", index="strategy", values="pct")
    lines = [__doc__.split("\n\n")[0], "", str(t), "", "How the fixed exit ended (% of contracts):", str(h), "",
             "Win rates: share of trades with net > 0 (< 0), unweighted and weighted by size; average net of winning / losing"
             " trades (cents per contract); share of traded games with positive net", str(pl.DataFrame(wins))]
    (OUT / "b16_exit_vs_hold.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
