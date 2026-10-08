"""When to decide: once a minute (the spec so far) or at every 5-s model step, on the test folds only.

The frozen candidates (120 s hold, frozen sizing, offset 1c, strictly-through fills), the frozen model (no
futures, average of 5 seeds' forecasts), both exits, three decision rules (b7_maker_trades.py):
    every 60 s              a decision every 60 s of game time; a contract can hold overlapping orders and
                            positions (a new order while the last trade on it is still open)
    every 60 s, one at a time   the same grid, but a contract with an open order or position is skipped
    every 5 s, one at a time    a decision at every model step for every free contract (DECIDE_S=5): acts as
                            soon as the signal appears, never stacks
Metrics as b17_holdout_report.py. Trade logs: b7_maker_trades.py with RUN_TAG=_h30-60-120-1800_tp_ens5 and
ONE_AT_A_TIME=1 [DECIDE_S=5 ROW_KINDS=spread,total FOLD=k], results/ingame/ens5, panel_v3.

    venv/bin/python -W ignore analysis/graph_2026_10/b19_decision_timing.py
    -> out/b19_decision_timing.txt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_stability as S                                  # noqa: E402  (b17 sets argv for it)
from b17_holdout_report import SIZE, exits, summary, wins   # noqa: E402
from b7_maker_liquidity import liquidity                 # noqa: E402
from b7_maker_robust import with_net, OUT                 # noqa: E402

TAG = os.environ.get("TAG", "_h30-60-120-1800_tp_ens5")   # TAG=_h30-60-120-1800_tp_pitch_ens5 LEAGUES=MLB: the MLB every-pitch model
LEAGUES = os.environ.get("LEAGUES", "NBA,MLB,NFL").split(",")
SUFFIX = "" if TAG == "_h30-60-120-1800_tp_ens5" else TAG.replace("_h30-60-120-1800", "")   # output name for another model
RULES = {"every 60 s": "", "every 60 s, one at a time": "_one", "every 5 s, one at a time": "_dec5_one"}
STRATS = [("NBA", "spread"), ("MLB", "total"), ("NBA", "total"), ("NFL", "total"),
          ("MLB", "spread"), ("NFL", "spread")]     # last two: dropped candidates, for reference


def load(L: str, suf: str) -> pl.DataFrame | None:
    files = sorted(OUT.glob(f"b7_maker_trades_{L}_d10{TAG}{suf}.parquet")) or sorted(OUT.glob(f"b7_maker_trades_{L}_d10{TAG}{suf}_fold*.parquet"))
    if not files:
        return None
    f = pl.concat([pl.read_parquet(x) for x in files], how="vertical_relaxed").filter(pl.col("k") == 1, pl.col("through") == 0,
                                                                                       pl.col("hold") == "120s")
    return (liquidity(with_net(f)).with_columns(SIZE.alias("size")).filter(pl.col("size") > 0)
            .with_columns(pl.col("first_play").dt.convert_time_zone("America/New_York").dt.date().alias("day")))


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_hide_column_data_types(True)
    main_rows, win_rows = [], []
    for L in LEAGUES:
        logs = {r: load(L, suf) for r, suf in RULES.items()}
        for _, kind in [s for s in STRATS if s[0] == L]:
            for rule, f in logs.items():
                if f is None:
                    continue
                x = f.filter(pl.col("kind") == kind)
                games, days = x["game_id"].unique().sort(), x.select("day").unique()
                for ex, g in exits(x).items():
                    main_rows.append(dict(strategy=f"{L} {kind}", exit=ex, rule=rule, games=len(games),
                                          trades_per_game=round(g.height / max(len(games), 1), 1)) | summary(g, games, days))
                    win_rows.append(dict(strategy=f"{L} {kind}", exit=ex, rule=rule) | wins(g))
    order = {r: i for i, r in enumerate(RULES)}
    t = pl.DataFrame(main_rows).with_columns(pl.col("rule").replace_strict(order).alias("_o")).sort("strategy", "exit", "_o").drop("_o")
    w = pl.DataFrame(win_rows).with_columns(pl.col("rule").replace_strict(order).alias("_o")).sort("strategy", "exit", "_o").drop("_o")
    lines = [__doc__.split("\n\n")[0], "", "Net / gross in cents per contract after / before fees; t = game-level t; all test folds.",
             str(t.drop("months_up")), "", "Win rates", str(w)]
    (OUT / f"b19_decision_timing{SUFFIX}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
