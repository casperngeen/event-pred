"""Scaling in: the 12 once-a-minute grids (decisions at second 0, 5, ..., 55 of each minute) at 1/12 size each,
against deciding every 5 s at full size one at a time, and a single once-a-minute grid. Test folds only; the
frozen setup (120 s hold, offset 1c, strictly-through fills, frozen sizing, minimum signal 5c); MLB: the
every-pitch model.

    12 grids x 1/12, stacking         every 5 s, a 1/12 piece whenever the signal clears the minimum (no blocking)
    12 grids x 1/12, one per grid     12 slots per contract, slot = the step's place in the minute; a slot takes
                                      no new piece while its last one is open (at most 12 pieces = 1 full position)
Trade logs: b7_maker_trades.py with DECIDE_OFFSET_S=5..55 [ONE_AT_A_TIME=1] ROW_KINDS=spread,total.
Game-level t: per-game net dollars over the games the every-5-s rule traded.

    venv/bin/python -W ignore analysis/graph_2026_10/b24_scale_in.py
    -> out/b24_scale_in.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b19_decision_timing as B                          # noqa: E402
from b17_holdout_report import exits                      # noqa: E402
from b22_league_thresholds import KINDS, TAGS            # noqa: E402
from b7_maker_robust import OUT                           # noqa: E402

OFFSETS = range(0, 60, 5)


def logs(L: str) -> dict[str, pl.DataFrame]:
    B.TAG = TAGS[L]
    grids = lambda one: pl.concat([B.load(L, (f"_off{o}" if o else "") + one).with_columns(pl.col("size") / 12) for o in OFFSETS],
                                  how="diagonal_relaxed")
    return {"every 5 s, one at a time": B.load(L, "_dec5_one"), "every 60 s, one at a time (grid 0)": B.load(L, "_one"),
            "every 60 s, stacking (grid 0)": B.load(L, ""), "12 grids x 1/12, stacking": grids(""),
            "12 grids x 1/12, one per grid": grids("_one")}


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_hide_column_data_types(True)
    rows = []
    for L in TAGS:
        lg = logs(L)
        for kind in KINDS:
            games = lg["every 5 s, one at a time"].filter(pl.col("kind") == kind)["game_id"].unique()
            for name, f in lg.items():
                for ex, g in exits(f.filter(pl.col("kind") == kind)).items():
                    pg = (pl.DataFrame({"game_id": games}).join(g.group_by("game_id").agg(pl.col("pnl").sum()), on="game_id", how="left")
                          .fill_null(0)["pnl"].to_numpy())
                    rows.append(dict(contract=f"{L} {kind}", exit=ex, rule=name, contracts=round(g["size"].sum()),
                                     net_c=round(g["pnl"].sum() / g["size"].sum(), 2),
                                     t_game=round(pg.mean() / pg.std(ddof=1) * np.sqrt(len(pg)), 2)))
        print(L, "done", flush=True)
    t = pl.DataFrame(rows)
    lines = [__doc__.split("\n\n")[0], "", "Net in cents per contract after fees; t = game-level t; all test folds.", str(t)]
    (OUT / "b24_scale_in.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
