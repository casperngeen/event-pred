"""Hold to the end of the game instead of exiting: the current model's maker entries on NBA spreads
(signal at the 120 s horizon >= 8c, same entry rule and fills as b7_maker_trades.py), each held to the
contract's result. Net = side x (100 y - entry) - maker fee at entry (no exit fee at settlement).
Compared with the fixed exit on the same entries; all test folds; with and without width >= 9c.

    venv/bin/python -W ignore analysis/graph_2026_10/b9_hold_to_end.py
    -> out/b9_hold_to_end.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_backtest import clustered_t                     # noqa: E402
from b6_maker_exit import MAKER_RATE                     # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402
from b9_candidates_by_model import entry_state          # noqa: E402
from stg.sports.loaders import scan_trades               # noqa: E402


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(60)
    rows, folds = [], []
    for d in (0, 5, 10):
        f = pl.read_parquet(OUT / f"b7_maker_trades_NBA_d{d}_h30-60-120-1800_tpf.parquet").filter(
            pl.col("k") == 1, pl.col("through") == 0, pl.col("kind") == "spread", pl.col("hold") == "120s", pl.col("signal").abs() >= 8)
        tr = (scan_trades(f["ticker"].str.extract(r"^([^-]+)").unique().to_list()).filter(pl.col("ticker").is_in(f["ticker"].unique().implode()))
              .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "taker_side").collect())
        q = (pl.col("entry") / 100).clip(0, 1)
        f = entry_state(with_net(f), tr).with_columns(
            (pl.col("side") * (100 * pl.col("y") - pl.col("entry")) - MAKER_RATE * q * (1 - q)).alias("hold_net"))
        for grp, cond in [("all", pl.lit(True)), ("width >= 9c", pl.col("width") >= 9)]:
            g = f.filter(cond)
            gm = g["game_id"].to_numpy()
            h = g["hold_net"].to_numpy()
            rows.append(dict(delay=d, group=grp, trades=g.height, games=g["game_id"].n_unique(),
                             fixed_exit_net=round(g["net"].mean(), 2), fixed_exit_t=round(clustered_t(g["net"].to_numpy(), gm), 1),
                             hold_net=round(h.mean(), 2), hold_t=round(clustered_t(h, gm), 1), hold_median=round(float(np.median(h)), 1),
                             hold_won=round(float((h > 0).mean()), 2), hold_sd=round(float(h.std()), 1),
                             mean_entry=round(g["entry"].mean(), 1)))
            for (k,), x in sorted(g.group_by("fold"), key=lambda z: z[0]):
                folds.append(dict(delay=d, group=grp, fold=k, trades=x.height, fixed_exit=round(x["net"].mean(), 1),
                                  hold=round(x["hold_net"].mean(), 1),
                                  hold_t=round(clustered_t(x["hold_net"].to_numpy(), x["game_id"].to_numpy()), 1)))
    text = ("Current model, NBA spreads, signal >= 8c (120 s horizon): fixed exit vs holding to the result (cents per contract)\n"
            + str(pl.DataFrame(rows)) + "\n\nby test fold\n"
            + str(pl.DataFrame(folds).pivot(on="delay", index=["group", "fold"], values=["fixed_exit", "hold", "hold_t"]).sort("group", "fold")))
    (OUT / "b9_hold_to_end.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
