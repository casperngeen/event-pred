"""Signal cut-off x width cut-off grid: current model, NBA spreads, 120 s hold, fixed exit, offset 1c,
strictly-through fills, all test folds. Net cents per contract (clustered t), feed delay 10 s; plus net by
1c signal band within width >= 5c and all widths.

    venv/bin/python -W ignore analysis/graph_2026_10/b9_cutoff_grid.py
    -> out/b9_cutoff_grid.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_backtest import clustered_t                     # noqa: E402
from b7_maker_liquidity import liquidity               # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402

SIG = [5, 6, 7, 8, 9, 10, 12, 15]
WID = [None, 3, 5, 7, 9, 11, 14]


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    f = pl.read_parquet(OUT / "b7_maker_trades_NBA_d10_h30-60-120-1800_tpf.parquet").filter(
        pl.col("k") == 1, pl.col("through") == 0, pl.col("kind") == "spread", pl.col("hold") == "120s")
    f = liquidity(with_net(f))
    cell = lambda g: f"{g['net'].mean():+.1f} ({clustered_t(g['net'].to_numpy(), g['game_id'].to_numpy()):.1f}) n={g.height}" if g.height >= 30 else "-"   # noqa: E731
    rows = []
    for s in SIG:
        r = {"signal >=": s}
        for w in WID:
            g = f.filter(pl.col("signal").abs() >= s)
            if w is not None:
                g = g.filter(pl.col("width") >= w)
            r["any width" if w is None else f"width >= {w}"] = cell(g)
        rows.append(r)
    band = []
    for lo in range(5, 16):
        hi = lo + 1 if lo < 15 else 1e9
        b = f.filter(pl.col("signal").abs() >= lo, pl.col("signal").abs() < hi)
        band.append({"signal band": f"{lo}..{lo + 1}" if lo < 15 else ">= 15", "any width": cell(b),
                     "width >= 5": cell(b.filter(pl.col("width") >= 5)), "width < 5 or missing": cell(b.filter(~(pl.col("width") >= 5)))})
    text = ("Current model, NBA spreads, 120 s hold, 10 s delay, all test folds: net cents per contract (clustered t), trades\n"
            "######## |signal| >= row and width >= column\n" + str(pl.DataFrame(rows)) +
            "\n\n######## by 1c signal band\n" + str(pl.DataFrame(band)))
    (OUT / "b9_cutoff_grid.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
