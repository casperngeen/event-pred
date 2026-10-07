"""Is width >= 9c a real threshold or one point on a smooth curve? Current model, NBA spreads, 120 s hold,
|signal| >= 8c, fixed exit, offset 1c, strictly-through fills, all test folds, feed delays 0 / 5 / 10 s.
Width = last YES-taker price - last NO-taker price, each within 120 s (b7_maker_liquidity.liquidity).
Net by width band, and for width >= cut over a range of cuts (by delay and by test fold).

    venv/bin/python -W ignore analysis/graph_2026_10/b9_width_curve.py
    -> out/b9_width_curve.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_backtest import clustered_t                     # noqa: E402
from b7_maker_liquidity import liquidity               # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402

BANDS = [-100, 1, 3, 5, 7, 9, 11, 13, 16, 21, 100]   # [lo, next lo)
CUTS = [3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20]


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    parts = []
    for d in (0, 5, 10):
        f = pl.read_parquet(OUT / f"b7_maker_trades_NBA_d{d}_h30-60-120-1800_tpf.parquet").filter(
            pl.col("k") == 1, pl.col("through") == 0, pl.col("kind") == "spread", pl.col("hold") == "120s", pl.col("signal").abs() >= 8)
        parts.append(with_net(f).with_columns(pl.lit(d).alias("delay")))
    f = liquidity(pl.concat(parts, how="diagonal_relaxed"))
    tt = lambda g: round(clustered_t(g["net"].to_numpy(), g["game_id"].to_numpy()), 1) if g.height >= 20 else None   # noqa: E731
    mn = lambda g: round(g["net"].mean(), 2) if g.height >= 20 else None   # noqa: E731
    rows = []
    for i in range(len(BANDS) - 1):
        lo, hi = BANDS[i], BANDS[i + 1]
        r = dict(width=f"{max(lo, -99)}..{hi - 1}" if lo > -100 else f"<= {hi - 1}")
        for d in (0, 5, 10):
            g = f.filter(pl.col("delay") == d, pl.col("width") >= lo, pl.col("width") < hi)
            r |= {f"d{d}_trades": g.height, f"d{d}_net": mn(g), f"d{d}_t": tt(g)}
        rows.append(r)
    r = dict(width="missing (a side > 120 s old)")
    for d in (0, 5, 10):
        g = f.filter(pl.col("delay") == d, pl.col("width").is_null())
        r |= {f"d{d}_trades": g.height, f"d{d}_net": mn(g), f"d{d}_t": tt(g)}
    rows.append(r)
    crow = []
    for c in CUTS:
        r = dict(cut=c)
        for d in (0, 5, 10):
            g = f.filter(pl.col("delay") == d, pl.col("width") >= c)
            r |= {f"d{d}_trades": g.height, f"d{d}_net": mn(g), f"d{d}_t": tt(g)}
        g = f.filter(pl.col("delay") == 10, pl.col("width") >= c)
        for k in sorted(f["fold"].unique().to_list()):
            x = g.filter(pl.col("fold") == k)
            r[f"d10_fold{k}"] = round(x["net"].mean(), 1) if x.height >= 10 else None
        crow.append(r)
    text = ("Current model, NBA spreads, 120 s hold, |signal| >= 8c, all test folds: net cents per contract by width at entry\n"
            "######## by width band\n" + str(pl.DataFrame(rows)) +
            "\n\n######## width >= cut (fold columns: 10 s delay)\n" + str(pl.DataFrame(crow)))
    (OUT / "b9_width_curve.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
