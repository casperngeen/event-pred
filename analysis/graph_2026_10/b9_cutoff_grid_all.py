"""Signal cut-off x width cut-off grids for every league / contract type (current model), plus the sizing
rules of b9_sizing.py. Hold 120 s (and 60 s for MLB totals), fixed exit, offset 1c, strictly-through fills,
all test folds, feed delay 10 s (sizing also at 0 and 5 s; NFL has no 5 s run). Net cents per contract (clustered t).

    venv/bin/python -W ignore analysis/graph_2026_10/b9_cutoff_grid_all.py
    -> out/b9_cutoff_grid_all.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_backtest import clustered_t                     # noqa: E402
from b7_maker_liquidity import liquidity               # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402
from b9_sizing import RULES, game_t                     # noqa: E402

STRATS = [("NBA", "total", "120s"), ("NBA", "winner", "120s"), ("MLB", "total", "60s"), ("MLB", "total", "120s"),
          ("MLB", "spread", "120s"), ("MLB", "winner", "120s"), ("NFL", "total", "120s"), ("NFL", "spread", "120s")]
SIG = [5, 6, 7, 8, 10, 12, 15]
WID = [None, 3, 5, 7, 9, 11, 14]
SHOW = ["hard: signal >= 8, width >= 9", "hard: signal >= 8, any width", "quadratic signal only",
        "quadratic signal x linear width (missing 1)", "quadratic signal x linear width (missing 0.5)"]


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_fmt_str_lengths(50)
    cell = lambda g: f"{g['net'].mean():+.1f} ({clustered_t(g['net'].to_numpy(), g['game_id'].to_numpy()):.1f}) n={g.height}" if g.height >= 30 else "-"   # noqa: E731
    lines = ["Current model, all test folds: net cents per contract (clustered t), trades. Grid at 10 s delay: |signal| >= row, width >= column."]
    for L in sorted({s[0] for s in STRATS}):
        parts = []
        for d in (0, 5, 10):
            fn = OUT / f"b7_maker_trades_{L}_d{d}_h30-60-120-1800_tpf.parquet"
            if not fn.exists():
                continue
            f = pl.read_parquet(fn).filter(pl.col("k") == 1, pl.col("through") == 0)
            parts.append(with_net(f).with_columns(pl.lit(d).alias("delay")))
        logs = liquidity(pl.concat(parts, how="diagonal_relaxed"))
        for _, kind, hold in [s for s in STRATS if s[0] == L]:
            x = logs.filter(pl.col("kind") == kind, pl.col("hold") == hold)
            f = x.filter(pl.col("delay") == 10)
            if f.height < 30:
                continue
            rows = []
            for s in SIG:
                r = {"signal >=": s}
                for w in WID:
                    g = f.filter(pl.col("signal").abs() >= s)
                    r["any width" if w is None else f"width >= {w}"] = cell(g if w is None else g.filter(pl.col("width") >= w))
                rows.append(r)
            sz = []
            for name in SHOW:
                r = dict(rule=name)
                for d in sorted(x["delay"].unique().to_list()):
                    y = x.filter(pl.col("delay") == d)
                    g = y.with_columns(RULES[name].alias("size")).filter(pl.col("size") > 0).with_columns((pl.col("size") * pl.col("net")).alias("pnl"))
                    c = g["size"].sum()
                    r |= {f"d{d}_trades": g.height, f"d{d}_total_size": round(c), f"d{d}_per_contract": round(g["pnl"].sum() / c, 2) if c else None,
                          f"d{d}_t": round(game_t(g, y["game_id"].unique()), 1) if c else None}
                sz.append(r)
            lines += [f"\n================ {L} {kind}, hold {hold}", str(pl.DataFrame(rows)), "sizing rules:", str(pl.DataFrame(sz))]
    (OUT / "b9_cutoff_grid_all.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
