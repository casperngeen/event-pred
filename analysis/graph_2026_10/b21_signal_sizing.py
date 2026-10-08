"""Sizing by the signal alone (no width / spread part), and results by signal size. Test folds only, the frozen
model (average of 5 seeds), 120 s hold, offset 1c, strictly-through fills, the three decision rules of
b19_decision_timing.py, both exits.

    signal x width (frozen)  size = ((|signal| - 5) / 10)^2 x (width / 10), each part capped at 1, width part 1
                             when width is missing
    signal only              size = ((|signal| - 5) / 10)^2, capped at 1
Then every trade counted once (size 1), by |signal| bucket: how net per contract rises with the signal.

    venv/bin/python -W ignore analysis/graph_2026_10/b21_signal_sizing.py
    -> out/b21_signal_sizing.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b17_holdout_report import exits, summary              # noqa: E402
from b19_decision_timing import RULES, STRATS, TAG       # noqa: E402
from b7_maker_liquidity import liquidity                 # noqa: E402
from b7_maker_robust import with_net, OUT                 # noqa: E402
from b9_sizing import RULES as SIZING                     # noqa: E402

SIZES = {"signal x width (frozen)": SIZING["quadratic signal x linear width (missing 1)"],
         "signal only": SIZING["quadratic signal only"]}
EDGES, LABELS = [6, 8, 10, 15], ["5-6c", "6-8c", "8-10c", "10-15c", ">=15c"]


def load_raw(L: str, suf: str) -> pl.DataFrame | None:
    files = sorted(OUT.glob(f"b7_maker_trades_{L}_d10{TAG}{suf}.parquet")) or sorted(OUT.glob(f"b7_maker_trades_{L}_d10{TAG}{suf}_fold*.parquet"))
    if not files:
        return None
    f = pl.concat([pl.read_parquet(x) for x in files], how="vertical_relaxed").filter(pl.col("k") == 1, pl.col("through") == 0,
                                                                                       pl.col("hold") == "120s")
    return liquidity(with_net(f)).with_columns(pl.col("first_play").dt.convert_time_zone("America/New_York").dt.date().alias("day"))


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_hide_column_data_types(True)
    rows, buckets = [], []
    for L in ("NBA", "MLB", "NFL"):
        logs = {r: load_raw(L, suf) for r, suf in RULES.items()}
        for _, kind in [s for s in STRATS if s[0] == L]:
            for rule, f in logs.items():
                if f is None:
                    continue
                x0 = f.filter(pl.col("kind") == kind)
                for sname, size in SIZES.items():
                    x = x0.with_columns(size.alias("size")).filter(pl.col("size") > 0)
                    games, days = x["game_id"].unique().sort(), x.select("day").unique()
                    for ex, g in exits(x).items():
                        s = summary(g, games, days)
                        rows.append(dict(strategy=f"{L} {kind}", rule=rule, sizing=sname, exit=ex, trades=g.height,
                                         contracts=s["contracts"], NET_c=s["NET_c"], t_game=s["t_game"], net_usd=s["net_usd"],
                                         max_dd_usd=s["max_dd_usd"]))
                # every trade once, by signal bucket
                x = x0.with_columns(pl.lit(1.0).alias("size"))
                for ex, g in exits(x).items():
                    b = (g.with_columns(pl.col("signal").abs().cut(EDGES, labels=LABELS, left_closed=True).cast(pl.String).alias("bucket"))
                         .group_by("bucket").agg(pl.len().alias("trades"), pl.col("game_id").n_unique().alias("games"),
                                                 (pl.col("pnl").mean()).round(2).alias("net_c"),
                                                 (100 * (pl.col("pnl") > 0).mean()).round(0).alias("win_pct")))
                    for r in b.iter_rows(named=True):
                        buckets.append(dict(strategy=f"{L} {kind}", rule=rule, exit=ex) | r)
    order = {r: i for i, r in enumerate(RULES)}
    t = pl.DataFrame(rows)
    piv = (t.with_columns(pl.format("{} ({})", "NET_c", "t_game").alias("v"))
           .pivot(on="rule", index=["strategy", "exit", "sizing"], values="v").sort("strategy", "exit", "sizing"))
    bk = pl.DataFrame(buckets).with_columns(pl.col("bucket").replace_strict({b: i for i, b in enumerate(LABELS)}).alias("_b"),
                                             pl.col("rule").replace_strict(order).alias("_r"))
    bpiv = (bk.with_columns(pl.format("{} ({})", "net_c", "trades").alias("v"))
            .pivot(on="bucket", index=["strategy", "exit", "rule", "_r"], values="v").sort("strategy", "exit", "_r").drop("_r")
            .select("strategy", "exit", "rule", *LABELS))
    lines = [__doc__.split("\n\n")[0], "",
             "1. Net cents per contract after fees (game-level t), by sizing and decision rule",
             str(piv.select("strategy", "exit", "sizing", *RULES)), "",
             "   detail", str(t.with_columns(pl.col("rule").replace_strict(order).alias("_r")).sort("strategy", "exit", "sizing", "_r").drop("_r")),
             "", "2. Every trade once, by |signal|: net cents per contract after fees (trades)", str(bpiv)]
    (OUT / "b21_signal_sizing.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4]))
    print("\n".join(lines[-1:]))


if __name__ == "__main__":
    main()
