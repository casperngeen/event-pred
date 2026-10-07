"""Maker backtest with the GRU model's forecasts vs the ridge-corrected forecasts
(b8_corrected_preds.py), on test folds 2-5 only (fold 1 cannot be corrected). Same trade logic and
liquidity measures (b7_maker_trades.py, b7_maker_liquidity.py): fixed exit, offset 1c,
strictly-through fills.

    venv/bin/python -W ignore analysis/graph_2026_10/b8_compare_corrected.py
    -> out/b8_compare_corrected.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b7_maker_liquidity import liquidity            # noqa: E402
from b7_maker_robust import with_net, summ, OUT     # noqa: E402

TAG = "_h30-60-120-1800_tpf"
STRATS = [("NBA", "spread", "120s"), ("NBA", "spread", "60s"), ("NBA", "total", "120s"), ("NBA", "total", "60s"),
          ("MLB", "total", "60s"), ("MLB", "total", "120s"), ("MLB", "spread", "120s"), ("NFL", "total", "120s"),
          ("NFL", "spread", "120s"), ("NBA", "winner", "120s"), ("MLB", "winner", "120s")]
SIGNALS = [5.0, 8.0, 10.0]


def load(L: str, tag: str) -> pl.DataFrame:
    parts = []
    for p in sorted(OUT.glob(f"b7_maker_trades_{L}_d*{tag}.parquet")):
        d = int(p.name.split("_d")[1].split("_")[0])
        parts.append(with_net(pl.read_parquet(p).filter(pl.col("k") == 1, pl.col("through") == 0, pl.col("fold") >= 2))
                     .with_columns(pl.lit(d).alias("delay")))
    return liquidity(pl.concat(parts, how="diagonal_relaxed"))


def main() -> None:
    pl.Config.set_tbl_rows(500)
    pl.Config.set_tbl_width_chars(230)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    rows = []
    for L in sorted({s[0] for s in STRATS}):
        logs = {"model": load(L, TAG), "corrected": load(L, TAG + "_ridge")}
        for _, kind, hold in [s for s in STRATS if s[0] == L]:
            for m in SIGNALS:
                for grp, cond in [("all", pl.lit(True)), ("width >= 9c", pl.col("width") >= 9)]:
                    for d in sorted(logs["corrected"]["delay"].unique().to_list()):
                        r = dict(league=L, kind=kind, hold=hold, min_signal=m, group=grp, delay=d)
                        for name, f in logs.items():
                            g = f.filter(pl.col("delay") == d, pl.col("kind") == kind, pl.col("hold") == hold,
                                         pl.col("signal").abs() >= m, cond)
                            s = summ(g, x=0)
                            r |= {f"{name}_trades": s["trades"], f"{name}_net": s.get("net"), f"{name}_t": s.get("t")}
                        rows.append(r)
    t = pl.DataFrame(rows)
    t.write_parquet(OUT / "b8_compare_corrected.parquet")
    lines = ["Maker backtest, folds 2-5: GRU model forecasts vs ridge-corrected forecasts (net cents per contract)"]
    for L in sorted({s[0] for s in STRATS}):
        lines += [f"\n================ {L}", str(t.filter(pl.col("league") == L).drop("league"))]
    (OUT / "b8_compare_corrected.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
