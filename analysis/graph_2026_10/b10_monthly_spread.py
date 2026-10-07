"""Spread of returns by month for the four candidates (current model, frozen sizing of b9_sizing.py,
120 s hold, fixed exit, offset 1c, strictly-through fills, all test folds), feed delay 10 s.
Per month: trades placed, total size, net per contract (size-weighted mean), t of the mean P&L per game,
size-weighted percentiles of the per-trade net (10 / 25 / 50 / 75 / 90), the size-weighted share of
winning trades, and the spread of P&L per game (cents at full size 1: 10th / 50th / 90th percentile).

    venv/bin/python -W ignore analysis/graph_2026_10/b10_monthly_spread.py
    -> out/b10_monthly_spread.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b7_maker_liquidity import liquidity               # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402
from b9_sizing import RULES                              # noqa: E402

CANDS = [("NBA", "spread"), ("NBA", "total"), ("MLB", "total"), ("NFL", "total")]
SIZE = RULES["quadratic signal x linear width (missing 1)"]


def wq(x: np.ndarray, w: np.ndarray, qs: list[float]) -> list[float]:
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return [float(x[o][np.searchsorted(c, q)]) for q in qs]


def stats(g: pl.DataFrame, label: str) -> dict:
    x, w = g["net"].to_numpy(), g["size"].to_numpy()
    per = g.group_by("game_id").agg((pl.col("size") * pl.col("net")).sum().alias("p"))["p"].to_numpy()
    p10, p25, p50, p75, p90 = wq(x, w, [0.1, 0.25, 0.5, 0.75, 0.9])
    return dict(month=label, trades=g.height, games=len(per), total_size=round(w.sum(), 1),
                per_contract=round(float((x * w).sum() / w.sum()), 2),
                t=round(float(per.mean() / per.std(ddof=1) * np.sqrt(len(per))), 1) if len(per) > 2 else None,
                p10=round(p10, 1), p25=round(p25, 1), median=round(p50, 1), p75=round(p75, 1), p90=round(p90, 1),
                won=round(float(w[x > 0].sum() / w.sum()), 2),
                game_p10=round(float(np.percentile(per, 10)), 1), game_median=round(float(np.median(per)), 1),
                game_p90=round(float(np.percentile(per, 90)), 1))


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(40)
    lines = []
    for L in ("NBA", "MLB", "NFL"):
        f = liquidity(with_net(pl.read_parquet(OUT / f"b7_maker_trades_{L}_d10_h30-60-120-1800_tpf.parquet").filter(
            pl.col("k") == 1, pl.col("through") == 0, pl.col("hold") == "120s")))
        f = f.with_columns(SIZE.alias("size")).filter(pl.col("size") > 0).with_columns(pl.col("first_play").dt.strftime("%Y-%m").alias("month"))
        for kind in [k for l_, k in CANDS if l_ == L]:
            g = f.filter(pl.col("kind") == kind)
            rows = [stats(x, m) for (m,), x in sorted(g.group_by("month"), key=lambda z: z[0])] + [stats(g, "ALL")]
            lines += [f"\n================ {L} {kind}", str(pl.DataFrame(rows))]
    text = ("Current model, frozen sizing, 120 s, 10 s delay, all test folds. Net cents per contract; percentiles of the per-trade net "
            "weighted by size; game_*: P&L per game in cents at full size 1\n" + "\n".join(lines))
    (OUT / "b10_monthly_spread.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
