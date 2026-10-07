"""Robustness of the signal-filtered maker strategy (fixed exit) from the per-trade logs of
b7_maker_trades.py: stability over the season (test fold, month), cut-off, fill strictness, order
offset, fees, concentration in a few games, contract price, side, game phase; bootstrap by game.

Net = gross - maker fee on a limit fill (MAKER_RATE x p (1 - p), x fee_mult) - taker fee on a taken
exit (FEE_RATE x p (1 - p)); none at settlement. t clustered by game.

    venv/bin/python analysis/graph_2026_10/b7_maker_robust.py 10 _h30-60-120-1800_tpf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.argv += [] if len(sys.argv) > 1 else ["10"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_backtest import FEE_RATE, clustered_t     # noqa: E402
from b6_maker_exit import MAKER_RATE             # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
DELAY = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "_h30-60-120-1800_tpf"
STRATS = [("NBA", "spread", "120s"), ("NBA", "spread", "60s"), ("NBA", "total", "120s"), ("NBA", "total", "60s"),
          ("MLB", "total", "60s"), ("MLB", "total", "120s")]
M = 8.0


def with_net(f: pl.DataFrame, fee_mult: float = 1.0) -> pl.DataFrame:
    q = lambda c: (pl.col(c) / 100).clip(0, 1)   # noqa: E731
    fee_out = (pl.when(pl.col("how") == "taken").then(FEE_RATE * q("exit") * (1 - q("exit")))
               .when(pl.col("how") == "limit filled").then(fee_mult * MAKER_RATE * q("exit") * (1 - q("exit"))).otherwise(0.0))
    f = f.with_columns((pl.col("side") * (pl.col("exit") - pl.col("entry"))).alias("gross"))
    return f.with_columns((pl.col("gross") - fee_mult * MAKER_RATE * q("entry") * (1 - q("entry")) - fee_out).alias("net"))


def summ(g: pl.DataFrame, **lab) -> dict:
    if g.height == 0:
        return dict(**lab, trades=0)
    gm = g["game_id"].to_numpy()
    return dict(**lab, trades=g.height, games=g["game_id"].n_unique(), gross=round(g["gross"].mean(), 2),
                net=round(g["net"].mean(), 2), t=round(clustered_t(g["net"].to_numpy(), gm), 1) if g.height >= 30 else None)


def boot_ci(g: pl.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    by = g.group_by("game_id").agg(pl.col("net").sum().alias("s"), pl.len().alias("n"))
    s, c = by["s"].to_numpy(), by["n"].to_numpy()
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(s), size=(n, len(s)))
    m = s[i].sum(1) / c[i].sum(1)
    return round(float(np.quantile(m, 0.025)), 2), round(float(np.quantile(m, 0.975)), 2)


def show(title: str, rows: list[dict]) -> None:
    print(f"\n######## {title}")
    print(pl.DataFrame(rows))


def main() -> None:
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    logs = {L: pl.read_parquet(OUT / f"b7_maker_trades_{L}_d{DELAY}{TAG}.parquet") for L in {s[0] for s in STRATS}}
    base = {L: with_net(f.filter(pl.col("k") == 1, pl.col("through") == 0)) for L, f in logs.items()}
    sel = lambda L, kind, hold, m=M: base[L].filter(pl.col("kind") == kind, pl.col("hold") == hold, pl.col("signal").abs() >= m)  # noqa: E731

    rows = []
    for L, kind, hold in STRATS:
        g = sel(L, kind, hold)
        lo, hi = boot_ci(g)
        rows.append(summ(g, league=L, kind=kind, hold=hold) | dict(ci95_lo=lo, ci95_hi=hi,
                    net_per_100_decisions=round(100 * g["net"].sum() / g["decisions"][0], 2)))
    show(f"headline: |signal| >= {M:g}c, fixed exit, feed delay {DELAY} s (95% CI: bootstrap over games)", rows)

    rows = []
    for L, kind, hold in STRATS:
        g = sel(L, kind, hold).with_columns(pl.col("first_play").dt.date().alias("d"))
        for (fold,), h in sorted(g.group_by("fold"), key=lambda x: x[0]):
            rows.append(summ(h, league=L, kind=kind, hold=hold, fold=fold, dates=f"{h['d'].min()}..{h['d'].max()}"))
    show("by test fold (in date order; each trained only on earlier games)", rows)

    rows = []
    for L, kind, hold in STRATS:
        g = sel(L, kind, hold).with_columns(pl.col("first_play").dt.strftime("%Y-%m").alias("month"))
        for (mo,), h in sorted(g.group_by("month"), key=lambda x: x[0]):
            rows.append(summ(h, league=L, kind=kind, hold=hold, month=mo))
    show("by month", rows)

    rows = []
    for L, kind, hold in STRATS:
        for m in [5, 6, 7, 8, 9, 10, 12, 15]:
            rows.append(summ(sel(L, kind, hold, m), league=L, kind=kind, hold=hold, min_signal=m))
    show("signal cut-off (the 8c cut-off was chosen after looking: is the curve smooth?)", rows)

    rows = []
    for L, kind, hold in STRATS:
        for k, thr in [(1, 0), (1, 1), (2, 0)]:
            g = with_net(logs[L].filter(pl.col("k") == k, pl.col("through") == thr, pl.col("kind") == kind,
                                        pl.col("hold") == hold, pl.col("signal").abs() >= M))
            rows.append(summ(g, league=L, kind=kind, hold=hold, offset_c=k,
                             fill_rule="strictly through" if thr == 0 else "2c+ through"))
    show("order offset and fill strictness (2c+ through: a fill needs a print at least 2c past our price)", rows)

    rows = []
    for L, kind, hold in STRATS:
        g0 = sel(L, kind, hold)
        for fm in [0.0, 1.0, 2.0, 4.0]:
            rows.append(summ(with_net(g0, fm), league=L, kind=kind, hold=hold, maker_fee_x=fm))
    show(f"maker fee multiple (1 = {MAKER_RATE} x p(1-p) cents, unverified)", rows)

    rows = []
    for L, kind, hold in STRATS:
        g = sel(L, kind, hold)
        by = g.group_by("game_id").agg(pl.col("net").sum().alias("s")).sort("s", descending=True)
        tot, n = by["s"].sum(), by.height
        top = by.head(max(1, n // 20))["game_id"]
        rows.append(dict(league=L, kind=kind, hold=hold, games=n, total_net_c=round(tot, 0),
                         share_from_top10_games=round(by.head(10)["s"].sum() / tot, 2) if tot else None,
                         games_positive=round((by["s"] > 0).mean(), 2),
                         **{k: v for k, v in summ(g.filter(~pl.col("game_id").is_in(top.implode())), x=0).items() if k in ("net", "t")},
                         ))
    show("concentration (net / t after dropping the best 5% of games)", rows)

    rows = []
    for L, kind, hold in STRATS:
        g = sel(L, kind, hold).with_columns(
            pl.col("entry").cut([20, 40, 60, 80], labels=["<20", "20-40", "40-60", "60-80", ">80"]).alias("price"))
        for (b,), h in sorted(g.group_by("price"), key=lambda x: str(x[0])):
            rows.append(summ(h, league=L, kind=kind, hold=hold, entry_price=b))
    show("by entry price (cents)", rows)

    rows = []
    for L, kind, hold in STRATS:
        g = sel(L, kind, hold)
        for s, nm in [(1, "buy YES"), (-1, "sell YES")]:
            rows.append(summ(g.filter(pl.col("side") == s), league=L, kind=kind, hold=hold, side=nm))
    show("by side", rows)

    rows = []
    for L, kind, hold in STRATS:
        g = sel(L, kind, hold).with_columns(((pl.col("t") - pl.col("first_play").dt.epoch("s")) / 60).alias("min"))
        q = g["min"].quantile(1 / 3), g["min"].quantile(2 / 3)
        g = g.with_columns(pl.when(pl.col("min") < q[0]).then(pl.lit("1 early")).when(pl.col("min") < q[1]).then(pl.lit("2 middle"))
                           .otherwise(pl.lit("3 late")).alias("phase"))
        for (ph,), h in sorted(g.group_by("phase"), key=lambda x: x[0]):
            rows.append(summ(h, league=L, kind=kind, hold=hold, phase=ph))
    show("by game phase (thirds of the trades by minutes since the first play)", rows)

    rows = []
    for L, kind, hold in STRATS:
        g = sel(L, kind, hold)
        for (how,), h in g.group_by("how"):
            rows.append(summ(h, league=L, kind=kind, hold=hold, exit=how) | dict(share=round(h.height / g.height, 2)))
    show("how the exit happened", sorted(rows, key=lambda r: (r["league"], r["kind"], r["hold"], r["exit"])))


if __name__ == "__main__":
    main()
