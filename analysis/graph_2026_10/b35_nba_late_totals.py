"""Does the late-game weakness of NBA totals show up in the walk-forward test folds, without any live data?

Two checks, both on test folds only (games.parquet fold 1-5):
  A. Mechanism, from the plays: points scored per minute and the share of plays that are free-throw trips, by
     minutes left and by the score gap, using the change in the game's total score from one play to the next.
  B. Money, from the frozen test-fold trade logs (NBA, every 5 s, one at a time, 120 s hold, fixed exit):
     net cents per contract (size-weighted) for totals and spreads by minutes left and by the score gap at the
     decision, with a 90% bootstrap interval over games.
Minutes left = tau x 48 (tau = share of regulation time left; overtime is clipped at tau = 0.001).

    venv/bin/python -W ignore analysis/graph_2026_10/b35_nba_late_totals.py
    -> out/b35_nba_late_totals.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b19_decision_timing as B                          # noqa: E402
from b17_holdout_report import exits                      # noqa: E402
from b7_maker_robust import OUT                           # noqa: E402

D = Path(__file__).resolve().parents[2] / "data/sports/derived/ingame/NBA"
CLOCK = [(0, 2, "last 2 min"), (2, 5, "2-5 min"), (5, 12, "5-12 min"), (12, 49, "earlier")]
GAP = [(0, 5, "gap <= 5"), (5, 10, "gap 6-10"), (10, 200, "gap > 10")]
rng = np.random.default_rng(0)


def clock_bin(m):
    e = pl.when(pl.lit(False)).then(pl.lit("")).otherwise(pl.lit("earlier"))
    for lo, hi, name in reversed(CLOCK):
        e = pl.when((m >= lo) & (m < hi)).then(pl.lit(name)).otherwise(e)
    return e


def gap_bin(g):
    e = pl.lit("gap > 10")
    for lo, hi, name in reversed(GAP):
        e = pl.when((g > lo - 1 if lo else g >= 0) & (g <= hi)).then(pl.lit(name)).otherwise(e)
    return e


def mechanism() -> str:
    folds = pl.read_parquet(D / "games.parquet").filter(pl.col("fold").is_between(1, 5)).select("game_id")
    p = (pl.read_parquet(D / "plays.parquet").join(folds, on="game_id").sort("game_id", "seq")
         .with_columns((pl.col("total") - pl.col("total").shift(1).over("game_id").fill_null(0)).alias("pts"),
                       (pl.col("tau") * 48).alias("m"), pl.col("margin").abs().alias("gap"),
                       pl.col("tau").shift(1).over("game_id").fill_null(1.0).alias("tau_prev"))
         .with_columns(((pl.col("tau_prev") - pl.col("tau")).clip(0, 1) * 48).alias("minutes")))
    p = p.filter(pl.col("play_type") != "period_end").with_columns(clock_bin(pl.col("m")).alias("clock"), gap_bin(pl.col("gap")).alias("g"))
    t = (p.group_by("clock", "g").agg(plays=pl.len(), games=pl.col("game_id").n_unique(),
                                     pts=pl.col("pts").sum(), minutes=pl.col("minutes").sum(),
                                     ft_share=(pl.col("play_type") == "ft_trip").mean() * 100,
                                     pts_per_play=pl.col("pts").mean())
         .with_columns((pl.col("pts") / pl.col("minutes")).round(2).alias("pts_per_min"),
                       pl.col("ft_share").round(1), pl.col("pts_per_play").round(2))
         .drop("pts", "minutes"))
    order = {n: i for i, (_, _, n) in enumerate(CLOCK)}
    gorder = {n: i for i, (_, _, n) in enumerate(GAP)}
    t = t.with_columns(pl.col("clock").replace_strict(order).alias("_c"), pl.col("g").replace_strict(gorder).alias("_g")).sort("_c", "_g").drop("_c", "_g")
    return "A. Scoring pace and free-throw share by game clock and score gap (test-fold games, all plays)\n" + str(t)


def interval(g: pl.DataFrame, games: np.ndarray) -> tuple[float, float, float]:
    c = g["size"].sum()
    if c == 0:
        return float("nan"), float("nan"), float("nan")
    per = g.group_by("game_id").agg(p=pl.col("pnl").sum(), s=pl.col("size").sum())
    per = pl.DataFrame({"game_id": games}).join(per, on="game_id", how="left").fill_null(0)
    p, s = per["p"].to_numpy(), per["s"].to_numpy()
    est = p.sum() / s.sum()
    idx = rng.integers(0, len(p), (2000, len(p)))
    bs = p[idx].sum(1) / np.maximum(s[idx].sum(1), 1e-9)
    return est, *np.percentile(bs, [5, 95])


def money() -> str:
    f = B.load("NBA", "_dec5_one")
    s = (pl.read_parquet(D / "steps.parquet").select("game_id", pl.col("t").dt.epoch("us").cast(pl.Float64).truediv(1e6).alias("t"),
                                                    pl.col("tau").fill_null(1.0), pl.col("margin").fill_null(0)).sort("t"))
    f = (f.sort("t").join_asof(s, on="t", by="game_id", strategy="backward")
         .with_columns((pl.col("tau") * 48).alias("m"), pl.col("margin").abs().alias("gap")))
    f = f.with_columns(clock_bin(pl.col("m")).alias("clock"), gap_bin(pl.col("gap")).alias("g"))
    f = exits(f)["fixed exit"]
    games_all = f["game_id"].unique().sort().to_numpy()
    rows = []
    for kind in ("total", "spread"):
        x = f.filter(pl.col("kind") == kind)
        for cname in [n for _, _, n in CLOCK]:
            for gname in [None] + [n for _, _, n in GAP]:
                y = x.filter(pl.col("clock") == cname) if gname is None else x.filter(pl.col("clock") == cname, pl.col("g") == gname)
                if y.height == 0:
                    continue
                est, lo, hi = interval(y, games_all)
                rows.append(dict(kind=kind, clock=cname, gap=gname or "all", trades=y.height, contracts=round(y["size"].sum(), 1),
                                 net_c=round(est, 2), lo90=round(lo, 2), hi90=round(hi, 2)))
        inside = x.filter(pl.col("m") < 5, pl.col("gap") <= 10)
        outside = x.filter(~((pl.col("m") < 5) & (pl.col("gap") <= 10)))
        for name, y in (("INSIDE live-style window (<5 min left, gap <= 10)", inside), ("OUTSIDE it", outside)):
            est, lo, hi = interval(y, games_all)
            rows.append(dict(kind=kind, clock=name, gap="", trades=y.height, contracts=round(y["size"].sum(), 1),
                             net_c=round(est, 2), lo90=round(lo, 2), hi90=round(hi, 2)))
    r = pl.DataFrame(rows)
    return ("B. Test-fold trades, net cents per contract after fees (fixed exit), 90% bootstrap interval over games\n"
            + str(r))


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_hide_column_data_types(True)
    out = __doc__.split("\n\n")[0] + "\n\n" + mechanism() + "\n\n" + money() + "\n"
    (OUT / "b35_nba_late_totals.txt").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
