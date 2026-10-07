"""Are the frozen strategy settings still well chosen for the 5-seed average, and was their choice seed luck?
The frozen settings were picked on seed 0's trades (b9_sizing.py, b9_cutoff_grid_all.py). Here, for each
version (seeds 0-4 and the average of 5) and candidate (NBA spreads, MLB totals, NBA totals, NFL totals),
feed delay 10 s, all test folds:

1. every sizing rule of b9_sizing.RULES (hold 120 s, offset 1c, strictly-through fills);
2. the frozen rule's shape:  size = clip((|signal| - START) / RAMP, 0, 1)^2 x clip(width / WSCALE, 0, 1)
   (missing width = 1), over START x RAMP x WSCALE; frozen = (5, 10, 10);
3. the frozen rule under the other trade settings in the logs: offset 1c / 2c, fills strictly through or
   at least 2c through, hold 60 s / 120 s.
Each cell: net cents per contract and the game-clustered t of per-game P&L; for the shape grid also the
frozen shape's rank among all shapes by t, per version. A setting is "still well chosen" if it is near the
top for the average AND for most single seeds; re-picking on these same folds would add selection, so
the bar to change a frozen setting is that it is clearly beaten across seeds, not only on the average.

    venv/bin/python -W ignore analysis/graph_2026_10/b13_sizing_check.py
    -> out/b13_sizing_check.txt
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b7_maker_liquidity import liquidity               # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402
from b9_sizing import RULES, game_t                     # noqa: E402

TAG = "_h30-60-120-1800_tpf"
VERSIONS = {"seed0": "", "seed1": "_seed1", "seed2": "_seed2", "seed3": "_seed3", "seed4": "_seed4", "average": "_ens5"}
STRATS = [("NBA", "spread"), ("MLB", "total"), ("NBA", "total"), ("NFL", "total")]
STARTS, RAMPS, WSCALES = [3, 4, 5, 6, 7, 8], [5, 10, 15, 20], [5, 10, 15, 20]
FROZEN = (5, 10, 10)


def shape(start: float, ramp: float, wscale: float) -> pl.Expr:
    a = pl.col("signal").abs()
    return ((a - start) / ramp).clip(0, 1) ** 2 * (pl.col("width") / wscale).clip(0, 1).fill_null(1)


def score(x: pl.DataFrame, size: pl.Expr, games: pl.Series) -> tuple[float | None, float | None]:
    g = x.with_columns(size.alias("size")).filter(pl.col("size") > 0).with_columns((pl.col("size") * pl.col("net")).alias("pnl"))
    c = g["size"].sum()
    if c == 0 or g.height < 30:
        return None, None
    return round(g["pnl"].sum() / c, 2), round(game_t(g, games), 1)


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_fmt_str_lengths(50)
    lines = ["Sizing and trade settings, seeds 0-4 and their average, feed delay 10 s, all test folds. Cells: net c/contract (t)."]
    for L in sorted({s[0] for s in STRATS}):
        logs = {v: liquidity(with_net(pl.read_parquet(OUT / f"b7_maker_trades_{L}_d10{TAG}{suf}.parquet"))) for v, suf in VERSIONS.items()}
        for _, kind in [s for s in STRATS if s[0] == L]:
            lines.append(f"\n================ {L} {kind}")
            base = {v: x.filter(pl.col("kind") == kind) for v, x in logs.items()}
            games = base["seed0"]["game_id"].unique()
            std = {v: x.filter(pl.col("k") == 1, pl.col("through") == 0, pl.col("hold") == "120s") for v, x in base.items()}
            # 1. the rules
            rows = []
            for name, size in RULES.items():
                r = {"rule": name}
                for v, x in std.items():
                    n, t = score(x, size, games)
                    r[v] = f"{n:+.1f} ({t:.1f})" if n is not None else "-"
                rows.append(r)
            lines += ["-- sizing rules (hold 120 s, offset 1c, strictly through)", str(pl.DataFrame(rows))]
            # 2. the shape grid
            grid = []
            for s0, ramp, ws in itertools.product(STARTS, RAMPS, WSCALES):
                for v, x in std.items():
                    n, t = score(x, shape(s0, ramp, ws), games)
                    grid.append(dict(start=s0, ramp=ramp, wscale=ws, version=v, net=n, t=t))
            g = pl.DataFrame(grid).drop_nulls()
            g = g.with_columns(pl.col("t").rank("min", descending=True).over("version").alias("rank"))
            n_shapes = len(STARTS) * len(RAMPS) * len(WSCALES)
            fz = g.filter(pl.col("start") == FROZEN[0], pl.col("ramp") == FROZEN[1], pl.col("wscale") == FROZEN[2])
            best = g.sort("t", descending=True).group_by("version", maintain_order=True).first()
            summ = fz.select("version", pl.col("net").alias("frozen_net"), pl.col("t").alias("frozen_t"), pl.col("rank").alias(f"frozen_rank_of_{n_shapes}")).join(
                best.select("version", pl.format("({}, {}, {})", "start", "ramp", "wscale").alias("best_shape"), pl.col("net").alias("best_net"),
                            pl.col("t").alias("best_t")), on="version")
            # the shape that does best on average rank across the six versions
            avg = (g.group_by("start", "ramp", "wscale").agg(pl.col("rank").mean().round(1).alias("mean_rank"), pl.col("t").min().alias("worst_t"),
                                                             pl.col("t").filter(pl.col("version") == "average").first().alias("t_average"))
                   .sort("mean_rank").head(8))
            lines += [f"-- shape grid: start x ramp x width scale ({n_shapes} shapes); frozen = {FROZEN}", str(summ),
                      "   shapes with the best mean rank over the six versions", str(avg)]
            # sensitivity of the average's t to each shape parameter, the others at their frozen values
            sens = []
            for p, vals in (("start", STARTS), ("ramp", RAMPS), ("wscale", WSCALES)):
                r = {"parameter": p}
                for val in vals:
                    q = dict(zip(("start", "ramp", "wscale"), FROZEN)) | {p: val}
                    c = g.filter(pl.col("version") == "average", pl.col("start") == q["start"], pl.col("ramp") == q["ramp"], pl.col("wscale") == q["wscale"])
                    r[str(val)] = f"{c['net'][0]:+.1f} ({c['t'][0]:.1f})" if c.height else "-"
                sens.append(r)
            lines += ["   average of 5: one parameter at a time (others frozen); columns = the parameter's value", str(pl.DataFrame(sens, infer_schema_length=None))]
            # 3. trade settings under the frozen rule
            rows = []
            for (k, thr, hold) in itertools.product((1, 2), (0, 1), ("60s", "120s")):
                r = {"offset": k, "through": thr, "hold": hold}
                for v, x in base.items():
                    n, t = score(x.filter(pl.col("k") == k, pl.col("through") == thr, pl.col("hold") == hold), shape(*FROZEN), games)
                    r[v] = f"{n:+.1f} ({t:.1f})" if n is not None else "-"
                rows.append(r)
            lines += ["-- trade settings, frozen rule (frozen: offset 1, through 0, hold 120s; through 1 = a stricter fill check, not a choice)",
                      str(pl.DataFrame(rows))]
    (OUT / "b13_sizing_check.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
