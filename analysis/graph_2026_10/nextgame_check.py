"""Next-game nodes: diagnostics of stg.sports.nextgame.nextgame_table.

    1. shape: rows, games, nights; share with both teams having played that night
    2. gaps: share of rows with each value and label
    3. as-of: every *_known_at <= t; labels end after t
    4. descriptive response (in-sample, night-clustered t): how the upcoming game's home win
       probability (cents) and implied home margin move with tonight's surprises
         total_x_close = a + b_z (z_home - z_away) + b_ms (ms_home - ms_away)     (x = p, mu)
         rem_x_close   = the same + b_moved * moved_x
       The difference form uses the symmetry: home good news and away bad news push the same way.

Writes data/sports/derived/samples/nextgame_<league>_<season>.parquet.

    venv/bin/python -W ignore analysis/graph_2026_10/nextgame_check.py > analysis/graph_2026_10/out/nextgame_check.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.checks import assert_asof                  # noqa: E402
from stg.sports.nextgame import nextgame_table             # noqa: E402

RUNS = [("MLB", 2025), ("NBA", 2025), ("NFL", 2025)]
OUT = Path("data/sports/derived/samples")


def ols_cluster(y, X, g):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    meat = sum(np.outer(X[g == k].T @ u[g == k], X[g == k].T @ u[g == k]) for k in np.unique(g))
    G = len(np.unique(g))
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1)
    return b, np.sqrt(np.diag(V))


def main() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    d = SportsData()
    OUT.mkdir(parents=True, exist_ok=True)
    tabs = []
    for lg, season in RUNS:
        t = nextgame_table(d, lg, season)
        t.write_parquet(OUT / f"nextgame_{lg}_{season}.parquet")
        tabs.append(t)
    s = pl.concat(tabs, how="diagonal_relaxed")

    print("1. shape")
    print(s.group_by("league").agg(pl.len().alias("rows"), pl.col("game_id").n_unique().alias("games"),
                                   pl.col("night").n_unique().alias("nights"),
                                   (pl.col("played_home") & pl.col("played_away")).mean().round(2).alias("both_played"),
                                   pl.col("days_to_game").median().round(1).alias("days_to_game_med")).sort("league"))

    print("\n2. gaps: share of rows present")
    cols = ["p_pre", "p_t", "mu_pre", "mu_t", "ms_home", "total_p_1h", "total_p_12h", "total_p_close", "rem_p_close",
            "total_mu_close", "y_home"]
    print(s.group_by("league").agg(*[pl.col(c).is_not_null().mean().round(2).alias(c) for c in cols]).sort("league"))

    print("\n3. as-of")
    assert_asof(s)
    late = {h: s.filter(pl.col(f"label_end_{h}") <= pl.col("t")).height for h in ("1h", "12h", "close")}
    print(f"   every *_known_at <= t: OK ({s.height:,} rows); label ends at or before t: {late}")
    print("   spread of the winner legs at t (c): median", s["spread_t"].median(), "| p90", s["spread_t"].quantile(0.9))

    print("\n4. descriptive response of the upcoming game (in-sample; t clustered by night)")
    out = []
    s = s.with_columns((pl.col("z_home") - pl.col("z_away")).alias("dz"),
                       (pl.col("ms_home").fill_null(0) - pl.col("ms_away").fill_null(0)).alias("dms"))
    for (lg,), g in s.group_by("league"):
        for x in ("p", "mu"):
            for lab, extra in ((f"total_{x}_close", []), (f"rem_{x}_close", [f"moved_{x}"])):
                ok = g.filter(pl.col(lab).is_not_null(), *[pl.col(c).is_not_null() for c in extra])
                if ok.height < 50:
                    continue
                cols_ = ["dz", "dms", *extra]
                X = np.column_stack([np.ones(ok.height)] + [ok[c].to_numpy() for c in cols_])
                b, se = ols_cluster(ok[lab].to_numpy(), X, ok["night"].cast(pl.Utf8).to_numpy())
                out.append({"league": lg, "label": lab, "n": ok.height,
                            **{c: f"{b[i + 1]:.2f} (t {b[i + 1] / se[i + 1]:.1f})" for i, c in enumerate(cols_)},
                            "r2": round(1 - ((ok[lab].to_numpy() - X @ b) ** 2).sum() / ((ok[lab].to_numpy() - ok[lab].mean()) ** 2).sum(), 3)})
    print(pl.DataFrame(out).sort("league", "label"))


if __name__ == "__main__":
    main()
