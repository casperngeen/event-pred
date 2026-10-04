"""Chunk 2 diagnostics: the training-sample table (stg.sports.samples.sample_table).

    1. shape: rows, nights, nodes, per league and kind; share of rows exposed tonight
    2. gaps: share of rows with each input and label present
    3. as-of: every *_known_at <= t (the decision time); labels end after t
    4. labels: distribution of the total and remaining moves, exposed rows
    5. a few rows

Writes data/sports/derived/samples/<league>_<season>.parquet.

    venv/bin/python -W ignore analysis/graph_2026_10/samples_check.py > analysis/graph_2026_10/out/samples_check.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.checks import assert_asof                  # noqa: E402
from stg.sports.graph import between_game_graph            # noqa: E402
from stg.sports.samples import HORIZONS, sample_table      # noqa: E402

RUNS = [("MLB", 2025), ("NBA", 2025), ("NFL", 2025)]
OUT = Path("data/sports/derived/samples")


def main() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(230)
    d = SportsData()
    edges = pl.read_parquet("data/sports/derived/edges.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    allr = []
    for lg, season in RUNS:
        s = sample_table(d, between_game_graph(d, edges, lg, season))
        s.write_parquet(OUT / f"{lg}_{season}.parquet")
        allr.append(s)
    s = pl.concat(allr, how="diagonal_relaxed")
    exposed = (pl.col("played") | (pl.col("n_rival_games") > 0))

    print("1. shape")
    print(s.group_by("league", "kind").agg(pl.len().alias("rows"), pl.col("night").n_unique().alias("nights"),
                                           pl.col("node").n_unique().alias("nodes"),
                                           exposed.mean().round(2).alias("share_exposed"),
                                           pl.col("played").mean().round(2).alias("share_played"))
          .sort("league", "kind"))

    print("\n2. gaps: share of rows with each field present")
    cols = ["value_pre", "value_t", "moved", "win_pct", *[f"total_{h}" for h in HORIZONS], *[f"rem_{h}" for h in HORIZONS]]
    print(s.group_by("league", "kind").agg(*[pl.col(c).is_not_null().mean().round(2).alias(c) for c in cols])
          .sort("league", "kind"))

    print("\n3. as-of")
    assert_asof(s)
    late = {h: s.filter(pl.col(f"label_end_{h}") <= pl.col("t")).height for h in HORIZONS}
    print(f"   every *_known_at <= t: OK ({s.height:,} rows); label ends at or before t: {late}")
    print("   decision delay after the night's last result (min):",
          ((s["t"] - s["last_result"]).dt.total_minutes()).describe()
          .filter(pl.col("statistic").is_in(["min", "50%", "max"])).rows())
    print("   decision lag after the night's first play (h):",
          ((s["t"] - s["t_pre"]).dt.total_minutes() / 60).describe()
          .filter(pl.col("statistic").is_in(["min", "50%", "max"])).rows())

    print("\n4. labels on exposed rows (cents; wins nodes in wins): median |move| and the share that moved at all")
    e = s.filter(exposed)
    print(e.group_by("league", "kind").agg(
        pl.len().alias("n"), pl.col("moved").abs().median().round(2).alias("|moved|_med"),
        *[pl.col(f"total_{h}").abs().median().round(2).alias(f"|total_{h}|_med") for h in HORIZONS],
        *[(pl.col(f"rem_{h}") != 0).mean().round(2).alias(f"rem_{h}_nonzero") for h in HORIZONS],
        pl.col("one_sided_t").mean().round(2).alias("one_sided_t")).sort("league", "kind"))

    print("\n5. example rows (NYY division, MLB)")
    print(s.filter(pl.col("node") == "NYY-division", pl.col("played")).select(
        "night", "z_own", "x_rival", "n_rival_games", "value_pre", "value_t", "moved", "total_1h", "rem_1h",
        "rem_24h", "win_pct", "games_behind", "frac_remaining").head(8))


if __name__ == "__main__":
    main()
