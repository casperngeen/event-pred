"""NFL: how fast do the markets price a play? Diagnostic only, prices from trade prints, no model.

For every touchdown, field goal and turnover, and every spread / total contract of that game trading between
15c and 85c just before the play:
    p0      last print 10 s before the play started (t_event)
    move    last print 120 s after the play was logged (known_at) - p0; only events with |move| >= 3c
    done(x) share of the move already in the price at time x: sum over events of sign(move) x (p(x) - p0)
            / sum of |move|, at x = the play's start, the play's log time + (-10, 0, 5, 10, 20, 30, 60) s
The model learns a play at known_at + 10 s (the feed delay), so 1 - done(+10 s) is what is left for it.
Periods: 2025 weeks 1-5 and 6-8 (training-only fold 0), the 2025-26 test folds, the 2026 holdout.

    venv/bin/python -W ignore analysis/graph_2026_10/b30_nfl_reaction_speed.py test
    STG_SPORTS_UNSEAL=final-test venv/bin/python -W ignore analysis/graph_2026_10/b30_nfl_reaction_speed.py holdout
    -> out/b30_nfl_reaction_speed_<test|holdout>.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

MODE = sys.argv[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b6_maker_exit import sweep_sorted                    # noqa: E402
from b7_maker_robust import OUT                           # noqa: E402
from stg.sports.loaders import derived_dir, scan_trades  # noqa: E402
from stg.splits import sports_holdout_unsealed           # noqa: E402

assert MODE in ("test", "holdout") and (MODE == "test") != sports_holdout_unsealed()
D = derived_dir() / "ingame" / "NFL"
LAGS = [-10, 0, 5, 10, 20, 30, 60]
PLAYS = ("touchdown", "field_goal", "turnover")


def epoch(c: str) -> pl.Expr:
    return pl.col(c).dt.epoch("us").cast(pl.Float64) / 1e6


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_fmt_str_lengths(40)
    pl.Config.set_tbl_hide_column_data_types(True)
    games = pl.read_parquet(D / "games.parquet").select("game_id", "fold", "first_play")
    season = pl.when(pl.col("first_play") >= pl.datetime(2026, 8, 1, time_zone="UTC")).then(pl.datetime(2026, 9, 9, time_zone="UTC")) \
        .otherwise(pl.datetime(2025, 9, 4, time_zone="UTC"))
    games = games.with_columns(((pl.col("first_play") - pl.duration(hours=4) - season).dt.total_days() // 7 + 1).alias("week"))
    games = games.with_columns(pl.when(pl.col("fold") == 99).then(pl.lit("2026 holdout (weeks 1-4)"))
                               .when(pl.col("fold") >= 1).then(pl.lit("2025-26 test folds (week 9 on)"))
                               .when(pl.col("week") <= 5).then(pl.lit("2025 weeks 1-5 (fold 0)"))
                               .otherwise(pl.lit("2025 weeks 6-8 (fold 0)")).alias("period"))
    plays = (pl.read_parquet(D / "plays.parquet").filter(pl.col("play_group").is_in(PLAYS))
             .select("game_id", "play_group", epoch("t_event").alias("t_start"), epoch("known_at").alias("t_known")))
    con = pl.read_parquet(D / "contracts.parquet").filter(pl.col("kind").is_in(["spread", "total"]), pl.col("game_id").is_in(games["game_id"].implode()))
    tr = sweep_sorted(scan_trades(con["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(con["ticker"].implode()))
                      .select("ticker", epoch("known_at").alias("t"), "yes_price", "taker_side", "trade_id").collect())
    tr = tr.join(con.select("ticker", "game_id", "kind"), on="ticker")
    out = []
    by_game = {g: d for (g,), d in plays.partition_by("game_id", as_dict=True).items()}
    for (g, tk, kind), d in tr.partition_by("game_id", "ticker", "kind", as_dict=True).items():
        pg = by_game.get(g)
        if pg is None:
            continue
        t, px = d["t"].to_numpy(), d["yes_price"].to_numpy().astype(float)

        def at(x: np.ndarray) -> np.ndarray:
            j = np.searchsorted(t, x, side="right") - 1
            return np.where(j >= 0, px[np.maximum(j, 0)], np.nan)

        ts, tk_ = pg["t_start"].to_numpy(), pg["t_known"].to_numpy()
        p0, p_end = at(ts - 10), at(tk_ + 120)
        row = dict(game_id=g, kind=kind, play=pg["play_group"].to_numpy(), p0=p0, move=p_end - p0, at_start=at(ts) - p0,
                   **{f"at_{l}": at(tk_ + l) - p0 for l in LAGS}, play_s=tk_ - ts)
        out.append(pl.DataFrame(row))
    ev = (pl.concat(out).filter(pl.col("p0").is_between(15, 85), pl.col("move").abs() >= 3)
          .join(games.select("game_id", "period"), on="game_id"))
    cols = ["at_start"] + [f"at_{l}" for l in LAGS]
    names = ["play start"] + [f"log {l:+d}s" for l in LAGS]
    rows = []
    for (period, kind, play), x in ev.group_by("period", "kind", "play", maintain_order=True):
        s = np.sign(x["move"].to_numpy())
        den = np.abs(x["move"].to_numpy()).sum()
        r = dict(period=period, kind=kind, play=play, events=x.height, median_move_c=round(float(x["move"].abs().median()), 1),
                 median_play_s=round(float(x["play_s"].median()), 1))
        for c, n in zip(cols, names):
            v = x[c].to_numpy()
            ok = np.isfinite(v)
            r[n] = round(100 * float((s[ok] * v[ok]).sum() / np.abs(x["move"].to_numpy()[ok]).sum()), 1) if ok.any() else None
        r["left_for_model_pct"] = round(100 - r["log +10s"], 1)
        rows.append(r)
    t = pl.DataFrame(rows).sort("kind", "play", "period")
    lines = [__doc__.split("\n\n")[0], "",
             f"######## {MODE}: % of the 2-min price move already in the last print (columns), events with |move| >= 3c",
             str(t)]
    (OUT / f"b30_nfl_reaction_speed_{MODE}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
