"""MLB: can live-visible signals tell that a pitch has been thrown which the model does not know yet?

Label at a moment t: 'new pitch' = at least one pitch started after the model's last known pitch (known after the
10 s feed delay), the pitches_since >= 1 of b20 / b25. Signals, all visible live at t (flag = 'new pitch'):
    time since last known pitch > X s     X = 12, 15, 18, 20, 25, 30 (the feed reports when its last pitch ended)
    A s <= time since last known pitch < B s   a window: long gaps (>= 60 s) are mostly breaks with no pitch
    game printed in the last W s          a print on any of the game's spread / total contracts, W = 5, 10
    contract printed in the last W s      a print on the contract itself (trade moments only)
    contract price moved since last known pitch   last print now vs last print when that pitch became known, >= 1c / 2c
Two sets of moments, test folds only:
    all 5-s steps         every model step of every test-fold game (spread / total trading hours)
    trade moments         the decision moments of scaling in (12 grids x 1/12, stacking; b24), spread + total
Per signal: share flagged, precision (new pitch when flagged), recall (share of new-pitch moments flagged),
accuracy, and 'up to date when not flagged' (what a filter that trades only unflagged moments would keep).

    venv/bin/python -W ignore analysis/graph_2026_10/b26_pitch_detect.py
    -> out/b26_pitch_detect.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b19_decision_timing as B                          # noqa: E402
from b6_maker_exit import sweep_sorted                    # noqa: E402
from b7_maker_robust import OUT                           # noqa: E402
from stg.models.ingame import Panel                      # noqa: E402
from stg.sports.loaders import scan_trades               # noqa: E402

D = Path("data/sports/derived/ingame/MLB_pitch")
DELAY_S = 10.0
TAG = "_h30-60-120-1800_tp_pitch_ens5"
AGES = [12, 15, 18, 20, 25, 30]


def epoch(c: str) -> pl.Expr:
    return pl.col(c).dt.epoch("us").cast(pl.Float64) / 1e6


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(60)
    pl.Config.set_tbl_hide_column_data_types(True)
    pn = Panel(D / "panel_v3")
    games = json.loads((D / "panel_v3" / "manifest.json").read_text())["games"]
    fold = dict(pl.read_parquet(D / "games.parquet").select("game_id", "fold").iter_rows())
    test = [i for i, g in enumerate(games) if fold.get(g, 0) >= 1]
    con = pl.read_parquet(D / "contracts.parquet").filter(pl.col("kind").is_in(["spread", "total"]))
    gids = [games[i] for i in test]
    tr = (scan_trades(con["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(con.filter(pl.col("game_id").is_in(gids))["ticker"].implode()))
          .select("ticker", epoch("known_at").alias("t"), "yes_price", "taker_side", "trade_id").collect())
    tr = sweep_sorted(tr).join(con.select("ticker", "game_id"), on="ticker")
    gprint = {g: np.sort(d["t"].to_numpy()) for (g,), d in tr.partition_by("game_id", as_dict=True).items()}
    ev = (pl.read_parquet("data/sports/pbp/events_MLB.parquet", columns=["game_id", "t", "is_pitch"])
          .filter(pl.col("is_pitch"), pl.col("game_id").is_in(gids)).select("game_id", epoch("t").alias("t")).sort("game_id", "t"))
    starts = {g: d["t"].to_numpy() for (g,), d in ev.partition_by("game_id", as_dict=True).items()}

    def base(gid: str, gi: int, t: np.ndarray) -> dict:
        """Age, label and game-print flags at times t of one game."""
        pt = pn.plays_t[pn.game_play_ptr[gi]:pn.game_play_ptr[gi + 1]]
        k = np.searchsorted(pt, t - DELAY_S, side="right")
        last = np.where(k > 0, pt[np.maximum(k - 1, 0)], np.nan)
        s = starts.get(gid, np.array([]))
        n_new = np.searchsorted(s, t, side="right") - np.searchsorted(s, np.nan_to_num(last, nan=-np.inf), side="right")
        p = gprint.get(gid, np.array([]))
        j = np.searchsorted(p, t, side="right")
        lastp = np.where(j > 0, p[np.maximum(j - 1, 0)], -np.inf)
        return {"t": t, "last_known": last, "age": t - last, "new_pitch": n_new >= 1,
                "game_print_5s": t - lastp <= 5, "game_print_10s": t - lastp <= 10}

    # 1. all 5-s steps (only from the game's first spread / total print on, so the market exists)
    parts = []
    for gi in test:
        gid = games[gi]
        t = pn.steps_t[pn.game_step_ptr[gi]:pn.game_step_ptr[gi + 1]].astype(float)
        p = gprint.get(gid)
        if p is None:
            continue
        t = t[t >= p[0]]
        b = base(gid, gi, t)
        parts.append(pl.DataFrame(b).with_columns(pl.lit(gid).alias("game_id")))
    steps = pl.concat(parts).filter(pl.col("last_known").is_not_nan())

    # 2. trade moments: scaling in, stacking (each grid's decisions), spread + total
    B.TAG = TAG
    f = pl.concat([B.load("MLB", f"_off{o}" if o else "") for o in range(0, 60, 5)], how="diagonal_relaxed").filter(
        pl.col("kind").is_in(["spread", "total"])).select("game_id", "ticker", "kind", "t")
    gidx = {g: i for i, g in enumerate(games)}
    parts = []
    for (gid,), d in f.partition_by("game_id", as_dict=True).items():
        b = base(gid, gidx[gid], d["t"].to_numpy())
        parts.append(d.with_columns(*[pl.Series(k, v) for k, v in b.items() if k != "t"]))
    mom = pl.concat(parts).filter(pl.col("last_known").is_not_nan()).sort("t")
    cp = tr.select("ticker", "t", pl.col("yes_price").alias("px"), pl.col("t").alias("tp")).sort("t")
    mom = (mom.join_asof(cp, on="t", by="ticker", strategy="backward")
           .with_columns(pl.col("last_known").alias("t_k")).sort("t_k")
           .join_asof(cp.rename({"t": "t_k", "px": "px_k", "tp": "tp_k"}), on="t_k", by="ticker", strategy="backward")
           .with_columns(((pl.col("t") - pl.col("tp")) <= 5).fill_null(False).alias("contract_print_5s"),
                         ((pl.col("t") - pl.col("tp")) <= 10).fill_null(False).alias("contract_print_10s"),
                         ((pl.col("px") - pl.col("px_k")).abs() >= 1).fill_null(False).alias("contract_moved_1c"),
                         ((pl.col("px") - pl.col("px_k")).abs() >= 2).fill_null(False).alias("contract_moved_2c")))

    def score(x: pl.DataFrame, sigs: dict[str, pl.Expr]) -> pl.DataFrame:
        y = x["new_pitch"].to_numpy()
        rows = []
        for name, e in sigs.items():
            s = x.select(e.alias("s"))["s"].to_numpy().astype(bool)
            tp, fp, fn, tn = (s & y).sum(), (s & ~y).sum(), (~s & y).sum(), (~s & ~y).sum()
            rows.append(dict(signal=name, flagged_pct=round(100 * s.mean(), 1), precision_pct=round(100 * tp / max(tp + fp, 1), 1),
                             recall_pct=round(100 * tp / max(tp + fn, 1), 1), accuracy_pct=round(100 * (tp + tn) / len(y), 1),
                             up_to_date_if_not_flagged_pct=round(100 * tn / max(tn + fn, 1), 1), kept_pct=round(100 * (~s).mean(), 1)))
        return pl.DataFrame(rows)

    common = {f"time since last known pitch > {a} s": pl.col("age") > a for a in AGES} | {
        **{f"{a} s <= time since last known pitch < {b} s": pl.col("age").is_between(a, b, closed="left")
           for a, b in [(15, 60), (15, 45), (18, 60), (20, 60)]},
        "game printed in last 5 s": pl.col("game_print_5s"), "game printed in last 10 s": pl.col("game_print_10s"),
        "time > 15 s OR game printed in last 10 s": (pl.col("age") > 15) | pl.col("game_print_10s"),
        "time > 15 s AND game printed in last 10 s": (pl.col("age") > 15) & pl.col("game_print_10s")}
    contract = {"contract printed in last 5 s": pl.col("contract_print_5s"), "contract printed in last 10 s": pl.col("contract_print_10s"),
                "contract price moved >= 1c since last known pitch": pl.col("contract_moved_1c"),
                "contract price moved >= 2c since last known pitch": pl.col("contract_moved_2c"),
                "time > 15 s OR contract moved >= 1c": (pl.col("age") > 15) | pl.col("contract_moved_1c"),
                "time > 15 s AND contract moved >= 1c": (pl.col("age") > 15) & pl.col("contract_moved_1c")}
    age_tab = lambda x: (x.with_columns(pl.col("age").cut([12, 15, 18, 20, 25, 30, 60], left_closed=True).cast(pl.String).alias("age_s"))
                         .group_by("age_s").agg(pl.len().alias("moments"), (100 * pl.col("new_pitch").mean()).round(1).alias("new_pitch_pct"))
                         .with_columns((100 * pl.col("moments") / pl.col("moments").sum()).round(1).alias("moments_pct"))
                         .sort(pl.col("age_s").str.extract(r"(-?[\d.]+|inf)").cast(pl.Float64, strict=False), nulls_last=False))
    lines = [__doc__.split("\n\n")[0], "",
             f"-- all 5-s steps: {steps.height:,} moments, new pitch at {100 * steps['new_pitch'].mean():.1f}%", str(score(steps, common)),
             "", "   new-pitch rate by time since the last known pitch", str(age_tab(steps)),
             "", f"-- trade moments (scaling in, stacking): {mom.height:,} moments, new pitch at {100 * mom['new_pitch'].mean():.1f}%",
             str(score(mom, common | contract)), "", "   new-pitch rate by time since the last known pitch", str(age_tab(mom))]
    (OUT / "b26_pitch_detect.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
