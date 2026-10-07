"""Simplifying the model, trading side: the frozen maker strategy run on the model without the futures contracts
(panel_v3) against the model with them (panel_v3f), per seed and for the average of the five seeds' forecasts.

Strategy as b11_ablation_pnl.py: feed delay 10 s, fixed exit, offset 1c, strictly-through fills, frozen sizing
"quadratic signal x linear width (missing 1)" (size 0-1 contract per trade), all test folds. Columns:
    contracts       sum of sizes
    gross_c / NET_c cents per contract before / after Kalshi fees (size-weighted); NET is the headline
    net_$           total net P&L in dollars at a 1-contract maximum per trade
    t_game          game-clustered t of the mean net P&L per game
    sharpe_ann      daily net P&L (US/Eastern game date, game days only) mean / sd x sqrt(game days per year,
                    from the test period); 1-contract scale, no capacity limit
    days_up_%       share of game days with positive net P&L
    max_dd_$        largest peak-to-trough fall of cumulative daily net P&L
    folds_up        test folds (of 5) with positive net P&L
    diff_$, diff_t  per-game net P&L minus the base (with futures, average of 5), summed; paired t over games
Trade logs: b7_maker_trades.py with RUN_TAG=_h30-60-120-1800_tp[_ens5], SEED=s, panel_v3 (and the _tpf logs).

    venv/bin/python -W ignore analysis/graph_2026_10/b14_nofutures_pnl.py
    -> out/b14_nofutures_pnl.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b7_maker_liquidity import liquidity               # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402
from b9_sizing import RULES                             # noqa: E402

TAG = "_h30-60-120-1800_tp"
BASE = "with futures, average of 5"
VERSIONS = {BASE: "f_ens5"} | {f"with futures, seed{s}": "f" + (f"_seed{s}" if s else "") for s in range(5)} \
    | {"without, average of 5": "_ens5"} | {f"without, seed{s}": f"_seed{s}" if s else "" for s in range(5)}
STRATS = [("NBA", "spread", "120s"), ("MLB", "total", "120s"), ("NBA", "total", "120s"), ("NFL", "total", "120s"),
          ("MLB", "spread", "120s"), ("NFL", "spread", "120s")]     # last two: dropped candidates, for reference
SIZE = RULES["quadratic signal x linear width (missing 1)"]


def per_game(g: pl.DataFrame, games: pl.Series) -> np.ndarray:
    return (pl.DataFrame({"game_id": games}).join(g.group_by("game_id").agg(pl.col("pnl").sum()), on="game_id", how="left")
            .fill_null(0).sort("game_id")["pnl"].to_numpy())


def metrics(g: pl.DataFrame, games: pl.Series, days: pl.DataFrame, base: np.ndarray, is_base: bool) -> dict:
    c = g["size"].sum()
    pg = per_game(g, games)
    d = (days.join(g.group_by("day").agg(pl.col("pnl").sum()), on="day", how="left").fill_null(0).sort("day")["pnl"].to_numpy() / 100)
    span_y = (days["day"].max() - days["day"].min()).days / 365.25
    per_year = len(d) / span_y if span_y > 0 else np.nan
    cum = np.cumsum(d)
    dd = float((np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]).max())
    fold = g.group_by("fold").agg(pl.col("pnl").sum())
    diff = pg - base
    return dict(trades=g.height, contracts=round(c), gross_c=round((g["size"] * g["gross"]).sum() / c, 2), NET_c=round(g["pnl"].sum() / c, 2),
                net_usd=round(g["pnl"].sum() / 100), t_game=round(pg.mean() / pg.std(ddof=1) * np.sqrt(len(pg)), 1),
                sharpe_ann=round(d.mean() / d.std(ddof=1) * np.sqrt(per_year), 1), days_up_pct=round(100 * (d > 0).mean()),
                max_dd_usd=round(dd), folds_up=f"{(fold['pnl'] > 0).sum()}/5",
                diff_usd=None if is_base else round(diff.sum() / 100), diff_t=None if is_base else round(diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff)), 1))


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_hide_column_data_types(True)
    lines = [__doc__.split("\n\n")[0], "", __doc__.split("\n\n")[1].split("\n    venv")[0]]
    summary = []
    for L in sorted({s[0] for s in STRATS}):
        logs = {}
        for v, suf in VERSIONS.items():
            f = pl.read_parquet(OUT / f"b7_maker_trades_{L}_d10{TAG}{suf}.parquet").filter(pl.col("k") == 1, pl.col("through") == 0)
            logs[v] = (liquidity(with_net(f)).with_columns(SIZE.alias("size")).filter(pl.col("size") > 0)
                       .with_columns((pl.col("size") * pl.col("net")).alias("pnl"),
                                     pl.col("first_play").dt.convert_time_zone("America/New_York").dt.date().alias("day")))
        for _, kind, hold in [s for s in STRATS if s[0] == L]:
            xs = {v: x.filter(pl.col("kind") == kind, pl.col("hold") == hold) for v, x in logs.items()}
            allx = pl.concat([x.select("game_id", "day") for x in xs.values()])
            games = allx["game_id"].unique().sort()          # every game any version traded (absent = 0 P&L)
            days = allx.select("day").unique()
            base = per_game(xs[BASE], games)
            rows = [dict(version=v) | metrics(g, games, days, base, v == BASE) for v, g in xs.items()]
            t = pl.DataFrame(rows)
            lines += [f"\n================ {L} {kind}, hold {hold}  ({len(games)} games, {days.height} game days)", str(t)]
            for v in (BASE, "without, average of 5"):
                r = t.filter(pl.col("version") == v).row(0, named=True)
                seeds = t.filter(pl.col("version").str.starts_with(v.split(",")[0]) & pl.col("version").str.contains("seed"))["NET_c"]
                summary.append(dict(strategy=f"{L} {kind}", version=v.replace(", average of 5", ""), gross_c=r["gross_c"], NET_c=r["NET_c"],
                                    NET_c_seed_range=f"{seeds.min():.2f} to {seeds.max():.2f}", net_usd=r["net_usd"], t_game=r["t_game"],
                                    sharpe_ann=r["sharpe_ann"], max_dd_usd=r["max_dd_usd"], folds_up=r["folds_up"], diff_usd=r["diff_usd"], diff_t=r["diff_t"]))
    lines = lines[:3] + ["", "SUMMARY (average-of-5 forecasts; seed range = NET_c over single seeds)", str(pl.DataFrame(summary))] + lines[3:]
    (OUT / "b14_nofutures_pnl.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
