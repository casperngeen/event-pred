"""Stability over the season of the maker candidates (b7_maker_candidates.py), and how the markets
themselves changed in their first season on Kalshi (spread / total contracts were listed from
2025-26: NFL 2025-09-05, MLB 2025-09-30, NBA 2025-10-21).

1. Candidates by month (of the game), at every delay: trades, net cents per contract, t (clustered
   by game), share stuck to settlement.
2. Market maturity by month, over ALL games of the panel (training months included), spread and
   total contracts only:
     width      rebuilt ask - bid of live, two-sided contracts, sampled every 60 s of game time
                (median, share >= 9c)
     trades     trade prints per game on the game's spread / total contracts, from the first play to
                the feed end (median over games)

    venv/bin/python -W ignore analysis/graph_2026_10/b7_maker_stability.py
    -> out/b7_maker_stability.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b7_maker_liquidity import liquidity                  # noqa: E402
from b7_maker_robust import with_net, summ, OUT           # noqa: E402
from stg.models.ingame import Panel                       # noqa: E402
from stg.sports.loaders import derived_dir, scan_trades   # noqa: E402

TAG = "_h30-60-120-1800_tpf"
CANDIDATES = [("NBA", "spread", "120s", 8.0, False), ("NBA", "total", "120s", 8.0, False), ("NBA", "total", "60s", 8.0, False),
              ("MLB", "total", "60s", 8.0, False), ("NFL", "total", "120s", 5.0, True), ("NFL", "total", "60s", 5.0, True)]
SAMPLE_STEPS = 12          # every 12th 5-s step = 60 s


def market_by_month(L: str) -> pl.DataFrame:
    D = derived_dir() / "ingame" / L
    pn = Panel(D / "panel_v3f")
    feats = json.loads((D / "panel_v3f" / "manifest.json").read_text())["node_market_features"]
    fi = {k: feats.index(k) for k in ("live", "has_bid", "has_ask", "bid", "ask")}
    games = json.loads((D / "panel_v3f" / "manifest.json").read_text())["games"]
    g_of_step = np.repeat(np.arange(len(games)), np.diff(pn.game_step_ptr))
    s = np.arange(0, len(pn.steps_t), SAMPLE_STEPS)
    a, b = pn.step_node_ptr[s], pn.step_node_ptr[s + 1]
    rows = np.concatenate([np.arange(x, y) for x, y in zip(a, b)])
    step_of_row = np.repeat(s, b - a)
    x = np.asarray(pn.nodes_x[rows][:, [fi["live"], fi["has_bid"], fi["has_ask"], fi["bid"], fi["ask"]]], dtype=np.float32)
    kind = np.asarray(pn.nodes_kind[rows])
    ok = (x[:, 0] > 0) & (x[:, 1] > 0) & (x[:, 2] > 0) & np.isin(kind, [1, 2])
    gm = pl.read_parquet(D / "games.parquet").select("game_id", pl.col("first_play").dt.strftime("%Y-%m").alias("month"), "fold",
                                                     "first_play", "feed_end")
    w = (pl.DataFrame({"game_id": np.asarray(games)[g_of_step[step_of_row[ok]]], "width": 100 * (x[ok, 4] - x[ok, 3])})
         .join(gm, on="game_id"))
    wm = w.group_by("month").agg(pl.col("width").median().round(1).alias("median_width_c"),
                                 (pl.col("width") >= 9).mean().round(2).alias("share_width_9c"),
                                 pl.col("game_id").n_unique().alias("games"), pl.col("fold").min().alias("first_fold"))
    con = pl.read_parquet(D / "contracts.parquet").filter(pl.col("kind").is_in(["spread", "total"])).select("ticker", "series_ticker", "game_id")
    tr = (scan_trades(con["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(con["ticker"].unique().implode()))
          .select("ticker", "known_at").collect().join(con.select("ticker", "game_id"), on="ticker").join(gm, on="game_id")
          .filter((pl.col("known_at") >= pl.col("first_play")) & (pl.col("known_at") <= pl.col("feed_end"))))
    tm = (tr.group_by("game_id", "month").len().group_by("month").agg(pl.col("len").median().alias("median_trades_per_game")))
    return wm.join(tm, on="month", how="left").sort("month")


def main() -> None:
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    lines = []
    out = lambda s: (print(s), lines.append(str(s)))   # noqa: E731
    logs = {}
    for L in sorted({c[0] for c in CANDIDATES}):
        files = sorted(OUT.glob(f"b7_maker_trades_{L}_d*{TAG}.parquet"))
        f = pl.concat([with_net(pl.read_parquet(p).filter(pl.col("k") == 1, pl.col("through") == 0))
                       .with_columns(pl.lit(int(p.name.split("_d")[1].split("_")[0])).alias("delay")) for p in files], how="diagonal_relaxed")
        logs[L] = liquidity(f).with_columns(pl.col("first_play").dt.strftime("%Y-%m").alias("month"))
    out("######## 1. candidates by month (net cents per contract; t clustered by game)")
    for L, kind, hold, m, busy in CANDIDATES:
        f = logs[L]
        g0 = f.filter(pl.col("kind") == kind, pl.col("hold") == hold, pl.col("signal").abs() >= m, pl.col("width") >= 9)
        if busy:
            g0 = g0.filter(pl.col("n_5m") >= f["n_5m"].median())
        rows = []
        for (mo, d), g in sorted(g0.group_by("month", "delay"), key=lambda x: x[0]):
            rows.append(summ(g, month=mo, delay=d) | dict(stuck=round((g["how"] == "settled").mean(), 2)))
        t = pl.DataFrame(rows)
        out(f"\n==== {L} {kind}, hold {hold}, signal >= {m:g}c, width >= 9c{', busy' if busy else ''}")
        out(t.pivot(on="delay", index="month", values=["trades", "net", "t"], sort_columns=True).sort("month"))
    out("\n######## 2. market maturity by month: spread / total contracts, all panel games (fold 0 = training only)")
    for L in ["NBA", "MLB", "NFL"]:
        out(f"\n==== {L}")
        out(market_by_month(L))
    (OUT / "b7_maker_stability.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
