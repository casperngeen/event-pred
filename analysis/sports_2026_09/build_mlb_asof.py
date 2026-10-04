#!/usr/bin/env python
"""The MLB 2025 pilot tables, rebuilt through the as-of data layer (stg.sports).

    venv/bin/python -W ignore analysis/sports_2026_09/build_mlb_asof.py > analysis/sports_2026_09/out/asof/build_mlb.txt

Same four tables and schemas as ``build_mlb.py``, written to ``out/asof/``, so
the downstream scripts run unchanged with ``SPORTS_PILOT_OUT=analysis/sports_2026_09/out/asof``.
What changes:

  start   the actual first pitch (MLB Stats API), known when it happens.
          ``build_mlb.py`` detected it from the game market's tape by looking
          back from the game's end -- a time chosen using the end.
  p_pre   the same VWAP (both team legs, the hour before ``start``), but on
          trades with known_at < start from the sports archive
          (``data/sports/``, which holds more of the tape than ``data/trades/``)
  won     from the final score, known at the game's end
  futures every futures trade, from the same archive

``graphs.py`` with ``SPORTS_PILOT_ASOF=1`` also fits its fill values and label
scale on each fold's days only (the other two leaks). Comparison:
``reports/sports_study.md`` and ``out/asof/*.txt``.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.sports.asof import SportsData  # noqa: E402

OUT = Path("analysis/sports_2026_09/out/asof")
OUT.mkdir(parents=True, exist_ok=True)
SLATE_HOUR = 12
END = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)       # the pilot's 2025 season
KIND = {"champ": "champ", "conference": "pennant", "playoffs": "playoffs", "division": "division"}
naive = lambda c: pl.col(c).dt.replace_time_zone(None)       # noqa: E731  the pilot's naive-UTC times

data = SportsData(cutoff=END - dt.timedelta(microseconds=1))
c = data.contracts.filter(pl.col("league") == "MLB")

# ------------------------------------------------------------------ teams (from the futures listing)
fut_c = c.filter(pl.col("kind").is_in(list(KIND)), pl.col("season") == 2025, pl.col("team").is_not_null())
league = {t: s[-2:] for t, s in fut_c.filter(pl.col("kind") == "conference").select("team", "series_ticker").iter_rows()}
division = {t: s[5:] for t, s in fut_c.filter(pl.col("kind") == "division").select("team", "series_ticker").iter_rows()}
teams = sorted(fut_c.filter(pl.col("kind") == "champ")["team"].unique())
assert len(teams) == 30 and set(league) == set(teams) and set(division) == set(teams), (len(teams), len(league))

# ------------------------------------------------------------------ games
gw = c.filter(pl.col("series_ticker") == "KXMLBGAME", pl.col("team").is_in(teams), pl.col("game_id").is_not_null())
two = gw.group_by("event_ticker").len().filter(pl.col("len") == 2)["event_ticker"]
gw = gw.filter(pl.col("event_ticker").is_in(two.implode()))
res = data.results.filter(pl.col("league") == "MLB").select(
    "game_id", "first_play", "home_score", "away_score", pl.col("known_at").alias("end"))
gw = gw.join(res, on="game_id", how="inner").filter(pl.col("end") < END)
print(f"games: {gw['event_ticker'].n_unique()} with both team legs and a final score "
      f"({gw['end'].min()} -> {gw['end'].max()})")

tr = (data.scan_quotes("KXMLBGAME").filter(pl.col("src") == "trade",
                                           pl.col("ticker").is_in(gw["ticker"].implode()))
      .select("ticker", "known_at", "price", "count").collect()
      .join(gw.select("ticker", "event_ticker", "team", "first_play"), on="ticker"))
pre = tr.filter(pl.col("known_at") < pl.col("first_play"))
hour = pre.filter(pl.col("known_at") >= pl.col("first_play") - pl.duration(hours=1))
last20 = pre.sort("known_at").group_by("ticker").tail(20)
vw = lambda f: f.group_by("ticker").agg(((pl.col("price") * pl.col("count")).sum() / pl.col("count").sum()).alias("vw"))  # noqa: E731
v = vw(hour).join(vw(last20).rename({"vw": "vw20"}), on="ticker", how="full", coalesce=True)
gw = gw.join(v.select("ticker", pl.coalesce("vw", "vw20").alias("vw")), on="ticker", how="left")

rows = []
for (ev,), g in gw.sort("team").partition_by("event_ticker", as_dict=True).items():
    a, b = g.row(0, named=True), g.row(1, named=True)
    pa = np.nanmean([a["vw"] if a["vw"] is not None else np.nan,
                     100 - b["vw"] if b["vw"] is not None else np.nan])
    for me, op, p in ((a, b, pa), (b, a, 100 - pa)):
        mine = me["home_score"] if me["side"] == "home" else me["away_score"]
        theirs = me["away_score"] if me["side"] == "home" else me["home_score"]
        won = mine > theirs
        rows.append(dict(event=ev, team=me["team"], opp=op["team"], start=me["first_play"], end=me["end"],
                         p_pre=p, won=won, z=(float(won) - p / 100) if np.isfinite(p) else np.nan,
                         start_detected=False, title=None))
games = (pl.DataFrame(rows).with_columns(naive("start"), naive("end"))
         .with_columns((pl.col("end") - pl.duration(hours=SLATE_HOUR)).dt.date().alias("slate"),
                       ((pl.col("end") - pl.col("start")).dt.total_minutes() / 60).alias("dur_h")))
games.write_parquet(OUT / "mlb_games.parquet")
g1 = games.filter(pl.col("z").is_finite())
print("start = actual first pitch for every team-game")
print(f"team-games: {games.height}, with p_pre: {g1.height}; game duration (start→end) h: "
      f"median {g1['dur_h'].median():.2f}, p5 {g1['dur_h'].quantile(.05):.2f}, p95 {g1['dur_h'].quantile(.95):.2f}")
print(f"p_pre: mean {g1['p_pre'].mean():.1f}, calibration: mean(won) {g1['won'].mean():.3f} vs "
      f"mean(p_pre)/100 {g1['p_pre'].mean() / 100:.3f}; mean z {g1['z'].mean():+.3f}, sd {g1['z'].std():.3f}")

# ------------------------------------------------------------------ futures
fin = data.finals.select("ticker", "result")
fut_c = fut_c.with_columns(pl.col("kind").replace_strict(KIND).alias("kind")).join(fin, on="ticker", how="left")
ft = (data.scan_quotes(fut_c["series_ticker"].unique().to_list())
      .filter(pl.col("src") == "trade", pl.col("ticker").is_in(fut_c["ticker"].implode()))
      .select("ticker", naive("known_at").alias("t"), pl.col("price").alias("yes_price"), "count").collect()
      .join(fut_c.select("ticker", pl.col("series_ticker").alias("series"), "kind", "team", "result"), on="ticker")
      .sort("t"))
ft.write_parquet(OUT / "mlb_futures.parquet")
print(f"futures trades: {ft.height} on {ft['ticker'].n_unique()} contracts; by kind "
      + ", ".join(f"{k} {n}" for k, n in ft.group_by("kind").len().sort("kind").iter_rows()))

tm = pl.DataFrame({"team": teams, "league": [league[t] for t in teams], "division": [division[t] for t in teams]})
tm.join(fut_c.select("team", "kind", "result").pivot(on="kind", index="team", values="result",
                                                      aggregate_function="first"), on="team"
        ).write_parquet(OUT / "mlb_teams.parquet")

# ------------------------------------------------------------------ daily panel (as build_mlb.py)
days = pl.date_range(games["slate"].min(), games["slate"].max(), "1d", eager=True)
snaps = pl.DataFrame({"slate": days}).with_columns(
    pl.col("slate").cast(pl.Datetime("us")).dt.offset_by(f"{SLATE_HOUR}h").alias("t0"),
    pl.col("slate").cast(pl.Datetime("us")).dt.offset_by(f"{24 + SLATE_HOUR}h").alias("t1"))
grid = snaps.join(pl.DataFrame({"team": teams}), how="cross")
zday = g1.group_by("slate", "team").agg(pl.col("z").sum().alias("z"), pl.len().alias("games"))
panel = grid.join(zday, on=["slate", "team"], how="left").with_columns(
    pl.col("z").fill_null(0.0), pl.col("games").fill_null(0))
for kind in ("champ", "pennant", "playoffs", "division"):
    f = ft.filter(pl.col("kind") == kind).select("team", "t", "yes_price").sort("t")
    for side, col in (("t0", "p0"), ("t1", "p1")):
        panel = panel.sort(side).join_asof(
            f.rename({"t": f"_{side}", "yes_price": f"{kind}_{col}"}), left_on=side,
            right_on=f"_{side}", by="team", strategy="backward", tolerance="7d").drop(f"_{side}")
    n = (panel.select("slate", "team", "t0", "t1").join(f, on="team")
         .filter((pl.col("t") > pl.col("t0")) & (pl.col("t") <= pl.col("t1")))
         .group_by("slate", "team").agg(pl.len().alias(f"{kind}_n")))
    panel = panel.join(n, on=["slate", "team"], how="left").with_columns(pl.col(f"{kind}_n").fill_null(0))
panel = panel.sort("slate", "team")
panel.write_parquet(OUT / "mlb_daily.parquet")
print(f"daily panel: {panel['slate'].n_unique()} slate days × {len(teams)} teams; "
      f"team-days with a game {int((panel['games'] > 0).sum())}")
for kind in ("champ", "pennant", "playoffs", "division"):
    ok = panel.filter((pl.col("games") > 0) & (pl.col(f"{kind}_n") > 0))
    print(f"  {kind:9} team-game-days whose futures traded that slate: {ok.height}")

# ------------------------------------------------------------------ start: detected vs actual
old = Path("analysis/sports_2026_09/out/mlb_games.parquet")
if old.exists():
    o = pl.read_parquet(old).select("event", "team", pl.col("start").alias("start_detected_t"), "start_detected")
    j = games.join(o, on=["event", "team"], how="inner").with_columns(
        ((pl.col("start_detected_t") - pl.col("start")).dt.total_minutes()).alias("lag_min"))
    lag = j["lag_min"]
    print(f"\nbuild_mlb.py's detected start minus the actual first pitch, {j.height} team-games: "
          f"median {lag.median():+.0f} min, p5 {lag.quantile(.05):+.0f}, p95 {lag.quantile(.95):+.0f}; "
          f"|lag| > 15 min: {(lag.abs() > 15).mean():.1%}; detected after the first pitch: {(lag > 0).mean():.1%}")
