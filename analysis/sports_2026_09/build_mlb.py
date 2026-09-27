#!/usr/bin/env python
"""Build the MLB 2025 game-resolution panel: the sports arm's trigger and target tables.

    venv/bin/python analysis/sports_2026_09/build_mlb.py > analysis/sports_2026_09/out/build_mlb.txt

Self-contained: reads only the in-sample archive (``data/markets/``,
``data/trades/``; never the ``_OOS_DO_NOT_USE_*`` folders) and writes only to
``analysis/sports_2026_09/out/``. Design: ``reports/sports_study_plan.md``.

Triggers — one row per (game, team):
  start   first pitch, detected from the game market's tape: the first 10-minute
          bin in the 6 h before the end whose trade count is ≥ 5× the median of
          the bins 8–5 h before the end (and ≥ 10 trades); fallback end − 3.25 h
  p_pre   the team's pre-game win probability, from the two game markets' trades
          in the hour before ``start`` (VWAP of the team's YES and 100 − the
          opponent's YES, averaged)
  z       1[team won] − p_pre / 100: the game's surprise for that team, the
          analogue of a release's (actual − expected)
  end     the game market's close time (the market closes when the game ends)

Targets — each team's futures, canonical team codes (the futures series spell
some teams differently: A/ATH, ARI/AZ, KAN/KC, FLA/MIA, WAS/WSH):
  champ     KXMLB           pennant   KXMLBAL / KXMLBNL
  playoffs  KXMLBPLAYOFFS   division  KXMLB{AL,NL}{EAST,CENT,WEST}
Division and league membership are read off those tickers, not hard-coded.

Outputs:
  out/mlb_games.parquet      one row per (game, team) with start, end, p_pre, z
  out/mlb_futures.parquet    every futures trade (team, kind, time, price, count)
  out/mlb_teams.parquet      team → league, division; futures settlement results
  out/mlb_daily.parquet      slate day × team: summed surprise, and per futures
                             kind the as-of price at the slate's start and end
                             (12:00 UTC to 12:00 UTC next day) and whether it
                             traded in between
"""
from __future__ import annotations

import glob

import numpy as np
import polars as pl

OUT = "analysis/sports_2026_09/out"
ALIAS = {"A": "ATH", "ARI": "AZ", "KAN": "KC", "FLA": "MIA", "WAS": "WSH"}
FUT = {"KXMLB": "champ", "KXMLBAL": "pennant", "KXMLBNL": "pennant", "KXMLBPLAYOFFS": "playoffs",
       **{f"KXMLB{lg}{dv}": "division" for lg in ("AL", "NL") for dv in ("EAST", "CENT", "WEST")}}
SLATE_HOUR = 12          # slate day d runs 12:00 UTC d → 12:00 UTC d+1 (08:00 ET)

files = sorted(glob.glob("data/markets/*.parquet"))
assert not any("OOS" in f for f in files)
canon = pl.col("code").replace(ALIAS)

# ------------------------------------------------------------------ metadata
mk = (pl.scan_parquet(files)
      .with_columns(pl.col("event_ticker").str.split("-").list.first().alias("series"),
                    pl.col("ticker").str.split("-").list.last().alias("code"))
      .filter(pl.col("series").is_in(["KXMLBGAME", *FUT]))
      .select("ticker", "event_ticker", "series", "code", "title", "result", "close_time")
      .collect().with_columns(canon.alias("team"))
      .unique("ticker"))

fut_mk = mk.filter(pl.col("series").is_in(list(FUT))).with_columns(
    pl.col("series").replace_strict(FUT).alias("kind"))
league = {r["team"]: r["series"][-2:] for r in fut_mk.filter(
    pl.col("series").is_in(["KXMLBAL", "KXMLBNL"])).iter_rows(named=True)}
division = {r["team"]: r["series"][5:] for r in fut_mk.filter(
    pl.col("kind") == "division").iter_rows(named=True)}
teams = sorted(set(fut_mk.filter(pl.col("series") == "KXMLB")["team"]))
assert len(teams) == 30 and set(league) == set(teams) and set(division) == set(teams)

# ------------------------------------------------------------------ games
gm = (mk.filter((pl.col("series") == "KXMLBGAME") & pl.col("team").is_in(teams)
                & pl.col("result").is_in(["yes", "no"])))
two = gm.group_by("event_ticker").agg(pl.len().alias("n")).filter(pl.col("n") == 2)["event_ticker"]
gm = gm.filter(pl.col("event_ticker").is_in(two))
print(f"games: {gm['event_ticker'].n_unique()} with both team markets settled "
      f"({gm['close_time'].min()} → {gm['close_time'].max()})")

UTC = pl.Datetime("us", "UTC")
naive = lambda c: pl.col(c).cast(UTC).dt.replace_time_zone(None)   # noqa: E731  naive UTC, µs
gt = (pl.scan_parquet("data/trades/*.parquet")
      .filter(pl.col("ticker").is_in(gm["ticker"].to_list()))
      .select("ticker", naive("created_time").alias("t"), "yes_price", "count").collect().sort("t"))
gt = gt.join(gm.select("ticker", "event_ticker", "team"), on="ticker")
end = {ev: np.datetime64(e.replace(tzinfo=None), "us") for ev, e in gm.group_by("event_ticker").agg(
    pl.col("close_time").first()).iter_rows()}
H, M10 = np.timedelta64(1, "h"), np.timedelta64(10, "m")


def detect_start(e: np.datetime64, t: np.ndarray) -> tuple[np.datetime64, bool]:
    """First 10-min bin in the last 6 h whose trade count is ≥ 5× the median
    bin 8–5 h before the end (and ≥ 10); else end − 3.25 h."""
    x = t[(t > e - 8 * H) & (t <= e)]
    if len(x) >= 20:
        counts = np.bincount(((x - (e - 8 * H)) // M10).astype(int), minlength=48)[:48]
        hot = np.nonzero((counts >= max(10, 5 * np.median(counts[:18]))) & (np.arange(48) >= 12))[0]
        if len(hot):
            return e - 8 * H + int(hot[0]) * M10, True
    return e - np.timedelta64(195, "m"), False


rows = []
for (ev,), tr in gt.partition_by("event_ticker", as_dict=True).items():
    e = end[ev]
    t = tr["t"].to_numpy()
    st, detected = detect_start(e, t)
    st_py, lo_py = st.item(), (st - H).item()
    pre = tr.filter((pl.col("t") < st_py) & (pl.col("t") >= lo_py))
    if pre.height == 0:
        pre = tr.filter(pl.col("t") < st_py).tail(20)
    two_t = gm.filter(pl.col("event_ticker") == ev).sort("team")
    vw = {}
    for tm in two_t["team"]:
        q = pre.filter(pl.col("team") == tm)
        vw[tm] = float((q["yes_price"] * q["count"]).sum() / q["count"].sum()) if q.height else np.nan
    a, b = two_t["team"].to_list()
    pa = float(np.nanmean([vw[a], 100 - vw[b]])) if np.isfinite([vw[a], vw[b]]).any() else np.nan
    for tm, opp, p in ((a, b, pa), (b, a, 100 - pa)):
        won = two_t.filter(pl.col("team") == tm)["result"][0] == "yes"
        rows.append(dict(event=ev, team=tm, opp=opp, start=st_py, end=e.item(), p_pre=p, won=won,
                         z=(float(won) - p / 100) if np.isfinite(p) else np.nan,
                         start_detected=detected, title=two_t["title"][0]))
games = pl.DataFrame(rows).with_columns(
    (pl.col("end") - pl.duration(hours=SLATE_HOUR)).dt.date().alias("slate"),
    ((pl.col("end") - pl.col("start")).dt.total_minutes() / 60).alias("dur_h"))
games.write_parquet(f"{OUT}/mlb_games.parquet")
g1 = games.filter(pl.col("z").is_finite())
print(f"start detected from the tape for {games['start_detected'].mean():.1%} of team-games")
print(f"team-games: {games.height}, with p_pre: {g1.height}; game duration (start→end) h: "
      f"median {g1['dur_h'].median():.2f}, p5 {g1['dur_h'].quantile(.05):.2f}, p95 {g1['dur_h'].quantile(.95):.2f}")
print(f"p_pre: mean {g1['p_pre'].mean():.1f}, calibration: mean(won) {g1['won'].mean():.3f} vs "
      f"mean(p_pre)/100 {g1['p_pre'].mean() / 100:.3f}; mean z {g1['z'].mean():+.3f}, sd {g1['z'].std():.3f}")

# ------------------------------------------------------------------ futures
ft = (pl.scan_parquet("data/trades/*.parquet")
      .filter(pl.col("ticker").is_in(fut_mk["ticker"].to_list()))
      .select("ticker", naive("created_time").alias("t"), "yes_price", "count").collect()
      .join(fut_mk.select("ticker", "series", "kind", "team", "result"), on="ticker").sort("t"))
ft.write_parquet(f"{OUT}/mlb_futures.parquet")
print(f"futures trades: {ft.height} on {ft['ticker'].n_unique()} contracts; by kind "
      + ", ".join(f"{k} {n}" for k, n in ft.group_by("kind").len().sort("kind").iter_rows()))

tm = pl.DataFrame({"team": teams, "league": [league[t] for t in teams],
                   "division": [division[t] for t in teams]})
res = fut_mk.select("team", "kind", "series", "result")
tm.join(res.pivot(on="kind", index="team", values="result", aggregate_function="first"),
        on="team").write_parquet(f"{OUT}/mlb_teams.parquet")

# ------------------------------------------------------------------ daily panel
days = pl.date_range(games["slate"].min(), games["slate"].max(), "1d", eager=True)
snaps = pl.DataFrame({"slate": days}).with_columns(
    pl.col("slate").cast(pl.Datetime("us")).dt.offset_by(f"{SLATE_HOUR}h").alias("t0"),
    pl.col("slate").cast(pl.Datetime("us")).dt.offset_by(f"{24 + SLATE_HOUR}h").alias("t1"))
grid = snaps.join(pl.DataFrame({"team": teams}), how="cross")
zday = g1.group_by("slate", "team").agg(pl.col("z").sum().alias("z"), pl.len().alias("games"))
panel = grid.join(zday, on=["slate", "team"], how="left").with_columns(
    pl.col("z").fill_null(0.0), pl.col("games").fill_null(0))
# one futures contract per (team, kind): champ, pennant, playoffs, division
for kind in ("champ", "pennant", "playoffs", "division"):
    f = ft.filter(pl.col("kind") == kind).select("team", "t", "yes_price").sort("t")
    for side, col in (("t0", "p0"), ("t1", "p1")):
        panel = panel.sort(side).join_asof(
            f.rename({"t": f"_{side}", "yes_price": f"{kind}_{col}"}), left_on=side,
            right_on=f"_{side}", by="team", strategy="backward",
            tolerance="7d").drop(f"_{side}")
    n = (panel.select("slate", "team", "t0", "t1")
         .join(f, on="team").filter((pl.col("t") > pl.col("t0")) & (pl.col("t") <= pl.col("t1")))
         .group_by("slate", "team").agg(pl.len().alias(f"{kind}_n")))
    panel = panel.join(n, on=["slate", "team"], how="left").with_columns(pl.col(f"{kind}_n").fill_null(0))
panel = panel.sort("slate", "team")
panel.write_parquet(f"{OUT}/mlb_daily.parquet")
print(f"daily panel: {panel['slate'].n_unique()} slate days × {len(teams)} teams; "
      f"team-days with a game {int((panel['games'] > 0).sum())}")
for kind in ("champ", "pennant", "playoffs", "division"):
    ok = panel.filter((pl.col("games") > 0) & (pl.col(f"{kind}_n") > 0))
    print(f"  {kind:9} team-game-days whose futures traded that slate: {ok.height}")
