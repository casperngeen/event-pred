"""NBA in-game events, cleaned: one event per possession, with each game's clock shifted to the market.

ESPN's NBA feed has about 485 rows per game (shots, rebounds, substitutions, timeouts...), and its
wall-clock timestamps can be off by a few seconds per game, so single rows are hard to tie to price
moves. This script:

1. Possessions. Splits each game into possessions. A possession ends at a made shot (or the and-one
   free throw after it), the last free throw of a trip to the line, a defensive rebound, a
   turnover, or the end of a period. Each possession gets the points each team scored in it, its
   ending type, and the change in ESPN's home win probability over it. A trip to the line is
   stamped at the foul that caused it (the moment the market learns of it), not at the free throws.
2. Clock shift per game. For each game, one time shift (-30..+30 s) that best lines up the
   possessions' win-probability changes with the home-win price changes in the 8 s after them
   (price from both winner contracts' trades, a NO contract read as 100 - p, 3-trade rolling median).
   One number per game, so the market cannot bend individual events. Also: how much the shift
   varies between games (a near-constant shift could be fixed in advance, which a live system needs).
3. Checks. Correlation of win-probability change with the 30 s price change, for raw rows vs
   possessions and before vs after the shift; spacing between possessions; and, for possessions
   with no other possession within ISO seconds, the share of the 2-min price change done by each
   second after the event.

Writes out/b1_nba_possessions.parquet (one row per possession, shifted times included).

    venv/bin/python -W ignore analysis/graph_2026_10/b1_nba_events.py > analysis/graph_2026_10/out/b1_nba_events.txt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.loaders import load_pbp, scan_trades       # noqa: E402

SHIFTS = np.arange(-30, 31)          # seconds tried per game
REACT = 8                            # seconds of price change scored after each event
MIN_DWP = 0.01                       # possessions used to fit the shift
ISO = 15                             # seconds with no other possession, either side
OUT = Path(__file__).resolve().parent / "out"
FT = re.compile(r"Free Throw - (?:Flagrant |Clear Path )?(\d) of (\d)")


def possessions(ev: pl.DataFrame) -> pl.DataFrame:
    """One row per possession for one league's play-by-play (sorted by game, seq)."""
    rows = []
    for (gid,), g in ev.partition_by("game_id", as_dict=True, maintain_order=True).items():
        g = g.sort("seq")
        # home team id: the team whose made shots raise the home score
        up = g.filter(pl.col("home_score").diff() > 0)["team_id"].drop_nulls()
        home_id = up.mode()[0] if len(up) else None
        et, desc, team = g["event_type"].to_list(), g["description"].fill_null("").to_list(), g["team_id"].to_list()
        t, clock, scoring = g["t"].to_list(), g["clock"].to_list(), g["is_scoring"].to_list()
        hs, as_, wp, sv = g["home_score"].to_list(), g["away_score"].to_list(), g["home_wp_post"].to_list(), g["score_value"].to_list()
        n = len(et)
        last_foul, trip_foul = None, None
        prev = dict(i=-1, hs=0, as_=0, wp=None, t=None)
        for i in range(n):
            e, d = et[i] or "", desc[i].lower()
            kind, stamp, offense = None, t[i], team[i]
            m = FT.match(e)
            if "Foul" in e and "Turnover" not in e:
                last_foul = i
            if m and int(m.group(1)) == 1:
                trip_foul = last_foul
            if e in ("End Period", "End Game"):
                kind, offense = "period_end", None
            elif m:
                if m.group(1) == m.group(2) and scoring[i]:
                    kind = "ft_trip"
                    if trip_foul is not None and (t[i] - t[trip_foul]).total_seconds() < 120:
                        stamp = t[trip_foul]
            elif scoring[i]:
                # and-one: a 1-of-1 free throw by the same team at the same game clock follows
                and_one = any(FT.match(et[j] or "") and (et[j] or "").endswith("1 of 1") and team[j] == team[i]
                              for j in range(i + 1, min(i + 6, n)) if clock[j] == clock[i])
                if not and_one:
                    kind = "made_3" if sv[i] == 3 else "made_2"
            elif "Defensive Rebound" in e or "defensive team rebound" in d:
                kind = "miss"
                offense = None if team[i] is None else ("away" if team[i] == home_id else "home")
            elif "Turnover" in e or e == "Traveling":
                kind = "turnover"
            if kind is None:
                continue
            if prev["t"] is not None and t[i] == prev["t"] and kind == "turnover":
                continue                                    # offensive foul + turnover rows: one event
            side = offense if offense in ("home", "away", None) else ("home" if offense == home_id else "away")
            if kind == "miss":
                side = offense
            rows.append(dict(game_id=gid, start_seq=prev["i"] + 1, end_seq=i, t=stamp, t_end=t[i], kind=kind,
                             offense=side, home_pts=hs[i] - prev["hs"], away_pts=as_[i] - prev["as_"],
                             wp_pre=prev["wp"], wp_post=wp[i], period=g["period"][i], clock=clock[i],
                             home_score=hs[i], away_score=as_[i]))
            prev = dict(i=i, hs=hs[i], as_=as_[i], wp=wp[i] if wp[i] is not None else prev["wp"], t=t[i])
    p = pl.DataFrame(rows).with_columns((pl.col("wp_post") - pl.col("wp_pre")).alias("dwp"))
    return p.sort("game_id", "t")


def price_paths(d: SportsData, game_ids: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """game_id -> (epoch seconds, home-win price) from both winner contracts' trades."""
    w = (d.contracts.filter(pl.col("league") == "NBA", pl.col("kind") == "winner", pl.col("game_id").is_in(game_ids),
                            pl.col("side").is_in(["home", "away"]))
         .select("ticker", "series_ticker", "side", "game_id"))
    tr = (scan_trades(w["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(w["ticker"].implode()))
          .select("ticker", "created_time", "yes_price").collect().join(w, on="ticker")
          .with_columns(pl.when(pl.col("side") == "home").then(pl.col("yes_price")).otherwise(100 - pl.col("yes_price"))
                        .alias("p"))
          .sort("game_id", "created_time")
          .with_columns(pl.col("p").rolling_median(3, min_samples=1).over("game_id").alias("p_med")))
    return {k: (v["created_time"].dt.epoch("us").to_numpy() / 1e6, v["p_med"].to_numpy())
            for (k,), v in tr.partition_by("game_id", as_dict=True).items()}


def price_at(ts: np.ndarray, pm: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Last trade price at or before each time in x (nan before the first trade)."""
    i = np.searchsorted(ts, x, side="right") - 1
    out = pm[np.clip(i, 0, None)].astype(float)
    out[i < 0] = np.nan
    return out


def fit_shift(ts, pm, t_ev, dwp) -> tuple[int, float, float]:
    """Best shift (s), its score, and the score at shift 0; score = corr(dWP, price change over REACT s)."""
    best, best_s, s0 = 0, -np.inf, np.nan
    for s in SHIFTS:
        a = price_at(ts, pm, t_ev + s - 1)
        b = price_at(ts, pm, t_ev + s + REACT)
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() < 20 or np.std(b[ok] - a[ok]) == 0:
            continue
        c = np.corrcoef(dwp[ok], b[ok] - a[ok])[0, 1]
        if s == 0:
            s0 = c
        if c > best_s:
            best, best_s = int(s), c
    return best, best_s, s0


def move_corr(paths, df: pl.DataFrame, tcol: str, horizon: int = 30) -> tuple[float, int]:
    xs, ys = [], []
    for (gid,), g in df.partition_by("game_id", as_dict=True).items():
        if gid not in paths:
            continue
        ts, pm = paths[gid]
        t0 = g[tcol].dt.epoch("us").to_numpy() / 1e6
        mv = price_at(ts, pm, t0 + horizon) - price_at(ts, pm, t0 - 1)
        ok = ~np.isnan(mv) & ~np.isnan(g["dwp"].to_numpy().astype(float))
        xs.append(g["dwp"].to_numpy()[ok])
        ys.append(mv[ok])
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def main() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    d = SportsData()
    ev = load_pbp("NBA").filter(~pl.col("t_outside_window")).sort("game_id", "seq")
    pos = possessions(ev)
    n_g = pos["game_id"].n_unique()
    print(f"######## 1. possessions: {pos.height:,} in {n_g} games ({pos.height / n_g:.0f} per game; "
          f"{ev.height / n_g:.0f} raw rows per game)")
    print(pos.group_by("kind").agg(pl.len().alias("n"), (pl.len() / n_g).round(1).alias("per_game"),
                                   (pl.col("dwp").abs().median() * 100).round(2).alias("dwp_med_pts"),
                                   (pl.col("dwp").abs().quantile(0.9) * 100).round(2).alias("dwp_p90_pts"))
          .sort("n", descending=True))
    gap = pos.with_columns(pl.col("t").diff().over("game_id").dt.total_milliseconds().alias("gap_ms") / 1000)
    print("seconds between consecutive possessions (quantiles 10/25/50/75/90):",
          [round(gap["gap_ms"].quantile(q), 1) for q in (0.1, 0.25, 0.5, 0.75, 0.9)])

    paths = price_paths(d, pos["game_id"].unique().to_list())
    shifts = []
    for (gid,), g in pos.filter(pl.col("dwp").abs() >= MIN_DWP).partition_by("game_id", as_dict=True).items():
        if gid not in paths:
            continue
        s, c, c0 = fit_shift(*paths[gid], g["t"].dt.epoch("us").to_numpy() / 1e6, g["dwp"].to_numpy())
        shifts.append(dict(game_id=gid, shift_s=s, corr_best=c, corr_0=c0, n=g.height))
    sh = pl.DataFrame(shifts)
    q = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    print(f"\n######## 2. clock shift per game ({sh.height} games; positive = the market reacts later than "
          f"ESPN's timestamp)")
    print("shift quantiles (5/10/25/50/75/90/95):", [sh["shift_s"].quantile(x) for x in q])
    print(f"share of games with shift within +-3 s of the median: "
          f"{((sh['shift_s'] - sh['shift_s'].median()).abs() <= 3).mean():.3f}; at the search edge (|shift| = 30): "
          f"{(sh['shift_s'].abs() == 30).mean():.3f}")
    print(f"8-s fit correlation, median over games: at shift 0 {sh['corr_0'].median():.3f}, at each game's best "
          f"{sh['corr_best'].median():.3f}, at the league median shift (below)")
    by_season = (sh.join(pos.group_by("game_id").agg(pl.col("t").min().dt.strftime("%Y-%m").alias("month")),
                         on="game_id").group_by("month").agg(pl.len().alias("games"),
                                                             pl.col("shift_s").median().alias("shift_med"),
                                                             pl.col("shift_s").quantile(0.25).alias("q25"),
                                                             pl.col("shift_s").quantile(0.75).alias("q75")).sort("month"))
    print("by month:")
    print(by_season)
    med = int(sh["shift_s"].median())
    pos = (pos.join(sh.select("game_id", "shift_s"), on="game_id", how="left")
           .with_columns((pl.col("t") + pl.duration(seconds=pl.col("shift_s").fill_null(med))).alias("t_game_shift"),
                         (pl.col("t") + pl.duration(seconds=med)).alias("t_league_shift")))

    print("\n######## 3. checks")
    raw = (ev.with_columns(pl.col("home_wp_post").forward_fill().shift(1).over("game_id", order_by="seq").alias("wp_pre"))
           .with_columns((pl.col("home_wp_post") - pl.col("wp_pre")).alias("dwp"))
           .filter(pl.col("dwp").abs() >= MIN_DWP))
    big = pos.filter(pl.col("dwp").abs() >= MIN_DWP)
    print(f"3a. corr(win-prob change, home-win price change from 1 s before to 30 s after), |dWP| >= {MIN_DWP}:")
    for name, df, col in [("raw rows, ESPN time", raw, "t"), ("possessions, ESPN time", big, "t"),
                          (f"possessions, league shift {med:+d} s", big, "t_league_shift"),
                          ("possessions, per-game shift", big, "t_game_shift")]:
        c, n = move_corr(paths, df, col)
        print(f"    {name:40s} corr {c:.3f}  (n {n:,})")
    iso = pos.with_columns(
        ((pl.col("t_game_shift") - pl.col("t_game_shift").shift(1).over("game_id")).dt.total_seconds() > ISO)
        .fill_null(True).alias("gap_before"),
        ((pl.col("t_game_shift").shift(-1).over("game_id") - pl.col("t_game_shift")).dt.total_seconds() > ISO)
        .fill_null(True).alias("gap_after"))
    iso = iso.filter(pl.col("gap_before"), pl.col("gap_after"))
    print(f"3b. isolated possessions (no other within {ISO} s either side): {iso.height:,} of {pos.height:,} "
          f"({iso.height / pos.height:.3f}); with |dWP| >= 0.03: {iso.filter(pl.col('dwp').abs() >= 0.03).height:,}")
    offs = [-10, -5, -2, 0, 2, 5, 10, 20, 30, 60]
    cur = {o: [] for o in offs}
    for (gid,), g in iso.filter(pl.col("dwp").abs() >= 0.03).partition_by("game_id", as_dict=True).items():
        if gid not in paths:
            continue
        ts, pm = paths[gid]
        for t0, dw in zip(g["t_game_shift"].dt.epoch("us").to_numpy() / 1e6, g["dwp"].to_numpy()):
            p0, p1 = price_at(ts, pm, np.array([t0 - 15, t0 + 120]))
            if np.isnan(p0) or np.isnan(p1) or abs(p1 - p0) < 2 or (p1 - p0) * dw <= 0:
                continue
            for o in offs:
                cur[o].append((price_at(ts, pm, np.array([t0 + o]))[0] - p0) / (p1 - p0))
    print(f"3c. isolated possessions with |dWP| >= 0.03 that moved >= 2c the WP's way ({len(cur[0])}): share of the "
          "price change (15 s before -> 120 s after) done by each second, median and share past half")
    print(pl.DataFrame({"seconds_after": offs, "median_share": [round(float(np.nanmedian(cur[o])), 2) for o in offs],
                        "past_half": [round(float(np.nanmean(np.array(cur[o]) >= 0.5)), 3) for o in offs]}))
    pos.write_parquet(OUT / "b1_nba_possessions.parquet")
    print(f"\nwrote {OUT / 'b1_nba_possessions.parquet'} ({pos.height:,} rows)")


if __name__ == "__main__":
    main()
