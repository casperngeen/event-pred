"""B1 preliminaries for the in-game graph (track B), one league: liveness, timing, events.

1. Liveness. Every game-level leg (winner, spread / total rungs, team totals, halves) on a 1-min
   grid from the first play to the feed's end. Each leg-minute is
       live      two-sided, spread <= LIVE_SPREAD, mid in [LIVE_LO, LIVE_HI], quote age <= LIVE_AGE
       input     a quote exists but fails one of those (one-sided, wide, extreme or stale)
       none      no quote yet
   Live = node and target; input = node value only (one-sided valued at ask/2 or (bid+100)/2, as in
   between-game-design); none = masked. Also flips (live <-> not) per leg-hour, and the live share
   of every leg around big plays against its share elsewhere.
2. Timing. Plays with |dWP| >= BIG_WP (the league feed's home win prob, after minus before). The
   home-win price path from both winner legs' trades (a NO leg read as 100 - p, 3-trade rolling
   median against stray prints): pre = mid at the play's timestamp t, post = mid at t + POST.
   Half-time = first time the path covers half of post - pre. Plays are timed at ``t_event`` (NFL:
   the play's end, not the snap). Also the share of the move already in place when the loader
   stamps the play known (``known_at``: NFL the next snap).
3. Events. Play groups (league-specific maps below): count, WP coverage, |dWP| quantiles, and the
   2-min home-win mid move. As-of check on the feed's WP: Brier of the pre-play WP against the
   market mid at the same moment, by period; a feed WP far better than the market would point to
   a WP fitted with hindsight.

    venv/bin/python -W ignore analysis/graph_2026_10/b1_prelim.py NFL > analysis/graph_2026_10/out/b1_prelim_NFL.txt
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.loaders import load_pbp, scan_trades       # noqa: E402
from stg.sports.prices import quotes_at, scan_quotes       # noqa: E402

LEAGUE = sys.argv[1] if len(sys.argv) > 1 else "NFL"
N_GAMES, SEED = 400, 0                 # liveness grid on a sample of games (all NFL games fit)
KINDS = ["winner", "spread", "total", "team_total", "winner_1h", "spread_1h", "total_1h",
         "winner_2h", "spread_2h", "total_2h"]
LIVE_SPREAD, LIVE_LO, LIVE_HI = 10.0, 3.0, 97.0
LIVE_AGE = dt.timedelta(minutes=5)
BIG_WP = 0.08
POST = dt.timedelta(seconds=120)
AROUND = {"-60s": -60, "+15s": 15, "+30s": 30, "+60s": 60, "+120s": 120}

# play groups: (group, predicate on event_type / description)
NFL_GROUPS = [
    ("review", pl.col("description").str.contains(r"(?i)challenged|reversed|replay")),
    ("injury", pl.col("description").str.contains(r"(?i)injured")),
    ("touchdown", pl.col("event_type").str.contains("Touchdown")),
    ("turnover", pl.col("event_type").is_in(["Pass Interception Return", "Fumble Recovery (Opponent)",
                                             "Sack Opp Fumble Recovery", "Muffed Punt Recovery (Opponent)",
                                             "Blocked Punt", "Blocked Field Goal"])),
    ("field_goal", pl.col("event_type").is_in(["Field Goal Good", "Field Goal Missed", "Safety"])),
    ("penalty", pl.col("event_type") == "Penalty"),
    ("stoppage", pl.col("event_type").is_in(["Timeout", "Official Timeout", "Two-minute warning", "End Period",
                                             "End of Half", "End of Regulation", "End of Game"])),
    ("special", pl.col("event_type").is_in(["Kickoff", "Punt"])),
    ("scrimmage", pl.lit(True)),
]
NBA_GROUPS = [
    ("injury", pl.col("description").str.contains(r"(?i)injur")),
    ("review", pl.col("description").str.contains(r"(?i)challenge|review|overturn")),
    ("made_3", pl.col("is_scoring") & (pl.col("score_value").cast(pl.Int64, strict=False) == 3)),
    ("made_2", pl.col("is_scoring") & (pl.col("score_value").cast(pl.Int64, strict=False) == 2)),
    ("free_throw", pl.col("event_type").str.contains("Free Throw")),
    ("turnover", pl.col("event_type").str.contains("(?i)turnover|steal|bad pass|lost ball|traveling|offensive foul")),
    ("foul", pl.col("event_type").str.contains("(?i)foul")),
    ("stoppage", pl.col("event_type").str.contains("(?i)timeout|end period|end game|substitution|jumpball")),
    ("miss_rebound", pl.lit(True)),
]
GROUPS = {"NFL": NFL_GROUPS, "NBA": NBA_GROUPS}
OUT = Path(__file__).resolve().parent / "out"


def legs_and_games(d: SportsData) -> tuple[pl.DataFrame, pl.DataFrame]:
    res = d.results.filter(pl.col("league") == LEAGUE).select("game_id", "first_play", "feed_end", "home_score",
                                                               "away_score")
    legs = (d.contracts.filter(pl.col("league") == LEAGUE, pl.col("scope") == "game", pl.col("kind").is_in(KINDS),
                               pl.col("game_id").is_not_null())
            .select("ticker", "series_ticker", "kind", "side", "strike", "game_id", "home", "away"))
    games = res.filter(pl.col("game_id").is_in(legs["game_id"].unique().implode()))
    return legs.filter(pl.col("game_id").is_in(games["game_id"].implode())), games


def classify(q: pl.DataFrame) -> pl.DataFrame:
    has = pl.col("bid").is_not_null() | pl.col("ask").is_not_null()
    age = pl.col("t") - pl.min_horizontal("bid_at", "ask_at")
    mid = (pl.col("bid") + pl.col("ask")) / 2
    live = ((pl.col("bid") > 0) & (pl.col("ask") < 100) & (pl.col("bid") <= pl.col("ask"))
            & (pl.col("ask") - pl.col("bid") <= LIVE_SPREAD) & mid.is_between(LIVE_LO, LIVE_HI) & (age <= LIVE_AGE))
    return q.with_columns(pl.when(live.fill_null(False)).then(pl.lit("live")).when(has).then(pl.lit("input"))
                          .otherwise(pl.lit("none")).alias("state"))


def section_liveness(legs: pl.DataFrame, games: pl.DataFrame, big: pl.DataFrame) -> None:
    g = games.sample(n=min(N_GAMES, games.height), seed=SEED)
    lg = legs.join(g, on="game_id")
    grid = (lg.with_columns(pl.datetime_ranges(pl.col("first_play").dt.truncate("1m") + pl.duration(minutes=1),
                                               pl.col("feed_end").dt.truncate("1m"), "1m").alias("t"))
            .explode("t").select("ticker", "kind", "game_id", "t"))
    series = lg["series_ticker"].unique().to_list()
    q = classify(quotes_at(scan_quotes(series), grid, max_spread=100.0))
    print(f"\n######## 1. liveness: {g.height} games, {lg.height} legs, {q.height:,} leg-minutes in play")
    print("1a. state share by kind (live = node + target; input = node value only; none = masked)")
    print(q.group_by("kind").agg(pl.col("ticker").n_unique().alias("legs"), pl.col("game_id").n_unique().alias("games"),
                                 pl.len().alias("leg_min"),
                                 *[(pl.col("state") == s).mean().round(3).alias(s) for s in ("live", "input", "none")])
          .sort("leg_min", descending=True))
    per_min = (q.filter(pl.col("state") == "live").group_by("game_id", "t", "kind").len()
               .group_by("kind").agg(pl.col("len").median().alias("live_legs_med"),
                                     pl.col("len").quantile(0.1).alias("p10"), pl.col("len").quantile(0.9).alias("p90")))
    print("1b. live legs per game-minute (minutes with at least one live leg of the kind)")
    print(per_min.sort("kind"))
    fl = (q.sort("ticker", "t").with_columns((pl.col("state") == "live").alias("lv"))
          .with_columns((pl.col("lv") != pl.col("lv").shift(1).over("ticker")).fill_null(False).alias("flip"))
          .group_by("kind").agg((pl.col("flip").sum() / (pl.len() / 60)).round(2).alias("flips_per_leg_hour"),
                                pl.col("lv").mean().round(3).alias("live")))
    print("1c. flips between live and not-live, per leg-hour")
    print(fl.sort("kind"))
    # around big plays: the state of every leg of that game at fixed offsets
    bp = big.join(g.select("game_id"), on="game_id").select("game_id", pl.col("t").alias("t_play"))
    at = pl.concat([bp.with_columns((pl.col("t_play") + pl.duration(seconds=s)).alias("t"), pl.lit(k).alias("offset"))
                    for k, s in AROUND.items()])
    at = at.join(lg.select("ticker", "kind", "game_id"), on="game_id").select("ticker", "kind", "game_id", "t", "offset")
    qa = classify(quotes_at(scan_quotes(series), at, max_spread=100.0))
    base = q.group_by("kind").agg((pl.col("state") == "live").mean().round(3).alias("all_min"))
    tab = (qa.group_by("kind", "offset").agg((pl.col("state") == "live").mean().round(3).alias("live"))
           .pivot(on="offset", index="kind", values="live").join(base, on="kind"))
    print(f"1d. live share around {bp.height} big plays (|dWP| >= {BIG_WP}) vs all in-play minutes")
    print(tab.select("kind", *AROUND.keys(), "all_min").sort("kind"))
    sp = (qa.filter(pl.col("kind") == "winner", pl.col("bid") > 0, pl.col("ask") < 100)
          .group_by("offset").agg(pl.col("spread").median().alias("spread_med"), pl.col("spread").mean().round(2)
                                  .alias("spread_mean")))
    print("1e. winner-leg spread around big plays (two-sided quotes only); all-minute median "
          f"{q.filter(pl.col('kind') == 'winner', pl.col('bid') > 0, pl.col('ask') < 100)['spread'].median()}")
    print(sp.sort("offset"))


def home_price_trades(legs: pl.DataFrame) -> pl.DataFrame:
    w = legs.filter(pl.col("kind") == "winner", pl.col("side").is_in(["home", "away"]))
    tr = (scan_trades(w["series_ticker"].unique().to_list())
          .filter(pl.col("ticker").is_in(w["ticker"].implode())).select("ticker", "created_time", "yes_price")
          .collect().join(w.select("ticker", "side", "game_id"), on="ticker")
          .with_columns(pl.when(pl.col("side") == "home").then(pl.col("yes_price"))
                        .otherwise(100 - pl.col("yes_price")).alias("p_home"))
          .sort("game_id", "created_time")
          .with_columns(pl.col("p_home").rolling_median(3, min_samples=1).over("game_id").alias("p_med")))
    return tr


def home_mid_at(legs: pl.DataFrame, at: pl.DataFrame) -> pl.DataFrame:
    """at: (game_id, t, ...) -> home-win mid at t, from the home leg (or 100 - the away leg)."""
    w = legs.filter(pl.col("kind") == "winner", pl.col("side").is_in(["home", "away"]))
    a = at.join(w.select("game_id", "ticker", "side"), on="game_id")
    q = quotes_at(scan_quotes(w["series_ticker"].unique().to_list()), a.select("ticker", "t").unique(),
                  max_spread=LIVE_SPREAD)
    a = a.join(q.select("ticker", "t", "mid").unique(["ticker", "t"]), on=["ticker", "t"], how="left").with_columns(
        pl.when(pl.col("side") == "home").then(pl.col("mid")).otherwise(100 - pl.col("mid")).alias("m"))
    keys = [c for c in at.columns]
    return a.group_by(keys).agg(pl.col("m").mean().alias("home_mid"))


def plays(legs: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    # every play is timed at t_event (NFL: the play's end; see loaders), its snap kept as t_snap
    ev = (load_pbp(LEAGUE).filter(~pl.col("t_outside_window"), pl.col("game_id").is_in(games["game_id"].implode()))
          .with_columns(pl.col("t").alias("t_snap"), pl.col("t_event").alias("t")).sort("game_id", "seq"))
    grp = pl.lit(None, pl.String)
    for name, pred in reversed(GROUPS[LEAGUE]):
        grp = pl.when(pred.fill_null(False)).then(pl.lit(name)).otherwise(grp)
    ev = ev.with_columns(
        grp.alias("group"),
        pl.col("home_wp_post").forward_fill().shift(1).over("game_id", order_by="seq").alias("wp_pre"))
    return ev.with_columns((pl.col("home_wp_post") - pl.col("wp_pre")).alias("dwp"))


def section_timing(legs: pl.DataFrame, ev: pl.DataFrame) -> None:
    big = ev.filter(pl.col("dwp").abs() >= BIG_WP).select("game_id", "seq", "t", "known_at", "dwp", "group")
    pts = pl.concat([big.select("game_id", "seq", pl.col("t").alias("t"), pl.lit("pre").alias("k")),
                     big.select("game_id", "seq", (pl.col("t") + POST).alias("t"), pl.lit("post").alias("k")),
                     big.select("game_id", "seq", pl.col("known_at").alias("t"), pl.lit("known").alias("k"))])
    m = home_mid_at(legs, pts).pivot(on="k", index=["game_id", "seq"], values="home_mid", aggregate_function="mean")
    b = big.join(m, on=["game_id", "seq"]).with_columns((pl.col("post") - pl.col("pre")).alias("move"))
    ok = b.filter(pl.col("move").abs() >= 3, (pl.col("move") * pl.col("dwp")) > 0)
    print(f"\n######## 2. timing: {big.height} plays with |dWP| >= {BIG_WP}; {b['move'].is_not_null().sum()} with "
          f"mids at t and t+{POST.seconds}s; {ok.height} moved >= 3c the WP's way")
    x, y = b.drop_nulls(["move"])["dwp"].to_numpy() * 100, b.drop_nulls(["move"])["move"].to_numpy()
    print(f"2a. market move vs feed dWP (cents per WP point): slope {np.polyfit(x, y, 1)[0]:.2f}, "
          f"corr {np.corrcoef(x, y)[0, 1]:.2f}, sign agreement {(np.sign(x) == np.sign(y)).mean():.2f}")
    tr = home_price_trades(legs)
    tg = {k: (v["created_time"].dt.epoch("us").to_numpy(), v["p_med"].to_numpy())
          for (k,), v in tr.partition_by("game_id", as_dict=True).items()}
    half = []
    for r in ok.iter_rows(named=True):
        ts, pm = tg.get(r["game_id"], (None, None))
        if ts is None:
            half.append(None)
            continue
        t0 = int(r["t"].timestamp() * 1e6)
        sel = (ts >= t0 - 60_000_000) & (ts <= t0 + POST.seconds * 1_000_000)
        sgn = np.sign(r["move"])
        hit = np.nonzero(((pm[sel] - r["pre"]) * sgn) >= abs(r["move"]) / 2)[0]
        half.append((ts[sel][hit[0]] - t0) / 1e6 if len(hit) else None)
    # trades-only absorption curve: share of (post - p0) done at each offset, p0 = last trade 60 s
    # before the timestamp, so a stale 1-min quote at t cannot fake an early move
    offs = [-30, -10, -5, 0, 5, 10, 20, 30, 60]
    curve = {o: [] for o in offs}
    for r in ok.iter_rows(named=True):
        ts, pm = tg.get(r["game_id"], (None, None))
        if ts is None:
            continue
        t0 = int(r["t"].timestamp() * 1e6)
        i0 = np.searchsorted(ts, t0 - 60_000_000, side="right") - 1
        if i0 < 0 or abs(r["post"] - pm[i0]) < 3:
            continue
        for o in offs:
            i = np.searchsorted(ts, t0 + o * 1_000_000, side="right") - 1
            curve[o].append((pm[i] - pm[i0]) / (r["post"] - pm[i0]))
    print(f"2b'. trades only, p0 = last trade at t-60s ({len(curve[0])} plays): share of the move to t+{POST.seconds}s "
          "done by each offset (median, and share of plays past half)")
    print(pl.DataFrame({"offset_s": offs, "median": [round(float(np.median(curve[o])), 2) for o in offs],
                        "past_half": [round(float(np.mean(np.array(curve[o]) >= 0.5)), 3) for o in offs]}))
    ok = ok.with_columns(pl.Series("half_s", half, dtype=pl.Float64),
                         ((pl.col("known_at") - pl.col("t")).dt.total_milliseconds() / 1000).alias("known_lag_s"),
                         ((pl.col("known") - pl.col("pre")) / pl.col("move")).alias("done_at_known"))
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    print("2b. seconds from the play timestamp to half the 2-min move (trades; pre = quote mid at t, which can be a stale 1-min candle, so before-t is overstated: use 2b'); "
          f"share before the timestamp {(ok['half_s'] < 0).mean():.3f}, never within 2 min {ok['half_s'].is_null().mean():.3f}")
    print(pl.DataFrame({"q": qs, "half_s": [ok["half_s"].quantile(q) for q in qs],
                        "known_lag_s": [ok["known_lag_s"].quantile(q) for q in qs],
                        "done_at_known": [round(ok["done_at_known"].quantile(q), 2) for q in qs]}))
    print("2c. by group (median half-time, share of the move done when the loader stamps it known)")
    print(ok.group_by("group").agg(pl.len().alias("n"), pl.col("half_s").median().alias("half_med"),
                                   (pl.col("half_s") < 0).mean().round(3).alias("before_t"),
                                   pl.col("known_lag_s").median().alias("known_lag_med"),
                                   pl.col("done_at_known").median().round(2).alias("done_at_known_med"))
          .sort("n", descending=True))


def section_events(legs: pl.DataFrame, ev: pl.DataFrame, games: pl.DataFrame) -> None:
    pts = pl.concat([ev.select("game_id", "seq", pl.col("t"), pl.lit("pre").alias("k")),
                     ev.select("game_id", "seq", (pl.col("t") + POST).alias("t"), pl.lit("post").alias("k"))])
    m = home_mid_at(legs, pts).pivot(on="k", index=["game_id", "seq"], values="home_mid", aggregate_function="mean")
    e = ev.join(m, on=["game_id", "seq"], how="left").with_columns((pl.col("post") - pl.col("pre")).alias("move"))
    n_g = e["game_id"].n_unique()
    print(f"\n######## 3. events: {e.height:,} plays in {n_g} games")
    print("3a. by group: WP coverage, |dWP| (WP points), share big, |2-min home-mid move| (cents)")
    print(e.group_by("group").agg(
        pl.len().alias("n"), (pl.len() / n_g).round(1).alias("per_game"),
        pl.col("home_wp_post").is_not_null().mean().round(3).alias("wp_cov"),
        (pl.col("dwp").abs().median() * 100).round(2).alias("dwp_med"),
        (pl.col("dwp").abs().quantile(0.9) * 100).round(2).alias("dwp_p90"),
        (pl.col("dwp").abs() >= 0.05).mean().round(3).alias("ge_5"),
        (pl.col("dwp").abs() >= 0.10).mean().round(3).alias("ge_10"),
        pl.col("move").abs().median().alias("move_med"),
        pl.col("move").abs().quantile(0.9).alias("move_p90"),
        pl.corr("dwp", "move").round(2).alias("corr_dwp_move")).sort("n", descending=True))
    # as-of check: the pre-play feed WP vs the market mid at the same moment, scored on the result
    y = games.select("game_id", (pl.col("home_score") > pl.col("away_score")).cast(pl.Float64).alias("y"),
                     (pl.col("home_score") == pl.col("away_score")).alias("tie"))
    s = (e.join(y, on="game_id").filter(~pl.col("tie"), pl.col("wp_pre").is_not_null(), pl.col("pre").is_not_null())
         .with_columns((pl.col("pre") / 100).alias("mkt")))
    print(f"3b. as-of check: Brier on the result, {s.height:,} plays with both (ties dropped); lower is better")
    print(s.group_by("period").agg(pl.len().alias("n"), ((pl.col("wp_pre") - pl.col("y")) ** 2).mean().round(4).alias("feed_wp"),
                                   ((pl.col("mkt") - pl.col("y")) ** 2).mean().round(4).alias("market"),
                                   (pl.col("wp_pre") - pl.col("mkt")).abs().mean().round(3).alias("mean_abs_gap"))
          .sort("period"))
    # does the feed WP lead the market? regress the next 2-min market move on (feed WP - market)
    s2 = s.drop_nulls(["move"])
    gap, mv = (s2["wp_pre"] - s2["mkt"]).to_numpy() * 100, s2["move"].to_numpy()
    print(f"3c. next 2-min market move on (feed WP - market) before the play: slope {np.polyfit(gap, mv, 1)[0]:.3f} "
          f"(0 = feed adds nothing; 1 = market moves fully to the feed), corr {np.corrcoef(gap, mv)[0, 1]:.3f}")


def main() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    d = SportsData()
    legs, games = legs_and_games(d)
    print(f"{LEAGUE}: {games.height} games with play-by-play and game-level legs; "
          f"legs by kind: {dict(legs.group_by('kind').len().sort('len', descending=True).iter_rows())}")
    ev = plays(legs, games)
    section_liveness(legs, games, ev.filter(pl.col("dwp").abs() >= BIG_WP))
    section_timing(legs, ev)
    section_events(legs, ev, games)


if __name__ == "__main__":
    main()
