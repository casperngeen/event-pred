"""Step 4 of the in-game preliminaries: how each contract's price responds to a play.

For every event (a play, possession or plate appearance that moved the home win probability by at
least MIN_DWP or scored), the mid price of each contract of that game at fixed offsets from the
event (time 0 = when the outcome happened: NFL the play's end, NBA the possession's end, MLB the
pitch's start, since its recorded end comes after the play finishes), measured from the mid REF seconds before. Mids come from the quote layer (1-min
candles plus trades; usable = two-sided, gap <= 10c, sides <= 10 min old; a one-sided quote at an
extreme, i.e. a contract all but settled, is valued halfway to 0 or 100 so it is not dropped).

Every move is signed so that positive = the direction the event should push the contract:
    winner contract of team X       + if the event raised X's win probability
    spread contract "X wins by > s" + if the event raised X's margin (else by X's win probability)
    total contract "total > s"      + (scoring events only: points were added)
    team futures (champion, ...)    + if the event raised the team's win probability

Win probability: NFL nflverse (ESPN when nflverse has none), NBA ESPN, MLB statsapi (per plate
appearance). Distance of a spread / total contract = its threshold minus the current margin /
total after the event (negative: already past the threshold).

Tables, per league:
    A  winner contracts by event type: mean signed move (cents) at each offset; "left after 60 s"
       = mean move from +60 s to +600 s, with a t-statistic clustered by game
    B  spread and total contracts by distance: same, plus the share with a usable mid
    C  team futures by kind: moves to +60 min, and the part left after +5 min
   Also, for big events: how soon each contract trades after the event and how much of its move to
   +120 s that first trade already shows (contracts that trade rarely get their mid from the 1-min
   candle, which can make them look slower than they are).

    venv/bin/python -W ignore analysis/graph_2026_10/b1_response.py NFL > analysis/graph_2026_10/out/b1_response_NFL.txt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.loaders import load_games, load_pbp        # noqa: E402
from stg.sports.prices import quotes_at, scan_quotes       # noqa: E402

LEAGUE = sys.argv[1] if len(sys.argv) > 1 else "NFL"
from b1_prelim import NFL_GROUPS                           # noqa: E402
from b2_futures_ingame import candles_1m, mids_at          # noqa: E402

N_GAMES = {"NFL": 10_000, "NBA": 300, "MLB": 500}
SEED = 0
MIN_DWP = 0.03
REF = 30
MIN_N = 30
OFFSETS = [0, 10, 30, 60, 120, 300, 600]
FUT_OFFSETS = [60, 300, 900, 3600]
GAP = 10.0
BINS = {"NFL": [-14, -7, -3, 0, 3, 7, 14], "NBA": [-20, -10, -5, 0, 5, 10, 20], "MLB": [-4, -2, -1, 0, 1, 2, 4]}
MLB_GROUPS = [("home_run", "home_run"), ("extra_base_hit", "double|triple"), ("single", "single"),
              ("walk_hbp", "walk|hit_by_pitch|intent"), ("strikeout", "strikeout"),
              ("double_play", "double_play|triple_play"), ("error", "error"), ("out_in_play", "out|sac|fielders_choice")]
# The first matching group wins, so the unanchored "double|triple" codes every double / triple play
# (grounded_into_double_play, ...) as an extra-base hit and double_play is never set. The frozen models were
# trained that way. MLB_GROUPS_FIX=1 matches a double, triple or strikeout exactly, so every double / triple play
# (strikeout_double_play included) is a double_play (a table built with it goes to its own folder).
MLB_GROUPS_FIX = os.environ.get("MLB_GROUPS_FIX") == "1"
if MLB_GROUPS_FIX:
    MLB_GROUPS[1] = ("extra_base_hit", "^(double|triple)$")
    MLB_GROUPS[4] = ("strikeout", "^strikeout$")
OUT = Path(__file__).resolve().parent / "out"


def nfl_events() -> pl.DataFrame:
    grp = pl.lit(None, pl.String)
    for name, pred in reversed(NFL_GROUPS):
        grp = pl.when(pred.fill_null(False)).then(pl.lit(name)).otherwise(grp)
    e = (load_pbp("NFL").filter(~pl.col("t_outside_window"), pl.col("play_type") != "stoppage").sort("game_id", "seq")
         .with_columns(pl.coalesce("nflverse_home_wp_post", "home_wp_post").alias("wp"), grp.alias("group")))
    return e.select("game_id", "seq", pl.col("t_event").alias("t"), "wp", "group", "home_score", "away_score")


def nba_events() -> pl.DataFrame:
    p = pl.read_parquet(OUT / "b1_nba_possessions.parquet").sort("game_id", "t")
    return p.select("game_id", pl.col("end_seq").alias("seq"), "t", pl.col("wp_post").alias("wp"),
                    pl.col("kind").alias("group"), "home_score", "away_score")


def mlb_events() -> pl.DataFrame:
    grp = pl.lit("other")
    for name, pat in reversed(MLB_GROUPS):
        grp = pl.when(pl.col("pa_event").str.contains(pat)).then(pl.lit(name)).otherwise(grp)
    e = load_pbp("MLB").filter(~pl.col("t_outside_window"), pl.col("pa_last_event").fill_null(False)).sort("game_id", "seq")
    return e.select("game_id", "seq", pl.col("t_event").alias("t"), pl.col("home_wp_post").alias("wp"),
                    grp.alias("group"), "home_score", "away_score")


def events() -> pl.DataFrame:
    e = {"NFL": nfl_events, "NBA": nba_events, "MLB": mlb_events}[LEAGUE]()
    e = e.with_columns(pl.col("wp").forward_fill().over("game_id"))
    prev = lambda c: pl.col(c).shift(1).over("game_id")   # noqa: E731
    e = e.with_columns((pl.col("wp") - prev("wp")).alias("dwp"),
                       ((pl.col("home_score") - pl.col("away_score")) - (prev("home_score") - prev("away_score")))
                       .fill_null(0).alias("dmargin"),
                       ((pl.col("home_score") + pl.col("away_score")) - (prev("home_score") + prev("away_score")))
                       .fill_null(0).alias("dpoints"))
    return e.filter((pl.col("dwp").abs() >= MIN_DWP) | (pl.col("dpoints") > 0)).drop_nulls("dwp")


def value(q: pl.DataFrame, max_age_s: float = 600.0) -> pl.Expr:
    """A contract's price: the mid of a two-sided quote (gap <= GAP); a one-sided quote at an extreme
    (contract all but settled) at (bid + 100) / 2 or ask / 2, as in between-game-design, so that
    contracts crossing their threshold are not dropped; sides <= 10 min old."""
    fresh = pl.col("quote_age") <= pl.duration(seconds=max_age_s)
    bid, ask = pl.col("bid").fill_null(0), pl.col("ask").fill_null(100)
    two = (bid > 0) & (ask < 100) & (ask >= bid) & (ask - bid <= GAP)
    return (pl.when(fresh & two).then((bid + ask) / 2)
            .when(fresh & (bid >= 90) & (ask >= 99)).then((bid + 100) / 2)
            .when(fresh & (ask <= 10) & (bid <= 1)).then(ask / 2))


def clustered_t(x: np.ndarray, g: np.ndarray) -> float:
    if len(x) < 10:
        return float("nan")
    d = pl.DataFrame({"x": x - x.mean(), "g": g}).group_by("g").agg(pl.col("x").sum())
    se = float(np.sqrt((d["x"] ** 2).sum())) / len(x)
    return float(x.mean() / se) if se > 0 else float("nan")


def summarise(df: pl.DataFrame, by: str, order: list[str] | None = None) -> pl.DataFrame:
    """Mean signed move per offset over rows with a usable mid at both the reference and that
    offset (null if fewer than MIN_N); the part left after 60 s on rows usable at 60 s and 600 s."""
    rows = []
    for (key,), g in df.partition_by(by, as_dict=True).items():
        r = {by: key, "n": g.height}
        for o in OFFSETS:
            v = g[f"m{o}"].drop_nulls()
            r[f"{o}s"] = round(float(v.mean()), 2) if len(v) >= MIN_N else None
        both = g.drop_nulls(["m60", "m600"])
        left = (both["m600"] - both["m60"]).to_numpy()
        r["usable_60_600"] = round(both.height / g.height, 3)
        r["left_after_60s"] = round(float(left.mean()), 2) if len(left) >= MIN_N else None
        r["t_left"] = round(clustered_t(left, both["game_id"].to_numpy()), 1) if len(left) >= MIN_N else None
        rows.append(r)
    out = pl.DataFrame(rows)
    if order:
        return out.with_columns(pl.col(by).replace_strict(order, list(range(len(order))), default=99).alias("_o")
                                ).sort("_o").drop("_o")
    return out.sort("n", descending=True)


def first_trades(x: pl.DataFrame, con: pl.DataFrame) -> pl.DataFrame:
    """For big events: delay to each contract's first trade after the event (within 120 s), and the
    share of the move to +120 s already in that trade's price."""
    from stg.sports.loaders import scan_trades
    b = x.filter(pl.col("dwp").abs() >= 0.10, pl.col("m120").is_not_null(), pl.col("m120").abs() >= 2,
                 (pl.col("kind") == "winner") | pl.col("dist").is_between(-7 if LEAGUE == "NFL" else -10, 7 if LEAGUE == "NFL" else 10))
    tr = (scan_trades(con["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(b["ticker"].unique().implode()))
          .select("ticker", pl.col("created_time").alias("tt"), pl.col("yes_price").cast(pl.Float64).alias("p")).collect().sort("tt"))
    b = b.select("ticker", "kind", pl.col("t").cast(pl.Datetime("us", "UTC")), "sgn", "m120", str(-REF))
    tr = tr.with_columns(pl.col("tt").cast(pl.Datetime("us", "UTC")))
    j = b.sort("t").join_asof(tr, left_on="t", right_on="tt", by="ticker", strategy="forward",
                              tolerance="120s", check_sortedness=False)
    j = j.with_columns(((pl.col("tt") - pl.col("t")).dt.total_milliseconds() / 1000).alias("delay_s"),
                       ((pl.col("sgn") * (pl.col("p") - pl.col(str(-REF)))) / pl.col("m120")).alias("share_at_first"))
    return j.group_by("kind").agg(pl.len().alias("n"), pl.col("delay_s").is_not_null().mean().round(3).alias("traded_within_120s"),
                                  pl.col("delay_s").median().round(1).alias("delay_med_s"),
                                  pl.col("delay_s").quantile(0.75).round(1).alias("delay_p75_s"),
                                  pl.col("share_at_first").median().round(2).alias("share_of_120s_move_at_first_trade"))


def event_contract_rows(d: SportsData) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(event, contract) rows of the sampled games, with the sign each event should push each contract
    (sgn) and the contract's distance to its threshold (dist); the contracts; the events."""
    ev = events()
    con = (d.contracts.filter(pl.col("league") == LEAGUE, pl.col("scope") == "game",
                              pl.col("kind").is_in(["winner", "spread", "total"]), pl.col("game_id").is_not_null())
           .select("ticker", "series_ticker", "kind", "team", "side", "strike", "game_id", "home", "away"))
    gids = con.filter(pl.col("kind") == ("spread" if LEAGUE == "MLB" else "winner"))["game_id"].unique()
    gids = gids.filter(gids.is_in(ev["game_id"].unique().implode()))
    gids = gids.sample(n=min(N_GAMES[LEAGUE], len(gids)), seed=SEED)
    ev = ev.filter(pl.col("game_id").is_in(gids.implode())).with_row_index("eid")
    con = con.filter(pl.col("game_id").is_in(gids.implode()))
    print(f"{LEAGUE}: {gids.len()} games, {ev.height:,} events (|dWP| >= {MIN_DWP} or scoring), "
          f"{con.height:,} contracts (winner / spread / total)")

    # one row per (event, contract) with the direction the event should push the contract
    x = ev.join(con, on="game_id").with_columns(
        pl.when(pl.col("side") == "home").then(1).when(pl.col("side") == "away").then(-1).otherwise(None).alias("hs"))
    margin_x = pl.col("hs") * (pl.col("home_score") - pl.col("away_score"))
    sign = (pl.when(pl.col("kind") == "winner").then((pl.col("hs") * pl.col("dwp")).sign())
            .when(pl.col("kind") == "spread").then(
                pl.when(pl.col("dmargin") != 0).then((pl.col("hs") * pl.col("dmargin")).sign())
                .otherwise((pl.col("hs") * pl.col("dwp")).sign()))
            .when(pl.col("dpoints") > 0).then(1.0).otherwise(None))
    dist = (pl.when(pl.col("kind") == "spread").then(pl.col("strike") - margin_x)
            .when(pl.col("kind") == "total").then(pl.col("strike") - (pl.col("home_score") + pl.col("away_score"))))
    x = x.with_columns(sign.alias("sgn"), dist.alias("dist")).filter(pl.col("sgn").is_not_null(), pl.col("sgn") != 0)
    return x, con, ev


def main() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(230)
    d = SportsData()
    x, con, ev = event_contract_rows(d)
    offs = [-REF] + OFFSETS
    at = pl.concat([x.select("eid", "ticker", (pl.col("t") + pl.duration(seconds=o)).alias("t"), pl.lit(o).alias("o"))
                    for o in offs])
    q = quotes_at(scan_quotes(con["series_ticker"].unique().to_list()), at.select("ticker", "t").unique(), max_spread=100.0)
    q = q.with_columns(value(q).alias("mid"))
    at = at.join(q.select("ticker", "t", "mid").unique(["ticker", "t"]), on=["ticker", "t"], how="left")
    wide = at.pivot(on="o", index=["eid", "ticker"], values="mid", aggregate_function="first")
    x = x.join(wide, on=["eid", "ticker"]).with_columns(
        [(pl.col("sgn") * (pl.col(str(o)) - pl.col(str(-REF)))).alias(f"m{o}") for o in OFFSETS])

    print(f"\n######## A. winner contracts: mean signed move (cents) from {REF} s before, by event type")
    print(summarise(x.filter(pl.col("kind") == "winner"), "group"))
    edges = BINS[LEAGUE]
    lab = [f"<{edges[0]}"] + [f"{a}..{b}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]}"]
    for kind in ("spread", "total"):
        k = x.filter(pl.col("kind") == kind).with_columns(pl.col("dist").cut(edges, labels=lab).cast(pl.String).alias("distance"))
        if k.height == 0:
            continue
        what = "threshold minus the team's current margin" if kind == "spread" else "threshold minus the current total"
        print(f"\n######## B. {kind} contracts by distance ({what}); "
              f"{'scoring events only' if kind == 'total' else 'all events'}")
        print(summarise(k, "distance", lab))
    print("    big events only (|dWP| >= 0.10), spread contracts:")
    print(summarise(x.filter(pl.col("kind") == "spread", pl.col("dwp").abs() >= 0.10)
                    .with_columns(pl.col("dist").cut(edges, labels=lab).cast(pl.String).alias("distance")), "distance", lab))
    print("    first trade after big events (|dWP| >= 0.10): is the slower spread / total response real, or the "
          "once-a-minute candle?")
    print(first_trades(x, con))

    # futures of the two teams
    fut = d.contracts.filter(pl.col("league") == LEAGUE, pl.col("scope") == "season",
                             pl.col("kind").is_in(["champ", "conference", "division", "playoffs"]),
                             pl.col("team").is_not_null()).select("ticker", "series_ticker", "kind", "team")
    cd = candles_1m(fut["series_ticker"].unique().to_list())
    g = load_games().filter(pl.col("league") == LEAGUE).select("game_id", "home", "away").unique("game_id")
    fe = pl.concat([ev.join(g, on="game_id").select("eid", "game_id", "t", "dwp", pl.col(s).alias("team"),
                                                    pl.lit(1 if s == "home" else -1).alias("hs")) for s in ("home", "away")])
    fe = fe.join(fut, on="team").with_columns((pl.col("hs") * pl.col("dwp")).sign().alias("sgn"))
    fat = pl.concat([fe.select("eid", "ticker", (pl.col("t") + pl.duration(seconds=o)).alias("t"), pl.lit(o).alias("o"))
                     for o in [-60] + FUT_OFFSETS])
    fm = mids_at(cd, fat).pivot(on="o", index=["eid", "ticker"], values="mid", aggregate_function="first")
    fx = fe.join(fm, on=["eid", "ticker"]).with_columns(
        [(pl.col("sgn") * (pl.col(str(o)) - pl.col("-60"))).alias(f"f{o}") for o in FUT_OFFSETS])
    rows = []
    for (kind,), gg in fx.partition_by("kind", as_dict=True).items():
        r = {"kind": kind, "n": gg.height}
        for o in FUT_OFFSETS:
            v = gg[f"f{o}"].drop_nulls()
            r[f"{o // 60}m"] = round(float(v.mean()), 3) if len(v) >= MIN_N else None
        both = gg.drop_nulls(["f300", "f3600"])
        left = (both["f3600"] - both["f300"]).to_numpy()
        r["usable"] = round(both.height / gg.height, 3)
        r["left_after_5m"] = round(float(left.mean()), 3) if len(left) else None
        r["t_left"] = round(clustered_t(left, both["game_id"].to_numpy()), 1) if len(left) else None
        rows.append(r)
    print("\n######## C. team futures: mean signed move (cents) from 60 s before, all events")
    print(pl.DataFrame(rows).sort("kind") if rows else "no futures with 1-min candles")


if __name__ == "__main__":
    main()
