"""What each sports market is: league, kind, team, strike, game and season.

Everything here is read from fields known when the market lists (ticker,
titles, ``rules_primary``, ``open_time``) plus the schedule, so the table
carries ``known_at = open_time``.

Kinds
-----
game level (``scope = "game"``), a game key in the event ticker:
    winner, spread, total, team_total, and their half / first-5 variants
    (``winner_1h``, ``spread_f5``, ...), ``rfi`` (first-inning run), ``prop``
    (player and event props: everything else on a game)
season level (``scope = "season"``):
    champ, conference, division, playoffs, series (a playoff series winner),
    wins (win-total ladders), ``season_other`` (awards, draft, leaders, ...)

Team codes are Kalshi's game-leg codes; futures series spell a few teams
differently (``ALIAS``). ``season`` is the league year containing
``open_time``: MLB from Dec 1, NBA from Jul 1, NFL from Feb 11 (after the Super
Bowl, so ``KXSB-26``, listed 2025-02-12, is the 2025 season).
"""
from __future__ import annotations

import datetime as dt
import re

import polars as pl

from stg.sports.loaders import event_key, load_games, load_listing

GAME_SERIES = re.compile(r"^KX(MLB|NBA|NFL)(1H|2H|F5)?(GAME|STGAME|WINNER|SPREAD|TOTAL|TEAMTOTAL)$")
BASE_KIND = {"GAME": "winner", "STGAME": "winner", "WINNER": "winner", "SPREAD": "spread",
             "TOTAL": "total", "TEAMTOTAL": "team_total"}
SEASON_KIND = {
    "KXMLB": "champ", "KXNBA": "champ", "KXSB": "champ",
    "KXMLBAL": "conference", "KXMLBNL": "conference", "KXNBAEAST": "conference",
    "KXNBAWEST": "conference", "KXNFLAFCCHAMP": "conference", "KXNFLNFCCHAMP": "conference",
    "KXAFC": "conference", "KXNFC": "conference",
    "KXMLBPLAYOFFS": "playoffs", "KXNBAPLAYOFF": "playoffs", "KXNFLPLAYOFF": "playoffs",
    "KXMLBSERIES": "series", "KXNBASERIES": "series",
}
DIVISION = re.compile(r"^KX(MLB(AL|NL)(EAST|CENT|WEST)|NFL(AFC|NFC)(EAST|NORTH|SOUTH|WEST))$")
WINS = re.compile(r"^KX(MLB|NBA|NFL)WINS")
LEAGUE_OF = {"KXSB": "NFL", "KXAFC": "NFL", "KXNFC": "NFL", "KXTEAMSINSB": "NFL",
             "KXTEAMSINNBAF": "NBA", "KXWSAL": "MLB", "KXWSNL": "MLB", "KXTEAMSINWS": "MLB",
             "KXALMVP": "MLB", "KXNLMVP": "MLB", "KXALALLSTAR": "MLB", "KXOHTANIMVPCY": "MLB"}
LEAGUE_PAT = [(re.compile(r"^KX(LEADERNFL|LEADERPINT|RECORDNFL)"), "NFL"),
              (re.compile(r"^KX(LEADERNBA|RECORDNBA|WINSTREAK)"), "NBA"),
              (re.compile(r"^KXLEADERMLB"), "MLB")]
ALIAS = {"MLB": {"A": "ATH", "ARI": "AZ", "FLA": "MIA", "KAN": "KC", "WAS": "WSH"},
         "NFL": {"JAX": "JAC", "LAR": "LA"}, "NBA": {}}
# source schedule abbreviations (statsapi / ESPN) -> Kalshi codes; inverse of fetch_pbp.TO_SRC
FROM_SRC = {"MLB": {"AZ": "AZ"},
            "NBA": {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"},
            "NFL": {"JAX": "JAC", "LAR": "LA", "WSH": "WAS"}}
SEASON_START = {"MLB": (12, 1), "NBA": (7, 1), "NFL": (2, 11)}   # (month, day) the league year turns

SCHEDULED = re.compile(r"originally scheduled for ([A-Z][a-z]{2} \d{1,2}, \d{4})(?: at (\d{1,2}:\d{2} [AP]M) (E[DS]T))?")


def league_of(series: str) -> str | None:
    m = re.match(r"^KX(MLB|NBA|NFL)", series)
    if m:
        return m.group(1)
    if series in LEAGUE_OF:
        return LEAGUE_OF[series]
    return next((lg for p, lg in LEAGUE_PAT if p.match(series)), None)


def kind_of(series: str, has_game_key: bool) -> tuple[str, str]:
    """(kind, scope) of a series."""
    m = GAME_SERIES.match(series)
    if m and has_game_key:
        part = {"1H": "_1h", "2H": "_2h", "F5": "_f5"}.get(m.group(2) or "", "")
        return BASE_KIND[m.group(3)] + part, "game"
    if series == "KXMLBRFI":
        return "rfi", "game"
    if series == "KXMLBF5" and has_game_key:
        return "winner_f5", "game"
    if series in SEASON_KIND:
        return SEASON_KIND[series], "season"
    if DIVISION.match(series):
        return "division", "season"
    if WINS.match(series):
        return "wins", "season"
    return ("prop", "game") if has_game_key else ("season_other", "season")


def season_expr(league: pl.Expr, t: pl.Expr) -> pl.Expr:
    """The league year containing ``t`` (see module docstring)."""
    y, md = t.dt.year(), t.dt.month().cast(pl.Int32) * 100 + t.dt.day().cast(pl.Int32)
    return (pl.when(league == "MLB").then(pl.when(md >= 1201).then(y + 1).otherwise(y))
            .when(league == "NBA").then(pl.when(md >= 701).then(y).otherwise(y - 1))
            .when(league == "NFL").then(pl.when(md >= 211).then(y).otherwise(y - 1))
            .otherwise(None)).cast(pl.Int32)


def parse_rules(rules: str | None) -> dict:
    """The scheduled start stated in ``rules_primary`` (ET), and the rule's shape."""
    out = {"rule_sched": None, "rule_shape": None}
    if not rules:
        return out
    m = SCHEDULED.search(rules)
    if m:
        d = dt.datetime.strptime(m.group(1), "%b %d, %Y")
        if m.group(2):
            tm = dt.datetime.strptime(m.group(2), "%I:%M %p").time()
            off = -4 if m.group(3) == "EDT" else -5
            out["rule_sched"] = dt.datetime.combine(d.date(), tm, dt.timezone(dt.timedelta(hours=off))
                                                    ).astimezone(dt.timezone.utc)
    low = rules.lower()
    for shape, pat in (("margin_over", "wins by more than"), ("margin_over", "by over"),
                       ("total_over", "collectively score more"), ("team_over", "score more than"),
                       ("win", " wins the "), ("win", "is the winner")):
        if pat in low:
            out["rule_shape"] = shape
            break
    return out


def _team_code(league: str | None, kind: str, suffix: str) -> str | None:
    if league is None:
        return None
    if kind in ("total", "total_1h", "total_2h", "total_f5", "rfi", "prop", "season_other", "wins"):
        return None
    m = re.match(r"^([A-Z]+?)(\d+)?$", suffix)
    if not m or m.group(1) in ("TIE",):
        return None
    return ALIAS[league].get(m.group(1), m.group(1))


def contracts() -> pl.DataFrame:
    """One row per market: league, kind, scope, team, strike, game key/id, season, sched start."""
    ls = load_listing()
    ev_suffix = event_key()
    ls = ls.with_columns(
        ev_suffix.alias("key"),
        pl.col("ticker").str.split("-").list.last().alias("leg"),
        pl.col("series_ticker").map_elements(league_of, return_dtype=pl.String).alias("league"))
    ls = ls.with_columns(pl.col("key").str.contains(r"^\d{2}[A-Z]{3}\d{2}").fill_null(False).alias("_gk"))
    ks = [kind_of(s, g) for s, g in ls.select("series_ticker", "_gk").iter_rows()]
    ls = ls.with_columns(pl.Series("kind", [k for k, _ in ks]), pl.Series("scope", [s for _, s in ks]))
    ls = ls.with_columns(
        pl.struct("league", "kind", "leg").map_elements(
            lambda r: _team_code(r["league"], r["kind"], r["leg"]), return_dtype=pl.String).alias("team"),
        # strike on one scale: YES iff the quantity is > strike ("N+" / "at least N" -> N - 0.5)
        pl.coalesce(pl.col("yes_sub_title").str.extract(r"(?i)over ([\d.]+)", 1).cast(pl.Float64, strict=False),
                    pl.col("title").str.extract(r"(?i)over ([\d.]+)", 1).cast(pl.Float64, strict=False),
                    pl.col("title").str.extract(r"(?i)more than ([\d.]+)", 1).cast(pl.Float64, strict=False),
                    pl.col("yes_sub_title").str.extract(r"(\d+)\+", 1).cast(pl.Float64, strict=False) - 0.5,
                    pl.col("title").str.extract(r"(?i)at least ([\d.]+)", 1).cast(pl.Float64, strict=False) - 0.5)
        .alias("strike"),
        season_expr(pl.col("league"), pl.col("open_time")).alias("season"))
    # wins ladders: the team is in the event ticker (KXNFLWINS-KC-25B)
    ls = ls.with_columns(pl.when(pl.col("kind") == "wins").then(
        pl.col("event_ticker").str.split("-").list.get(1, null_on_oob=True)).otherwise(pl.col("team")).alias("team"))
    ls = ls.with_columns(pl.struct("league", "team").map_elements(
        lambda r: ALIAS.get(r["league"] or "", {}).get(r["team"], r["team"]) if r["team"] else None,
        return_dtype=pl.String).alias("team"))
    rules = [parse_rules(r) for r in ls["rules_primary"]]
    ls = ls.with_columns(pl.Series("rule_sched", [r["rule_sched"] for r in rules], dtype=pl.Datetime("us", "UTC")),
                         pl.Series("rule_shape", [r["rule_shape"] for r in rules], dtype=pl.String))

    # game-level markets -> the matched source game (on the event key, as fetch_pbp.py says)
    g = load_games()
    gk = (g.select("key", "game_id", "league", "start_utc", "home", "away", "game_type")
          .unique("key", keep="first")
          .with_columns(*(pl.struct(pl.col("league"), pl.col(c)).map_elements(
              lambda r, c=c: FROM_SRC[r["league"]].get(r[c], r[c]), return_dtype=pl.String).alias(c)
              for c in ("home", "away")))
          .rename({"league": "_glg", "start_utc": "sched_start"}))
    ls = ls.join(gk, on="key", how="left").with_columns(
        pl.when(pl.col("scope") == "game").then(pl.col(c)).otherwise(None).alias(c)
        for c in ("game_id", "sched_start", "home", "away", "game_type"))
    ls = ls.with_columns(
        pl.when(pl.col("team").is_null()).then(None)
        .when(pl.col("team") == pl.col("home")).then(pl.lit("home"))
        .when(pl.col("team") == pl.col("away")).then(pl.lit("away")).alias("side"),
        pl.when(pl.col("team") == pl.col("home")).then(pl.col("away"))
        .when(pl.col("team") == pl.col("away")).then(pl.col("home")).alias("opp"),
        pl.when(pl.col("scope") == "game").then(season_expr(pl.col("league"), pl.col("sched_start")))
        .otherwise(pl.col("season")).alias("season"))
    return ls.select("ticker", "event_ticker", "series_ticker", "league", "kind", "scope", "team",
                     "opp", "side", "strike", "strike_type", "key", "game_id", "home", "away",
                     "game_type", "sched_start", "rule_sched", "rule_shape", "season", "open_time",
                     "expiration_time", "known_at")
