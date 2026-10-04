"""The between-game graph (reports/sports_research_plan.md §8).

One graph per league-season. Nodes are ``<team>-<kind>`` futures (division,
playoffs, conference, champ, wins), each live from its first listing until its
last leg closes. Game results enter as shocks on a team's nodes, not as nodes.

    between_game_graph(data, edges, league, season) -> BetweenGameGraph
        .nodes          node, idx, team, kind, conf, division, tickers, known_at, valid_to
        .relations      src, dst (node names), relation, sign_prior, known_at, valid_to
        .active(t)      nodes live at t
        .adjacency(t)   {relation: (N, N)} live at t; [i, j] = influence of i on j
        .candidates(t)  every live off-diagonal pair (the dense within-league graph)
    node_values(data, graph, at)            each node's value at t: the mid (cents), or for a wins
                                            node the expected wins implied by its ladder
    game_shocks(data, league, season)       one row per (game, team): z = y - p_pre
    pair_attributes(data, graph, t)         as-of attributes of every candidate pair at t

Relations: ``rival_<kind>`` (-: legs of one exclusive futures event, or for
playoffs, teams of one conference competing for its spots) and ``hierarchy``
(+: a team's division -> playoffs -> conference -> champ, wins -> playoffs).

``valid_to`` comes from the legs' close times, as in :mod:`stg.sports.edges`:
test liveness with ``known_at <= t < valid_to``, which only asks whether the
close has happened by t.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import norm

from stg.sports.asof import SportsData
from stg.sports.edges import conference_map
from stg.sports.features import team_state
from stg.sports.snapshots import pregame_snapshots

NODE_KINDS = ("division", "playoffs", "conference", "champ", "wins")
# MLB's wins ladders are too thin to value (5 strikes 10 wins apart, listed in May, 41% of legs
# never traded, median two-sided spread 31c): left out of its graph (decided 2026-10-03)
EXCLUDE_NODES = {("MLB", "wins")}
RELATIONS = ("rival_division", "rival_playoffs", "rival_conference", "rival_champ", "hierarchy")


def _lit(t: dt.datetime) -> pl.Expr:
    return pl.lit(t, dtype=pl.Datetime("us", "UTC"))


@dataclass
class BetweenGameGraph:
    league: str
    season: int
    nodes: pl.DataFrame
    relations: pl.DataFrame

    @property
    def names(self) -> list[str]:
        return self.nodes["node"].to_list()

    def _live(self, df: pl.DataFrame, t: dt.datetime) -> pl.DataFrame:
        return df.filter(pl.col("known_at") <= _lit(t), pl.col("valid_to").is_null() | (pl.col("valid_to") > _lit(t)))

    def active(self, t: dt.datetime) -> np.ndarray:
        live = set(self._live(self.nodes, t)["node"].to_list())
        return np.array([n in live for n in self.names])

    def adjacency(self, t: dt.datetime, relations=RELATIONS, signed: bool = False,
                  symmetric: bool = False) -> dict[str, np.ndarray]:
        """One (N, N) matrix per relation, edges live at t (both ends live)."""
        n, idx = len(self.names), dict(zip(self.names, range(len(self.names))))
        on = self.active(t)
        live = self._live(self.relations, t)
        out = {}
        for r in relations:
            a = np.zeros((n, n), dtype=np.float32)
            for s, d, sg in live.filter(pl.col("relation") == r).select("src", "dst", "sign_prior").iter_rows():
                i, j = idx[s], idx[d]
                if on[i] and on[j]:
                    a[i, j] = sg if signed else 1.0
            out[r] = np.maximum(a, a.T) if symmetric and not signed else a
        return out

    def candidates(self, t: dt.datetime) -> np.ndarray:
        on = self.active(t)
        m = np.outer(on, on)
        np.fill_diagonal(m, False)
        return m


def between_game_graph(data: SportsData, edges: pl.DataFrame, league: str, season: int) -> BetweenGameGraph:
    kinds = [k for k in NODE_KINDS if (league, k) not in EXCLUDE_NODES]
    c = data.contracts.filter(pl.col("league") == league, pl.col("season") == season,
                              pl.col("kind").is_in(kinds), pl.col("team").is_not_null())
    conf = conference_map(data.contracts).filter(pl.col("league") == league, pl.col("season") == season)
    # a conference node holds only the team's own conference's leg (see conference_map)
    c = (c.join(conf, on=["league", "season", "team"], how="left")
         .filter((pl.col("kind") != "conference")
                 | pl.col("series_ticker").str.contains(pl.col("conf").fill_null("~"), literal=True)))
    close = data.finals.select("ticker", "close_time")
    c = c.join(close, on="ticker", how="left")
    div = (c.filter(pl.col("kind") == "division").group_by("team")
           .agg(pl.col("event_ticker").first().alias("division"), pl.col("open_time").min().alias("div_known_at")))
    nodes = (c.group_by("team", "kind")
             .agg(pl.col("ticker").sort().alias("tickers"), pl.col("conf").first(), pl.col("conf_known_at").first(),
                  pl.col("open_time").min().alias("known_at"),
                  # live until its last leg closes; null if any leg is still open
                  pl.when(pl.col("close_time").is_null().any()).then(None)
                  .otherwise(pl.col("close_time").max()).alias("valid_to"))
             .join(div, on="team", how="left")
             .with_columns(pl.format("{}-{}", "team", "kind").alias("node"))
             .sort("kind", "team").with_row_index("idx")
             .select("node", "idx", "team", "kind", "conf", "conf_known_at", "division", "div_known_at",
                     "tickers", "known_at", "valid_to"))

    t2n = nodes.select("node", "tickers").explode("tickers").rename({"tickers": "ticker"})
    e = (edges.filter(pl.col("edge_type").is_in(["rival", "hierarchy"]))
         .join(t2n.rename({"ticker": "src", "node": "src_node"}), on="src")
         .join(t2n.rename({"ticker": "dst", "node": "dst_node"}), on="dst")
         .filter(pl.col("src_node") != pl.col("dst_node"))
         .join(nodes.select(pl.col("node").alias("src_node"), pl.col("kind").alias("src_kind")), on="src_node"))
    rel = (e.with_columns(pl.when(pl.col("edge_type") == "rival")
                          .then(pl.format("rival_{}", "src_kind")).otherwise(pl.col("edge_type")).alias("relation"))
           .group_by("src_node", "dst_node", "relation")
           .agg(pl.col("sign_prior").first(), pl.col("known_at").min(),
                pl.when(pl.col("valid_to").is_null().any()).then(None)
                .otherwise(pl.col("valid_to").max()).alias("valid_to"))
           .rename({"src_node": "src", "dst_node": "dst"})
           .sort("relation", "src", "dst"))
    return BetweenGameGraph(league, season, nodes, rel)


# --------------------------------------------------------------------------- node values

NODE_MAX_SPREAD = 10.0                    # futures books are thin: wider than the game-market default
NODE_MAX_AGE = dt.timedelta(days=1)       # a side not updated within a day is stale


def _threshold(strike: pl.Expr, strike_type: pl.Expr) -> pl.Expr:
    """The k with a leg = P(W >= k): 'more than 90' -> 91, 'at least 74.5' -> 75."""
    return (pl.when(strike_type == "greater").then(strike.floor() + 1)
            .otherwise(strike.ceil()))


def fit_wins_ladder(k: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Expected wins and spread from ladder prices P(W >= k), assuming W ~ Normal(mu, sigma):
    probit(p) = (mu + 0.5 - k) / sigma, fitted by least squares weighted by p(1-p) (legs near 50c
    say most about mu). Needs two legs; returns (nan, nan) if the ladder is not decreasing."""
    p = np.clip(p, 0.005, 0.995)
    if len(k) < 2 or np.ptp(k) == 0:
        return np.nan, np.nan
    w = p * (1 - p)
    A = np.column_stack([np.ones_like(k), k]) * np.sqrt(w)[:, None]
    a, b = np.linalg.lstsq(A, norm.ppf(p) * np.sqrt(w), rcond=None)[0]
    if b >= 0:
        return np.nan, np.nan
    sigma = -1 / b
    return a * sigma - 0.5, sigma


EXTREME_C = 3.0                           # a one-sided book this close to 0 / 100 is read as a value


def node_values(data: SportsData, graph: BetweenGameGraph, at: pl.DataFrame,
                max_spread: float = NODE_MAX_SPREAD, max_age: dt.timedelta = NODE_MAX_AGE,
                extremes: bool = True) -> pl.DataFrame:
    """at: node, t. Returns node, t, kind, value, value_known_at, spread, quote_age, one_sided, and
    for wins nodes the fitted sigma and the legs used (extreme_legs of them one-sided). Value: the
    mid in cents (single-leg kinds) or the expected wins (wins), from quotes known at t only; null
    if no valid quote. spread / quote_age are of the leg (wins: the widest / oldest leg used), kept
    so later steps can weight or filter without rebuilding.

    A node listed more than once (Kalshi relists: the NFL wins ladders -25 / -25B) is read from
    its newest event listed by t. ``extremes``: a one-sided book with ask <= EXTREME_C (no bid)
    is valued at ask / 2, and one with bid >= 100 - EXTREME_C (no ask) at (bid + 100) / 2, so
    longshots and near-certainties keep a (coarse) value. A wins ladder is fitted only when at
    least one leg is two-sided; extreme legs then anchor its tails."""
    nd = graph.nodes.select("node", "kind", "tickers")
    legs = (at.join(nd, on="node").explode("tickers").rename({"tickers": "ticker"})
            .join(data.contracts.select("ticker", "event_ticker", "open_time", "strike", "strike_type"),
                  on="ticker", how="left")
            .filter(pl.col("open_time") <= pl.col("t"))
            .with_columns(pl.col("open_time").min().over("event_ticker").alias("_ev_open")))
    legs = legs.filter(pl.col("_ev_open") == pl.col("_ev_open").max().over("node", "t"))
    q = data.quotes_at(legs.select("ticker", "t").unique(), max_spread=max_spread, max_age=max_age)
    q = q.select("ticker", "t", "mid", "bid", "ask", "bid_at", "ask_at", "quote_known_at").unique(["ticker", "t"])
    legs = legs.join(q, on=["ticker", "t"], how="left")
    if extremes:
        fresh = lambda c: (pl.col("t") - pl.col(c)) <= max_age          # noqa: E731
        low = (pl.col("bid").fill_null(0) <= 0) & (pl.col("ask") <= EXTREME_C) & fresh("ask_at")
        high = (pl.col("ask").fill_null(100) >= 100) & (pl.col("bid") >= 100 - EXTREME_C) & fresh("bid_at")
        legs = legs.with_columns(
            pl.col("mid").is_not_null().alias("_two_sided"),
            pl.when(pl.col("mid").is_not_null()).then(pl.col("mid"))
            .when(low).then(pl.col("ask") / 2).when(high).then((pl.col("bid") + 100) / 2).alias("mid"))
    else:
        legs = legs.with_columns(pl.col("mid").is_not_null().alias("_two_sided"))
    legs = legs.with_columns(
        pl.when(pl.col("_two_sided")).then(pl.col("ask") - pl.col("bid")).alias("spread"),
        pl.max_horizontal(pl.col("t") - pl.col("bid_at"), pl.col("t") - pl.col("ask_at")).alias("quote_age"),
        (pl.col("mid").is_not_null() & ~pl.col("_two_sided")).alias("one_sided"))

    single = (legs.filter(pl.col("kind") != "wins")
              .select("node", "t", "kind", pl.col("mid").alias("value"),
                      pl.when(pl.col("mid").is_not_null()).then(pl.col("quote_known_at")).alias("value_known_at"),
                      "spread", pl.when(pl.col("mid").is_not_null()).then(pl.col("quote_age")).alias("quote_age"),
                      "one_sided"))
    w = (legs.filter(pl.col("kind") == "wins", pl.col("mid").is_not_null())
         .with_columns(_threshold(pl.col("strike"), pl.col("strike_type")).alias("k")))
    rows = []
    for (node, t), g in w.group_by("node", "t"):
        if not g["_two_sided"].any():          # tails alone only bound the ladder: no fit
            continue
        mu, sigma = fit_wins_ladder(g["k"].to_numpy().astype(float), g["mid"].to_numpy() / 100)
        rows.append({"node": node, "t": t, "value": mu, "sigma": sigma, "legs": g.height,
                     "extreme_legs": int(g["one_sided"].sum()), "spread": g["spread"].max(),
                     "quote_age": g["quote_age"].max(), "value_known_at": g["quote_known_at"].max()})
    wins = (at.join(nd.filter(pl.col("kind") == "wins").select("node", "kind"), on="node")
            .join(pl.DataFrame(rows, schema={"node": pl.String, "t": at.schema["t"], "value": pl.Float64,
                                             "sigma": pl.Float64, "legs": pl.Int64, "extreme_legs": pl.Int64,
                                             "spread": pl.Float64, "quote_age": pl.Duration("us"),
                                             "value_known_at": pl.Datetime("us", "UTC")}),
                  on=["node", "t"], how="left")
            .with_columns(pl.col("value").fill_nan(None), pl.col("sigma").fill_nan(None),
                          pl.col("legs").fill_null(0), pl.lit(False).alias("one_sided")))
    return (pl.concat([single, wins], how="diagonal_relaxed")
            .with_columns(pl.when(pl.col("value").is_null()).then(None)
                          .otherwise(pl.col("value_known_at")).alias("value_known_at"))
            .sort("node", "t"))


# --------------------------------------------------------------------------- shocks

def game_shocks(data: SportsData, league: str, season: int) -> pl.DataFrame:
    """One row per (game, team): the pre-game win probability p_pre (the winner
    outcome's mid at first pitch / tip-off / kickoff), the result y (1 win, 0
    loss, 0.5 tie), and the surprise z = y - p_pre, known at the final score."""
    s = pregame_snapshots(data, offsets=(dt.timedelta(0),), kinds=("winner",), leagues=(league,))
    s = s.filter(pl.col("season") == season, pl.col("team").is_not_null(), pl.col("mid").is_not_null())
    g = data.contracts.select("ticker", "opp", "home", "away", "game_type")
    r = data.results.select("game_id", "home_score", "away_score", pl.col("known_at").alias("result_known_at"))
    s = (s.select("league", "season", "game_id", "ticker", "team", "anchor", pl.col("mid").alias("p_pre"),
                  "quote_known_at")
         .join(g, on="ticker", how="left").join(r, on="game_id", how="inner"))
    me = pl.when(pl.col("team") == pl.col("home")).then(pl.col("home_score")).otherwise(pl.col("away_score"))
    op = pl.when(pl.col("team") == pl.col("home")).then(pl.col("away_score")).otherwise(pl.col("home_score"))
    s = s.with_columns(pl.col("p_pre") / 100,
                       pl.when(me > op).then(1.0).when(me < op).then(0.0).otherwise(0.5).alias("y"))
    ref = s.select("league", "season", "game_id", "game_type", "team", "opp", pl.col("anchor").alias("first_play"),
                   "p_pre", "y", "quote_known_at", pl.col("result_known_at").alias("known_at"))
    other = ref.with_columns(pl.col("opp").alias("team"), pl.col("team").alias("opp"),
                             (1 - pl.col("p_pre")).alias("p_pre"), (1 - pl.col("y")).alias("y"))
    return (pl.concat([ref, other]).with_columns((pl.col("y") - pl.col("p_pre")).alias("z"))
            .sort("known_at", "game_id", "team"))


# --------------------------------------------------------------------------- pair attributes

def pair_attributes(data: SportsData, graph: BetweenGameGraph, t: dt.datetime,
                    meeting_window: dt.timedelta = dt.timedelta(days=7)) -> pl.DataFrame:
    """Every candidate pair live at t, with neutral as-of attributes: structure
    (same team / kind / division / conference), standings gap, games remaining,
    and scheduled meetings in the next ``meeting_window`` (games listed by t).

    Only what is known at t: memberships from listings opened by t (else null),
    standings from games finished by t, meetings from games listed by t."""
    tt = _lit(t)
    # a membership is used only once the listing that establishes it is known
    nd = graph.nodes.filter(pl.Series(graph.active(t))).with_columns(
        pl.when(pl.col("conf_known_at") <= tt).then(pl.col("conf")).alias("conf"),
        pl.when(pl.col("div_known_at") <= tt).then(pl.col("division")).alias("division"))
    pairs = (nd.select(pl.col("node").alias("src"), pl.col("team").alias("src_team"), pl.col("kind").alias("src_kind"),
                       pl.col("conf").alias("src_conf"), pl.col("division").alias("src_div"))
             .join(nd.select(pl.col("node").alias("dst"), pl.col("team").alias("dst_team"),
                             pl.col("kind").alias("dst_kind"), pl.col("conf").alias("dst_conf"),
                             pl.col("division").alias("dst_div")), how="cross")
             .filter(pl.col("src") != pl.col("dst")))
    teams = nd.select("team").unique().with_columns(pl.lit(graph.league).alias("league"),
                                                    pl.lit(graph.season).alias("season"), _lit(t).alias("t"))
    st = team_state(data, teams.select("league", "season", "team", "t")).select(
        "team", "win_pct", "games_behind", "frac_remaining", "standings_known_at")
    meet = (data.contracts.filter(pl.col("league") == graph.league, pl.col("kind") == "winner",
                                  pl.col("open_time") <= tt, pl.col("sched_start") > tt,
                                  pl.col("sched_start") <= _lit(t + meeting_window),
                                  pl.col("team").is_not_null(), pl.col("opp").is_not_null())
            .group_by("team", "opp").agg(pl.col("game_id").n_unique().alias("meetings_next")))
    out = (pairs
           .join(st.select(pl.all().name.prefix("src_")).rename({"src_team": "src_team"}), on="src_team", how="left")
           .join(st.select(pl.all().name.prefix("dst_")).rename({"dst_team": "dst_team"}), on="dst_team", how="left")
           .join(meet.rename({"team": "src_team", "opp": "dst_team"}), on=["src_team", "dst_team"], how="left")
           .with_columns(
               _lit(t).alias("t"),
               (pl.col("src_team") == pl.col("dst_team")).alias("same_team"),
               (pl.col("src_kind") == pl.col("dst_kind")).alias("same_kind"),
               # null = not yet known at t (or no such grouping, e.g. NBA divisions)
               (pl.col("src_conf") == pl.col("dst_conf")).alias("same_conf"),
               (pl.col("src_div") == pl.col("dst_div")).alias("same_div"),
               (pl.col("src_win_pct") - pl.col("dst_win_pct")).alias("win_pct_gap"),
               (pl.col("src_games_behind") - pl.col("dst_games_behind")).alias("gb_gap"),
               pl.col("meetings_next").fill_null(0),
               pl.max_horizontal("src_standings_known_at", "dst_standings_known_at").alias("standings_known_at"))
           .drop("src_standings_known_at", "dst_standings_known_at", "src_conf", "dst_conf", "src_div", "dst_div"))
    return out.sort("src", "dst")
