"""The sports data layer's safeguards (reports/sports_research_plan.md §4).

Unit tests run on synthetic frames; the data-backed ones (truncation test,
as-of checks) skip when ``data/sports/`` or its derived layer is absent.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import polars as pl
import pytest

from stg.splits import SPORTS_HOLDOUT_START
from stg.sports import loaders
from stg.sports.checks import assert_asof, final_column_reads, truncation_test
from stg.sports.contracts import kind_of, parse_rules, season_expr
from stg.sports.prices import QuoteBook, build_quotes, mid_exprs

UTC = dt.timezone.utc
ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parents[1] / "stg" / "sports"
T0 = dt.datetime(2025, 6, 1, 18, 0, tzinfo=UTC)

have_data = pytest.mark.skipif(
    not (Path("data/sports/markets.parquet").exists() and Path("data/sports/derived/quotes").exists()),
    reason="sports archive / derived layer absent: run scripts/build_sports_layer.py from event-pred/")


# --------------------------------------------------------------------------- holdout

def test_holdout_boundary_matches_the_pull_script():
    spec = importlib.util.spec_from_file_location("fetch_sports", ROOT / "scripts" / "fetch_sports.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.HOLDOUT_START == SPORTS_HOLDOUT_START


def test_safe_path_refuses_the_holdout():
    with pytest.raises(PermissionError):
        loaders.safe_path("_HOLDOUT_DO_NOT_USE", "markets.parquet")
    loaders.safe_path("trades", "KXNBAGAME")   # fine


# --------------------------------------------------------------------------- quotes

@pytest.fixture
def synthetic_trades(tmp_path, monkeypatch):
    d = tmp_path / "trades" / "KXTEST"
    d.mkdir(parents=True)
    s = lambda k: T0 + dt.timedelta(seconds=k)  # noqa: E731
    pl.DataFrame({
        "trade_id": ["a", "b", "c", "d", "e"], "ticker": ["X"] * 5, "count": [1.0] * 5,
        "yes_price": [40.0, 42.0, 43.0, 44.0, 41.0], "no_price": [60.0, 58.0, 57.0, 56.0, 59.0],
        "taker_side": ["no", "yes", "yes", "no", "no"],
        "created_time": [s(0), s(10), s(20), s(30), s(3600)], "is_block_trade": [False] * 5,
    }).write_parquet(d / "1.parquet")
    monkeypatch.setenv("KALSHI_SPORTS_DIR", str(tmp_path))
    return build_quotes("KXTEST")


def test_trade_proxies_fill_each_side(synthetic_trades):
    q = synthetic_trades
    assert q["bid"].to_list() == [40.0, 40.0, 40.0, 44.0, 41.0]
    assert q["ask"].to_list() == [None, 42.0, 43.0, 43.0, 43.0]
    r = q.with_columns(mid_exprs(pl.col("known_at")))
    # row 0 one-sided; rows 1-2 valid; row 3 crossed (bid 44 > ask 43); row 4 ask an hour stale
    assert r["two_sided"].to_list() == [False, True, True, False, False]
    assert r["mid"][1] == 41.0


def test_quotebook_uses_rows_known_at_or_before_t(synthetic_trades):
    b = QuoteBook(synthetic_trades)
    assert b.at("X", T0 - dt.timedelta(microseconds=1)) is None
    at10 = b.at("X", T0 + dt.timedelta(seconds=10))          # the trade at exactly t counts
    assert at10["ask"] == 42.0 and at10["mid"] == 41.0
    assert b.at("X", T0 + dt.timedelta(seconds=9.999))["ask"] != at10["ask"]


# --------------------------------------------------------------------------- parsing

def test_kinds_and_rules():
    assert kind_of("KXNBAGAME", True) == ("winner", "game")
    assert kind_of("KXNBA1HSPREAD", True) == ("spread_1h", "game")
    assert kind_of("KXMLBF5", True) == ("winner_f5", "game")
    assert kind_of("KXMLBALEAST", False) == ("division", "season")
    assert kind_of("KXNFLWINS-KC", False) == ("wins", "season")
    assert kind_of("KXMLBHR", True) == ("prop", "game")
    r = parse_rules("If Cleveland wins by more than 2.5 runs in the Cleveland vs Kansas City professional "
                    "baseball game originally scheduled for May 6, 2026 at 7:40 PM EDT, then ...")
    assert r["rule_shape"] == "margin_over"
    assert r["rule_sched"] == dt.datetime(2026, 5, 6, 23, 40, tzinfo=UTC)


def test_season_boundaries():
    f = pl.DataFrame({"lg": ["NFL", "NFL", "NBA", "NBA", "MLB"],
                      "t": [dt.datetime(2025, 2, 12, tzinfo=UTC), dt.datetime(2025, 1, 23, tzinfo=UTC),
                            dt.datetime(2025, 7, 12, tzinfo=UTC), dt.datetime(2025, 4, 18, tzinfo=UTC),
                            dt.datetime(2025, 12, 5, tzinfo=UTC)]})
    assert f.select(season_expr(pl.col("lg"), pl.col("t")).alias("s"))["s"].to_list() == [2025, 2024, 2025, 2024, 2026]


# --------------------------------------------------------------------------- leakage rules

def test_feature_code_reads_no_final_listing_fields():
    assert final_column_reads(PKG / "features.py") == []


def test_assert_asof_catches_a_future_feature():
    f = pl.DataFrame({"t": [T0], "mid_known_at": [T0 + dt.timedelta(seconds=1)]})
    with pytest.raises(AssertionError):
        assert_asof(f)
    assert_asof(f.with_columns(pl.col("t") + dt.timedelta(seconds=1)))


# --------------------------------------------------------------------------- data-backed

CUT = dt.datetime(2025, 7, 1, tzinfo=UTC)


def _build(data):
    from stg.sports.features import market_state, team_state
    from stg.sports.snapshots import pregame_snapshots
    s = pregame_snapshots(data, kinds=("winner", "total"), leagues=("MLB",))
    s = s.filter(pl.col("t") >= dt.datetime(2025, 5, 1, tzinfo=UTC))
    ms = market_state(data, s.select("ticker", "t")).drop("ticker", "t").select(pl.all().name.prefix("ms_"))
    ts = team_state(data, s.filter(pl.col("team").is_not_null()).select("league", "season", "team", "t").unique())
    return (pl.concat([s, ms], how="horizontal")
            .join(ts, on=["league", "season", "team", "t"], how="left"))


@have_data
def test_truncation_snapshots_and_features():
    r = truncation_test(_build, CUT, key=["outcome_id", "ticker", "offset_h"])
    assert r["samples"] > 1000


@have_data
def test_features_are_asof():
    from stg.sports.asof import SportsData
    assert_asof(_build(SportsData(cutoff=CUT)))


@have_data
def test_asof_view_hides_results_until_settled():
    from stg.sports.asof import SportsData
    d = SportsData()
    fin = d.finals.filter(pl.col("result") == "yes").sort("known_at").row(0, named=True)
    before = d.as_of(fin["known_at"] - dt.timedelta(seconds=1)).markets(open_only=False)
    after = d.as_of(fin["known_at"]).markets(open_only=False)
    assert before.filter(pl.col("ticker") == fin["ticker"])["result"][0] is None
    assert after.filter(pl.col("ticker") == fin["ticker"])["result"][0] == "yes"
    with pytest.raises(ValueError):
        SportsData(cutoff=CUT).as_of(CUT + dt.timedelta(seconds=1))


# --------------------------------------------------------------------------- the between-game graph

def test_conference_map_is_asof_majority():
    from stg.sports.edges import conference_map
    t = lambda m, d: dt.datetime(2000 + m // 100, m % 100, d, tzinfo=UTC)   # noqa: E731
    c = pl.DataFrame({
        "league": ["NBA"] * 4 + ["NFL"] * 2,
        "season": [2024, 2025, 2025, 2025, 2024, 2025],
        "team": ["DAL", "DAL", "DAL", "BOS", "KC", "KC"],
        "kind": ["conference"] * 6,
        "series_ticker": ["KXNBAWEST", "KXNBAWEST", "KXNBAEAST", "KXNBAEAST", "KXAFC", "KXNFLNFCCHAMP"],
        "event_ticker": ["KXNBAWEST-25", "KXNBAWEST-26", "KXNBAEAST-26", "KXNBAEAST-26", "KXAFC-25",
                         "KXNFLNFCCHAMP-25"],
        "open_time": [t(2501, 23), t(2507, 12), t(2507, 12), t(2507, 12), t(2501, 23), t(2503, 28)],
    })
    m = {(r["league"], r["season"], r["team"]): (r["conf"], r["conf_known_at"])
         for r in conference_map(c).iter_rows(named=True)}
    assert m == {
        ("NBA", 2024, "DAL"): ("WEST", t(2501, 23)),          # only what was listed by then
        ("NBA", 2025, "DAL"): ("WEST", t(2507, 12)),          # 2 West listings vs 1 East
        ("NBA", 2025, "BOS"): ("EAST", t(2507, 12)),
        ("NFL", 2024, "KC"): ("AFC", t(2501, 23)),            # the 2025 NFC listing is not used yet
    }                                                          # NFL 2025 KC: one each -> tie, dropped


@pytest.fixture(scope="module")
def mlb_graph():
    from stg.sports.asof import SportsData
    from stg.sports.graph import between_game_graph
    d = SportsData()
    return d, between_game_graph(d, pl.read_parquet("data/sports/derived/edges.parquet"), "MLB", 2025)


@have_data
def test_graph_matches_league_structure(mlb_graph):
    _, g = mlb_graph
    n = dict(g.relations.group_by("relation").len().iter_rows())
    # 6 divisions of 5, 2 leagues of 15, 30 teams; 5 hierarchy links per team
    # MLB wins nodes are left out (EXCLUDE_NODES), so 4 hierarchy links per team, not 5
    assert n == {"rival_division": 6 * 5 * 4, "rival_playoffs": 2 * 15 * 14, "rival_conference": 2 * 15 * 14,
                 "rival_champ": 30 * 29, "hierarchy": 30 * 4}
    assert set(g.nodes["kind"].unique()) == {"division", "playoffs", "conference", "champ"}
    conf = dict(g.nodes.select("node", "conf").iter_rows())
    po = g.relations.filter(pl.col("relation") == "rival_playoffs")
    assert all(conf[s] == conf[d] for s, d in po.select("src", "dst").iter_rows())


@have_data
def test_graph_adjacency_signs(mlb_graph):
    import numpy as np
    _, g = mlb_graph
    a = g.adjacency(dt.datetime(2025, 8, 15, tzinfo=UTC), signed=True)
    for r, m in a.items():
        assert np.all(np.diag(m) == 0)
        assert set(np.unique(m)) <= ({-1.0, 0.0} if r.startswith("rival") else {0.0, 1.0})
        if r.startswith("rival"):
            assert np.array_equal(m, m.T)


@have_data
def test_results_are_never_known_before_the_game_starts():
    from stg.sports.loaders import game_results
    r = game_results()
    assert r.filter(pl.col("known_at") <= pl.col("first_play")).height == 0


@have_data
def test_game_shocks(mlb_graph):
    from stg.sports.graph import game_shocks
    d, _ = mlb_graph
    s = game_shocks(d, "MLB", 2025)
    assert s.filter((pl.col("z") - (pl.col("y") - pl.col("p_pre"))).abs() > 1e-9).height == 0
    per_game = s.group_by("game_id").agg(pl.len().alias("n"), pl.col("z").sum().abs().alias("zsum"))
    assert per_game["n"].max() == 2 and per_game["zsum"].max() < 1e-9
    assert s.filter(pl.col("quote_known_at") > pl.col("first_play")).height == 0
    assert s.filter(pl.col("known_at") <= pl.col("first_play")).height == 0


@have_data
def test_game_shocks_truncation():
    from stg.sports.graph import game_shocks
    r = truncation_test(lambda d: game_shocks(d, "MLB", 2025), CUT, key=["game_id", "team"], label_end="known_at")
    assert r["samples"] > 500


@have_data
def test_pair_attributes_are_asof(mlb_graph):
    from stg.sports.graph import pair_attributes
    d, g = mlb_graph
    t = dt.datetime(2025, 8, 15, 16, tzinfo=UTC)
    pa = pair_attributes(d, g, t)
    assert pa.height == int(g.candidates(t).sum())
    assert_asof(pa)


@have_data
def test_memberships_unknown_before_their_listing(mlb_graph):
    from stg.sports.graph import pair_attributes
    d, g = mlb_graph
    t = dt.datetime(2025, 3, 1, tzinfo=UTC)          # champ / conference live, divisions listed 2025-03-26
    pa = pair_attributes(d, g, t)
    assert pa.height > 0 and pa["same_div"].null_count() == pa.height
    e = pl.read_parquet("data/sports/derived/edges.parquet").filter(
        pl.col("provenance").str.starts_with("playoffs: same conference"))
    from stg.sports.edges import conference_map
    cm = conference_map(d.contracts)
    late = (e.join(d.contracts.select(pl.col("ticker").alias("src"), "league", "season", "team"), on="src")
            .join(cm, on=["league", "season", "team"]).filter(pl.col("known_at") < pl.col("conf_known_at")))
    assert late.height == 0


# --------------------------------------------------------------------------- node values

def test_wins_ladder_fit_recovers_a_normal():
    import numpy as np
    from scipy.stats import norm
    from stg.sports.graph import fit_wins_ladder
    k = np.array([40.0, 45.0, 50.0, 55.0, 60.0])
    p = norm.cdf((47.3 + 0.5 - k) / 6.0)
    mu, sigma = fit_wins_ladder(k, p)
    assert abs(mu - 47.3) < 1e-6 and abs(sigma - 6.0) < 1e-6
    assert np.isnan(fit_wins_ladder(k[:1], p[:1])[0])                 # one leg: no fit
    assert np.isnan(fit_wins_ladder(k, p[::-1])[0])                   # increasing ladder: no fit


def test_wins_threshold():
    from stg.sports.graph import _threshold
    f = pl.DataFrame({"strike": [90.0, 6.5, 74.5], "st": ["greater", "greater", "greater_or_equal"]})
    k = f.select(_threshold(pl.col("strike"), pl.col("st")).alias("k"))["k"].to_list()
    assert k == [91, 7, 75]          # more than 90 -> >= 91; more than 6.5 -> >= 7; at least 74.5 -> >= 75


@have_data
def test_node_values_asof_and_bounds(mlb_graph):
    from stg.sports.graph import EXTREME_C, NODE_MAX_AGE, NODE_MAX_SPREAD, node_values
    d, g = mlb_graph
    days = pl.datetime_range(dt.datetime(2025, 7, 1, 16, tzinfo=UTC), dt.datetime(2025, 7, 10, 16, tzinfo=UTC),
                             "1d", time_zone="UTC", eager=True)
    at = g.nodes.select("node").join(pl.DataFrame({"t": days}), how="cross")
    v = node_values(d, g, at)
    have = v.filter(pl.col("value").is_not_null())
    assert have.height > 0.5 * v.height
    assert_asof(have)
    two = have.filter(~pl.col("one_sided"))
    assert two["spread"].max() <= NODE_MAX_SPREAD and two["quote_age"].max() <= NODE_MAX_AGE
    one = have.filter(pl.col("one_sided"))
    assert one.filter((pl.col("value") > EXTREME_C / 2) & (pl.col("value") < 100 - EXTREME_C / 2)).height == 0


@have_data
def test_node_values_read_the_newest_listing():
    """The NFL wins ladders were relisted (-25 in May, -25B in late August): read only -25B after."""
    import numpy as np
    from stg.sports.asof import SportsData
    from stg.sports.graph import _threshold, between_game_graph, fit_wins_ladder, node_values
    d = SportsData()
    g = between_game_graph(d, pl.read_parquet("data/sports/derived/edges.parquet"), "NFL", 2025)
    t = dt.datetime(2025, 9, 15, 16, tzinfo=UTC)
    v = node_values(d, g, pl.DataFrame({"node": ["BUF-wins"], "t": [t]}))
    legs = (d.contracts.filter(pl.col("event_ticker") == "KXNFLWINS-BUF-25B")
            .with_columns(_threshold(pl.col("strike"), pl.col("strike_type")).alias("k")))
    q = d.quotes_at(legs.select("ticker", pl.lit(t).alias("t")), max_spread=10.0, max_age=dt.timedelta(days=1))
    ok = q.join(legs.select("ticker", "k"), on="ticker").filter(pl.col("mid").is_not_null())
    mu, _ = fit_wins_ladder(ok["k"].to_numpy().astype(float), ok["mid"].to_numpy() / 100)
    assert v["value"][0] is not None
    if ok.height == v["legs"][0]:          # same legs (no one-sided extremes added): same fit
        assert abs(v["value"][0] - mu) < 1e-9
    assert 9 < v["value"][0] < 14          # Buffalo, priced around 11-12 wins in September 2025


@have_data
def test_result_times_come_from_the_leagues_not_kalshi():
    """Kalshi closes markets on its own schedule (13 MLB markets of 2025-04-18 closed on
    2026-03-25; the GB-DAL tie 8 h before kickoff): result times must not follow it."""
    from stg.sports import loaders
    r = loaders.game_results()
    hours = (pl.col("known_at") - pl.col("first_play")).dt.total_minutes() / 60
    bounds = {"MLB": 7.5, "NBA": 4.0, "NFL": 5.0}
    for lg, hi in bounds.items():
        x = r.filter(pl.col("league") == lg)
        assert x.filter(hours <= 0).height == 0, lg
        assert x.filter(hours > hi).height == 0, f"{lg}: implausible game lengths"
    # the 2025-04-18 MLB games are known that day, not in 2026
    apr18 = r.filter(pl.col("league") == "MLB", pl.col("first_play").dt.date() == dt.date(2025, 4, 18))
    assert apr18.height > 0 and apr18["known_at"].max() < dt.datetime(2025, 4, 20, tzinfo=UTC)
    # MLB: final scores agree with statsapi's official ones (incomplete feeds take the official)
    off = loaders.mlb_official_end()
    if off.height:
        m = r.filter(pl.col("league") == "MLB").join(off, on="game_id")
        assert m.filter((pl.col("home_score") != pl.col("official_home"))
                        | (pl.col("away_score") != pl.col("official_away"))).height == 0


def test_result_time_ignores_kalshi_close(monkeypatch):
    """Moving every Kalshi close by a year leaves the result times unchanged."""
    from stg.sports import loaders
    ev = pl.DataFrame({
        "game_id": ["g"] * 3, "seq": [0, 1, 2], "league": ["NBA"] * 3,
        "t": [T0, T0 + dt.timedelta(hours=2), T0 + dt.timedelta(hours=2, seconds=5)],
        "event_type": ["Jumpball", "Jump Shot", "End Game"], "home_score": [0, 100, 100],
        "away_score": [0, 98, 98], "t_outside_window": [False] * 3, "is_pitch": [False] * 3,
        "t_end": [None] * 3,
    })
    for shift in (dt.timedelta(0), dt.timedelta(days=365)):
        e = ev.with_columns(pl.lit(T0 + shift).alias("game_close"), pl.col("t").alias("known_at"))
        monkeypatch.setattr(loaders, "LEAGUES", ("NBA",))
        monkeypatch.setattr(loaders, "load_pbp", lambda lg, e=e: e)
        r = loaders.game_results()
        assert r["known_at"][0] == T0 + dt.timedelta(hours=2, seconds=5) + loaders.RESULT_LAG


@have_data
def test_mlb_2025_standings_are_complete():
    """Kalshi's MLB game markets start 2025-04-16; the league schedule fills in the rest."""
    from stg.sports.asof import SportsData
    from stg.sports.features import team_state
    d = SportsData()
    if d.external_results.is_empty():
        pytest.skip("data/sports/pbp/mlb_schedule.parquet absent: run scripts/fetch_mlb_schedule.py")
    teams = d.contracts.filter(pl.col("league") == "MLB", pl.col("season") == 2025, pl.col("kind") == "division")["team"].unique()
    q = pl.DataFrame({"league": "MLB", "season": 2025, "team": teams,
                      "t": dt.datetime(2025, 9, 30, tzinfo=UTC)})
    s = team_state(d, q)
    assert s["gp"].to_list() == [162] * 30
    assert s["wins"].sum() == s["losses"].sum()
    assert s.filter(pl.col("team") == "NYY").select("wins", "losses").row(0) == (94.0, 68.0)


# --------------------------------------------------------------------------- training samples

def test_decision_time_is_after_the_first_hourly_close():
    from stg.sports.samples import CANDLE_READ, DECISION_DELAY, decision_time
    r = [dt.datetime(2025, 6, 1, 3, 12, tzinfo=UTC), dt.datetime(2025, 6, 1, 3, 57, tzinfo=UTC),
         dt.datetime(2025, 6, 1, 3, 55, tzinfo=UTC)]
    t = pl.DataFrame({"r": r}).select(decision_time(pl.col("r")).alias("t"))["t"].to_list()
    assert t == [dt.datetime(2025, 6, 1, 4, 0, tzinfo=UTC) + CANDLE_READ,      # 03:12 + 5 min -> 04:00 close
                 dt.datetime(2025, 6, 1, 5, 0, tzinfo=UTC) + CANDLE_READ,      # 03:57 + 5 min passes 04:00
                 dt.datetime(2025, 6, 1, 4, 0, tzinfo=UTC) + CANDLE_READ]      # exactly on the hour
    assert all(a >= b + DECISION_DELAY for a, b in zip(t, r))


@have_data
def test_samples_are_asof(mlb_graph):
    from stg.sports.samples import HORIZONS, sample_table
    d, g = mlb_graph
    s = sample_table(d, g)
    assert s.height > 1000
    assert_asof(s)
    assert s.filter(pl.col("t") < pl.col("last_result")).height == 0
    for h in HORIZONS:
        assert s.filter(pl.col(f"label_end_{h}") <= pl.col("t")).height == 0
    lv = s.filter(pl.col("live"), pl.col("kind") != "wins")
    assert lv["value_pre"].min() >= 3 and lv["value_t"].max() <= 97 and not lv["one_sided_t"].any()
    one = s.filter(pl.col("total_1h").is_not_null())          # labels are differences of node values
    assert one.filter((pl.col("total_1h") - pl.col("moved") - pl.col("rem_1h")).abs() > 1e-9).height == 0


def _toy_nights(n_nights: int = 20, per: int = 6, seed: int = 0) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(seed)
    t0 = dt.datetime(2025, 4, 1, 4, tzinfo=dt.timezone.utc)
    rows = []
    for i in range(n_nights):
        t = t0 + dt.timedelta(days=i)
        for j in range(per):
            x = rng.normal()
            rows.append({"night": t.date(), "node": f"n{j}", "t": t, "label_end": t + dt.timedelta(hours=36),
                         "x": x, "y": 2 * x + rng.normal(scale=0.1), "played": j % 2 == 0})
    return pl.DataFrame(rows)


def test_walk_forward_fn_purges_overlapping_labels():
    from stg.sports.walkforward import walk_forward_fn
    d = _toy_nights()
    seen = []

    def fit(train):
        seen.append(train["label_end"].max())
        return lambda rows: rows["x"].to_numpy() * 0
    out = walk_forward_fn(d, "y", "label_end", fit, min_train=1)
    starts = out.group_by("fold").agg(pl.col("t").min()).sort("fold")["t"].to_list()
    assert len(seen) == len(starts) > 0
    assert all(e < s for e, s in zip(seen, starts))          # no training label ends inside the test fold


def test_scoring_perfect_and_paired():
    from stg.sports.scoring import score
    d = _toy_nights()
    good = d.with_columns((2 * pl.col("x")).alias("pred"))
    zero = d.with_columns(pl.lit(0.0).alias("pred"))
    s = score(good, "y", base=zero)
    assert s["r2"] > 0.99 and s["r2_quiet"] > 0.99 and abs(s["calib"] - 1) < 0.05 and s["ic"] > 0.9
    assert s["gain_vs_base"] > 0.99 and s["t_vs_base"] > 5
    assert score(zero, "y", base=good)["t_vs_base"] < -5
