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


# --------------------------------------------------------------------------- result times

@have_data
def test_results_are_never_known_before_the_game_starts():
    from stg.sports.loaders import game_results
    r = game_results()
    assert r.filter(pl.col("known_at") <= pl.col("first_play")).height == 0


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
