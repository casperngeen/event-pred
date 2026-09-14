import polars as pl
import pytest

from stg.panel import (
    build_surprise_panel, event_counts, is_same_release, same_release_pairs,
    target_universe, trigger_universe, universe,
)
from stg.panel.registry import (
    SPECS, canonical, assert_trade_coverage, trade_coverage,
)
from stg.panel.surprise import MAX_MASS, MIN_COVERAGE, MIN_MASS

from conftest import requires_archive


# --- registry (no data needed) -------------------------------------------
def test_alias_merge_before_filter():
    assert canonical("PROLLS") == "PAYROLLS"
    assert canonical("JOBLESS") == "JOBLESSCLAIMS"
    assert canonical("CPI") == "CPI"


def test_same_release_is_symmetric_and_curated():
    assert is_same_release("CPI", "CPIYOY")
    assert is_same_release("CPIYOY", "CPI")
    assert is_same_release("PAYROLLS", "U3")
    assert not is_same_release("CPI", "WTI")
    for pair in same_release_pairs():
        a, b = tuple(pair)
        assert SPECS[a].same_release == SPECS[b].same_release


# --- data-backed --------------------------------------------------------
@requires_archive
def test_universe_is_derived_from_min_events(markets):
    assert set(universe(1000, markets)) <= set(universe(5, markets))
    n5 = len(universe(5, markets))
    assert 15 <= n5 <= 30, f"unexpected universe size {n5}"


@requires_archive
def test_event_counts_cover_universe(markets):
    c = event_counts(markets)
    in_u = c.filter(pl.col("in_universe"))
    assert in_u.height == len(universe(5, markets))
    assert (in_u["n_events"] >= 5).all()


@requires_archive
def test_surprise_panel_is_in_sample_and_sane(surprise_panel):
    from stg.splits import assert_no_oos
    assert_no_oos(surprise_panel, time_col="close_time")
    # surprise = resolved - implied_mean, must be finite and not absurd
    assert surprise_panel["surprise"].is_finite().all()
    # WTI's ladder mass is a known coherence violation (median ~1.2 after the
    # two-sided gate, ~1.26 before it). The gate is what bounds it now: an
    # ungated panel reached 2.96, which renormalisation cannot rescue.
    wti = surprise_panel.filter(pl.col("series") == "WTI")
    if wti.height:
        assert 1.0 < wti["ladder_mass"].median() < 1.6
    assert surprise_panel["ladder_mass"].is_between(MIN_MASS, MAX_MASS).all()
    assert (surprise_panel["coverage"] >= MIN_COVERAGE).all()


@requires_archive
def test_surprise_panel_row_counts_match_research_log(surprise_panel):
    """Loose bounds against research_log.md §7 / structure_discovery output.

    Counts are post-gate. The quality gates took the panel from 799 rows to
    519 — WTI 368 -> 197 is the bulk of it, since the bucket path is where the
    mass violations live. The bounds below are the *gated* floors; they moved
    down deliberately, and no trigger series was lost (14 clear n>=10 before
    and after), which is the property worth pinning.
    """
    n = dict(surprise_panel.group_by("series").agg(pl.len().alias("n")).iter_rows())
    assert n.get("CPI", 0) >= 40
    assert n.get("CPICORE", 0) >= 30
    assert n.get("WTI", 0) >= 190
    assert "FEDDECISION" not in n  # categorical, target only


# --- trigger / target universes are separate ------------------------------
def test_asset_targets_can_never_be_triggers():
    """``can_trigger`` is structural, so no gate setting exposes them."""
    assets = [c for c, s in SPECS.items() if not s.can_trigger]
    assert assets, "expected asset-price targets in the registry"
    for c in assets:
        assert SPECS[c].role == "target_only"
        # a target needs a price path, not an information event
        assert not SPECS[c].scheduled_release


def test_scheduled_release_annotates_the_wti_finding():
    """edge_economics.md §2(b): a settle is not an information release."""
    assert not SPECS["WTI"].scheduled_release
    assert not SPECS["WTIW"].scheduled_release
    # ...but WTI stays trigger-eligible until the open decision is made
    assert SPECS["WTI"].can_trigger


@requires_archive
def test_trigger_universe_excludes_asset_targets(markets):
    trigs = set(trigger_universe(5, markets))
    tgts = set(target_universe(5, markets))
    assets = {c for c, s in SPECS.items() if not s.can_trigger}
    assert trigs.isdisjoint(assets)
    assert assets <= tgts, "asset targets must still be usable as targets"
    assert trigs < tgts, "triggers are a strict subset of targets"


@requires_archive
def test_require_release_gate_drops_only_wti(markets):
    base = set(trigger_universe(5, markets))
    gated = set(trigger_universe(5, markets, require_release=True))
    assert base - gated == {"WTI", "WTIW"}


@requires_archive
def test_universe_alias_matches_target_universe(markets):
    assert universe(5, markets) == target_universe(5, markets)


@requires_archive
def test_event_counts_order_is_deterministic(markets):
    """Ties on n_events broke ordering run-to-run before the canon tiebreak."""
    runs = [target_universe(5, markets) for _ in range(4)]
    assert all(r == runs[0] for r in runs)


# --- trade coverage: metadata volume is not a price path ------------------
@requires_archive
def test_every_registered_series_has_traded_events(markets):
    """The guard itself: no registered series may lack a price path."""
    assert_trade_coverage(min_events_traded=5, markets=markets)


@requires_archive
def test_trade_coverage_flags_a_metadata_only_series(markets):
    """NASDAQ100D is listed with real aggregate volume but has zero trades.

    It is deliberately absent from SPECS; this asserts the guard would have
    caught it, so the three near-misses of 2026-09-08 cannot recur silently.
    """
    import polars as pl
    from stg.panel._io import scan_trades
    tr = scan_trades(is_only=True)
    traded = (tr.select("ticker")
              .with_columns(pl.col("ticker").str.replace(r"^KX", "")
                            .str.split("-").list.first().alias("s"))
              .filter(pl.col("s").is_in(["NASDAQ100D", "TNOTED", "USDJPYH"]))
              .select(pl.len()).collect().item())
    assert traded == 0, "these three are the metadata-only cases; expected no trades"
    listed = markets.filter(
        pl.col("series_raw").is_in(["NASDAQ100D", "TNOTED", "USDJPYH"])).height
    assert listed > 0, "they should still be present in the markets metadata"


@requires_archive
def test_registered_asset_targets_actually_trade(markets):
    cov = trade_coverage(markets)
    assets = cov.filter(~pl.col("can_trigger"))
    assert assets.height == 3
    assert (assets["n_events_traded"] >= 300).all(), assets


# --- true settlement values ----------------------------------------------
def test_parse_settlement_strips_formatting():
    """Kalshi writes the same number five ways across a ladder."""
    from stg.panel._io import _parse_settlement
    assert _parse_settlement("0.5%") == 0.5
    assert _parse_settlement(".9%") == 0.9
    assert _parse_settlement("224,000") == 224000.0
    assert _parse_settlement("$68.10") == 68.1
    assert _parse_settlement("0.00") == 0.0


def test_parse_settlement_rejects_categorical():
    """FEDDECISION settles to prose, which is not a point on a ladder."""
    from stg.panel._io import _parse_settlement
    for v in ("Hike 25bps", "No hike or cut", "Fed maintains rate", "", None):
        assert _parse_settlement(v) is None


def test_resolved_prefers_true_value_and_records_source():
    from stg.panel.surprise import _resolved

    def boom():  # must not be reached when the true value is present
        raise AssertionError("fallback called despite a true value")

    assert _resolved("CPI-24JAN", {"CPI-24JAN": 0.3}, boom) == (0.3, "expiration_value")
    assert _resolved("CPI-24FEB", {}, lambda: 0.25) == (0.25, "ladder")
    assert _resolved("CPI-24MAR", {}, lambda: None) == (None, None)


@requires_archive
def test_settlement_values_respect_the_oos_wall():
    """The re-pull spans 2026; its result fields must not cross the wall."""
    from stg.panel._io import load_settlement_values, markets_glob
    from stg.splits import OOS_START
    sv = load_settlement_values(is_only=True)
    assert sv.height > 500, f"expected the archive's ~847 IS events, got {sv.height}"
    assert sv["event_ticker"].n_unique() == sv.height, "one row per event"
    closes = (pl.scan_parquet(markets_glob()).select("event_ticker", "close_time")
              .group_by("event_ticker").agg(pl.col("close_time").min())
              .filter(pl.col("event_ticker").is_in(sv["event_ticker"].to_list()))
              .collect())
    assert (closes["close_time"] < OOS_START).all()
    assert load_settlement_values(is_only=False).height > sv.height


@requires_archive
def test_surprise_panel_mostly_uses_the_true_value(surprise_panel):
    """The ladder fallback should be the exception, not the rule."""
    src = surprise_panel["resolved_source"].value_counts().to_dicts()
    frac = (surprise_panel["resolved_source"] == "expiration_value").mean()
    assert frac > 0.75, f"true-value coverage fell to {frac:.1%}: {src}"


# --- quality gates -------------------------------------------------------
def _fake_panel(masses, coverages=None, legs=None) -> pl.DataFrame:
    n = len(masses)
    return pl.DataFrame({
        "series": ["WTI"] * n,
        "ladder_mass": masses,
        "coverage": coverages or [1.0] * n,
        "n_legs": legs or [5] * n,
        "resolved_source": ["expiration_value"] * n,
    })


def test_mass_gate_is_two_sided():
    """The old MIN_MASS floor let a ladder summing to 2.96 through."""
    from stg.panel.surprise import gate_panel
    out = gate_panel(_fake_panel([0.4, 0.75, 1.0, 1.26, 1.49, 1.85, 2.96]),
                min_mass=0.7, max_mass=1.5, min_coverage=0.0, min_legs=3)
    assert sorted(out["ladder_mass"].to_list()) == [0.75, 1.0, 1.26, 1.49]


def test_coverage_and_legs_gates_bite():
    from stg.panel.surprise import gate_panel
    out = gate_panel(_fake_panel([1.0] * 4, coverages=[0.23, 0.5, 0.625, 1.0]),
                min_mass=0.7, max_mass=1.5, min_coverage=0.5, min_legs=3)
    assert out.height == 3, "coverage 0.23 should be dropped"
    out = gate_panel(_fake_panel([1.0] * 3, legs=[2, 3, 4]),
                min_mass=0.7, max_mass=1.5, min_coverage=0.0, min_legs=3)
    assert out["n_legs"].to_list() == [3, 4]


def test_gate_report_accounts_for_every_row():
    from stg.panel.surprise import gate_panel, gate_report
    panel = _fake_panel([0.4, 1.0, 1.26, 2.96], coverages=[1.0, 0.3, 1.0, 1.0])
    rep = gate_report(panel, min_mass=0.7, max_mass=1.5, min_coverage=0.5,
                      min_legs=3).row(0, named=True)
    assert rep["n"] == 4
    assert rep["n_kept"] == 1
    assert rep["dropped_mass"] == 2      # 0.4 and 2.96
    assert rep["dropped_coverage"] == 1  # the 0.3
    assert rep["n_kept"] == gate_panel(panel, min_mass=0.7, max_mass=1.5,
                                  min_coverage=0.5, min_legs=3).height


@requires_archive
def test_gates_cost_rows_but_no_trigger_series(markets):
    """A gate that silently removed a node would change the graph, not clean it."""
    from stg.panel.surprise import gate_panel, usable_triggers
    ungated = build_surprise_panel(["CPI", "CPICORE", "U3"], markets=markets,
                                   gated=False)
    gated = gate_panel(ungated, min_mass=0.7, max_mass=1.5, min_coverage=0.5,
                  min_legs=3)
    assert gated.height < ungated.height, "expected the gates to bite at all"
    assert set(gated["series"]) == set(ungated["series"])
    assert usable_triggers(gated, 10) == usable_triggers(ungated, 10)


# --- node panel freshness ------------------------------------------------
@requires_archive
def test_node_panel_fresh_rule_matches_the_surprise_panel(markets):
    """Both panels must apply one staleness rule, not two opposite ones."""
    from stg.panel.nodes import _threshold_daily
    from stg.panel._io import scan_trades
    tr = scan_trades(is_only=True).collect()

    fresh = _threshold_daily("CPI", markets, tr, freshness="fresh")
    filled = _threshold_daily("CPI", markets, tr, freshness="filled")

    assert fresh.height < filled.height, "forward-fill should add rows"
    # every retained day carried a real cross-section
    assert (fresh["n_fresh_legs"] >= 3).all()
    # the old path let three quarters of rows through with a stale leg
    assert (filled["max_stale_days"] > 0).mean() > 0.5
    assert set(fresh.columns) == set(filled.columns)


@requires_archive
def test_node_panel_rejects_unknown_freshness(markets):
    from stg.panel.nodes import _threshold_daily
    from stg.panel._io import scan_trades
    tr = scan_trades(is_only=True).collect()
    with pytest.raises(ValueError, match="freshness"):
        _threshold_daily("CPI", markets, tr, freshness="ffill")


@requires_archive
def test_bucket_snapshot_day_is_deterministic(markets):
    """Identical rebuilds must produce an identical panel.

    ``_bucket_surprise`` picked its snapshot day with ``max(day_px, key=len)``,
    so ties were broken by dict insertion order — which follows the order
    polars happens to emit groups in. Three builds of WTI gave three different
    ``ladder_mass`` totals and 197 vs 199 rows surviving the gates. Third
    instance of this failure mode in the codebase, after
    ``registry.event_counts`` and ``targets.representative_tickers``.
    """
    from stg.panel._io import load_settlement_values, scan_trades
    tr = scan_trades(is_only=True)
    sv = load_settlement_values(is_only=True)
    runs = [build_surprise_panel(["WTIW"], markets=markets, trades=tr,
                                 settlements=sv, gated=False) for _ in range(3)]
    assert runs[0].height > 0
    for r in runs[1:]:
        assert r.equals(runs[0])


# --- causal representative-leg selection ---------------------------------
@requires_archive
def test_representative_leg_uses_only_pre_trigger_prints(markets):
    """The instrument must not be chosen with trades from after the decision."""
    from stg.panel._io import scan_trades
    from stg.panel.targets import response_panel, target_frames
    tr = scan_trades(is_only=True)
    sp = pl.read_parquet("artifacts/panels/surprise_panel.parquet") \
        if __import__("pathlib").Path(
            "artifacts/panels/surprise_panel.parquet").exists() else None
    if sp is None:
        pytest.skip("surprise panel not built")
    fr = target_frames("FED", "any", markets=markets, trades=tr)
    rp = response_panel(sp.filter(pl.col("series") == "CPI"), "FED", "any",
                        markets=markets, trades=tr, frames=fr)
    assert rp.height > 0
    # every chosen leg had at least one print before the trigger resolved,
    # which is both the p0 requirement and the selection criterion
    assert (rp["n_pre"] > 0).all()
    assert (rp["t_p0"] <= rp["t0"]).all() if "t0" in rp.columns else True


@requires_archive
def test_representative_tickers_as_of_restricts_the_count(markets):
    """``as_of`` is what makes the choice causal; without it the count is
    over the event's whole life."""
    import datetime as dt
    from stg.panel._io import scan_trades
    from stg.panel.targets import representative_tickers
    tr = scan_trades(is_only=True)
    early = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    full = representative_tickers("CPI", markets=markets, trades=tr)
    cut = representative_tickers("CPI", as_of=early, markets=markets, trades=tr)
    assert cut.height < full.height, "an earlier cutoff must see fewer events"
    joined = full.join(cut, on="event_ticker", suffix="_cut")
    assert (joined["n_trades_cut"] <= joined["n_trades"]).all()


@requires_archive
def test_target_legs_returns_every_traded_leg(markets):
    from stg.panel._io import scan_trades
    from stg.panel.targets import representative_tickers, target_legs
    tr = scan_trades(is_only=True)
    legs = target_legs("CPI", markets=markets, trades=tr)
    reps = representative_tickers("CPI", markets=markets, trades=tr)
    assert legs.height > reps.height, "a ladder has more legs than events"
    assert set(reps["event_ticker"]) == set(legs["event_ticker"])
    assert (legs["n_trades"] > 0).all()


# --- node panel: clearance and date-based windows ------------------------
@requires_archive
def test_clearance_days_is_actually_applied(markets):
    """It was disabled by a literal ``if False``, so nodes appeared on their
    own resolution day (days_to_close p05 was 0)."""
    from stg.panel import build_node_panel
    p = build_node_panel(["CPI", "CPICORE"], cadence="daily", clearance_days=16,
                         markets=markets)
    if p.is_empty():
        pytest.skip("no node rows for these series")
    assert (p["days_to_close"] >= 16).all()
    loose = build_node_panel(["CPI", "CPICORE"], cadence="daily",
                             clearance_days=0, markets=markets)
    assert loose.height > p.height
    assert int(loose["days_to_close"].min()) < 16


def test_expanding_z_scores_do_not_see_the_future():
    """``mean().over(event)`` standardised day 3 against day 40."""
    import datetime as dt
    from stg.io.kalshi import KalshiOHLCV
    base = pl.DataFrame({
        "event_ticker": ["E"] * 6,
        "date": [dt.date(2024, 1, d) for d in range(1, 7)],
        "close": [10.0, 12, 14, 16, 18, 20], "volume": [1] * 6,
        "ticker": ["t"] * 6,
        "close_time": [dt.datetime(2024, 2, 1)] * 6, "result": ["yes"] * 6})
    a = KalshiOHLCV.aggregate_to_event_level(base).sort("date")
    perturbed = base.with_columns(
        pl.when(pl.col("date") == dt.date(2024, 1, 6)).then(999.0)
        .otherwise(pl.col("close")).alias("close"))
    b = KalshiOHLCV.aggregate_to_event_level(perturbed).sort("date")
    for col in ("implied_mean_norm", "log_volume_norm", "price_spread_norm"):
        assert a[col].to_list()[:5] == pytest.approx(b[col].to_list()[:5]), col
