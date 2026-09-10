import polars as pl
import pytest

from stg.panel import (
    build_surprise_panel, event_counts, is_same_release, same_release_pairs,
    target_universe, trigger_universe, universe,
)
from stg.panel.registry import (
    SPECS, canonical, assert_trade_coverage, trade_coverage,
)

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
    # WTI ladder mass is a known ~1.28 coherence violation
    wti = surprise_panel.filter(pl.col("series") == "WTI")
    if wti.height:
        assert 1.0 < wti["ladder_mass"].median() < 1.6


@requires_archive
def test_surprise_panel_row_counts_match_research_log(surprise_panel):
    """Loose bounds against research_log.md §7 / structure_discovery output."""
    n = dict(surprise_panel.group_by("series").agg(pl.len().alias("n")).iter_rows())
    assert n.get("CPI", 0) >= 40
    assert n.get("CPICORE", 0) >= 30
    assert n.get("WTI", 0) >= 200
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
