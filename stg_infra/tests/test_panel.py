import polars as pl
import pytest

from stg.panel import (
    build_surprise_panel, event_counts, is_same_release, same_release_pairs,
    universe,
)
from stg.panel.registry import SPECS, canonical

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
