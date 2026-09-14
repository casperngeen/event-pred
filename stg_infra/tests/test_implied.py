"""Coverage guards for the threshold/bucket parsers.

Every parsing bug fixed in August failed *silently* (parse_threshold returning
None for a whole series). These assertions are the cheap insurance the TODO
asked for: if a parser regresses, coverage drops below 100% and CI is red.
"""

import numpy as np
import polars as pl
import pytest

from stg.events.implied import (
    BUCKET, THRESHOLD, classify_contract, infer_spacing, parse_bucket,
    parse_threshold, pit, pmf_bin_index, recover_pdf, resolved_value,
    surprisal,
)
from stg.panel.registry import SPECS, series_filter_expr, universe

from conftest import requires_archive

THRESHOLD_SERIES = [c for c, s in SPECS.items() if s.kind == "threshold"]


def test_parse_threshold_rejects_categorical_and_bucket():
    assert parse_threshold("FEDDECISION-24JUL-C25", "Cut 25bps") is None
    assert parse_threshold("KXWTI-25JAN-B67.5", "$67 to 67.99") is None
    assert parse_threshold("CPI-24JAN-T3.2", "Above 3.2%") == 3.2


def test_infer_spacing_is_per_ladder():
    assert infer_spacing(np.array([3.1, 3.2, 3.3])) == pytest.approx(0.1)
    assert infer_spacing(np.array([200_000.0, 225_000.0, 250_000.0])) == pytest.approx(25_000)


def test_recover_pdf_sums_to_one():
    mids, probs = recover_pdf(np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.5, 0.2]))
    assert probs.sum() == pytest.approx(1.0)
    assert (probs >= 0).all()


@requires_archive
@pytest.mark.parametrize("canon", THRESHOLD_SERIES)
def test_threshold_parse_coverage_is_total(markets, canon):
    """Every contract that classifies as THRESHOLD must yield a numeric strike."""
    sub = markets.filter(series_filter_expr(canon)).select("ticker", "yes_sub_title")
    if sub.height == 0:
        pytest.skip(f"{canon}: no contracts in archive")
    misses = [
        (t, s) for t, s in zip(sub["ticker"], sub["yes_sub_title"])
        if classify_contract(t, s) == THRESHOLD and parse_threshold(t, s) is None
    ]
    assert not misses, f"{canon}: {len(misses)} THRESHOLD contracts parsed to None, e.g. {misses[:3]}"


@requires_archive
def test_bucket_parse_coverage_for_wti(markets):
    sub = markets.filter(series_filter_expr("WTI")).select("ticker", "yes_sub_title")
    misses = [
        (t, s) for t, s in zip(sub["ticker"], sub["yes_sub_title"])
        if classify_contract(t, s) == BUCKET and parse_bucket(t, s) is None
    ]
    assert not misses, f"WTI: {len(misses)} BUCKET contracts parsed to None"


# --------------------------------------------------------------------------
# PIT / surprisal — the distribution-relative surprise measures (§1.4, §1.5)
# --------------------------------------------------------------------------
def test_pmf_bin_index_recovers_the_original_thresholds():
    """Edges are reconstructed from midpoints, so they must land on the strikes."""
    mids, _ = recover_pdf(np.array([3.0, 3.1, 3.2]), np.array([0.8, 0.5, 0.2]))
    assert pmf_bin_index(mids, 2.80) == 0     # below the lowest strike
    assert pmf_bin_index(mids, 3.05) == 1     # (3.0, 3.1]
    assert pmf_bin_index(mids, 3.15) == 2     # (3.1, 3.2]
    assert pmf_bin_index(mids, 3.40) == 3     # above the highest strike


def test_mid_pit_is_half_at_the_centre_of_a_symmetric_ladder():
    mids, probs = recover_pdf(np.array([3.0, 3.1]), np.array([0.75, 0.25]))
    # mass 0.25 / 0.50 / 0.25 -> the middle bin's mid-PIT is 0.25 + 0.5*0.5
    assert pit(mids, probs, 3.05) == pytest.approx(0.5)


def test_pit_is_monotone_and_bounded():
    mids, probs = recover_pdf(np.array([3.0, 3.1, 3.2]), np.array([0.8, 0.5, 0.2]))
    us = [pit(mids, probs, v) for v in (2.8, 3.05, 3.15, 3.4)]
    assert us == sorted(us)
    assert all(0.0 <= u <= 1.0 for u in us)


def test_surprisal_is_larger_for_the_less_likely_outcome():
    mids, probs = recover_pdf(np.array([3.0, 3.1, 3.2]), np.array([0.95, 0.90, 0.05]))
    # nearly all mass sits in (3.1, 3.2]; a print below 3.0 is the surprise
    assert surprisal(mids, probs, 2.8) > surprisal(mids, probs, 3.15)
    assert surprisal(mids, probs, 3.15) == pytest.approx(-np.log(probs[2]))


def test_surprisal_floors_a_zero_mass_outcome():
    mids = np.array([0.0, 1.0, 2.0])
    probs = np.array([0.5, 0.0, 0.5])
    assert np.isfinite(surprisal(mids, probs, 1.0))


@requires_archive
def test_surprise_panel_carries_the_pit_columns(surprise_panel):
    for c in ("pit", "s_pit", "surprisal", "implied_median", "surprise_median"):
        assert c in surprise_panel.columns, f"{c} missing from the surprise panel"
    u = surprise_panel["pit"].drop_nulls().to_numpy()
    assert u.size == surprise_panel.height
    assert ((u >= 0.0) & (u <= 1.0)).all()
    s = surprise_panel["s_pit"].to_numpy()
    assert ((s >= -1.0) & (s <= 1.0)).all()
    assert (surprise_panel["surprisal"].to_numpy() >= 0).all()
