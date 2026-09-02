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
    parse_threshold, recover_pdf, resolved_value,
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
