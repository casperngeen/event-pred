"""Unit tests for reconstruction-free ladder statistics."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stg.events.quantile import (
    crossing, isotonic_decreasing, ladder_pit, prob_at, quantile_moments,
)


def test_isotonic_leaves_monotone_untouched():
    y = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    assert np.allclose(isotonic_decreasing(y), y)


def test_isotonic_pools_only_the_violating_run():
    # one bad print in the middle; the tail must NOT be dragged down the way a
    # running-minimum envelope would drag it.
    y = np.array([0.9, 0.4, 0.6, 0.3, 0.1])
    out = isotonic_decreasing(y)
    assert np.all(np.diff(out) <= 1e-12)
    assert out[0] == pytest.approx(0.9)
    assert out[-1] == pytest.approx(0.1)
    assert out[1] == pytest.approx(0.5)      # 0.4 and 0.6 pooled
    assert out[2] == pytest.approx(0.5)


def test_crossing_interpolates_linearly():
    k = np.array([0.0, 1.0, 2.0])
    p = np.array([0.8, 0.4, 0.2])
    # 0.5 sits between 0.8 and 0.4 -> 3/4 of the way from 0 to 1
    assert crossing(k, p, 0.5) == pytest.approx(0.75)


def test_crossing_returns_none_when_not_bracketed():
    k = np.array([0.0, 1.0, 2.0])
    p = np.array([0.4, 0.3, 0.2])
    assert crossing(k, p, 0.5) is None       # level above the highest price
    assert crossing(k, p, 0.1) is None       # level below the lowest price


def test_crossing_handles_a_flat_run_at_the_level():
    k = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([0.8, 0.5, 0.5, 0.2])
    assert crossing(k, p, 0.5) == pytest.approx(1.5)


def test_iqr_is_immune_to_a_missing_wing_leg():
    k = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    p = np.array([0.95, 0.80, 0.50, 0.20, 0.05])
    full = quantile_moments(k, p)
    trimmed = quantile_moments(k[1:-1], p[1:-1])      # drop both 5% wings
    assert full["iqr"] == pytest.approx(trimmed["iqr"])
    assert full["q50"] == pytest.approx(trimmed["q50"])


def test_pit_is_uniform_for_a_calibrated_ladder():
    # ladder IS the true cdf; outcomes drawn from it must give uniform PIT
    from math import erf
    k = np.linspace(-3, 3, 25)
    p = 1.0 - np.array([0.5 * (1 + erf(x / np.sqrt(2))) for x in k])
    rng = np.random.default_rng(0)
    pits = []
    for v in rng.normal(size=400):
        if not (-3 < v < 3):
            continue
        pit, censor = ladder_pit(k, p, float(v))
        if censor is None:
            pits.append(pit)
    a = np.array(pits)
    assert abs(a.mean() - 0.5) < 0.05
    assert abs(np.median(a) - 0.5) < 0.06


def test_pit_flags_censoring_outside_the_traded_range():
    k = np.array([0.0, 1.0, 2.0])
    p = np.array([0.8, 0.5, 0.2])
    _, c_lo = ladder_pit(k, p, -5.0)
    _, c_hi = ladder_pit(k, p, 5.0)
    _, c_in = ladder_pit(k, p, 1.5)
    assert c_lo == "left" and c_hi == "right" and c_in is None


def test_too_few_legs_returns_none():
    assert crossing([0.0, 1.0], [0.6, 0.4], 0.5) is None
    assert prob_at([0.0, 1.0], [0.6, 0.4], 0.5) == (None, None)


def test_bowley_skew_is_zero_for_a_symmetric_ladder():
    k = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    p = np.array([0.95, 0.75, 0.50, 0.25, 0.05])
    assert quantile_moments(k, p)["skew_q"] == pytest.approx(0.0, abs=1e-9)
