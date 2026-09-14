import numpy as np
import pytest

from stg.structure.stats import (
    benjamini_hochberg, bh_critical, partial_spearman, permutation_p, spearman,
    spearman_p,
)


def test_spearman_monotone_recovery():
    x = np.arange(30.0)
    y = x ** 2  # monotone increasing
    assert spearman(x, y) == 1.0
    assert spearman(x, -y) == -1.0


def test_spearman_p_small_for_strong_edge():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = x + rng.normal(scale=0.3, size=40)
    r = spearman(x, y)
    assert r > 0.7
    assert spearman_p(r, 40) < 0.001


def test_permutation_p_recovers_planted_edge():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    y = 0.8 * x + rng.normal(scale=0.5, size=50)
    assert permutation_p(x, y, n_perm=1000, rng=rng) < 0.01
    # and is ~uniform-high for pure noise
    z = rng.normal(size=50)
    assert permutation_p(x, z, n_perm=1000, rng=rng) > 0.10


def test_partial_spearman_collapses_a_mediated_edge():
    rng = np.random.default_rng(2)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.2, size=200)      # B carries A
    c = b + rng.normal(scale=0.2, size=200)      # C driven by B only
    assert spearman(a, c) > 0.8
    assert abs(partial_spearman(a, b, c)) < 0.25   # collapses once B controlled


def test_bh_monotone_and_order_independent():
    p = np.array([0.001, 0.2, 0.02, 0.8, 0.009])
    m = benjamini_hochberg(p, q=0.10)
    # survivors must be a prefix in p-order
    order = np.argsort(p)
    surv_sorted = m[order]
    assert list(surv_sorted) == sorted(surv_sorted, reverse=True)
    crit = bh_critical(p, 0.10)
    assert np.all(np.diff(crit[order]) >= 0)


def test_bh_empty_and_all_null():
    assert not benjamini_hochberg(np.array([np.nan, np.nan])).any()
    assert benjamini_hochberg(np.array([1e-9, 1e-9, 1e-9]), 0.1).all()


# --- tie-corrected ranks and exact p-values -------------------------------
def test_spearman_averages_ties():
    """response is a difference of integer cents — 98% of it is tied."""
    from scipy import stats as sps
    from stg.structure.stats import spearman
    rng = np.random.default_rng(3)
    for n in (10, 15, 25, 46):
        a = rng.normal(size=n).round(1)
        b = rng.integers(-3, 4, n).astype(float)   # heavy ties by construction
        assert spearman(a, b) == pytest.approx(sps.spearmanr(a, b).statistic)


def test_spearman_p_is_exact_t_not_normal():
    """The normal approximation was anti-conservative at these n."""
    from scipy import stats as sps
    from stg.structure.stats import spearman, spearman_p
    rng = np.random.default_rng(4)
    for n in (10, 12, 19, 34):
        a = rng.normal(size=n)
        b = a + rng.normal(size=n)
        rho = spearman(a, b)
        assert spearman_p(rho, n) == pytest.approx(sps.spearmanr(a, b).pvalue)
    # the specific regression: rho=0.6 at n=12 was reported ~2x too small
    assert spearman_p(0.6, 12) > 2 * 0.0177


def test_permutation_p_has_a_floor():
    """p = 0.0 is not a possible estimate from a finite number of draws."""
    from stg.structure.stats import permutation_p
    p = permutation_p(np.arange(20.0), np.arange(20.0), n_perm=2000)
    assert p == pytest.approx(1 / 2001)
    assert p > 0


def test_permutation_p_does_not_depend_on_call_order():
    """A shared module RNG made every p depend on how many ran before it."""
    from stg.structure.stats import permutation_p
    a = np.arange(15.0)
    b = np.roll(a, 4)
    first = permutation_p(a, b, n_perm=500)
    for _ in range(3):                      # burn draws off any shared stream
        permutation_p(np.arange(10.0), np.arange(10.0)[::-1], n_perm=500)
    assert permutation_p(a, b, n_perm=500) == first


def test_bh_critical_gives_tied_pvalues_the_same_threshold():
    from stg.structure.stats import bh_critical, benjamini_hochberg
    p = np.array([0.001, 0.005, 0.005, 0.02, 0.3])
    crit = bh_critical(p, 0.10)
    assert crit[1] == crit[2], "identical p-values must get identical thresholds"
    assert list(benjamini_hochberg(p, 0.10)) == list(p <= crit)


def test_block_permutation_p_has_a_floor():
    from stg.structure.stats import block_permutation_sign_p
    cells = [("CPI", np.arange(1.0, 21.0), np.arange(1.0, 21.0))]
    out = block_permutation_sign_p(cells, n_perm=500)
    assert out["p"] >= 1 / 501
