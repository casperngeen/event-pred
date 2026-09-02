import numpy as np

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
