"""Rank statistics and multiplicity control for the structure estimator.

Rank/sign based throughout: the CPI/CPIYOY consistency check (research_log.md
§2) showed the signal is in the *direction* of a surprise, not its magnitude,
so Pearson dilutes it toward zero.

Ties
----
``response`` is a difference of integer cent prices, so it takes only ~103
distinct values over 5,315 panel rows — **98.1% of the mass is tied** (and
``surprise`` is 86.2% tied). Ranking therefore has to average ties: the old
``argsort(argsort(x))`` handed arbitrary distinct ranks to equal values, which
moved individual edge weights by up to |Δρ| = 0.238 against a tie-corrected
Spearman (``JOBLESSCLAIMS->FED``: 0.329 vs 0.090).

P-values
--------
Two are available per pair and they are not interchangeable:

``spearman_p``
    the t-approximation. Exact-t now, not normal — the old version evaluated
    the t statistic against a standard normal, which is anti-conservative at
    the n = 10-46 this grid runs on (``CPIYOY->FEDDECISION/cut``: 0.014
    reported, 0.062 actual). Still only an *approximation*: under this much
    tying its distributional assumption is not met, so treat it as a screen.

``permutation_p``
    exact by construction. A permutation test stays valid with a non-standard
    statistic, ties included, because the null is generated with the same
    statistic. **This is the one to select on.** Applying BH to the asymptotic
    p instead cost the published edge table six of its eight survivors; see
    ``estimate_adjacency(select_on=...)``.

Randomness
----------
Every permutation routine takes an explicit ``seed``. The module previously
held one shared ``default_rng(0)`` that all callers drew from in sequence, so
adding a pair — or a ladder rung — silently changed every p-value computed
after it.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata, t as _t_dist

DEFAULT_SEED = 0


def _rng(rng, seed: int):
    """A generator that does not depend on how many callers ran before us."""
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _rank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged — required, see the module docstring."""
    return rankdata(np.asarray(x, float)).astype(float)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def spearman_p(rho: float, n: int) -> float:
    """Two-sided p from the t-approximation on ranks, with ``n - 2`` df.

    Matches ``scipy.stats.spearmanr``. A *screen*, not the test of record --
    see the module docstring on ties. Prefer :func:`permutation_p` for
    anything that gets selected on or reported.
    """
    if not np.isfinite(rho) or abs(rho) >= 1 or n < 4:
        return float("nan")
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    return float(min(max(2.0 * _t_dist.sf(abs(t), n - 2), 0.0), 1.0))


def permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int = 2000,
                  rng: np.random.Generator | None = None,
                  seed: int = DEFAULT_SEED) -> float:
    """Two-sided permutation p: shuffle ``a`` against fixed ``b``.

    Uses the ``(1 + k) / (1 + n_perm)`` estimator, so the floor is
    ``1 / (1 + n_perm)`` rather than zero. A reported ``p = 0.0000`` is not a
    possible estimate from a finite number of draws, and it breaks BH and any
    log-scale plot downstream.
    """
    rng = _rng(rng, seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    obs = spearman(a, b)
    if not np.isfinite(obs):
        return float("nan")
    null = np.array([spearman(rng.permutation(a), b) for _ in range(n_perm)])
    k = int((np.abs(null) >= abs(obs)).sum())
    return float((1 + k) / (1 + n_perm))


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman(x, z) controlling for y, on ranks (residual-on-residual)."""
    X, Y, Z = _rank(x), _rank(y), _rank(z)

    def resid(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        v0 = v - v.mean()
        beta = np.dot(u - u.mean(), v0) / max(np.dot(v0, v0), 1e-9)
        return u - u.mean() - beta * v0

    rx, rz = resid(X, Y), resid(Z, Y)
    d = math.sqrt(float(np.dot(rx, rx)) * float(np.dot(rz, rz)))
    return float(np.dot(rx, rz) / d) if d > 0 else float("nan")


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Boolean mask of hypotheses surviving BH-FDR at level ``q``.

    Order-independent: input order is preserved in the output.
    """
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    m = int(ok.sum())
    out = np.zeros(p.shape, bool)
    if m == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    crit = q * (np.arange(1, m + 1)) / m
    passing = np.where(p[order] <= crit)[0]
    if passing.size:
        out[order[: passing.max() + 1]] = True
    return out


def bh_critical(pvals: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Per-hypothesis BH threshold q*rank/m, aligned to input order.

    Ties take the *largest* rank in the tie group, which is what BH's step-up
    actually grants them: if the last member of a tie passes, all of them do.
    ``argsort(argsort(p))`` broke ties arbitrarily, so two hypotheses with an
    identical p could be shown different thresholds — and permutation p-values
    tie constantly, being multiples of ``1 / (1 + n_perm)``.
    """
    p = np.asarray(pvals, float)
    m = len(p)
    ranks = rankdata(p, method="max")
    return q * ranks / m


def block_permutation_sign_p(
    cells: list[tuple[str, np.ndarray, np.ndarray]],
    blocks: dict[str, list[str]] | None = None,
    n_perm: int = 5000,
    rng: np.random.Generator | None = None,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Pooled sign-agreement test with a block-permutation null.

    ``cells`` is a list of ``(trigger, aligned_surprise, response)`` arrays,
    where *aligned* means already multiplied by the theory-predicted sign so a
    positive product is a hit. The null shuffles surprises **within each
    trigger series**, which — because CPI and CPIYOY resolve from the same
    print — preserves their r≈0.69 dependence and gives the correct (wider)
    null. The naive per-cell shuffle overstates significance ~10x
    (research_log.md §3).
    """
    rng = _rng(rng, seed)

    def pooled(cs: list[tuple[str, np.ndarray, np.ndarray]]) -> float:
        hits = tot = 0
        for _, s, r in cs:
            m = (s != 0) & (r != 0)
            hits += int((s[m] * r[m] > 0).sum())
            tot += int(m.sum())
        return hits / tot if tot else float("nan")

    obs = pooled(cells)
    by_trig: dict[str, list[int]] = {}
    for i, (trig, _, _) in enumerate(cells):
        by_trig.setdefault(trig, []).append(i)

    null = np.empty(n_perm)
    for k in range(n_perm):
        perm_cells = list(cells)
        for _trig, idxs in by_trig.items():
            pooled_s = np.concatenate([cells[i][1] for i in idxs])
            shuffled = rng.permutation(pooled_s)
            cur = 0
            for i in idxs:
                n = len(cells[i][1])
                perm_cells[i] = (cells[i][0], shuffled[cur:cur + n], cells[i][2])
                cur += n
        null[k] = pooled(perm_cells)
    return {
        "sign_agreement": obs,
        "null_mean": float(np.nanmean(null)),
        "null_sd": float(np.nanstd(null)),
        # (1 + k) / (1 + n_perm): see permutation_p
        "p": float((1 + int((null >= obs).sum())) / (1 + n_perm)),
        "n": int(sum(len(s) for _, s, _ in cells)),
    }
