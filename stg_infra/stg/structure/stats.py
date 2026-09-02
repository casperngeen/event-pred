"""Rank statistics and multiplicity control for the structure estimator.

Rank/sign based throughout: the CPI/CPIYOY consistency check (research_log.md
§2) showed the signal is in the *direction* of a surprise, not its magnitude,
so Pearson dilutes it toward zero.
"""

from __future__ import annotations

import math

import numpy as np

_RNG = np.random.default_rng(0)


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


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
    """Two-sided asymptotic p-value (normal approx to the t on ranks, n >= 10)."""
    if not np.isfinite(rho) or abs(rho) >= 1 or n < 4:
        return float("nan")
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    z = abs(t) / math.sqrt(2.0)
    return float(min(max(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z))), 0.0), 1.0))


def permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int = 2000,
                  rng: np.random.Generator | None = None) -> float:
    """Two-sided permutation p: shuffle ``a`` against fixed ``b``."""
    rng = rng or _RNG
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    obs = spearman(a, b)
    if not np.isfinite(obs):
        return float("nan")
    null = np.array([spearman(rng.permutation(a), b) for _ in range(n_perm)])
    return float((np.abs(null) >= abs(obs)).mean())


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
    """Per-hypothesis BH threshold q*rank/m, aligned to input order."""
    p = np.asarray(pvals, float)
    m = len(p)
    ranks = np.argsort(np.argsort(p)) + 1
    return q * ranks / m


def block_permutation_sign_p(
    cells: list[tuple[str, np.ndarray, np.ndarray]],
    blocks: dict[str, list[str]] | None = None,
    n_perm: int = 5000,
    rng: np.random.Generator | None = None,
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
    rng = rng or _RNG

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
        "p": float((null >= obs).mean()),
        "n": int(sum(len(s) for _, s, _ in cells)),
    }
