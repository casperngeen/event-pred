"""Reconstruction-free ladder statistics.

``implied.py`` recovers a pdf from the ladder and integrates it. Every moment
that comes back therefore inherits an assumption about the strikes that are
*not* there -- and ``relations_findings.md`` item 1 measured that mean ladder
coverage is only 0.64-0.83, with the absent strikes disproportionately
far-from-the-money because those trade least. ``recover_pdf`` handles the gap by
placing tail mass ``spacing / 2`` past the extreme observed strikes, which
biases ``implied_std`` down and inflates ``surprisal``.

This module never integrates. Everything here is read off the traded ladder by
interpolation between adjacent strikes, so it depends only on the region where
contracts actually traded:

    q50   the strike where P(X > K) crosses 0.50          -- location
    IQR   K(p = 0.25) - K(p = 0.75)                       -- width
    PIT   1 - P(X > resolved), interpolated at the outcome

The 25% and 75% crossings are interior to the liquid part of any ladder that
brackets them, so a missing 2c wing leg cannot move them. Where a ladder does
*not* bracket a level, the statistic is **undefined and returned as None**
rather than extrapolated -- an explicit gate instead of a silent bias.

Sign convention throughout: ``probs[i] = P(X > thresholds[i])``, so ``probs`` is
non-increasing in ``thresholds``. That is the same convention ``recover_pdf``
takes, so the two are drop-in comparable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# A ladder needs at least this many traded legs before any statistic is
# attempted. Two points can define a crossing but not a credible one.
MIN_LEGS = 3


def isotonic_decreasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Least-squares non-increasing fit (pool adjacent violators).

    ``coherence.py`` measured that ~2% of adjacent strike pairs violate
    monotonicity even when both legs traded within five minutes, so a raw
    ladder is not always a valid survival function and interpolating one
    directly can produce a non-monotone quantile function. PAVA is the minimal
    projection onto the monotone cone: it leaves a coherent ladder untouched and
    pools only the violating runs, rather than the running-minimum envelope,
    which would drag every later strike down after a single bad print.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy()
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    # Work on the reversed series so the problem is "non-decreasing".
    vals = y[::-1].copy()
    wts = w[::-1].copy()
    level_val: list[float] = []
    level_wt: list[float] = []
    level_n: list[int] = []
    for i in range(n):
        level_val.append(vals[i])
        level_wt.append(wts[i])
        level_n.append(1)
        while len(level_val) > 1 and level_val[-2] > level_val[-1]:
            v2, w2, n2 = level_val.pop(), level_wt.pop(), level_n.pop()
            v1, w1, n1 = level_val.pop(), level_wt.pop(), level_n.pop()
            tw = w1 + w2
            level_val.append((v1 * w1 + v2 * w2) / tw if tw else (v1 + v2) / 2)
            level_wt.append(tw)
            level_n.append(n1 + n2)
    out = np.empty(n)
    idx = 0
    for v, cnt in zip(level_val, level_n):
        out[idx:idx + cnt] = v
        idx += cnt
    return out[::-1]


def _prepare(thresholds, probs, monotone: bool = True):
    """Sort by strike, drop non-finite, optionally project onto monotone."""
    k = np.asarray(thresholds, dtype=float)
    p = np.asarray(probs, dtype=float)
    ok = np.isfinite(k) & np.isfinite(p)
    k, p = k[ok], p[ok]
    if k.size == 0:
        return k, p
    order = np.argsort(k)
    k, p = k[order], p[order]
    # collapse duplicate strikes by averaging their prices
    if np.any(np.diff(k) == 0):
        uk, inv = np.unique(k, return_inverse=True)
        p = np.bincount(inv, weights=p) / np.bincount(inv)
        k = uk
    if monotone and k.size > 1:
        p = isotonic_decreasing(p)
    return k, p


def crossing(thresholds, probs, level: float, monotone: bool = True) -> Optional[float]:
    """Strike at which ``P(X > K)`` crosses ``level``, by linear interpolation.

    Returns ``None`` when the ladder does not bracket ``level`` -- i.e. the
    quantile lies outside the traded strike range. That is the whole point:
    the caller learns the statistic is unavailable instead of receiving an
    extrapolation.
    """
    k, p = _prepare(thresholds, probs, monotone)
    if k.size < MIN_LEGS:
        return None
    if level > p[0] or level < p[-1]:
        return None                      # not bracketed: censored, not zero
    hit = np.where(p == level)[0]
    if hit.size:                         # flat run exactly at the level
        return float((k[hit[0]] + k[hit[-1]]) / 2.0)
    i = int(np.searchsorted(-p, -level, side="left"))
    i = max(1, min(i, len(p) - 1))
    p_hi, p_lo = p[i - 1], p[i]
    if p_hi == p_lo:
        return float((k[i - 1] + k[i]) / 2.0)
    frac = (p_hi - level) / (p_hi - p_lo)
    return float(k[i - 1] + frac * (k[i] - k[i - 1]))


def prob_at(thresholds, probs, value: float, monotone: bool = True):
    """``P(X > value)`` interpolated at an arbitrary value.

    Returns ``(prob, censor)`` where ``censor`` is ``"left"`` if ``value`` sits
    below the lowest strike, ``"right"`` if above the highest, else ``None``.
    A censored value still returns the nearest bound, so the caller can choose
    between dropping it and treating it as an inequality.
    """
    k, p = _prepare(thresholds, probs, monotone)
    if k.size < MIN_LEGS:
        return None, None
    if value < k[0]:
        return float(p[0]), "left"
    if value > k[-1]:
        return float(p[-1]), "right"
    i = int(np.searchsorted(k, value, side="left"))
    if i == 0:
        return float(p[0]), None
    if k[i] == value:
        return float(p[i]), None
    frac = (value - k[i - 1]) / (k[i] - k[i - 1])
    return float(p[i - 1] + frac * (p[i] - p[i - 1])), None


def ladder_pit(thresholds, probs, value: float, monotone: bool = True):
    """``F(value) = 1 - P(X > value)``, read off the ladder.

    The reconstruction-free counterpart of ``implied.pit``. Returns
    ``(pit, censor)``; under a calibrated ladder the uncensored ``pit`` values
    are uniform on (0, 1).
    """
    pr, censor = prob_at(thresholds, probs, value, monotone)
    return (None if pr is None else 1.0 - pr), censor


def quantile_moments(thresholds, probs, monotone: bool = True) -> dict:
    """Location, width and asymmetry, none of which touch the ladder's tails.

    Keys: ``q10 q25 q50 q75 q90 iqr sigma_iqr skew_q n_legs span_lo span_hi``.
    ``sigma_iqr`` is ``IQR / 1.349``, the Gaussian-equivalent standard
    deviation, so it is comparable in magnitude with ``implied_std`` without
    inheriting its tail assumption. ``skew_q`` is Bowley's coefficient, which
    is bounded on [-1, 1] and defined entirely by the three quartiles.

    Any statistic whose crossing is not bracketed comes back ``None``.
    """
    k, p = _prepare(thresholds, probs, monotone)
    out: dict = {"n_legs": int(k.size),
                 "span_lo": float(k[0]) if k.size else None,
                 "span_hi": float(k[-1]) if k.size else None}
    for name, lvl in (("q10", 0.90), ("q25", 0.75), ("q50", 0.50),
                      ("q75", 0.25), ("q90", 0.10)):
        out[name] = crossing(k, p, lvl, monotone=False) if k.size >= MIN_LEGS else None
    q25, q50, q75 = out["q25"], out["q50"], out["q75"]
    out["iqr"] = (q75 - q25) if (q25 is not None and q75 is not None) else None
    out["sigma_iqr"] = (out["iqr"] / 1.349) if out["iqr"] is not None else None
    out["skew_q"] = (((q75 - q50) - (q50 - q25)) / (q75 - q25)
                     if None not in (q25, q50, q75) and (q75 - q25) > 0 else None)
    return out
