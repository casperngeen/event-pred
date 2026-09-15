#!/usr/bin/env python
"""Does the lead-lag signal predict settlement, and where in price space?

    venv/bin/python analysis/leadlag_2026_09/signal_model.py

Four passes, weakest assumptions first. The order is deliberate: if the
model-free pass is flat, a fitted model that "works" is fitting noise, and at
this sample size that is the likelier outcome.

1. **Model-free, by price bucket.** Does a positive aligned signal predict the
   target leg settles YES *more often than its price implies*? The residual
   ``win - p_entry/100`` is the market's error; the test is whether the signal
   correlates with it. Zero parameters.
2. **The null that matters: block permutation.** Shuffle ``z_surprise`` among
   trigger events **within each trigger series**, rebuild the signal, recompute
   the statistic. This preserves the ladder, the target outcomes, the price
   levels and the number of legs, and destroys only which trigger surprise is
   paired with which target event -- which is the lead-lag claim itself. It
   also preserves the r~0.69 dependence between CPI and CPIYOY, which resolve
   from one print; ``research_log.md`` §3 measured that a naive per-row shuffle
   overstates significance ~10x.
3. **Walk-forward delta.** ``p_hat = clip(p + delta*signal)``, ``delta`` chosen
   on prior years only. Addendum 2's estimator, re-run on the full ladder
   rather than one representative leg.
4. **Walk-forward logistic, fit separately within each price bucket.**
   ``p_hat = sigmoid(a + b*logit(p) + c*signal)``. Per-bucket fits rather than
   one model with an interaction term, because the question *is* whether the
   relationship differs by price, and a per-bucket fit answers it without
   assuming the interaction is linear.

Scoring is always **against the market price at entry**, never a base rate --
beating the base rate here is trivial (Brier 0.091 vs 0.250) and means nothing.
Bootstraps cluster on ``target_event``: one print settles every leg of an
event, and the same target event is matched by several different triggers.

The falsification cell
----------------------
``U3`` and ``JOBLESSCLAIMS`` carry ``HAWKISH = -1``, so a hawkish CPI surprise
predicts U3 prints *lower* and U3's YES legs get less likely. Pass 5 checks
that the raw (unaligned) association reverses in those cells while the aligned
one does not. If both flip, the economics is doing no work.

In-sample only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("analysis/leadlag_2026_09/out")
N_BOOT = 10000
N_PERM = 2000
EPS = 1e-6

BUCKETS = [(1, 5), (5, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99)]


# --------------------------------------------------------------------------
# clustered bootstrap -- precomputed cluster sums, so a resample is two
# gathers rather than a concatenate of hundreds of arrays.
# --------------------------------------------------------------------------
def _sums(vals, inv, k):
    return np.bincount(inv, weights=vals, minlength=k), \
        np.bincount(inv, minlength=k).astype(float)


def cluster_boot(vals: np.ndarray, groups: np.ndarray, seed: int = 0):
    uniq, inv = np.unique(groups, return_inverse=True)
    k = len(uniq)
    s, c = _sums(vals, inv, k)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    boot = s[pick].sum(axis=1) / np.maximum(c[pick].sum(axis=1), 1e-9)
    return (float(vals.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), float((boot <= 0).mean()))


def cluster_boot_diff(v_hi, v_lo, g_hi, g_lo, seed: int = 0):
    """Clustered bootstrap of mean(hi) - mean(lo), resampling whole events.

    Both sides share one event universe, so an event enters or leaves the
    resample carrying whatever rows it has on each side.
    """
    uniq = np.unique(np.concatenate([g_hi, g_lo]))
    k = len(uniq)
    ih = np.searchsorted(uniq, g_hi)
    il = np.searchsorted(uniq, g_lo)
    sh, ch = _sums(v_hi, ih, k)
    sl, cl = _sums(v_lo, il, k)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    mh = sh[pick].sum(axis=1) / np.maximum(ch[pick].sum(axis=1), 1e-9)
    ml = sl[pick].sum(axis=1) / np.maximum(cl[pick].sum(axis=1), 1e-9)
    boot = mh - ml
    obs = float(v_hi.mean() - v_lo.mean())
    return (obs, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)),
            float((boot <= 0).mean()))


def logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic(X, y, l2=1.0, iters=200):
    """Newton-IRLS with ridge. Small, dense, well-conditioned -- no sklearn."""
    n, k = X.shape
    w = np.zeros(k)
    pen = l2 * np.eye(k)
    pen[0, 0] = 0.0
    for _ in range(iters):
        mu = sigmoid(X @ w)
        s = np.clip(mu * (1 - mu), 1e-9, None)
        g = X.T @ (mu - y) + pen @ w
        H = (X * s[:, None]).T @ X + pen
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def fit_logistic_offset(X, y, offset, l2=1.0, iters=200):
    """Newton-IRLS with a fixed offset: eta = offset + X @ w.

    The offset is the market's own logit. Fixing its coefficient at 1 means
    the model can only *tilt* the quote; it cannot re-slope it, which is what
    a free coefficient on logit(p) does and what costs pass 4 its skill.
    """
    n, k = X.shape
    w = np.zeros(k)
    pen = l2 * np.eye(k)
    pen[0, 0] = 0.0
    for _ in range(iters):
        mu = sigmoid(offset + X @ w)
        s = np.clip(mu * (1 - mu), 1e-9, None)
        g = X.T @ (mu - y) + pen @ w
        H = (X * s[:, None]).T @ X + pen
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def tercile_diff(sig_v, resid_v):
    """mean resid in the top signal tercile minus the bottom. NaN if degenerate."""
    t = np.percentile(sig_v, [33.3, 66.7])
    hi, lo = sig_v > t[1], sig_v <= t[0]
    if hi.sum() < 10 or lo.sum() < 10:
        return np.nan, hi, lo
    return float(resid_v[hi].mean() - resid_v[lo].mean()), hi, lo


def main() -> None:
    d = pl.read_parquet(OUT / "leadlag_legs.parquet")
    p = d["p_entry"].to_numpy() / 100.0
    y = d["win"].to_numpy().astype(float)
    ev = d["target_event"].to_numpy()
    yr = d["yr"].to_numpy()
    px = d["p_entry"].to_numpy()
    direction = d["direction"].to_numpy().astype(float)
    resid = y - p

    # Winsorise z, not the product: z_surprise has heavy tails and one 8-sigma
    # print should not set a fitted coefficient. Monotone, so it changes no sign.
    z_raw = d["z_surprise"].to_numpy()
    lim = float(np.percentile(np.abs(z_raw), 99))
    z = np.clip(z_raw, -lim, lim)
    sig = direction * z

    # Index rows to their trigger event, for the block permutation.
    trig_ev = d["trigger_event"].to_numpy()
    trig_sr = d["trigger"].to_numpy()
    uniq_te, row_te = np.unique(trig_ev, return_inverse=True)
    te_z = np.zeros(len(uniq_te))
    te_series = np.empty(len(uniq_te), dtype=object)
    for i, te in enumerate(uniq_te):
        j = np.argmax(trig_ev == te)
        te_z[i] = z[j]
        te_series[i] = trig_sr[j]
    series_groups = [np.where(te_series == s)[0] for s in np.unique(te_series)]

    print(f"rows {len(y)}   target events {len(np.unique(ev))}   "
          f"trigger events {len(uniq_te)}   |z| winsorised at {lim:.2f}\n")

    # ------------------------------------------------------------- pass 1+2
    print("=== 1. model-free, by entry price, with a block-permutation null ===")
    print("diff_pp = mean(win - price) in the top signal tercile minus the")
    print("bottom, in percentage points. The bootstrap CI treats the ladder as")
    print("clustered; the permutation p-value is the one that tests lead-lag.\n")

    rng = np.random.default_rng(0)
    # Pre-draw every permutation once, so all buckets are tested against the
    # SAME null draws -- otherwise bucket-to-bucket p-values are not comparable.
    perm_sig = np.empty((N_PERM, len(y)))
    for b in range(N_PERM):
        zp = te_z.copy()
        for g in series_groups:
            zp[g] = te_z[rng.permutation(g)]
        perm_sig[b] = direction * zp[row_te]

    rows = []
    for lo, hi in BUCKETS:
        m = (px >= lo) & (px < hi)
        if m.sum() < 60:
            continue
        s_m, r_m, e_m = sig[m], resid[m], ev[m]
        obs, hi_m, lo_m = tercile_diff(s_m, r_m)
        if np.isnan(obs):
            continue
        _, clo, chi, _ = cluster_boot_diff(r_m[hi_m], r_m[lo_m],
                                           e_m[hi_m], e_m[lo_m])
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            nb, _, _ = tercile_diff(perm_sig[b][m], r_m)
            null[b] = nb
        null = null[np.isfinite(null)]
        rows.append(dict(band=f"{lo}-{hi}c", n=int(m.sum()),
                         n_ev=int(len(np.unique(e_m))),
                         resid_lo=100 * float(r_m[lo_m].mean()),
                         resid_hi=100 * float(r_m[hi_m].mean()),
                         diff_pp=100 * obs,
                         ci_lo=100 * clo, ci_hi=100 * chi,
                         null_mean=100 * float(null.mean()),
                         null_sd=100 * float(null.std()),
                         perm_p=float((null >= obs).mean())))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=230):
        print(pl.DataFrame(rows))
    print("perm_p is one-sided: the share of block permutations reaching the")
    print("observed difference. The theory predicts a POSITIVE difference.")

    print("\n=== 2. the same, pooled across all prices ===")
    obs, hi_m, lo_m = tercile_diff(sig, resid)
    _, clo, chi, _ = cluster_boot_diff(resid[hi_m], resid[lo_m], ev[hi_m], ev[lo_m])
    null = np.array([tercile_diff(perm_sig[b], resid)[0] for b in range(N_PERM)])
    print(f"diff {100*obs:+.2f}pp   clustered CI [{100*clo:+.2f}, {100*chi:+.2f}]   "
          f"block-perm null {100*null.mean():+.2f} +/- {100*null.std():.2f}   "
          f"p = {(null >= obs).mean():.4f}")

    # ------------------------------------------------------------- pass 3
    print("\n=== 3. walk-forward delta:  p_hat = clip(p + delta*signal) ===")
    print("delta chosen on prior years only, by log loss.\n")
    years = sorted(np.unique(yr))
    grid = np.arange(0.0, 0.1001, 0.005)
    rows = []
    for Y in years[1:]:
        tr_m, te_m = yr < Y, yr == Y
        if te_m.sum() < 50 or tr_m.sum() < 200:
            continue
        losses = [logloss(np.clip(p[tr_m] + g * sig[tr_m], EPS, 1 - EPS),
                          y[tr_m]).mean() for g in grid]
        best = grid[int(np.argmin(losses))]
        ph = np.clip(p[te_m] + best * sig[te_m], EPS, 1 - EPS)
        db = brier(p[te_m], y[te_m]) - brier(ph, y[te_m])
        _, blo, bhi, _ = cluster_boot(db, ev[te_m])
        rows.append(dict(year=int(Y), n_test=int(te_m.sum()), delta=float(best),
                         mkt_brier=float(brier(p[te_m], y[te_m]).mean()),
                         mdl_brier=float(brier(ph, y[te_m]).mean()),
                         brier_skill=float(db.mean()), sk_lo=blo, sk_hi=bhi))
    with pl.Config(tbl_rows=20, float_precision=4, tbl_width_chars=220):
        print(pl.DataFrame(rows))

    # ------------------------------------------------------------- pass 4
    print("\n=== 4. walk-forward logistic, fit separately within each price bucket ===")
    print("skill = market loss - model loss, so POSITIVE means the model beat")
    print("the market. Clustered on target_event.\n")
    rows = []
    for lo, hi in BUCKETS:
        m = (px >= lo) & (px < hi)
        if m.sum() < 120:
            continue
        ph = np.full(int(m.sum()), np.nan)
        sub_yr, sub_p, sub_s, sub_y = yr[m], p[m], sig[m], y[m]
        X = np.column_stack([np.ones(int(m.sum())), logit(sub_p), sub_s])
        coefs = []
        for Y in years[1:]:
            tr_i, te_i = sub_yr < Y, sub_yr == Y
            if te_i.sum() == 0 or tr_i.sum() < 80:
                continue
            w = fit_logistic(X[tr_i], sub_y[tr_i])
            coefs.append(w[2])
            ph[te_i] = sigmoid(X[te_i] @ w)
        ok = ~np.isnan(ph)
        if ok.sum() < 60:
            continue
        db = brier(sub_p[ok], sub_y[ok]) - brier(ph[ok], sub_y[ok])
        dl = logloss(sub_p[ok], sub_y[ok]) - logloss(ph[ok], sub_y[ok])
        _, lo_b, hi_b, _ = cluster_boot(db, ev[m][ok])
        _, lo_l, hi_l, _ = cluster_boot(dl, ev[m][ok])
        rows.append(dict(band=f"{lo}-{hi}c", n_scored=int(ok.sum()),
                         n_ev=int(len(np.unique(ev[m][ok]))),
                         mean_coef=float(np.mean(coefs)) if coefs else np.nan,
                         brier_skill=float(db.mean()), b_lo=lo_b, b_hi=hi_b,
                         ll_skill=float(dl.mean()), l_lo=lo_l, l_hi=hi_l))
    with pl.Config(tbl_rows=20, float_precision=4, tbl_width_chars=230):
        print(pl.DataFrame(rows))

    # ------------------------------------------------------------- pass 4b
    print("\n=== 4b. the same, but with the market price as an OFFSET ===")
    print("p_hat = sigmoid(logit(p) + a + c*signal). The coefficient on logit(p)")
    print("is FIXED at 1, so the model can only tilt the market's quote, never")
    print("re-slope it. Pass 4 re-estimated that slope and paid for it: the")
    print("market's calibration is already good, and distorting it costs more")
    print("than the signal gains. Two free parameters, walk-forward.\n")
    rows = []
    for lo, hi in BUCKETS:
        m = (px >= lo) & (px < hi)
        if m.sum() < 120:
            continue
        ph = np.full(int(m.sum()), np.nan)
        sub_yr, sub_p, sub_s, sub_y = yr[m], p[m], sig[m], y[m]
        off = logit(sub_p)
        X = np.column_stack([np.ones(int(m.sum())), sub_s])
        coefs = []
        for Y in years[1:]:
            tr_i, te_i = sub_yr < Y, sub_yr == Y
            if te_i.sum() == 0 or tr_i.sum() < 80:
                continue
            w = fit_logistic_offset(X[tr_i], sub_y[tr_i], off[tr_i])
            coefs.append(w[1])
            ph[te_i] = sigmoid(off[te_i] + X[te_i] @ w)
        ok = ~np.isnan(ph)
        if ok.sum() < 60:
            continue
        db = brier(sub_p[ok], sub_y[ok]) - brier(ph[ok], sub_y[ok])
        dl = logloss(sub_p[ok], sub_y[ok]) - logloss(ph[ok], sub_y[ok])
        _, lo_b, hi_b, pneg_b = cluster_boot(db, ev[m][ok])
        _, lo_l, hi_l, _ = cluster_boot(dl, ev[m][ok])
        rows.append(dict(band=f"{lo}-{hi}c", n_scored=int(ok.sum()),
                         n_ev=int(len(np.unique(ev[m][ok]))),
                         mean_coef=float(np.mean(coefs)) if coefs else np.nan,
                         brier_skill=float(db.mean()), b_lo=lo_b, b_hi=hi_b,
                         p_le0=pneg_b,
                         ll_skill=float(dl.mean()), l_lo=lo_l, l_hi=hi_l))
    with pl.Config(tbl_rows=20, float_precision=4, tbl_width_chars=240):
        print(pl.DataFrame(rows))

    # -------------------------------------------------------- pass 5
    print("\n=== 5. falsification: the dovish cells must reverse ===")
    print("U3/JOBLESSCLAIMS carry HAWKISH = -1. If the economics is carrying")
    print("the result, `raw z` reverses between the two direction blocks while")
    print("`aligned` does not. If aligned flips too, the alignment does nothing.\n")
    rows = []
    for dirn in (+1, -1):
        m = direction == dirn
        if m.sum() < 100:
            continue
        for label, feat in [("aligned", sig[m]), ("raw z", z[m])]:
            o, h, l = tercile_diff(feat, resid[m])
            if np.isnan(o):
                continue
            _, clo, chi, _ = cluster_boot_diff(resid[m][h], resid[m][l],
                                               ev[m][h], ev[m][l])
            rows.append(dict(direction=dirn, feature=label, n=int(m.sum()),
                             n_ev=int(len(np.unique(ev[m]))),
                             diff_pp=100 * o, ci_lo=100 * clo, ci_hi=100 * chi))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
