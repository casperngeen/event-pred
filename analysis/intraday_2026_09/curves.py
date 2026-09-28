#!/usr/bin/env python
"""Absorption curves: how much of each channel's response has arrived by τ (Q1, Q2).

    venv/bin/python -W ignore analysis/intraday_2026_09/curves.py \
        > analysis/intraday_2026_09/out/curves.txt        # needs build_paths.py; writes out/curves.parquet

For each channel and each bar τ, regress the target's move r(τ) = p(τ) − p₀
(cents) on the theory-signed surprise signal across (release, target) cells,
with an intercept:

  own        the releasing series' next contract on its own z (Kalshi arm)
  channel    source family → target type: cross cells of that target type,
             jointly on one signal column per source family (Σ z·HAWKISH·HAWKISH),
             so a CPI + GDP instant is split between its channels
  pooled     all cross cells on the summed signal ``sig``; also split into
             liquid / thin targets at the median pre-release print count

β(τ) > 0 is the theory direction. Absorption = β(τ)/β(24 h) (pooled slopes, not
per-cell ratios, which blow up when r(24 h) ≈ 0). Half-life = first τ with
β(τ) ≥ ½β(24 h). CIs: bootstrap over release instants. p: one-sided, share of
per-instant sign flips with β(24 h) at least the real one. BH at q = 0.10 over
the channels of each arm.

Paths stop at the next Kalshi release instant or the contract's close
(``trunc_k``). Robustness rows (pooled): bounce-robust price (mean of the last
two prints), and truncation at any release, calendar ones included
(``trunc_any``). Placebo arms: the same fits at times with no release.

Trade time (Q2): the move at the target's k-th post-release print (k = 1, 2, 3,
5, 10; within 24 h) against its 24 h clock-time move, on the cells that reach k
prints. The ratio of the two slopes is the share of the 24 h response already in
the price after k trades.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.structure.stats import benjamini_hochberg

sys.path.insert(0, "analysis/intraday_2026_09")
from _common import (OUT, TAU_POST, TTYPES, cells, fit, instant_index, response_matrix,  # noqa: E402
                     signal_columns, tau_label)

rng = np.random.default_rng(0)
KS = (1, 2, 3, 5, 10)
REPORT = [5, 15, 30, 60, 120, 240, 480, 1440, 4320, 10080]
TAUS = TAU_POST
I24 = int(np.nonzero(TAUS == 1440)[0][0])
IREP = [int(np.nonzero(TAUS == t)[0][0]) for t in REPORT]
PRE = [-60, -30, -15]
rows_out = []
SURVIVE: dict[str, set] = {}     # channels passing BH in clock time; the only ones shown in trade time


def half_life(b: np.ndarray) -> float:
    if not np.isfinite(b[I24]) or b[I24] <= 0:
        return np.nan
    hit = np.nonzero(b[:I24 + 1] >= 0.5 * b[I24])[0]
    return float(TAUS[hit[0]]) if len(hit) else np.nan


def fmt_min(m: float) -> str:
    return "—" if not np.isfinite(m) else tau_label(int(m))


def design(c: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(c.height)] + [c[k].to_numpy() for k in cols])


def run(arm: str, group: str, c: pl.DataFrame, cols: list[str], names: list[str], *,
        col="r", trunc="trunc_k", null=True, boot=True):
    """Fit one design; returns per-signal-column summaries."""
    if c.height < 30:
        return []
    Y = response_matrix(c, TAUS, col, trunc)
    X = design(c, cols)
    est, bs, nl = fit(X, Y, instant_index(c), rng, n_boot=1000 if boot else 0,
                      n_flip=1000 if null else 0)
    out = []
    for j, name in enumerate(names, start=1):
        b = est[:, j]
        r = dict(arm=arm, group=group, channel=name, n=int(np.isfinite(Y[:, I24]).sum()),
                 nz=int((X[:, j] != 0).sum()), b24=b[I24], hl=half_life(b),
                 ratio=b / b[I24] if b[I24] != 0 else np.full_like(b, np.nan))
        if bs is not None:
            bb = bs[:, :, j]
            r["b24_ci"] = np.percentile(bb[:, I24], [2.5, 97.5])
            hl = np.array([half_life(x) for x in bb])
            r["hl_ci"] = np.nanpercentile(hl, [2.5, 97.5], method="nearest") if np.isfinite(hl).mean() > 0.5 else (np.nan, np.nan)
            ok = bb[:, I24] > 0
            r1h = bb[ok, IREP[3]] / bb[ok, I24]
            r["r1h_ci"] = np.percentile(r1h, [2.5, 97.5]) if ok.mean() > 0.5 else (np.nan, np.nan)
            lo, hi = np.nanpercentile(bb, [2.5, 97.5], axis=0)
        else:
            lo = hi = np.full_like(b, np.nan)
        if nl is not None:
            r["p"] = (1 + (nl[:, I24, j] >= b[I24]).sum()) / (1 + nl.shape[0])
        for t, v, l_, h_ in zip(TAUS, b, lo, hi):
            rows_out.append(dict(arm=arm, group=group, channel=name, tau_min=int(t), beta=float(v),
                                 lo=float(l_), hi=float(h_), ratio=float(v / b[I24]) if b[I24] else None,
                                 variant=f"{col}/{trunc}"))
        out.append(r)
    return out


HDR = (f"{'channel':30} {'n':>5} {'nz':>5} {'β24h ¢/z':>8} {'95% CI':>17} {'p':>6} {'BH':>3} | "
       + " ".join(f"{tau_label(t):>6}" for t in REPORT[:7]) + " | "
       + f"{'3d':>5} {'7d':>5} | {'½-life':>6} {'95% CI':>15} | {'share 1h CI':>15}")


def show(r: dict, bh: bool | None = None):
    ci = r.get("b24_ci", (np.nan, np.nan))
    hc = r.get("hl_ci", (np.nan, np.nan))
    rc = r.get("r1h_ci", (np.nan, np.nan))
    p = r.get("p", np.nan)
    ratios = " ".join(f"{r['ratio'][i]:>6.2f}" for i in IREP[:7])
    print(f"{r['channel']:30} {r['n']:>5} {r['nz']:>5} {r['b24']:>+8.3f} [{ci[0]:>+6.2f},{ci[1]:>+6.2f}] "
          f"{p:>6.3f} {('yes' if bh else '') if bh is not None else '':>3} | {ratios} | "
          f"{r['ratio'][IREP[8]]:>5.2f} {r['ratio'][IREP[9]]:>5.2f} | {fmt_min(r['hl']):>6} "
          f"[{fmt_min(hc[0]):>6},{fmt_min(hc[1]):>6}] | [{rc[0]:>+5.2f},{rc[1]:>+5.2f}]", flush=True)


def pooled_groups(arm: str):
    c = cells.filter(pl.col("arm") == arm)
    own = c.filter(pl.col("role") == "own")
    cross = c.filter(pl.col("role") == "cross")
    med = cross["n_pre7"].median()
    return [("own next contract", own, ["x_own"]),
            ("cross, pooled", cross, ["sig"]),
            (f"cross, liquid (≥{med:g} prints/7d)", cross.filter(pl.col("n_pre7") >= med), ["sig"]),
            (f"cross, thin (<{med:g} prints/7d)", cross.filter(pl.col("n_pre7") < med), ["sig"])]


def clock_time(arm: str):
    print(f"\n{'=' * 150}\nCLOCK TIME — arm '{arm}': β(τ)/β(24h) at τ; p = sign-flip, one-sided (theory direction)"
          f"\n{'=' * 150}\n{HDR}")
    for name, c, cols in pooled_groups(arm):
        for r in run(arm, "pooled", c, cols, [name]):
            show(r)
    if arm.endswith("placebo"):
        return
    res = []
    c = cells.filter((pl.col("arm") == arm) & (pl.col("role") == "cross"))
    for tt in TTYPES:
        ct = c.filter(pl.col("ttype") == tt)
        cols = signal_columns(ct, arm)
        if cols:
            res += run(arm, "channel", ct, cols, [f"{k[2:]} → {tt}" for k in cols])
    bh = benjamini_hochberg(np.array([r["p"] for r in res]), 0.10)
    print("-" * 150)
    for r, b in sorted(zip(res, bh), key=lambda x: x[0]["p"]):
        show(r, bool(b))
    print(f"{len(res)} channels, {int(bh.sum())} survive BH at q = 0.10")
    SURVIVE[arm] = {r["channel"] for r, b in zip(res, bh) if b}

    print(f"\nrobustness, pooled (no null): bounce-robust price (2-print mean), and truncation at any release")
    for name, c, cols in pooled_groups(arm)[:2]:
        for col, trunc, lab in (("r2", "trunc_k", "2-print mean"), ("r", "trunc_any", "cut at any release")):
            for r in run(arm, f"robust:{lab}", c, cols, [f"{name} [{lab}]"], col=col, trunc=trunc, null=False):
                show(r)

    print("\npre-release drift, pooled: β at τ < 0 (should be ≈ 0)")
    for name, c, cols in pooled_groups(arm)[:2]:
        if c.height < 30:
            continue
        Y = response_matrix(c, PRE)
        est, bs, _ = fit(design(c, cols), Y, instant_index(c), rng, n_boot=1000, n_flip=0)
        print(f"  {name:28}" + "  ".join(
            f"{tau_label(t)}: {est[i, 1]:+.3f} [{np.percentile(bs[:, i, 1], 2.5):+.3f},"
            f"{np.percentile(bs[:, i, 1], 97.5):+.3f}]" for i, t in enumerate(PRE)))


def trade_time(arm: str):
    print(f"\n{'=' * 118}\nTRADE TIME — arm '{arm}': slope at the k-th post-release print ÷ slope at +24 h, "
          f"same cells (those reaching k prints ≤ 24 h);\npooled groups and the channels passing BH in clock "
          f"time (a ratio to a 24 h slope near 0 is noise)\n{'=' * 118}")
    print(f"{'channel':30} " + " ".join(f"{'k=' + str(k):>24}" for k in KS))
    c_all = cells.filter(pl.col("arm") == arm)
    groups = [(n, c, cols, [n]) for n, c, cols in pooled_groups(arm)]
    cross = c_all.filter(pl.col("role") == "cross")
    for tt in TTYPES:
        ct = cross.filter(pl.col("ttype") == tt)
        cols = signal_columns(ct, arm)
        if cols:
            groups.append((tt, ct, cols, [f"{k[2:]} → {tt}" for k in cols]))
    for _, c, cols, names in groups:
        if c.height < 30:
            continue
        y24 = response_matrix(c, [1440])[:, 0]
        X = design(c, cols)
        inst = instant_index(c)
        line = {n: [] for n in names}
        for k in KS:
            yk = (c[f"p_k{k}"] - c["p0"]).to_numpy().astype(float)
            m = np.isfinite(yk) & np.isfinite(y24)
            Y = np.column_stack([np.where(m, yk, np.nan), np.where(m, y24, np.nan)])
            est, bs, _ = fit(X, Y, inst, rng, n_boot=500, n_flip=0)
            for j, n in enumerate(names, start=1):
                if (X[m, j] != 0).sum() < 15:
                    line[n].append(f"{'—':>24}")
                    continue
                rat = est[0, j] / est[1, j]
                ok = bs[:, 1, j] > 0
                ci = (np.percentile(bs[ok, 0, j] / bs[ok, 1, j], [2.5, 97.5])
                      if est[1, j] > 0 and ok.mean() > 0.5 else (np.nan, np.nan))
                line[n].append(f"{rat:>5.2f} [{ci[0]:>+5.2f},{ci[1]:>+5.2f}] n={int(m.sum()):<4}")
        for n in names:
            if n in dict.fromkeys(g[0] for g in pooled_groups(arm)) or n in SURVIVE.get(arm, ()):
                print(f"{n:30} " + " ".join(line[n]), flush=True)


for arm in ("kalshi", "calendar", "kalshi_placebo", "calendar_placebo"):
    clock_time(arm)
for arm in ("kalshi", "calendar"):
    trade_time(arm)

pl.DataFrame(rows_out).write_parquet(OUT / "curves.parquet")
print(f"\nwrote {OUT / 'curves.parquet'}")
