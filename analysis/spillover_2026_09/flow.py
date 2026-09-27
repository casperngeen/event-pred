#!/usr/bin/env python
"""Signed order flow as the shock: does one-sided buying in A move B?

    venv/bin/python -W ignore analysis/spillover_2026_09/flow.py > analysis/spillover_2026_09/out/flow.txt

A jump needs the price to move. Order flow does not: the archive records the
taker side of every trade, so a burst of one-sided YES buying on a series'
threshold contracts is an information event in its own right (Kyle-style
informed trading), and there are many more of them.

Shocks in source A: 1-hour bars of A's net signed flow (Σ over its threshold
contracts of +count for YES-taker trades, −count for NO-taker; + = buying the
"above K" side, i.e. expecting a higher value), in A's top 5% (and top 1%) of
|flow| over its non-zero bars, more than 24 h from any release or settlement,
merged within 1 h. Direction = sign(flow).

Responses, placebo and theory signs are exactly as in ``jumps.py``
(``_tape.py``): B's lead contract, from the end of the flow bar to +1 h … +72 h,
theory-signed by sign(flow)·HAWKISH[A]·HAWKISH[B], minus the same measurement at
a random quiet time 3–10 days away. "own" is A's own lead contract (sign =
sign(flow)): it checks that the flow is informative for A itself.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.panel.registry import is_same_release
from stg.structure.stats import benjamini_hochberg

sys.path.insert(0, "analysis/spillover_2026_09")
from _tape import (FLOW, H, HAWKISH, HORIZONS, LEGS, TAPE, TYPE,  # noqa: E402
                   lead_contract, near_release, placebo_time, response)

OUT = "analysis/spillover_2026_09/out"
QUANTILES = (0.95, 0.99)
COMOVE_C, N_BOOT = 5.0, 2000
rng = np.random.default_rng(0)


def flow_bars(series):
    """Net signed flow per 1-hour bar over the series' threshold contracts."""
    t = np.concatenate([TAPE[tk][0] for tk, _ in LEGS[series]])
    f = np.concatenate([FLOW[tk] for tk, _ in LEGS[series]])
    b = t.astype("datetime64[h]")
    u, inv = np.unique(b, return_inverse=True)
    return u.astype("datetime64[us]"), np.bincount(inv, f)


def shocks_of(series, q):
    bars, flow = flow_bars(series)
    nz = flow != 0
    if nz.sum() < 50:
        return []
    cut = np.quantile(np.abs(flow[nz]), q)
    out = []
    for t0, f in zip(bars, flow):
        if abs(f) < cut or near_release(t0) or (out and t0 - out[-1][0] <= H):
            continue
        out.append((t0, t0 + H, float(np.sign(f)), float(abs(f))))
    return out


rows = []
for q in QUANTILES:
    print(f"flow shocks, top {100 * (1 - q):.0f}% of |net flow| per series:", flush=True)
    for A in LEGS:
        S = shocks_of(A, q)
        print(f"  {A:<14} {len(S):>5} shocks", flush=True)
        for sid, (t0, t1, dA, size) in enumerate(S):
            tp = placebo_time(t0, rng)
            for B in LEGS:
                s = dA * (1 if B == A else HAWKISH[A] * HAWKISH[B])
                for kind, win in (("real", (t0, t1)), ("placebo", (tp, tp + H) if tp is not None else None)):
                    if win is None:
                        continue
                    tb = lead_contract(B, win[0])
                    if tb is None:
                        continue
                    same, after = response(tb, *win)
                    rows.append(dict(q=q, A=A, B=B, own=A == B, shock=f"{q}:{A}#{sid}", t0=t0.item(),
                                     kind=kind, size=size, same_release=A != B and is_same_release(A, B),
                                     channel="own" if A == B else f"{TYPE[A]}→{TYPE[B]}",
                                     comove=bool(np.isfinite(same) and abs(same) >= COMOVE_C),
                                     same=s * same if np.isfinite(same) else np.nan,
                                     **{h: s * v if np.isfinite(v) else np.nan for h, v in after.items()}))
R = pl.DataFrame(rows)
R.write_parquet(f"{OUT}/flow.parquet")


def compare(d, h, exclude_comove=True):
    real = d.filter((pl.col("kind") == "real") & pl.col(h).is_finite())
    if exclude_comove:
        real = real.filter(~pl.col("comove"))
    plac = d.filter((pl.col("kind") == "placebo") & pl.col(h).is_finite())
    if real.height < 10 or plac.height < 10:
        return None
    rj = dict(real.group_by("shock").agg(pl.col(h).sum(), pl.len()).select(
        "shock", pl.struct(h, "len")).iter_rows())
    pj = dict(plac.group_by("shock").agg(pl.col(h).sum(), pl.len()).select(
        "shock", pl.struct(h, "len")).iter_rows())
    keys = sorted(set(rj) | set(pj))
    RS = np.array([rj[k][h] if k in rj else 0.0 for k in keys]); RN = np.array([rj[k]["len"] if k in rj else 0 for k in keys])
    PS = np.array([pj[k][h] if k in pj else 0.0 for k in keys]); PN = np.array([pj[k]["len"] if k in pj else 0 for k in keys])
    est = RS.sum() / RN.sum() - PS.sum() / PN.sum()
    bs = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(keys), len(keys))
        if RN[k].sum() and PN[k].sum():
            bs.append(RS[k].sum() / RN[k].sum() - PS[k].sum() / PN[k].sum())
    bs = np.array(bs)
    nz = real.filter(pl.col(h) != 0)[h].to_numpy()
    return dict(n=real.height, shocks=real["shock"].n_unique(), diff=est, lo=np.percentile(bs, 2.5),
                hi=np.percentile(bs, 97.5), p=max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / N_BOOT),
                agree=float((nz > 0).mean()) if len(nz) else np.nan)


def line(lab, c):
    return (f"{lab:30} {c['n']:>6} {c['shocks']:>6} {c['diff']:>+8.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] "
            f"{c['agree']:>6.3f}")


for q in QUANTILES:
    RQ = R.filter(pl.col("q") == q)
    print("\n" + "#" * 96 + f"\nFLOW SHOCKS: top {100 * (1 - q):.0f}% of |net flow|  "
          f"({RQ.filter(pl.col('kind') == 'real')['shock'].n_unique()} shocks)\n" + "#" * 96)
    print(f"{'':30} {'n':>6} {'shocks':>6} {'real−placebo [95% CI]':>26} {'agree':>6}")
    for h in HORIZONS:
        c = compare(RQ.filter(pl.col("own")), h, exclude_comove=False)
        if c:
            print(line(f"OWN contract {h}", c))
    for h in HORIZONS:
        c = compare(RQ.filter(~pl.col("own") & ~pl.col("same_release")), h)
        if c:
            print(line(f"other series {h}", c))
    print(f"\nBY CHANNEL, +24h (other series)")
    for (ch, sr), d in sorted(RQ.filter(~pl.col("own")).group_by("channel", "same_release"),
                              key=lambda kv: str(kv[0])):
        c = compare(d, "+24h")
        if c:
            print(line(ch + (" (same release)" if sr else ""), c))
    pairs = []
    for (a, b), d in sorted(RQ.filter(~pl.col("own")).group_by("A", "B"), key=lambda kv: kv[0]):
        c = compare(d, "+24h")
        if c and c["n"] >= 30:
            pairs.append(dict(pair=f"{a}→{b}", same_release=d["same_release"][0], **c))
    if pairs:
        P = pl.DataFrame(pairs).sort("p")
        P = P.with_columns(pl.Series("bh", benjamini_hochberg(P["p"].to_numpy(), 0.10)))
        print(f"\nBY PAIR, +24h: {P.height} pairs with ≥ 30 responses, {int(P['bh'].sum())} survive BH (q = 0.10); top 8:")
        for r in P.head(8).iter_rows(named=True):
            print(f"  {r['pair'] + (' *' if r['same_release'] else ''):28} n {r['n']:>4} {r['diff']:>+7.3f} "
                  f"[{r['lo']:+.3f}, {r['hi']:+.3f}] p {r['p']:.3f}{'  BH' if r['bh'] else ''}")
