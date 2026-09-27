#!/usr/bin/env python
"""Do price jumps in one macro market spill over to another, away from releases?

    venv/bin/python -W ignore analysis/spillover_2026_09/jumps.py > analysis/spillover_2026_09/out/jumps.txt

The release studies use scheduled releases as the shocks: 154 instants in four
years. Between releases, the same markets reprice on unscheduled news and on
each other. This script treats every sizeable, persistent price jump in one
series as a shock and asks whether the other series follow in the direction
theory predicts.

Universe: the 17 threshold ("Above K") series of the event-time panel, each a
source and a target (``HAWKISH`` signs as in ``event_time_2026_09``). In-sample
archive only (``load_markets``/``scan_trades`` with ``is_only=True``).

Jumps in source A, at each of four thresholds (5c, 10c, 15c, 20c), detected
separately:
  - a threshold contract's 15-minute last price moves ≥ the threshold from its previous
    traded bar, and that bar is ≤ 2 h earlier
  - persistent: the first print in the next hour does not undo more than half
  - more than 24 h from every macro release or settlement (any event close of
    the 17 series), so neither a release nor settlement convergence drives it
  - A's jumping contracts in one bar are one jump (direction = sign of their
    mean move); jumps of A within 1 h of the previous one are merged

Response in target B (≠ A): B's lead contract (the threshold contract with the
most prints in the prior 7 days, among B's events closing later), from the end
of the jump bar to +1 h, +4 h, +24 h, +72 h (a print in that window is
required; the start is B's last print before the jump bar ends, ≤ 7 d old).
Signed by theory: s = sign(ΔA)·HAWKISH[A]·HAWKISH[B]; response = s·ΔB, so a
positive mean is spillover in the theory direction.

  co-jump    B's lead contract also moved ≥ 5c inside the jump bar: common news
             rather than diffusion; reported separately, excluded from the
             lead-lag rows
  placebo    the same measurement at a random time 3–10 days before or after,
             at the same time of day and also outside the release windows

Reported by channel (type → type) and by pair: mean signed response, real minus
placebo with a 95% CI from a bootstrap over source jumps, share of responses in
the theory direction, and BH-FDR over pairs (q = 0.10).
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.panel.registry import is_same_release
from stg.structure.stats import benjamini_hochberg

OUT = "analysis/spillover_2026_09/out"
sys.path.insert(0, "analysis/spillover_2026_09")
from _tape import (D, H, HAWKISH, HORIZONS, LEGS, M15, RELEASE_EXCL, TAPE, TYPE,  # noqa: E402
                   asof, lead_contract, near_release, placebo_time, response)

THRESHOLDS = (5.0, 10.0, 15.0, 20.0)       # jump sizes studied, cents
COJUMP_C, GAP_MAX = 5.0, 2 * H
N_BOOT = 2000
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ jumps
def jumps_of(series, thresh):
    """Persistent ≥ ``thresh`` 15-min jumps of any threshold contract of
    ``series``, away from releases; one jump per bar, merged within 1 h."""
    per_bar = defaultdict(list)
    for tk, close in LEGS[series]:
        ts, px = TAPE[tk]
        if len(ts) < 3:
            continue
        b = (ts - ts[0]) // M15
        last = np.r_[np.nonzero(np.diff(b))[0], len(b) - 1]      # last print of each bar
        bt, bp = ts[last], px[last]
        bar_start = ts[0] + b[last] * M15
        for j in range(1, len(bt)):
            dp = bp[j] - bp[j - 1]
            if abs(dp) < thresh or bt[j] - bt[j - 1] > GAP_MAX or bt[j] > close - RELEASE_EXCL:
                continue
            k = np.searchsorted(ts, bt[j], side="right")
            if k < len(ts) and ts[k] - bt[j] <= H and np.sign(px[k] - bp[j]) == -np.sign(dp) \
                    and abs(px[k] - bp[j]) > abs(dp) / 2:
                continue                                          # reverted: bounce, not news
            per_bar[bar_start[j]].append((dp, bt[j]))
    out = []
    for t0 in sorted(per_bar):
        dps = [d for d, _ in per_bar[t0]]
        t_end = t0 + M15
        if near_release(t0) or (out and t0 - out[-1][0] <= H):
            continue
        direction = np.sign(np.mean(dps))
        if direction != 0:
            out.append((t0, t_end, direction, float(np.mean(np.abs(dps))), len(dps)))
    return out


rows = []
for thresh in THRESHOLDS:
    print(f"jump threshold {thresh:.0f}c:", flush=True)
    for A in LEGS:
        J = jumps_of(A, thresh)
        print(f"  {A:<14} {len(J):>5} jumps", flush=True)
        for jid, (t0, t_end, dA, size, nleg) in enumerate(J):
            tp = placebo_time(t0, rng)                 # one placebo time per jump, shared by all targets
            for B in LEGS:
                if B == A:
                    continue
                s = dA * HAWKISH[A] * HAWKISH[B]
                for kind, win in (("real", (t0, t_end)), ("placebo", (tp, tp + M15) if tp else None)):
                    if win is None:
                        continue
                    u0, u1 = win
                    tb = lead_contract(B, u0)
                    if tb is None:
                        continue
                    same, after = response(tb, u0, u1)
                    rows.append(dict(thresh=thresh, A=A, B=B, jump=f"{thresh:.0f}:{A}#{jid}", t0=t0.item(),
                                     kind=kind, sign=s, size=size,
                                     same_release=is_same_release(A, B),
                                     channel=f"{TYPE[A]}→{TYPE[B]}",
                                     cojump=bool(np.isfinite(same) and abs(same) >= COJUMP_C),
                                     same=s * same if np.isfinite(same) else np.nan,
                                     **{h: s * v if np.isfinite(v) else np.nan for h, v in after.items()}))
R = pl.DataFrame(rows)
R.write_parquet(f"{OUT}/jumps.parquet")


# ------------------------------------------------------------------ summaries
def compare(d, h):
    """real − placebo mean of the signed response at horizon h (non-co-jump
    rows), bootstrap over source jumps; share of real responses > 0 among ≠ 0."""
    real = d.filter((pl.col("kind") == "real") & ~pl.col("cojump") & pl.col(h).is_finite())
    plac = d.filter((pl.col("kind") == "placebo") & pl.col(h).is_finite())
    if real.height < 10 or plac.height < 10:
        return None
    rj = real.group_by("jump").agg(pl.col(h).sum().alias("s"), pl.len().alias("n"))
    pj = plac.group_by("jump").agg(pl.col(h).sum().alias("s"), pl.len().alias("n"))
    jumps = sorted(set(rj["jump"]) | set(pj["jump"]))
    ri = dict(zip(rj["jump"], zip(rj["s"], rj["n"])))
    pi = dict(zip(pj["jump"], zip(pj["s"], pj["n"])))
    RS, RN = np.array([ri.get(j, (0, 0))[0] for j in jumps]), np.array([ri.get(j, (0, 0))[1] for j in jumps])
    PS, PN = np.array([pi.get(j, (0, 0))[0] for j in jumps]), np.array([pi.get(j, (0, 0))[1] for j in jumps])
    est = RS.sum() / RN.sum() - PS.sum() / PN.sum()
    bs = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(jumps), len(jumps))
        if RN[k].sum() and PN[k].sum():
            bs.append(RS[k].sum() / RN[k].sum() - PS[k].sum() / PN[k].sum())
    bs = np.array(bs)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    nz = real.filter(pl.col(h) != 0)[h].to_numpy()
    return dict(n=real.height, jumps=real["jump"].n_unique(), real=RS.sum() / RN.sum(),
                placebo=PS.sum() / PN.sum(), diff=est, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                p=max(p, 1 / N_BOOT), agree=float((nz > 0).mean()) if len(nz) else np.nan)


PAIRS = []
for thresh in THRESHOLDS:
    RT = R.filter(pl.col("thresh") == thresh)
    print("\n" + "#" * 100 + f"\nJUMP THRESHOLD {thresh:.0f}c\n" + "#" * 100)
    real_all = RT.filter(pl.col("kind") == "real")
    print(f"\nsource jumps: {real_all['jump'].n_unique()}; (jump, target) pairs with a target contract: "
          f"{real_all.height}; co-jumps (target moved ≥ 5c in the same bar): "
          f"{real_all['cojump'].mean():.1%}")
    print("\nALL PAIRS — theory-signed response of the target after the jump bar (cents), real vs placebo")
    print(f"{'horizon':8} {'n':>6} {'jumps':>6} {'real':>7} {'placebo':>8} {'real−placebo [95% CI]':>26} {'agree':>6}")
    for h in HORIZONS:
        c = compare(RT.filter(~pl.col("same_release")), h)
        if c:
            print(f"{h:8} {c['n']:>6} {c['jumps']:>6} {c['real']:>+7.3f} {c['placebo']:>+8.3f} "
                  f"{c['diff']:>+8.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] {c['agree']:>6.3f}")
    print("(same-release pairs excluded; they are listed in the channel table as their own rows)")

    for h in ("+4h", "+24h"):
        print(f"\nBY CHANNEL — horizon {h}")
        print(f"{'channel':26} {'n':>6} {'jumps':>6} {'real−placebo [95% CI]':>26} {'agree':>6} {'co-jump':>7}")
        for (ch, same_rel), d in sorted(RT.group_by("channel", "same_release"), key=lambda kv: str(kv[0])):
            c = compare(d, h)
            if c:
                cj = d.filter(pl.col("kind") == "real")["cojump"].mean()
                lab = ch + (" (same release)" if same_rel else "")
                print(f"{lab:26} {c['n']:>6} {c['jumps']:>6} {c['diff']:>+8.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] "
                      f"{c['agree']:>6.3f} {cj:>7.1%}")

    if thresh not in (5.0, 10.0):
        continue
    print("\nBY PAIR — horizon +24h, pairs with ≥ 30 responses; BH-FDR over the pairs shown (q = 0.10)")
    pairs = []
    for (a, b), d in sorted(RT.group_by("A", "B"), key=lambda kv: kv[0]):
        c = compare(d, "+24h")
        if c and c["n"] >= 30:
            pairs.append(dict(A=a, B=b, same_release=d["same_release"][0], **c))
    if not pairs:
        print("  no pair with ≥ 30 responses")
        continue
    P = pl.DataFrame(pairs).sort("p")
    P = P.with_columns(pl.Series("bh", benjamini_hochberg(P["p"].to_numpy(), 0.10)))
    print(f"{'pair':30} {'n':>5} {'jumps':>5} {'real−placebo [95% CI]':>26} {'p':>6} {'BH':>3} {'agree':>6}")
    for r in P.head(25).iter_rows(named=True):
        lab = f"{r['A']}→{r['B']}" + (" *" if r["same_release"] else "")
        print(f"{lab:30} {r['n']:>5} {r['jumps']:>5} {r['diff']:>+8.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] "
              f"{r['p']:>6.3f} {'yes' if r['bh'] else '':>3} {r['agree']:>6.3f}")
    print(f"... {P.height} pairs tested, {int(P['bh'].sum())} survive BH at q = 0.10 (* = same release)")
    PAIRS.append(P.with_columns(pl.lit(thresh).alias("thresh")))
pl.concat(PAIRS).write_parquet(f"{OUT}/jumps_pairs.parquet")
