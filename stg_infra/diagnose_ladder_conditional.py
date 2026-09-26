"""
diagnose_ladder_conditional.py

IS THE LADDER GAP A RANDOM WALK, OR IS alpha ~ 0.99 A MIXTURE ARTIFACT?

The project's ladder conclusion rests on alpha ~ 0.99: regressing gap(t+h)
on gap(t) gives a coefficient that barely shrinks, in every gap-magnitude
bucket, so the gap was called a random walk with no level to revert to.

diagnose_ladder_reversion.py bucketed by gap MAGNITUDE specifically to
defeat the dilution objection, and that was the right control for the
question it asked. But it never conditioned on the thing that most
obviously drives the coefficient: WHETHER EITHER LEG'S PRICE MOVED AT ALL.

That matters because of two numbers already measured elsewhere in this
project, which have never been put side by side:

  - ~90% of consecutive snapshot observations show NO price change. A pair
    where neither leg reprices has gap(t+h) == gap(t) EXACTLY. It
    contributes alpha = 1 by arithmetic, not by economics. It is not
    evidence about mean reversion; it is evidence that nothing happened.

  - diagnose_staleness.py found that when AT LEAST ONE LEG REPRICES,
    violation survival collapses to 33.5% at h=1 and 25.1% at h=12. Two
    thirds of violations resolve once the legs actually move.

Those two facts together predict exactly the pooled number we observed: a
large frozen majority pinning alpha at 1, averaged with a moving minority
that may revert strongly. 0.9 * 1.0 + 0.1 * (something well below 1) lands
near 0.99 without the gap being a random walk in any economically
meaningful sense.

The same stratification was already decisive once in this project. On the
NODE forecast, the pooled ratio was 1.010x -- apparently worse than
persistence -- while the 'moved' stratum was 0.990x, better than
persistence. 74% flat positions were hiding the result. Nobody applied it
to the ladder gap.

WHAT THIS MEASURES. Everything diagnose_ladder_reversion.py reported, but
split three ways:

  ALL     every pair, reproducing the published number as a check
  MOVED   at least one leg's price changed between t and t+h
  FROZEN  neither leg moved (gap is arithmetically identical; alpha must
          be exactly 1.000 and MSE exactly 0 -- if it is not, this script
          has a bug and nothing else in it should be believed)

READING IT. If MOVED shows alpha materially below 1 while ALL sits at 0.99,
the ladder gap mean-reverts whenever the market actually trades it, and the
"random walk / no level to revert to" conclusion is wrong as stated. The
honest revised claim would be that ladder violations persist because the
legs do not reprice, not because the gap has no restoring force -- a
limits-to-arbitrage finding rather than a no-signal finding.

If MOVED still shows alpha near 1, the original conclusion survives a
harder test than it has yet faced, and can be stated with more confidence
than before.

Either way the answer changes what the ladder chapter says. Read-only; no
model, no checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "examples"))

from model.chunking import build_split_ranges, chunk_ranges  # noqa: E402
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402
from model.train import true_prices  # noqa: E402

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]
MOVE_EPS = 0.005          # half a cent: below this a price is unchanged
BUCKETS = [(-9.99, -0.05, "gap <= -5c   (deep non-viol)"),
           (-0.05, -0.01, "-5c..-1c     (non-viol)"),
           (-0.01, 0.01, "-1c..+1c     (at parity)"),
           (0.01, 0.03, "+1c..+3c     (small viol)"),
           (0.03, 0.08, "+3c..+8c     (mid viol)"),
           (0.08, 9.99, "gap >= +8c   (large viol)")]


def _alpha(x, y):
    """alpha minimising sum (alpha*x - y)^2, plus that minimum as an MSE."""
    sxx = float((x * x).sum())
    if sxx <= 0 or not len(x):
        return float("nan"), float("nan")
    sxy = float((x * y).sum())
    syy = float((y * y).sum())
    return sxy / sxx, max((syy - sxy * sxy / sxx) / len(x), 0.0)


def _table(x, y, thr, title, n_ref):
    if not len(x):
        print(f"\n  {title}: no pairs")
        return
    a_all, mse_all = _alpha(x, y)
    pers = float(((x - y) ** 2).mean())
    print(f"\n  {title}   n={len(x):,} ({100*len(x)/max(n_ref,1):.1f}% of all pairs)")
    print(f"  POOLED alpha={a_all:.4f}   persistence MSE={pers:.6f}   "
          f"shrinkage MSE={mse_all:.6f}   ({mse_all/pers if pers>0 else float('nan'):.3f}x)")
    print(f"  {'bucket':<30} {'n':>9} {'mean gap t':>11} {'mean t+h':>10} "
          f"{'alpha':>8} {'persist':>10} {'shrink':>10} {'ratio':>7} {'P(viol)':>8}")
    print("  " + "-" * 100)
    for lo, hi, name in BUCKETS:
        sel = (x > lo) & (x <= hi)
        k = int(sel.sum())
        if k < 50:
            continue
        xb, yb = x[sel], y[sel]
        al, msh = _alpha(xb, yb)
        pb = float(((xb - yb) ** 2).mean())
        pv = float((yb > thr).float().mean())
        print(f"  {name:<30} {k:>9,} {float(xb.mean()):>11.4f} {float(yb.mean()):>10.4f} "
              f"{al:>8.4f} {pb:>10.6f} {msh:>10.6f} "
              f"{msh/pb if pb>0 else float('nan'):>7.3f} {100*pv:>7.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--chunk-len", type=int, default=168)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 12])
    ap.add_argument("--threshold", type=float, default=0.01)
    args = ap.parse_args()

    store = MonthlyBundleStore(build_month_paths(args.months or PILOT_MONTHS,
                                                 _REPO_ROOT / "cache"), verbose=False)
    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=args.chunk_len, min_chunk_len=4)
              if c.split == args.split]
    if not chunks:
        raise SystemExit(f"no '{args.split}' chunks")

    # per horizon: gap(t), gap(t+h), and whether either leg actually moved
    data = {h: [[], [], []] for h in args.horizons}
    for ci, c in enumerate(chunks):
        ct = store.materialize_chunk(c)
        mask, prices = ct["mask"], true_prices(ct["features"])
        lad = ct["adjacency_by_type"].get("ladder_monotonic")
        if lad is None:
            del ct
            continue
        T = prices.shape[0]
        for h in args.horizons:
            xs, ys, ms = data[h]
            for t in range(T - h):
                e = lad[t]
                if e.edge_index.numel() == 0:
                    continue
                a, b = e.edge_index[0], e.edge_index[1]
                ok = mask[t, a] & mask[t, b] & mask[t + h, a] & mask[t + h, b]
                if not bool(ok.any()):
                    continue
                a, b = a[ok], b[ok]
                pa_t, pb_t = prices[t, a], prices[t, b]
                pa_h, pb_h = prices[t + h, a], prices[t + h, b]
                xs.append(pb_t - pa_t)
                ys.append(pb_h - pa_h)
                ms.append(((pa_h - pa_t).abs() > MOVE_EPS) |
                          ((pb_h - pb_t).abs() > MOVE_EPS))
        del ct
        print(f"  chunk {ci + 1}/{len(chunks)} scanned", flush=True)

    for h in args.horizons:
        xs, ys, ms = data[h]
        if not xs:
            continue
        x, y, moved = torch.cat(xs), torch.cat(ys), torch.cat(ms)
        n = len(x)
        print("\n" + "=" * 106)
        print(f"LADDER GAP, CONDITIONED ON WHETHER THE LEGS MOVED -- "
              f"{args.split.upper()}, h={h} ({h*2}h)   n={n:,}")
        print("=" * 106)
        print(f"  pairs where at least one leg repriced: {int(moved.sum()):,} "
              f"({100*float(moved.float().mean()):.1f}%)")
        _table(x, y, args.threshold, "ALL PAIRS (reproduces the published number)", n)
        _table(x[moved], y[moved], args.threshold, "MOVED -- at least one leg repriced", n)
        _table(x[~moved], y[~moved], args.threshold,
               "FROZEN -- neither leg repriced (alpha MUST be 1.000)", n)

        # integrity check: frozen pairs are identical by arithmetic
        fz = ~moved
        if int(fz.sum()):
            drift = float((x[fz] - y[fz]).abs().max())
            ok = drift <= 2 * MOVE_EPS
            print(f"\n  integrity: max |gap(t) - gap(t+h)| among FROZEN pairs = {drift:.6f} "
                  f"-> {'OK' if ok else 'FAIL'}")
            if not ok:
                print("  FAIL means the 'frozen' test does not match how gap is computed.")
                print("  Fix that before reading anything above.")

    print("\n" + "=" * 106)
    print("HOW TO READ THIS")
    print("=" * 106)
    print("  FROZEN pairs carry NO information about mean reversion. Their alpha is 1")
    print("  because neither price moved, which is arithmetic, not economics. If they")
    print("  are the majority, the pooled alpha is mostly measuring how often Kalshi")
    print("  ladder legs go untraded at a new price -- not whether the gap reverts.")
    print()
    print("  So read the MOVED block as the real test:")
    print()
    print("    alpha well below 1 in MOVED  ->  the gap DOES revert once the market")
    print("      reprices, and 'the ladder gap is a random walk' is wrong as stated.")
    print("      The correct claim becomes: violations persist because the legs do not")
    print("      reprice, which is a LIMITS-TO-ARBITRAGE result, not a no-signal one.")
    print("      It also means the training metric was diluted, and the ladder loss")
    print("      should be restricted to pairs where a leg actually moved.")
    print()
    print("    alpha near 1 in MOVED too    ->  the original conclusion survives a test")
    print("      it had not previously faced, and can be stated more strongly.")
    print()
    print("  Either way, compare P(viol) in MOVED against ALL. diagnose_staleness.py")
    print("  already reported survival collapsing to 33.5% at h=1 once a leg reprices;")
    print("  this should reproduce something close to that, and if it does not, the two")
    print("  measurements disagree and that has to be resolved before either is used.")


if __name__ == "__main__":
    main()