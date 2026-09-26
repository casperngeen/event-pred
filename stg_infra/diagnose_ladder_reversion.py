"""
diagnose_ladder_reversion.py

IS THE LADDER GAP UNFORECASTABLE, OR IS THE METRIC DILUTED?

Training reported alpha = 0.995 for the ladder gap -- the best single
shrinkage coefficient barely shrinks -- and concluded the gap is a random
walk. That conclusion may be an artifact of how the coefficient was fitted.

82% of ladder pairs have gap <= $0.01: deep non-violations where the lower
rung is correctly priced above the upper one and nothing is happening. A
single alpha fitted over ALL pairs is dominated by that mass, and the MSE
is too. If LARGE violations revert while ordinary gaps drift, the pooled
alpha is ~1 and the pooled MSE says "random walk" no matter how strongly
the violations mean-revert.

There is already direct evidence the conditional structure exists:
diagnose_staleness.py found violation survival drops to 33.5% once either
leg reprices. Violations do close. A linear coefficient over the whole
distribution simply cannot express "big gaps revert, small ones do not" --
but a model can, which is the entire argument for using one.

WHAT THIS MEASURES, with no model and no checkpoint:

  for each bucket of gap(t), and each horizon h:
    - n, mean gap(t), mean gap(t+h)
    - alpha fitted WITHIN the bucket (closed form, see _alpha)
    - persistence MSE  (predict gap(t+h) = gap(t))
    - shrinkage MSE    (predict alpha * gap(t), alpha from this bucket)
    - P(still a violation at t+h)

READING IT. If alpha in the violation buckets is materially below 1 while
the pooled alpha is 0.995, the ladder signal is real and the training
metric was diluting it -- the fix is to weight the ladder loss toward
violations rather than to abandon the mechanism. If alpha stays near 1 in
every bucket, the gap genuinely is a random walk at this resolution and
the negative result stands.

Alpha is fitted on the split being scored, which makes the shrinkage
column an ORACLE and therefore optimistic. That is deliberate: it is the
strongest form of the "a scalar could do this" objection.
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

    # collect (gap_t, gap_t+h) for every horizon, across all chunks
    pairs = {h: [[], []] for h in args.horizons}
    for c in chunks:
        ct = store.materialize_chunk(c)
        mask, prices = ct["mask"], true_prices(ct["features"])
        lad = ct["adjacency_by_type"].get("ladder_monotonic")
        if lad is None:
            del ct
            continue
        T = prices.shape[0]
        for h in args.horizons:
            xs, ys = pairs[h]
            for t in range(T - h):
                e = lad[t]
                if e.edge_index.numel() == 0:
                    continue
                a, b = e.edge_index[0], e.edge_index[1]
                ok = mask[t, a] & mask[t, b] & mask[t + h, a] & mask[t + h, b]
                if not bool(ok.any()):
                    continue
                a, b = a[ok], b[ok]
                xs.append(prices[t, b] - prices[t, a])
                ys.append(prices[t + h, b] - prices[t + h, a])
        del ct
        print(f"  chunk scanned", flush=True)

    for h in args.horizons:
        xs, ys = pairs[h]
        if not xs:
            continue
        x = torch.cat(xs)
        y = torch.cat(ys)
        a_all, mse_all = _alpha(x, y)
        pers_all = float(((x - y) ** 2).mean())
        print("\n" + "=" * 104)
        print(f"LADDER GAP REVERSION -- {args.split.upper()} split, h={h} "
              f"({h * 2}h ahead)   n={len(x):,}")
        print(f"POOLED: alpha={a_all:.4f}  persistence MSE={pers_all:.6f}  "
              f"shrinkage MSE={mse_all:.6f}  ({mse_all/pers_all:.3f}x)")
        print("=" * 104)
        print(f"{'bucket':<30} {'n':>9} {'share':>7} {'mean gap t':>11} "
              f"{'mean t+h':>10} {'alpha':>8} {'persist':>10} {'shrink':>10} "
              f"{'ratio':>7} {'P(viol)':>8}")
        print("-" * 104)
        for lo, hi, name in BUCKETS:
            sel = (x > lo) & (x <= hi)
            k = int(sel.sum())
            if k < 50:
                continue
            xb, yb = x[sel], y[sel]
            al, msh = _alpha(xb, yb)
            pb = float(((xb - yb) ** 2).mean())
            pv = float((yb > args.threshold).float().mean())
            print(f"{name:<30} {k:>9,} {100*k/len(x):>6.1f}% {float(xb.mean()):>11.4f} "
                  f"{float(yb.mean()):>10.4f} {al:>8.4f} {pb:>10.6f} {msh:>10.6f} "
                  f"{msh/pb if pb>0 else float('nan'):>7.3f} {100*pv:>7.1f}%")
        print("-" * 104)

    print("\n" + "=" * 104)
    print("HOW TO READ THIS")
    print("=" * 104)
    print("  alpha << 1 in the violation buckets while the POOLED alpha is ~1 means the")
    print("  ladder gap DOES mean-revert where it matters, and the training metric was")
    print("  diluted by the 80%+ of pairs sitting at parity. The fix is to weight the")
    print("  ladder loss toward violations -- not to drop the mechanism.")
    print()
    print("  alpha ~ 1 in EVERY bucket means the gap is a random walk at this resolution")
    print("  and the negative result stands on its own terms.")
    print()
    print("  'mean t+h' against 'mean gap t' is the same story without the algebra:")
    print("  if large violations shrink toward zero, reversion is real.")


if __name__ == "__main__":
    main()