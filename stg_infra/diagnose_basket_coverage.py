"""
diagnose_basket_coverage.py

HOW MUCH OF THE MECE MARKET CAN A NAIVE STAT-ARB TRADER ACTUALLY SEE?

This answers the objection that a model is redundant because "traders can
just run static arbitrage in real time".

They can -- but only on baskets where EVERY leg has a usable current price.
The basket sum is a sum over all N legs; if two of five legs have not
traded, there is no sum to compute, no deviation to measure, and nothing to
trade. Those baskets are invisible to the naive rule no matter how fast it
runs.

That invisible population is exactly what this project's Direction B was
aimed at: price the thin legs from their liquid neighbours so that
under-covered baskets become analysable at all.

So the model's territory is not "baskets the naive rule trades badly". It
is "baskets the naive rule cannot see". This script measures how big that
territory is.

WHAT IT REPORTS, per basket-snapshot:

  - coverage = legs observed now / legs the basket should have
  - the share at FULL coverage      -> the naive trader's whole universe
  - the share PARTIALLY covered     -> model-only territory
  - the share with too little to work with at all
  - how many legs are missing, among the partial ones

HOW TO READ IT. If full coverage is rare, a real-time static arbitrageur is
operating on a small slice of the market and the model is addressing the
rest -- which is the justification for building one. If full coverage is
the norm, the model's addressable population is small and the "just run
stat arb" objection largely stands.

This is a genuine question with an unknown answer, not a rhetorical one,
and the number should be reported whichever way it falls.

IMPORTANT CAVEAT, which must be quoted alongside any favourable number.
Reaching the invisible population is necessary but not sufficient. The
coverage-extension evaluation in the results doc (§4) found that on
partially-covered baskets the model's own estimation error was 1.49x the
size of the signal it was trying to detect, so a deviation computed from
imputed legs books the model's error as profit. Being able to SEE a basket
is not the same as being able to PRICE it.

Read-only. No model, no checkpoint.
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

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]
SLOT_LEGS_TOTAL = 3          # basket hub feature slots, stg/nodes/kalshi.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--chunk-len", type=int, default=168)
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = _REPO_ROOT / cache
    store = MonthlyBundleStore(build_month_paths(args.months or PILOT_MONTHS, cache),
                               verbose=False)
    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=args.chunk_len, min_chunk_len=4)
              if c.split == args.split]
    if not chunks:
        raise SystemExit(f"no '{args.split}' chunks")

    n_seen = []          # legs with a price right now
    n_total = []         # legs the basket should have
    legs_by_size = {}    # legs_total -> [coverage, ...]

    for ci, c in enumerate(chunks):
        ct = store.materialize_chunk(c)
        features, mask = ct["features"], ct["mask"]
        mece_adj = ct["adjacency_by_type"].get("mece_leg_to_basket")
        if mece_adj is None:
            del ct
            continue
        T = features.shape[0]
        for t in range(T):
            e = mece_adj[t]
            if e.edge_index.numel() == 0:
                continue
            leg, hub = e.edge_index[0], e.edge_index[1]
            for h in torch.unique(hub).tolist():
                sel = (hub == h).nonzero(as_tuple=True)[0]
                legs = leg[sel]
                if legs.numel() == 0:
                    continue
                total = float(features[t, h, SLOT_LEGS_TOTAL])
                if total <= 0:
                    continue
                seen = int(mask[t, legs].sum())
                n_seen.append(seen)
                n_total.append(total)
                legs_by_size.setdefault(int(round(total)), []).append(seen / total)
        del ct
        print(f"  chunk {ci + 1}/{len(chunks)} scanned "
              f"({len(n_seen):,} basket-snapshots)", flush=True)

    if not n_seen:
        raise SystemExit("no MECE baskets found on this split")

    seen = torch.tensor(n_seen, dtype=torch.float)
    tot = torch.tensor(n_total, dtype=torch.float)
    cov = seen / tot
    n = cov.numel()

    full = int((cov >= 0.999).sum())
    partial = int(((cov < 0.999) & (cov >= 0.5)).sum())
    sparse = int((cov < 0.5).sum())

    print("\n" + "=" * 92)
    print(f"MECE BASKET COVERAGE -- {args.split.upper()} split   "
          f"{n:,} basket-snapshots")
    print("=" * 92)
    print(f"  {'population':<44} {'count':>12} {'share':>9}")
    print("  " + "-" * 68)
    print(f"  {'FULL coverage -- naive stat arb can act':<44} {full:>12,} "
          f"{100*full/n:>8.1f}%")
    print(f"  {'PARTIAL (50-99%) -- invisible to naive rule':<44} {partial:>12,} "
          f"{100*partial/n:>8.1f}%")
    print(f"  {'SPARSE (<50%) -- too thin for anything':<44} {sparse:>12,} "
          f"{100*sparse/n:>8.1f}%")
    print("  " + "-" * 68)
    print(f"  {'model-addressable but naive-invisible':<44} {partial:>12,} "
          f"{100*partial/n:>8.1f}%")

    print(f"\n  mean coverage {float(cov.mean()):.3f}   "
          f"median {float(cov.median()):.3f}   "
          f"mean legs/basket {float(tot.mean()):.2f}")

    # Missing-leg count among the partial ones -- one missing leg is a very
    # different imputation problem from four.
    miss = (tot - seen)[(cov < 0.999) & (cov >= 0.5)]
    if miss.numel():
        print("\n  among PARTIAL baskets, legs missing:")
        for k in (1, 2, 3):
            c_ = int((miss.round() == k).sum())
            print(f"    exactly {k}: {c_:>10,}  ({100*c_/miss.numel():>5.1f}% of partial)")
        c_ = int((miss.round() >= 4).sum())
        print(f"    4 or more: {c_:>8,}  ({100*c_/miss.numel():>5.1f}% of partial)")

    print("\n  coverage by basket size:")
    print(f"    {'legs':>6} {'snapshots':>12} {'mean coverage':>15} {'% at full':>11}")
    for k in sorted(legs_by_size):
        v = torch.tensor(legs_by_size[k])
        if v.numel() < 50:
            continue
        print(f"    {k:>6} {v.numel():>12,} {float(v.mean()):>15.3f} "
              f"{100*float((v >= 0.999).float().mean()):>10.1f}%")

    print("\n" + "=" * 92)
    print("HOW TO READ THIS")
    print("=" * 92)
    print("  The FULL row is the entire universe available to a real-time static")
    print("  arbitrageur. Without a price for every leg there is no basket sum, so")
    print("  there is no deviation to see and nothing to trade -- however fast they")
    print("  are. The PARTIAL row is what a model that prices thin legs from their")
    print("  neighbours would add.")
    print()
    print("  A large PARTIAL share is the justification for building the model at")
    print("  all: it is not competing with the naive rule, it is reaching a")
    print("  population the naive rule cannot reach.")
    print()
    print("  A small PARTIAL share means the naive rule already sees most of the")
    print("  market and the 'just run stat arb' objection largely stands. Report")
    print("  the number either way.")
    print()
    print("  EITHER WAY, quote this next to it: the coverage-extension evaluation")
    print("  found the model's estimation error on partially-covered baskets was")
    print("  1.49x the signal being detected. Seeing a basket is necessary but not")
    print("  sufficient -- pricing it accurately enough to trade is a further step")
    print("  this model has not yet cleared.")


if __name__ == "__main__":
    main()