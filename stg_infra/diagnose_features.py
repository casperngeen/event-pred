"""
diagnose_features.py

Fast, read-only check of the per-feature value scales in the cached
month bundles. Does NOT train, build, or modify anything -- it just loads
cache/bundle_month_*.pt and reports each feature slot's range over the
positions that are actually observed (mask=True), which is the only place
real values live.

WHY: a neural network fed features whose magnitudes differ by orders of
magnitude learns almost nothing useful from the small ones. The large
feature dominates the first Linear layer's output, attention softmaxes
saturate, and the gradient signal for everything else is swamped. The
symptom is exactly what this project's 5-month run produced: loss drops
slightly in the first few epochs, then flatlines, with train and val both
stuck and val sitting near or below train (underfitting, not overfitting).

Per stg/nodes/kalshi.py, the ticker feature slots are expected to be:
    0 last_yes_price   (0-100 cents)      1 yes_vwap        (0-100)
    2 price_return     (-100..100)        3 price_std       (0..~50)
    4 window_volume    (0..thousands)     5 net_flow        (-1..1)
    6 buy_ratio        (0..1)             7 trade_intensity (trades/sec)
    8 time_to_close    (SECONDS)          9 is_mece_basket  (0/1 indicator)

Slot 8 in particular is seconds-to-close: a market 30 days out is
2,592,000, against slots 5/6 which live in [-1, 1]. Run this to see the
actual measured spread rather than reasoning about it from the docstring.

Run from stg_infra/:  python diagnose_features.py
"""
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))

SLOT_NAMES = [
    "last_yes_price", "yes_vwap", "price_return", "price_std", "window_volume",
    "net_flow", "buy_ratio", "trade_intensity", "time_to_close", "is_mece_basket",
]


# (slot, human name, low, high) -- physically possible range per
# stg/nodes/kalshi.py's documented feature semantics. A value outside
# these is a DATA BUG, not an unusual market.
VALID_RANGES = [
    (0, "last_yes_price", 0.0, 100.0),      # a YES price is 0-100 cents, full stop
    (1, "yes_vwap", 0.0, 100.0),            # a volume-weighted average of prices in 0-100
    (2, "price_return", -100.0, 100.0),     # difference of two values each in 0-100
    (3, "price_std", 0.0, 100.0),           # a standard deviation is non-negative
    (4, "window_volume", 0.0, float("inf")),  # contracts traded cannot be negative
    (5, "net_flow", -1.0, 1.0),             # documented as a normalised signed ratio
    (6, "buy_ratio", 0.0, 1.0),             # documented as a ratio
    (7, "trade_intensity", 0.0, float("inf")),  # trades per second, non-negative
    (8, "time_to_close", 0.0, 2 * 365 * 24 * 3600.0),  # 2 years: generous for event contracts
]


def _report_validity(totals: torch.Tensor):
    """Counts values that are physically impossible rather than merely
    extreme. This is a separate question from feature SCALING: rescaling
    a corrupt value just produces a well-scaled corrupt value.

    The price slot matters most. true_prices() feeds slot 0 / 100 to the
    loss as the regression TARGET, while MeceOutputHead (softmax) and
    LadderOutputHead (sigmoid) are both bounded to [0, 1] BY
    CONSTRUCTION. A price above 100 therefore becomes a target above 1.0
    that the model cannot reach at any parameter setting -- it
    contributes irreducible loss forever and drags the reported MSE
    regardless of how well the model actually learns.
    """
    print("\n" + "=" * 84)
    print("DATA VALIDITY -- values that are physically impossible, not merely extreme")
    print("=" * 84)
    n = totals.shape[0]
    print(f"{'slot':>4} {'name':<17} {'valid range':>22} {'violations':>12} {'% of rows':>11}")
    print("-" * 84)
    any_bad = False
    for slot, name, lo, hi in VALID_RANGES:
        if slot >= totals.shape[1]:
            continue
        col = totals[:, slot]
        bad = int(((col < lo) | (col > hi)).sum())
        if bad:
            any_bad = True
        hi_s = "inf" if hi == float("inf") else f"{hi:,.0f}"
        print(f"{slot:>4} {name:<17} {f'[{lo:,.0f}, {hi_s}]':>22} {bad:>12,} {100 * bad / n:>10.4f}%")

    # The one that directly breaks the training objective.
    price = totals[:, 0] / 100.0
    unreachable = int(((price < 0.0) | (price > 1.0)).sum())
    print("-" * 84)
    print(f"TARGETS OUTSIDE [0,1] after true_prices() (slot 0 / 100): "
          f"{unreachable:,} ({100 * unreachable / n:.4f}% of observations)")
    if unreachable:
        worst = float(price.max())
        print(f"  worst target value: {worst:.3f}  (heads can only ever output <= 1.0)")
        print(f"  --> these positions contribute irreducible loss no matter what the model learns.")
    if not any_bad:
        print("\n--> No impossible values found; the data is clean on these checks.")


def main():
    cache_dir = _REPO_ROOT / "cache"
    paths = sorted(cache_dir.glob("bundle_month_*.pt"))
    if not paths:
        print(f"No cached month bundles found in {cache_dir}.")
        print("Run run_real_training.py at least once first (it writes them).")
        return

    print(f"Found {len(paths)} cached month bundle(s) in {cache_dir}\n")

    totals = None
    for p in paths:
        bundle = torch.load(p, weights_only=False)
        feats, mask = bundle.features, bundle.mask
        active = feats[mask]  # (n_observed, F) -- real observations only, no padding
        if totals is None:
            totals = active
        else:
            totals = torch.cat([totals, active], dim=0)
        print(f"  {p.name}: T={feats.shape[0]} N={feats.shape[1]} "
              f"observed={active.shape[0]:,}")
        del bundle, feats, mask, active

    F = totals.shape[1]
    print(f"\nPooled over {totals.shape[0]:,} real observations across all months.\n")
    print(f"{'slot':>4} {'name':<17} {'min':>14} {'max':>16} {'mean':>14} {'std':>14}")
    print("-" * 84)
    stds = []
    for i in range(F):
        col = totals[:, i]
        name = SLOT_NAMES[i] if i < len(SLOT_NAMES) else f"slot{i}"
        print(f"{i:>4} {name:<17} {col.min().item():>14.3f} {col.max().item():>16.3f} "
              f"{col.mean().item():>14.3f} {col.std().item():>14.3f}")
        stds.append(max(col.std().item(), 1e-12))

    # The actual number that matters: how far apart are the feature scales?
    biggest, smallest = max(stds), min(stds)
    print("\n" + "=" * 84)
    print(f"Largest feature std:  {biggest:,.4f}")
    print(f"Smallest feature std: {smallest:,.6f}")
    print(f"RATIO: {biggest / smallest:,.0f}x")
    print("=" * 84)
    _report_validity(totals)

    if biggest / smallest > 100:
        print(
            "\n--> VERDICT: feature scales differ by far more than a network can absorb.\n"
            "    The largest-magnitude feature dominates the input projection, so the\n"
            "    embedding mostly encodes that one feature and the rest contribute\n"
            "    almost nothing. This is consistent with a loss that flatlines well\n"
            "    above a useful level while train and val stay close together.\n"
            "    FIX: standardize features (per-slot mean/std computed on the TRAIN\n"
            "    split only) before the projection layer."
        )
    else:
        print("\n--> Feature scales are within a reasonable range; look elsewhere for the\n"
              "    cause of the flat loss (capacity, learning rate, or the task's own floor).")


if __name__ == "__main__":
    main()