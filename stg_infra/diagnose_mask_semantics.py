"""
diagnose_mask_semantics.py

WHAT DOES mask=True ACTUALLY MEAN?

The liveness filter added to train_forecast.py was a no-op: mechanism
counts were identical with and without it, which can only happen if
window_volume > 0 wherever mask is True. That fact matters far beyond the
filter, because the project's headline staleness result rests on the
opposite assumption.

  If mask means "this leg has a last_yes_price, possibly carried forward
  from an old trade", then 90% of consecutive observations showing no
  price change means the prints are STALE -- the reading this project has
  been using.

  If mask means "this leg TRADED in this window", then the same 90% means
  legs are trading twice, two hours apart, at the SAME PRICE. That is a
  sticky price in an actively trading market, not a data artifact, and the
  staleness interpretation has to be rewritten.

These are opposite conclusions from the same number, so this script reads
the actual feature values rather than inferring.

Reports, over masked TICKER positions only:
  - the distribution of window_volume (slot 4) and trade_intensity (slot 7)
  - what fraction have volume exactly 0
  - the same for the subset where the price did NOT change from t-1,
    which is the population the staleness claim is about

Read-only. No model, no training.
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
SLOTS = {0: "last_yes_price", 1: "yes_vwap", 2: "price_return", 3: "price_std",
         4: "window_volume", 5: "net_flow", 6: "buy_ratio", 7: "trade_intensity",
         8: "time_to_close"}


def _q(v, p):
    if not len(v):
        return float("nan")
    s, n = v.sort().values, len(v)
    return float(s[min(n - 1, max(0, int(p * n)))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--chunk-len", type=int, default=168)
    ap.add_argument("--max-chunks", type=int, default=2)
    args = ap.parse_args()

    store = MonthlyBundleStore(build_month_paths(args.months or PILOT_MONTHS,
                                                 _REPO_ROOT / "cache"), verbose=False)
    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=args.chunk_len, min_chunk_len=4)
              if c.split == args.split][:args.max_chunks]
    if not chunks:
        raise SystemExit(f"no '{args.split}' chunks")

    vals = {s: [] for s in SLOTS}
    frozen_vals = {s: [] for s in SLOTS}
    n_masked = n_frozen = 0

    for c in chunks:
        ct = store.materialize_chunk(c)
        f, m = ct["features"], ct["mask"]
        prices = true_prices(f)
        is_t = f[..., -1] == 0.0
        sel = m & is_t
        n_masked += int(sel.sum())
        # positions whose price is unchanged from the previous snapshot AND
        # observed at both -- the exact population the staleness claim covers
        froz = torch.zeros_like(sel)
        froz[1:] = sel[1:] & sel[:-1] & ((prices[1:] - prices[:-1]).abs() <= 0.005)
        n_frozen += int(froz.sum())
        for s in SLOTS:
            if f.shape[-1] > s:
                vals[s].append(f[..., s][sel])
                frozen_vals[s].append(f[..., s][froz])
        del ct

    print(f"split={args.split}  chunks={len(chunks)}  "
          f"masked ticker positions={n_masked:,}  of which price-unchanged "
          f"vs previous snapshot={n_frozen:,} ({100*n_frozen/max(n_masked,1):.1f}%)\n")

    print("=" * 92)
    print("FEATURE VALUES ON MASKED TICKER POSITIONS")
    print("=" * 92)
    print(f"{'slot':<5} {'name':<16} {'zero%':>8} {'p25':>12} {'median':>12} "
          f"{'p75':>12} {'max':>12}")
    print("-" * 92)
    for s, name in SLOTS.items():
        if not vals[s]:
            continue
        v = torch.cat(vals[s])
        z = float((v == 0).float().mean()) * 100
        print(f"{s:<5} {name:<16} {z:>7.1f}% {_q(v,0.25):>12.4f} {_q(v,0.50):>12.4f} "
              f"{_q(v,0.75):>12.4f} {float(v.max()):>12.4f}")

    print("\n" + "=" * 92)
    print("SAME, RESTRICTED TO POSITIONS WHOSE PRICE DID NOT CHANGE")
    print("(the population behind the '90% unchanged' staleness result)")
    print("=" * 92)
    print(f"{'slot':<5} {'name':<16} {'zero%':>8} {'p25':>12} {'median':>12} "
          f"{'p75':>12} {'max':>12}")
    print("-" * 92)
    for s, name in SLOTS.items():
        if not frozen_vals[s]:
            continue
        v = torch.cat(frozen_vals[s])
        if not len(v):
            continue
        z = float((v == 0).float().mean()) * 100
        print(f"{s:<5} {name:<16} {z:>7.1f}% {_q(v,0.25):>12.4f} {_q(v,0.50):>12.4f} "
              f"{_q(v,0.75):>12.4f} {float(v.max()):>12.4f}")

    print("\n" + "-" * 92)
    vol = torch.cat(vals[4]) if vals[4] else torch.tensor([])
    fvol = torch.cat(frozen_vals[4]) if frozen_vals[4] else torch.tensor([])
    zero_all = float((vol == 0).float().mean()) if len(vol) else float("nan")
    zero_frz = float((fvol == 0).float().mean()) if len(fvol) else float("nan")
    print(f"  window_volume == 0 on all masked positions : {100*zero_all:.1f}%")
    print(f"  window_volume == 0 on unchanged-price ones : {100*zero_frz:.1f}%")
    print()
    if zero_all < 0.01:
        print("  VERDICT: mask IMPLIES a trade. Every observed leg traded in its window,")
        print("  so a liveness filter is vacuous -- and the '90% unchanged' result does")
        print("  NOT mean stale prints. It means legs trade repeatedly at the SAME price:")
        print("  a sticky price under active trading. The staleness sections of")
        print("  claude/stgat-results-and-limitations.md need rewriting, and the")
        print("  limits-to-arbitrage story changes from 'the data is stale' to 'the")
        print("  market genuinely quotes through these violations'.")
    elif zero_frz > 0.5:
        print("  VERDICT: unchanged prices are mostly ZERO-VOLUME carry-forwards. The")
        print("  staleness reading stands, and the liveness filter should have fired --")
        print("  check that VOLUME_SLOT matches this table before trusting the run.")
    else:
        print("  VERDICT: MIXED. Some unchanged prices carry volume, some do not.")
        print("  Split them and report separately; neither single story is right.")
    print("-" * 92)


if __name__ == "__main__":
    main()