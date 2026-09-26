"""
diagnose_persistence.py

THE FEASIBILITY CHECK THAT DECIDES WHETHER FORECASTING IS WORTH BUILDING.

The results document established that selecting among OBSERVED arbitrage
opportunities is arithmetic, not prediction -- every term of the PnL is
visible at decision time except execution cost. The one remaining route to a
genuine prediction task is forecasting: predict gap_{t+k} from information
at t, which is NOT observable and so cannot be computed by any naive rule.

Before writing a forecasting head, one measurement decides whether that task
is worth defining at all:

    how long does a monotonicity violation actually last?

Two consequences, both decisive:

  1. FORECASTABILITY. If a violation is gone by the next 2-hourly snapshot,
     there is nothing to forecast -- the signal has no persistence for a
     model to learn. If it survives for hours, forecasting is live.

  2. MAKER VS TAKER. The measured-spread result is that trading every
     violation loses $15,700 at full spread, and that entire loss is the
     cost of CROSSING the spread as a taker. A resting limit order reverses
     the sign of that term -- but only if the violation is still there when
     the order fills. Persistence is therefore a precondition for the only
     mechanism that could make this strategy profitable at scale.

This script ALSO computes baseline 1 of the redesign: the persistence
predictor gap_{t+k} = gap_t. That is the number a trained forecaster has to
beat, and per correction 5.6 in the results document it is named and measured
BEFORE any model is trained, not after.

No model is loaded. Pure measurement over the cached graph.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "examples"))

from model.chunking import build_split_ranges, chunk_ranges  # noqa: E402
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402
from model.train import true_prices  # noqa: E402

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]
GAP_THRESHOLD = 0.01


def _median(v):
    if not v:
        return float("nan")
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val", "train"], default="test")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--chunk-len", type=int, default=84)
    ap.add_argument("--threshold", type=float, default=GAP_THRESHOLD)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 6, 12])
    args = ap.parse_args()

    months = args.months or PILOT_MONTHS
    store = MonthlyBundleStore(build_month_paths(months, _REPO_ROOT / "cache"), verbose=False)

    # Global timestamp index, so a pair can be followed ACROSS chunk
    # boundaries. Without this, every violation in the last k snapshots of a
    # chunk would be recorded as "did not persist" purely because the chunk
    # ended -- a censoring artifact that would bias every horizon downward.
    t_index = {ts: i for i, ts in enumerate(store.timestamps)}

    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=args.chunk_len, min_chunk_len=4)
              if c.split == args.split]
    if not chunks:
        raise SystemExit(f"No '{args.split}' chunks for months {months}.")

    # (ticker_a, ticker_b) -> {global_t: gap}.  gap = p_b - p_a, so a
    # violation is gap > 0. Negative values are recorded too: they are what a
    # forecaster must also get right, and they are how we tell "the pair is
    # still quoted but no longer mispriced" apart from "the pair is gone".
    series = defaultdict(dict)
    observed_at = defaultdict(set)

    for ci, c in enumerate(chunks):
        ct = store.materialize_chunk(c)
        mask = ct["mask"]
        prices = true_prices(ct["features"])
        ladder_adj = ct["adjacency_by_type"].get("ladder_monotonic")
        node_ids = ct.get("node_ids")
        if ladder_adj is None or not node_ids:
            del ct
            continue
        for t in range(prices.shape[0]):
            edges = ladder_adj[t]
            if edges.edge_index.numel() == 0:
                continue
            gt = t_index.get(ct["timestamps"][t])
            if gt is None:
                continue
            a_idx, b_idx = edges.edge_index[0], edges.edge_index[1]
            for k in range(a_idx.shape[0]):
                a, b = int(a_idx[k]), int(b_idx[k])
                if not (bool(mask[t, a]) and bool(mask[t, b])):
                    continue
                key = (str(node_ids[a]), str(node_ids[b]))
                series[key][gt] = float(prices[t, b]) - float(prices[t, a])
                observed_at[key].add(gt)
        del ct
        print(f"  chunk {ci + 1}/{len(chunks)} scanned", flush=True)

    violations = [(key, gt, g) for key, d in series.items()
                  for gt, g in d.items() if g > args.threshold]
    if not violations:
        raise SystemExit("No violations found.")

    print(f"\nladder pairs tracked: {len(series):,}")
    print(f"violation sightings (gap > ${args.threshold:.3f}): {len(violations):,}")
    print(f"snapshot spacing: {store.timestamps[1] - store.timestamps[0]} "
          f"(horizon h = h snapshots ahead)")

    print("\n" + "=" * 100)
    print(f"VIOLATION PERSISTENCE -- {args.split.upper()} split")
    print("=" * 100)
    print(f"{'h':>3} {'still quoted':>14} {'still violation':>17} "
          f"{'| of quoted':>12} {'med gap t':>11} {'med gap t+h':>12} {'persist MAE':>12}")
    print("-" * 100)

    for h in args.horizons:
        quoted = still = 0
        g0, gh, abs_err = [], [], []
        for key, gt, g in violations:
            nxt = series[key].get(gt + h)
            if nxt is None:
                continue
            quoted += 1
            g0.append(g)
            gh.append(nxt)
            abs_err.append(abs(nxt - g))          # persistence predictor: g_{t+h} = g_t
            if nxt > args.threshold:
                still += 1
        n_v = len(violations)
        if quoted == 0:
            print(f"{h:>3} {'0':>14} {'-':>17} {'-':>12} {'-':>11} {'-':>12} {'-':>12}")
            continue
        print(f"{h:>3} {quoted:>8,} {100.0 * quoted / n_v:>5.1f}% "
              f"{still:>10,} {100.0 * still / n_v:>5.1f}% "
              f"{100.0 * still / quoted:>11.1f}% "
              f"${_median(g0):>10.4f} ${_median(gh):>11.4f} "
              f"${sum(abs_err) / len(abs_err):>11.4f}")

    print("-" * 100)
    print("  'still quoted'    = both legs still have a price h snapshots later")
    print("  'still violation' = and the gap is STILL above threshold")
    print("  '| of quoted'     = survival CONDITIONAL on the pair still being quoted,")
    print("                      which separates mispricing decay from data dropout")
    print("  'persist MAE'     = error of the baseline gap_{t+h} = gap_t.")
    print("                      THIS IS THE NUMBER A TRAINED FORECASTER MUST BEAT.")

    # Half-life, read off the conditional survival curve.
    print()
    surv = []
    for h in range(1, max(args.horizons) + 1):
        q = s = 0
        for key, gt, g in violations:
            nxt = series[key].get(gt + h)
            if nxt is None:
                continue
            q += 1
            if nxt > args.threshold:
                s += 1
        if q:
            surv.append((h, s / q))
    half = next((h for h, r in surv if r < 0.5), None)
    if half is None:
        print("  VERDICT: conditional survival never drops below 50% within the horizons")
        print("  tested. Violations are persistent -- forecasting is well posed, and a")
        print("  resting limit order has time to fill. The maker route is open.")
    elif half == 1:
        print("  VERDICT: more than half of violations are gone by the NEXT snapshot.")
        print("  These are transient. Forecasting them is poorly posed at this sampling")
        print("  rate, and a resting order would rarely fill. Report the mispricings as")
        print("  microstructure noise rather than a forecastable signal -- and note that")
        print("  a finer-grained graph (sub-2-hourly) would be needed to say more.")
    else:
        print(f"  VERDICT: conditional half-life is ~{half} snapshots. Violations persist")
        print("  long enough to forecast and long enough to rest an order against.")
        print("  Forecast horizons should sit at or inside that half-life.")


if __name__ == "__main__":
    main()